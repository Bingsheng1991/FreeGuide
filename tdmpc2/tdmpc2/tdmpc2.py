import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict


class AdaptiveBeta:
	"""Adaptive beta controller for FreeGuide epistemic bonus scaling."""

	def __init__(self, cfg):
		fg = cfg.freeguide
		self.beta = fg['beta_init']
		self.beta_min = fg['beta_min']
		self.beta_max = fg['beta_max']
		self.lr = fg['beta_lr']
		self.rho = fg['rho']
		self.ema = None
		self.target = None
		self._step = 0
		self.calibration_steps = fg['calibration_steps']
		self.enabled = fg['use_adaptive_beta']

	def update(self, ig_mean):
		if not self.enabled:
			return self.beta
		self._step += 1
		self.ema = ig_mean if self.ema is None else 0.99 * self.ema + 0.01 * ig_mean
		if self._step >= self.calibration_steps and self.target is None:
			self.target = self.rho * self.ema
		if self.target is not None:
			self.beta -= self.lr * (self.ema - self.target)
			self.beta = max(self.beta_min, min(self.beta_max, self.beta))
		return self.beta


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self._freeguide_cfg = cfg.freeguide if hasattr(cfg, 'freeguide') else cfg.get('freeguide', {'enabled': False})
		self._freeguide_enabled = self._freeguide_cfg.get('enabled', False)
		self._rnd_cfg = cfg.rnd if hasattr(cfg, 'rnd') else cfg.get('rnd', {'enabled': False})
		self._rnd_enabled = self._rnd_cfg.get('enabled', False)
		# Mutual exclusivity check
		if self._freeguide_enabled and self._rnd_enabled:
			raise ValueError('FreeGuide and RND cannot both be enabled. Set one to false.')

		# Build optimizer param groups
		optim_groups = [
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
		]
		self.optim = torch.optim.Adam(optim_groups, lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)

		# FreeGuide: separate optimizer for ensemble dynamics heads
		if self._freeguide_enabled:
			self.ensemble_optim = torch.optim.Adam(
				self.model._dynamics_ensemble.parameters(), lr=self.cfg.lr
			)
			self.adaptive_beta = AdaptiveBeta(cfg)
			# Running normalization stats for info gain
			self._ig_mean = 0.0
			self._ig_std = 1.0
			self._ig_count = 0
			# Logging accumulators
			self._fg_log = {
				'info_gain_edd': 0.0,
				'info_gain_qev': 0.0,
				'info_gain_normalized': 0.0,
				'ensemble_loss': 0.0,
				'beta': self.adaptive_beta.beta,
				'ig_running_mean': 0.0,
				'ig_running_std': 1.0,
			}

		# RND: separate optimizer for predictor network + running normalization stats
		if self._rnd_enabled:
			self.rnd_optim = torch.optim.Adam(
				self.model._rnd_predictor.parameters(), lr=self.cfg.lr
			)
			# Running normalization for RND bonus
			self._rnd_bonus_mean = 0.0
			self._rnd_bonus_std = 1.0
			# Logging accumulators
			self._rnd_log = {
				'rnd_bonus_raw': 0.0,
				'rnd_bonus_normalized': 0.0,
				'rnd_predictor_loss': 0.0,
				'rnd_bonus_running_mean': 0.0,
				'rnd_bonus_running_std': 1.0,
			}

		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		if self._freeguide_enabled:
			print(f'FreeGuide enabled: K={self._freeguide_cfg["ensemble_K"]}, '
			      f'alpha={self._freeguide_cfg["alpha"]}, '
			      f'beta_init={self._freeguide_cfg["beta_init"]}')
		if self._rnd_enabled:
			print(f'RND enabled: bonus_coef={self._rnd_cfg["bonus_coef"]}')
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _estimate_value_freeguide(self, z, actions, task):
		"""Estimate value with FreeGuide epistemic bonus.

		Uses proportional scaling: the epistemic term is scaled to be a fixed
		fraction (beta) of the extrinsic term's magnitude. This ensures the
		bonus is task-agnostic regardless of observation/action dimensionality.
		"""
		fg = self._freeguide_cfg
		G_extrinsic, G_epistemic = 0, 0
		discount = 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)

		total_edd = 0.0
		total_qev = 0.0

		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)

			# Use ensemble for next state prediction + disagreement
			z_next, edd = self.model.ensemble_dynamics(z, actions[t], task)

			# Q-value ensemble variance (epistemic uncertainty in value)
			qev = torch.zeros(self.cfg.num_samples, device=z.device)
			if fg['use_qev']:
				q_all = self.model.Q(z, actions[t], task, return_type='all')  # [num_q, B, num_bins]
				q_vals = math.two_hot_inv(q_all, self.cfg)  # [num_q, B, 1]
				qev = q_vals.squeeze(-1).var(0)  # [B]

			# Combine raw info gain
			ig_raw = torch.zeros(self.cfg.num_samples, device=z.device)
			if fg['use_edd']:
				ig_raw = ig_raw + edd
			if fg['use_qev']:
				ig_raw = ig_raw + fg['alpha'] * qev

			# Accumulate (no normalization yet — we scale after the loop)
			G_extrinsic = G_extrinsic + discount * (1 - termination) * reward
			survival = (1 - termination).squeeze(-1)
			G_epistemic = G_epistemic + discount * survival * ig_raw

			total_edd += edd.mean().item()
			total_qev += qev.mean().item()

			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			z = z_next  # Use ensemble mean for state transition (paper Algorithm 1, line 26)
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)

		# Terminal value
		action, _ = self.model.pi(z, task)
		terminal_value = self.model.Q(z, action, task, return_type='avg')
		G_extrinsic = G_extrinsic + discount * (1 - termination) * terminal_value

		# Update running stats (EMA) for logging purposes
		batch_mean = ig_raw.mean().item()
		batch_var = ig_raw.var().item()
		self._ig_mean = 0.99 * self._ig_mean + 0.01 * batch_mean
		self._ig_std = (0.99 * self._ig_std ** 2 + 0.01 * batch_var) ** 0.5
		self._ig_count += 1

		# Zero-mean + variance-matched scaling for epistemic term.
		# 1. Center epistemic to zero mean (prevents epistemic mean from
		#    shifting the ranking away from extrinsic signal)
		# 2. Scale epistemic std to match extrinsic std * beta
		#    (so MPPI differentiation is meaningful but bounded)
		# 3. When extrinsic std is tiny (early training), use a floor
		#    so epistemic can still drive exploration
		G_ext_flat = G_extrinsic.squeeze(-1)  # [B]
		epi_centered = G_epistemic - G_epistemic.mean()
		epi_std = G_epistemic.std() + 1e-8
		ext_std = G_ext_flat.std()
		# Floor: at minimum, epistemic std targets 1% of |ext_mean|
		target_std = torch.max(ext_std, G_ext_flat.abs().mean() * 0.01 + 1e-8)
		epi_scaled = epi_centered * (target_std / epi_std)

		beta = self.adaptive_beta.beta
		score = G_extrinsic + beta * epi_scaled.unsqueeze(-1)

		# Diagnostic logging
		self._diag_J_ext = G_ext_flat
		self._diag_J_epi = G_epistemic
		self._diag_beta = beta
		self._diag_ig_raw_mean = ig_raw.mean().item()
		self._diag_ig_norm_mean = epi_scaled.mean().item()
		self._diag_epi_std = G_epistemic.std().item()
		self._diag_ext_std = G_ext_flat.std().item()
		self._diag_target_std = target_std.item() if isinstance(target_std, torch.Tensor) else target_std

		# Update logging accumulators
		self._fg_log['info_gain_edd'] = total_edd / self.cfg.horizon
		self._fg_log['info_gain_qev'] = total_qev / self.cfg.horizon
		self._fg_log['info_gain_normalized'] = epi_scaled.mean().item()
		self._fg_log['beta'] = beta
		self._fg_log['ig_running_mean'] = self._ig_mean
		self._fg_log['ig_running_std'] = self._ig_std

		return score

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			if self._freeguide_enabled:
				value = self._estimate_value_freeguide(z, actions, task).nan_to_num(0)
			else:
				value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _update(self, obs, action, reward, terminated, task=None):
		# RND: compute exploration bonus and add to reward before targets
		rnd_predictor_loss = None
		if self._rnd_enabled:
			with torch.no_grad():
				# Encode observations to get latent states for RND
				z_for_rnd = self.model.encode(obs[:-1], task)  # [horizon, B, D]
				# Compute raw RND bonus per timestep
				rnd_bonus_raw = self.model.rnd_bonus(z_for_rnd)  # [horizon, B, 1]
				# Update running normalization (EMA)
				batch_mean = rnd_bonus_raw.mean().item()
				batch_std = rnd_bonus_raw.std().item() + 1e-8
				self._rnd_bonus_mean = 0.99 * self._rnd_bonus_mean + 0.01 * batch_mean
				self._rnd_bonus_std = (0.99 * self._rnd_bonus_std**2 + 0.01 * batch_std**2) ** 0.5
				# Normalize bonus
				rnd_bonus_norm = (rnd_bonus_raw - self._rnd_bonus_mean) / (self._rnd_bonus_std + 1e-8)
				# Add bonus to reward
				bonus_coef = self._rnd_cfg['bonus_coef']
				reward = reward + bonus_coef * rnd_bonus_norm
				# Update logging
				self._rnd_log['rnd_bonus_raw'] = batch_mean
				self._rnd_log['rnd_bonus_normalized'] = rnd_bonus_norm.mean().item()
				self._rnd_log['rnd_bonus_running_mean'] = self._rnd_bonus_mean
				self._rnd_log['rnd_bonus_running_std'] = self._rnd_bonus_std

		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# FreeGuide: train ensemble dynamics heads
		ensemble_loss_val = 0.0
		if self._freeguide_enabled:
			ensemble_loss_val = self._update_ensemble(obs, action, task)

		# RND: train predictor network
		if self._rnd_enabled:
			rnd_predictor_loss = self._update_rnd_predictor(obs, task)

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		if self._freeguide_enabled:
			info.update(TensorDict({
				"freeguide_ensemble_loss": torch.tensor(ensemble_loss_val),
			}))
		if self._rnd_enabled:
			info.update(TensorDict({
				"rnd_predictor_loss": torch.tensor(rnd_predictor_loss if rnd_predictor_loss is not None else 0.0),
			}))
		return info.detach().mean()

	def _update_ensemble(self, obs, action, task):
		"""Train ensemble dynamics heads with joint-embedding prediction loss."""
		with torch.no_grad():
			# Get target latent states from encoder
			z_targets = self.model.encode(obs[1:], task)  # [horizon, B, D]
			z0 = self.model.encode(obs[0], task)  # [B, D]

		# Train ensemble heads: predict next latent from (z, a)
		self.model._dynamics_ensemble.train()
		ensemble_loss = 0
		z = z0.detach()
		for t in range(self.cfg.horizon):
			a = action[t]
			target = z_targets[t].detach()
			# Predict with each ensemble member
			if self.cfg.multitask:
				z_inp = self.model.task_emb(z, task)
			else:
				z_inp = z
			za = torch.cat([z_inp, a], dim=-1)
			for head in self.model._dynamics_ensemble:
				pred = head(za)
				ensemble_loss = ensemble_loss + F.mse_loss(pred, target)
			# Use main dynamics for next step
			with torch.no_grad():
				z = self.model.next(z, a, task)

		K = self._freeguide_cfg['ensemble_K']
		ensemble_loss = ensemble_loss / (self.cfg.horizon * K)
		ensemble_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._dynamics_ensemble.parameters(), self.cfg.grad_clip_norm)
		self.ensemble_optim.step()
		self.ensemble_optim.zero_grad(set_to_none=True)
		self.model._dynamics_ensemble.eval()

		loss_val = ensemble_loss.item()
		self._fg_log['ensemble_loss'] = loss_val
		return loss_val

	def update_freeguide_beta(self):
		"""Update adaptive beta (call once per episode)."""
		if self._freeguide_enabled:
			ig_mean = self._fg_log.get('ig_running_mean', 0.0)
			self.adaptive_beta.update(ig_mean)
			self._fg_log['beta'] = self.adaptive_beta.beta

	def get_freeguide_metrics(self):
		"""Return FreeGuide metrics for logging."""
		if not self._freeguide_enabled:
			return {}
		return {
			'freeguide/beta': self._fg_log['beta'],
			'freeguide/info_gain_edd': self._fg_log['info_gain_edd'],
			'freeguide/info_gain_qev': self._fg_log['info_gain_qev'],
			'freeguide/info_gain_normalized': self._fg_log['info_gain_normalized'],
			'freeguide/ensemble_loss': self._fg_log['ensemble_loss'],
			'freeguide/ig_running_mean': self._fg_log['ig_running_mean'],
			'freeguide/ig_running_std': self._fg_log['ig_running_std'],
		}

	def _update_rnd_predictor(self, obs, task):
		"""Train RND predictor network to match fixed target network output."""
		with torch.no_grad():
			z = self.model.encode(obs[:-1], task)  # [horizon, B, D]
		# Compute predictor loss
		rnd_loss = self.model.rnd_loss(z)
		rnd_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._rnd_predictor.parameters(), self.cfg.grad_clip_norm)
		self.rnd_optim.step()
		self.rnd_optim.zero_grad(set_to_none=True)
		loss_val = rnd_loss.item()
		self._rnd_log['rnd_predictor_loss'] = loss_val
		return loss_val

	def get_rnd_metrics(self):
		"""Return RND metrics for logging."""
		if not self._rnd_enabled:
			return {}
		return {
			'rnd/bonus_raw': self._rnd_log['rnd_bonus_raw'],
			'rnd/bonus_normalized': self._rnd_log['rnd_bonus_normalized'],
			'rnd/predictor_loss': self._rnd_log['rnd_predictor_loss'],
			'rnd/bonus_running_mean': self._rnd_log['rnd_bonus_running_mean'],
			'rnd/bonus_running_std': self._rnd_log['rnd_bonus_running_std'],
		}

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)
