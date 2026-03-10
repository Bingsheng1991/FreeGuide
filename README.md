# FreeGuide

**Expected Free Energy Guided Exploration for Model-Based Reinforcement Learning**

FreeGuide injects an Expected Free Energy (EFE) scoring term into TD-MPC2's MPPI planner, using ensemble dynamics disagreement as an information gain proxy. This enables the agent to simultaneously pursue high reward and reduce world model uncertainty during planning.

## Key Ideas

- **Ensemble Dynamics Disagreement (EDD):** K independent dynamics heads predict next latent states; their variance estimates epistemic uncertainty.
- **Q-Value Ensemble Variance (QEV):** Variance across Q-function heads provides an additional epistemic signal in value space.
- **Variance-Matched Scaling:** The epistemic bonus is automatically scaled relative to the extrinsic reward signal, ensuring task-agnostic behavior across different observation/action dimensionalities.
- **Adaptive Beta:** A per-episode controller adjusts the exploration-exploitation trade-off based on an EMA of information gain.

## Project Structure

```
FreeGuide/
├── CLAUDE.md                     # Automated execution protocol
├── tdmpc2/                       # TD-MPC2 source with FreeGuide modifications
│   └── tdmpc2/
│       ├── tdmpc2.py             # FreeGuide MPPI scoring + adaptive beta
│       ├── common/
│       │   └── world_model.py    # Ensemble dynamics heads
│       ├── config.yaml           # Hydra config with freeguide block
│       └── trainer/
│           └── online_trainer.py # Training loop with FreeGuide hooks
├── analysis/                     # Plotting and analysis scripts
├── experiments/scripts/          # Experiment launch scripts
├── paper/                        # Paper draft and figures
└── logs/                         # Experiment logs
```

## Quick Start

```bash
# Setup
conda create -n freeguide python=3.10 -y
conda activate freeguide
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
cd tdmpc2 && pip install -e .
pip install dm_control mujoco hydra-core omegaconf matplotlib pandas scipy gymnasium==0.29.1

# Run baseline TD-MPC2
cd tdmpc2
python train.py task=walker-run steps=200000 freeguide.enabled=false enable_wandb=false wandb_project=freeguide

# Run FreeGuide
python train.py task=walker-run steps=200000 freeguide.enabled=true enable_wandb=false wandb_project=freeguide
```

## FreeGuide Config

```yaml
freeguide:
  enabled: true
  ensemble_K: 3          # Number of ensemble dynamics heads
  alpha: 0.5             # Weight for QEV relative to EDD
  use_edd: true           # Use ensemble dynamics disagreement
  use_qev: true           # Use Q-value ensemble variance
  use_adaptive_beta: true
  beta_init: 0.1          # Initial exploration weight
  beta_min: 0.0
  beta_max: 1.0
  beta_lr: 0.0001
  rho: 0.3
  calibration_steps: 10000
```

## How It Works

During MPPI planning, FreeGuide modifies trajectory scoring:

```
score = J_extrinsic + beta * scaled(J_epistemic)
```

Where:
- `J_extrinsic` = standard TD-MPC2 trajectory value (rewards + terminal Q-value)
- `J_epistemic` = accumulated information gain (EDD + alpha * QEV) along the trajectory
- `scaled()` = variance-matched normalization (zero-mean, std matched to extrinsic)
- `beta` = adaptive exploration weight

The epistemic bonus is computed under `torch.no_grad()` during planning only — it does not modify the training loss or reward signal.

## DMControl Tasks

Evaluated on 5 tasks: `cheetah-run`, `walker-run`, `quadruped-run`, `humanoid-run`, `dog-run`.

## References

- Hansen et al. (2024). TD-MPC2: Scalable, Robust World Models for Continuous Control. *ICLR 2024*.
- Friston (2009). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.
- Friston et al. (2017). Active inference and learning. *Neuroscience & Biobehavioral Reviews*.
- Lakshminarayanan et al. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS 2017*.

## License

This project builds on [TD-MPC2](https://github.com/nicklashansen/tdmpc2) (MIT License).
