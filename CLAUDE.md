# CLAUDE.md — FreeGuide 全自动执行协议

## 项目简介

FreeGuide 在 TD-MPC2 的 MPPI 规划器中注入 Expected Free Energy 评分，用 ensemble dynamics disagreement 近似 information gain，使 agent 在规划时同时追求高回报和减少世界模型的不确定性。

## 路径与环境

- **项目根目录**：`/home/miller/Desktop/FreeGuide`
- **Conda 环境名**：`freeguide`
- **Python 版本**：3.10
- **所有命令都必须先 `conda activate freeguide`**
- 本地 GPU：RTX 4090 24GB
- 服务器（可选）：3×A800 80GB

## 下载源（必须遵守）

所有 pip 和 conda 安装优先使用中国镜像源：

```bash
# pip 全局设置（Phase 0 中执行一次）
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com

# conda 添加清华源（Phase 0 中执行一次）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes

# PyTorch 安装使用官方 cu118 源（国内镜像不一定有最新版）
# 如果官方源太慢，备选：https://mirror.sjtu.edu.cn/pytorch-wheels/cu118
```

每次 `pip install` 和 `conda install` 时不需要手动加 `-i` 参数，全局配置已生效。如果某个包从阿里源下载失败，自动回退到 `https://pypi.tuna.tsinghua.edu.cn/simple/`。

## 自主执行协议

本项目设计为 Claude Code 全自动执行。收到启动指令后，按 Phase 0→1→2→3→4→5 顺序执行。

### 状态检查

每次开始工作前，先检查进度：
```bash
cd /home/miller/Desktop/FreeGuide
ls -la phase*.done 2>/dev/null
```
从最后一个完成的 phase 的下一个开始。如果没有任何 .done 文件，从 Phase 0 开始。

### Phase 完成标记

每个 Phase 完成后创建标记：
```bash
echo "$(date)" > /home/miller/Desktop/FreeGuide/phase{N}.done
```

### 错误处理

- 遇到报错时，先读 traceback，自行诊断修复，最多重试 3 次
- 如果 3 次都失败，把错误信息写入 `/home/miller/Desktop/FreeGuide/logs/error_phase{N}.log`，创建 `phase{N}.failed` 标记，跳到能继续的下一步
- OOM → 缩小 replay buffer 到 500K 或降低 ensemble_K 到 2
- NaN → 检查 normalization 除零保护，加 eps=1e-8

---

## 目录结构

```
/home/miller/Desktop/FreeGuide/
├── CLAUDE.md                    # 本文件
├── tdmpc2/                      # TD-MPC2 源码（clone 后修改）
├── experiments/
│   ├── scripts/                 # 实验启动脚本
│   └── results/                 # 实验输出
├── analysis/                    # 画图与分析脚本
├── paper/
│   ├── FreeGuide_Paper_Draft.md # 论文草稿（参考用）
│   ├── figures/
│   └── tables/
├── logs/                        # 日志
└── phase0.done ~ phase5.done    # 进度标记
```

---

## Phase 0：环境搭建

### 0.1 配置中国镜像源 + 创建 conda 环境

```bash
# 配置 conda 清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes

# 创建环境
conda create -n freeguide python=3.10 -y
conda activate freeguide

# 配置 pip 阿里源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com

# PyTorch（官方 cu118 源，国内镜像可能缺包）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 如果上面太慢，备选上海交大镜像：
# pip install torch torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu118
```

### 0.2 Clone 并安装 TD-MPC2

```bash
mkdir -p /home/miller/Desktop/FreeGuide
cd /home/miller/Desktop/FreeGuide
git clone https://github.com/nicklashansen/tdmpc2.git tdmpc2
cd tdmpc2
pip install -e .
pip install dm_control mujoco hydra-core omegaconf wandb matplotlib seaborn pandas scipy tensorboard
```

### 0.3 创建目录

```bash
cd /home/miller/Desktop/FreeGuide
mkdir -p experiments/scripts experiments/results analysis paper/figures paper/tables logs
```

### 0.4 验证 TD-MPC2

```bash
conda activate freeguide
cd /home/miller/Desktop/FreeGuide/tdmpc2
python -m tdmpc2.train task=walker-run steps=5000
```

必须无报错地跑完 5000 步。如果报错，排查修复后重试。

### 0.5 代码结构侦察

阅读 TD-MPC2 源码，找到并记录以下关键位置（写入 `logs/code_map.txt`）：
- 世界模型类的定义文件和类名
- dynamics model 的 forward 方法位置
- Q-function ensemble 的定义和使用位置
- MPPI planning 的 trajectory scoring 代码位置（精确到函数名和行号范围）
- 训练主循环的位置（在哪里调 plan、在哪里算 loss、在哪里 update）
- Hydra config 的加载方式和默认 config 文件路径
- reward prediction 和 termination prediction（如果有）的位置

这些信息后续修改代码时要用。

### 0.6 标记完成

```bash
echo "$(date)" > /home/miller/Desktop/FreeGuide/phase0.done
```

---

## Phase 1：核心实现

> **重要**：所有修改都在 `/home/miller/Desktop/FreeGuide/tdmpc2/` 下进行。
> 修改前先 `git add -A && git commit -m "pre-freeguide baseline"` 保存原始代码。

### 1.1 Hydra Config 扩展

在 TD-MPC2 的 config 系统中添加 freeguide 配置块：

```yaml
freeguide:
  enabled: false
  ensemble_K: 3
  alpha: 0.5
  use_edd: true
  use_qev: true
  use_adaptive_beta: true
  beta_init: 0.1
  beta_min: 0.0
  beta_max: 1.0
  beta_lr: 0.0001
  rho: 0.3
  calibration_steps: 10000
```

确保 `freeguide.enabled=false` 时代码行为与原版 TD-MPC2 完全一致。

### 1.2 Ensemble Dynamics Heads

在世界模型类中添加：

```python
# __init__ 中（使用 NormedLinear + SimNorm，与主 dynamics 架构一致）：
if cfg.freeguide.enabled:
    K = cfg.freeguide.ensemble_K
    self._dynamics_ensemble = nn.ModuleList([
        nn.Sequential(
            layers.NormedLinear(cfg.latent_dim + cfg.action_dim + cfg.task_dim, cfg.mlp_dim),
            layers.NormedLinear(cfg.mlp_dim, cfg.mlp_dim),
            layers.NormedLinear(cfg.mlp_dim, cfg.latent_dim, act=layers.SimNorm(cfg))
        ) for _ in range(K)
    ])

# 新方法：
def ensemble_dynamics(self, z, a, task):
    if self.cfg.multitask:
        z_inp = self.task_emb(z, task)
    else:
        z_inp = z
    za = torch.cat([z_inp, a], dim=-1)
    preds = torch.stack([h(za) for h in self._dynamics_ensemble], dim=0)  # [K,B,D]
    mean_pred = preds.mean(0)       # [B,D]
    disagree = ((preds - mean_pred.unsqueeze(0))**2).mean(dim=(0,2))  # [B]
    return mean_pred, disagree
```

在训练 loss 中添加 ensemble heads 的 joint-embedding prediction loss（与主 dynamics 相同的 loss 类型）。

### 1.3 FreeGuide MPPI Scoring

修改 MPPI planning 函数。核心逻辑：

```python
# 在 trajectory scoring 中（torch.no_grad() 下）：

if freeguide_enabled:
    # Running normalization 统计量
    # 在 planning 类的 __init__ 中初始化：
    #   self._ig_mean = 0.0
    #   self._ig_std = 1.0
    #   self._ig_count = 0
    
    for i in range(horizon):
        z_next, edd = model.ensemble_dynamics(z, a_i)   # 用 ensemble mean 做 rollout
        qev = torch.stack([Q_k(z, a_i) for Q_k in Q_ensemble]).var(0)
        ig_raw = edd + alpha * qev
        ig_norm = (ig_raw - self._ig_mean) / (self._ig_std + 1e-8)
        J_epistemic += survival_weight * (gamma ** i) * ig_norm
        z = z_next
    
    score = J_extrinsic + beta * J_epistemic
    
    # 更新 running stats（EMA）
    batch_mean = ig_raw.mean().item()
    batch_var = ig_raw.var().item()
    self._ig_mean = 0.99 * self._ig_mean + 0.01 * batch_mean
    self._ig_std = (0.99 * self._ig_std**2 + 0.01 * batch_var) ** 0.5
else:
    score = J_extrinsic  # 原版 TD-MPC2
```

### 1.4 Adaptive Beta

```python
class AdaptiveBeta:
    def __init__(self, cfg):
        self.beta = cfg.freeguide.beta_init
        self.beta_min = cfg.freeguide.beta_min
        self.beta_max = cfg.freeguide.beta_max
        self.lr = cfg.freeguide.beta_lr
        self.rho = cfg.freeguide.rho
        self.ema = None
        self.target = None
        self._step = 0
        self.calibration_steps = cfg.freeguide.calibration_steps
        self.enabled = cfg.freeguide.use_adaptive_beta

    def update(self, ig_mean):
        if not self.enabled:
            return self.beta
        self._step += 1
        self.ema = ig_mean if self.ema is None else 0.99*self.ema + 0.01*ig_mean
        if self._step >= self.calibration_steps and self.target is None:
            self.target = self.rho * self.ema
        if self.target is not None:
            self.beta -= self.lr * (self.ema - self.target)
            self.beta = max(self.beta_min, min(self.beta_max, self.beta))
        return self.beta
```

在训练主循环中，每个 episode 结束后调用 `update()`。

### 1.5 Logging

确保 wandb（或 tensorboard/csv）中记录以下量：
- `episode_return`（每个 episode）
- `freeguide/beta`
- `freeguide/info_gain_edd`（raw，非归一化）
- `freeguide/info_gain_qev`（raw）
- `freeguide/info_gain_normalized`
- `freeguide/ensemble_loss`
- `freeguide/ig_running_mean`
- `freeguide/ig_running_std`
- `reward_loss`（世界模型 reward head 的训练 loss，**Fig 5 分析需要**）
- `elapsed_time`（累计训练时间，**Table 2 需要**）

**注意**：`reward_loss` 和 `elapsed_time` 对所有方法（TD-MPC2, +RND, FreeGuide）都要记录，不仅仅是 FreeGuide。

**Latent State Dump**（Fig 5b 需要）：
训练过程中每 100K 步自动保存一批 latent states 到 `{work_dir}/latent_states/latent_{step}.npz`。
文件包含 `z`（encoder 输出，shape `[B, latent_dim]`）和 `step`。
Phase 4 的 `plot_reward_vs_planning.py` 会用 `latent_500000.npz` 做 PCA 2D 投影对比 TD-MPC2 vs FreeGuide 的 latent coverage。

### 1.6 RND Baseline 实现

实现 TD-MPC2 + RND 作为 exploration baseline。**RND 把 bonus 加到 reward 信号中，与 FreeGuide 改 planning 形成对照。**

```python
# 在世界模型类中添加（操作原始观测，不是 latent space，避免 SimNorm 坍缩）：
if cfg.rnd.enabled:
    rnd_obs_dim = list(cfg.obs_shape.values())[0][0]  # 原始观测维度
    rnd_hidden = 256
    # 固定的随机目标网络（不训练）
    self.rnd_target = nn.Sequential(
        nn.Linear(rnd_obs_dim, rnd_hidden), nn.ReLU(), nn.Linear(rnd_hidden, rnd_hidden)
    )
    for p in self.rnd_target.parameters():
        p.requires_grad = False

    # 可训练的预测网络（同架构）
    self.rnd_predictor = nn.Sequential(
        nn.Linear(rnd_obs_dim, rnd_hidden), nn.ReLU(), nn.Linear(rnd_hidden, rnd_hidden)
    )

def rnd_bonus(self, obs):
    """计算 RND exploration bonus（输入为原始观测）。"""
    target = self.rnd_target(obs)
    pred = self.rnd_predictor(obs)
    bonus = (pred - target).pow(2).mean(dim=-1)  # [B]
    return bonus
```

**训练**：rnd_predictor 的 loss = MSE(pred, target.detach())，单独 optimizer 训练。

**使用**：在 reward prediction 后，把 RND bonus 加到 reward 上：
```python
if cfg.rnd.enabled:
    bonus = model.rnd_bonus(obs)  # 原始观测，不是 latent state
    bonus_normalized = (bonus - rnd_running_mean) / (rnd_running_std + 1e-8)
    reward_augmented = reward + cfg.rnd.bonus_coef * bonus_normalized
```

**Hydra config**：
```yaml
rnd:
  enabled: false
  bonus_coef: 0.01
```

**互斥约束**：`rnd.enabled` 和 `freeguide.enabled` 不能同时为 true。加一个 assert 检查。

### 1.7 Commit

```bash
cd /home/miller/Desktop/FreeGuide/tdmpc2
git add -A && git commit -m "feat: FreeGuide + RND baseline implementation"
```

### 1.8 快速验证

运行 4 个实验（各 200K 步），**必须串行运行**以节省 VRAM：

```bash
conda activate freeguide
cd /home/miller/Desktop/FreeGuide/tdmpc2

# Baseline Walker-Run
python -m tdmpc2.train task=walker-run steps=200000 seed=1 \
  freeguide.enabled=false \
  exp_name=validate_baseline_walker-run

# FreeGuide Walker-Run
python -m tdmpc2.train task=walker-run steps=200000 seed=1 \
  freeguide.enabled=true \
  exp_name=validate_freeguide_walker-run

# Baseline Humanoid-Run
python -m tdmpc2.train task=humanoid-run steps=200000 seed=1 \
  freeguide.enabled=false \
  exp_name=validate_baseline_humanoid-run

# FreeGuide Humanoid-Run
python -m tdmpc2.train task=humanoid-run steps=200000 seed=1 \
  freeguide.enabled=true \
  exp_name=validate_freeguide_humanoid-run
```

跑完后，写一个 python 脚本画出对比 learning curve（2 个 subplot，每个任务一张），保存到 `logs/validation_curves.png`。

### 1.9 验证判定

读取 4 个实验的最终 return，判定：

- ✅ **通过**：FreeGuide 在 humanoid-run 上的 200K 步 return ≥ baseline 的 95%，且训练中段（50K-150K）有明显领先 → 继续 Phase 2
- ⚠️ **边缘**：差距不明显 → 执行以下调试：
  1. 打印 `J_extrinsic` 和 `beta * J_epistemic` 的比值
  2. 如果 epistemic 项 < extrinsic 的 1%，尝试 fixed beta=0.3
  3. 如果归一化后 ig 方差过小，关闭归一化直接用 raw * 0.01
  4. 尝试 ensemble_K=5
  5. 换 dog-run 任务（38 DoF）试试
- ❌ **失败**：所有调试后仍无提升 → 写入 `logs/validation_failed.md` 记录所有尝试结果，等待人工决策

### 1.10 标记完成

```bash
echo "$(date)" > /home/miller/Desktop/FreeGuide/phase1.done
```

---

## Phase 2：主实验 + 消融实验（A800 服务器）

> **所有正式实验在 3×A800 服务器上跑。不要在本地 4090 上跑正式实验。**
> 本地 4090 仅用于代码开发、快速验证（Phase 1）和 debug。

### 2.0 服务器环境准备

```bash
# 在 A800 服务器上执行（假设 SSH 可达）
# 1. 把本地代码同步到服务器
rsync -avz /home/miller/Desktop/FreeGuide/ server:/home/miller/FreeGuide/

# 2. 在服务器上创建同样的 conda 环境
conda create -n freeguide python=3.10 -y
conda activate freeguide
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
cd /home/miller/FreeGuide/tdmpc2 && pip install -e .
pip install dm_control mujoco hydra-core omegaconf wandb matplotlib seaborn pandas scipy tensorboard
```

### 2.1 Logging 要求（重要）

在实验脚本生成之前，**确认训练代码 log 了以下所有量**（Phase 4 分析需要用到）：

必须 log 到 csv/wandb 的量：
- `episode_return`（每个 episode）
- `freeguide/info_gain_edd`（raw）
- `freeguide/info_gain_qev`（raw）
- `freeguide/beta`
- `freeguide/ensemble_loss`
- `reward_loss`（世界模型的 reward head 训练 loss，**Fig 5 需要**）
- `elapsed_time`（累计训练时间）

如果 `reward_loss` 还没有被 log，在代码中加上再同步到服务器。

### 2.2 实验总览与优先级

全部实验按 3 个优先级分批，高优先级先跑：

```
═══════════════════════════════════════════════════════
 优先级 P1（必须最先跑完，论文核心结果）
 → Table 1, Fig 2, Fig 3, Fig 4
═══════════════════════════════════════════════════════

主实验：3 methods × 5 tasks × 5 seeds = 75 runs × 3M steps

方法：
  tdmpc2:        freeguide.enabled=false rnd.enabled=false
  tdmpc2_rnd:    freeguide.enabled=false rnd.enabled=true
  freeguide:     freeguide.enabled=true  rnd.enabled=false

任务：
  cheetah-run    (半猎豹, 6 DoF)
  walker-run     (双足, 6 DoF)
  quadruped-run  (简单四足, 12 DoF)
  humanoid-run  (人形, 21 DoF)
  dog-run        (复杂四足, 38 DoF)

Seeds：1 2 3 4 5

═══════════════════════════════════════════════════════
 优先级 P2（P1 跑完后立刻开始，消融分析）
 → Fig 6, Fig 7
═══════════════════════════════════════════════════════

组件消融：7 variants × 2 tasks × 3 seeds = 42 runs × 3M steps

任务：walker-run (6 DoF), humanoid-run (21 DoF)
Seeds：1 2 3

变体：
  tdmpc2:        freeguide.enabled=false
  qev_only:      enabled=true use_edd=false use_qev=true use_adaptive_beta=true
  edd_only:      enabled=true use_edd=true use_qev=false use_adaptive_beta=true
  fixed_beta_01: enabled=true use_adaptive_beta=false beta_init=0.1
  fixed_beta_03: enabled=true use_adaptive_beta=false beta_init=0.3
  fixed_beta_05: enabled=true use_adaptive_beta=false beta_init=0.5
  freeguide:     enabled=true（完整版）

注意：tdmpc2 和 freeguide 的 walker-run/humanoid-run 已在 P1 中跑过，
可以直接复用 P1 的 seed=1,2,3 结果，实际只需额外跑：
  5 variants (qev_only, edd_only, fixed_beta_01, fixed_beta_03, fixed_beta_05) × 2 tasks × 3 seeds = 30 runs

Ensemble K 消融（walker-run only）：
  K = 2, 3, 5  seeds=1,2,3 = 9 runs × 3M steps
  K=3 可以复用 P1 的 freeguide walker-run seed=1,2,3
  实际只需额外跑：2 K-values × 3 seeds = 6 runs

P2 实际新增 runs：30 + 6 = 36 runs

═══════════════════════════════════════════════════════
 优先级 P3（P1 数据可用后就能做，纯分析无需额外实验）
 → Fig 4, Fig 5, Fig 8, Table 2
═══════════════════════════════════════════════════════

以下图表从 P1/P2 的实验 log 中提取，不需要额外跑实验：
  - Fig 4 DoF scaling：从 P1 主实验的 sample efficiency 数据计算
  - Fig 5 Reward vs Planning 分析：从 P1 的 reward_loss log 提取 + latent state PCA
  - Fig 8 Information dynamics：从 P1 全部 5 个任务的 FreeGuide 实验 log 提取 info_gain, beta, ensemble_loss
  - Table 2 Wall-clock overhead：从 P1 的 elapsed_time 计算

总实际 runs：75 (P1) + 36 (P2) = 111 runs
```

### 2.3 GPU 分配策略

```
3 张 A800，每张同时跑 4 个实验（80GB VRAM 足够），共 12 个并行槽。

GPU 0: humanoid-run 和 dog-run 的实验（高维任务，最重要）
GPU 1: quadruped-run 和 walker-run 的实验
GPU 2: cheetah-run 的实验 + P2 消融实验

估计时间：
  P1: 75 runs / 12 slots × ~15h/run ≈ 4-5 天
  P2: 36 runs / 12 slots × ~15h/run ≈ 2-3 天
  总计：约 7-8 天
```

### 2.4 生成实验脚本

生成以下脚本到 `experiments/scripts/`：

**所有实验的公共参数**：
```
steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000
```

**`run_p1_main.sh`**：P1 全部 75 个实验
- 3 张 GPU 各分配任务，nohup 后台运行
- 每张 GPU 串行排队 4 个实验
- 命名规则：`{method}_{task}_s{seed}`
- 日志输出到 `logs/{exp_name}.log`
- 实验结果输出到 `experiments/results/{exp_name}/`

**`run_p2_ablations.sh`**：P2 全部 36 个实验
- 复用 P1 的 tdmpc2 和 freeguide 结果（检查 `experiments/results/` 下是否已有）
- 只跑实际需要的 36 个新实验

**`check_progress.sh`**：检查实验进度
- 扫描 `experiments/results/` 下所有 eval.csv
- 打印每个实验的进度（当前步数 / 3M）
- 汇总：完成 X 个 / 运行中 X 个 / 未开始 X 个

**P1 内部的启动顺序**（按重要性）：
```
第 1 批（立刻启动）：humanoid-run × 3 methods × 5 seeds = 15 runs
第 2 批：dog-run × 3 methods × 5 seeds = 15 runs
第 3 批：quadruped-run × 3 methods × 5 seeds = 15 runs
第 4 批：walker-run × 3 methods × 5 seeds = 15 runs
第 5 批：cheetah-run × 3 methods × 5 seeds = 15 runs
```

### 2.5 标记完成

当 P1 全部 75 个实验跑完后：
```bash
echo "$(date)" > /home/miller/Desktop/FreeGuide/phase2.done
```

当 P2 全部 36 个实验跑完后：
```bash
echo "$(date)" > /home/miller/Desktop/FreeGuide/phase3.done
```

---

## Phase 4：分析与出图

> 可以在 P1 部分实验完成后就开始，不需要等全部跑完。
> 用已有数据先出初版图，后续补齐。

### 4.1 画图配置

创建 `analysis/plot_config.py`：
- matplotlib 配置：serif 字体，10pt，300 DPI
- 配色 dict：TD-MPC2=#4C72B0（蓝）, +RND=#8172B3（紫）, FreeGuide=#C44E52（红）, FreeGuide-QEV=#DD8452（橙）, FreeGuide-EDD=#55A868（绿）, Fixed-β=#937860（棕）
- 通用函数：load_data, smooth, compute_ci, save_fig

### 4.2 生成所有图表

创建以下分析脚本（每个可独立运行），**图表编号与论文一一对应**：

| 脚本 | 输出 | 论文编号 | 数据来源 | 内容 |
|------|------|---------|---------|------|
| `plot_main_results.py` | `fig2_main.pdf` | Fig 2 | P1 | 5 任务 learning curve (2×3 grid)，3 条线（TD-MPC2, +RND, FreeGuide），按 DoF 从低到高排列 |
| `plot_sample_efficiency.py` | `fig3_efficiency.pdf` | Fig 3 | P1 | 达到 80% TD-MPC2 asymptotic return 的步数 bar chart，5 tasks × 3 methods |
| `plot_dof_scaling.py` | `fig4_dof_scaling.pdf` | Fig 4 | P1 | X 轴 DoF，Y 轴 sample efficiency 提升 %。两条线：FreeGuide 和 +RND |
| `plot_reward_vs_planning.py` | `fig5_reward_vs_planning.pdf` | **Fig 5** | P1 | **(a)** reward prediction loss over training: 3 条线（TD-MPC2 世界模型学真实 reward，+RND 世界模型学 distorted reward，FreeGuide 世界模型学真实 reward）→ 预期 FreeGuide ≈ TD-MPC2 < +RND。**(b)** latent state coverage heatmap at 500K steps: PCA 2D 投影，TD-MPC2 vs FreeGuide |
| `plot_ablations.py` | `fig6_ablations.pdf` | Fig 6 | P2 | 消融 learning curve (2×1: walker-run + humanoid-run)，5 条线 |
| `plot_ensemble_k.py` | `fig7_ensemble_k.pdf` | Fig 7 | P2 | K 敏感性 bar chart (humanoid-run)，K=2,3,5 |
| `plot_info_dynamics.py` | `fig8_dynamics.pdf` | Fig 8 | P1 | 三面板：(a) info_gain_ema (b) beta (c) ensemble_loss，在全部 5 个任务上画（每个任务一条线或选 humanoid-run 为主 + 其他任务为辅），5 seeds 均值 + CI |
| `compute_tables.py` | `table1.md`, `table2.md` | Table 1,2 | P1 | 主结果表 + wall-clock overhead 表 |
| `statistical_tests.py` | `stats.md` | — | P1 | Welch t-test + Cohen's d |

创建 `analysis/generate_all.py` 一键调用所有脚本。

> 如果实验数据还不完整（只有部分 seed 跑完），脚本应该能优雅处理——用已有数据画图，在标题中标注 "(N seeds)" 数量。

### 4.3 标记完成

```bash
echo "$(date)" > /home/miller/Desktop/FreeGuide/phase4.done
```

---

## Phase 5：论文写作

### 5.1 更新论文草稿

直接在 `/home/miller/Desktop/FreeGuide/paper/FreeGuide_Paper_Draft.md` 上修改，不写 LaTeX。

具体操作：
1. 读取 Phase 4 生成的所有图表和表格数据
2. 把论文草稿中所有实验结果的占位描述替换为**实际数字**（mean ± std）
3. 把"Expected Results Structure"、"Hypothesis"等草稿标记删掉，替换为基于真实结果的分析段落
4. 把"Figures to produce"替换为对实际生成图片的引用：`![Figure N](figures/figN_xxx.pdf)`
5. 补充 Abstract 中的具体数字（样本效率提升百分比、任务数量等）
6. 如果某些实验数据尚不完整，保留占位符 `**XX.X ± X.X**` 并在旁边加注释 `<!-- TODO: fill when experiment completes -->`

### 5.2 补充内容

在草稿末尾添加 References 部分，用 Markdown 格式列出所有引用：

```markdown
## References

[1] Hansen, N., Su, H., & Wang, X. (2024). TD-MPC2: Scalable, Robust World Models for Continuous Control. *ICLR 2024*.
[2] Hansen, N., et al. (2025). Hierarchical World Models as Visual Whole-Body Humanoid Controllers. *ICLR 2025*.
...
```

必须包含的核心引用：
- Hansen et al. 2024 (TD-MPC2)
- Hansen et al. 2025 (Puppeteer, ICLR 2025)
- Friston 2009 (FEP)
- Friston et al. 2017 (Active inference)
- Haarnoja et al. 2018 (SAC)
- Hafner et al. 2023 (DreamerV3)
- Lakshminarayanan et al. 2017 (Ensemble uncertainty)
- Agarwal et al. 2021 (Statistical precipice)

### 5.3 标记完成

```bash
echo "$(date)" > /home/miller/Desktop/FreeGuide/phase5.done
```

---

## 关键实现约束（必须遵守）

1. **ensemble disagreement 只在 MPPI planning 时计算（torch.no_grad），不加入 reward 信号，不产生训练梯度**
2. **ensemble dynamics heads 用与主 dynamics 相同类型的 loss 独立训练**
3. **adaptive beta 每个 episode 更新一次，不是每步**
4. **info gain 必须做 running normalization 后再乘以 beta**
5. **MPPI rollout 时用 ensemble mean 做 state transition，用 variance 做 epistemic value**
6. **不要修改 TD-MPC2 的 encoder、reward head、Q-function 的训练逻辑**
7. **freeguide.enabled=false 时，代码路径必须与原版 TD-MPC2 完全一致**
8. **所有命令前先 `conda activate freeguide`**
9. **绝对禁止轮询等待实验完成。所有训练实验用 `nohup ... &` 后台启动后立即返回。不要用 sleep/wait/tail -f 等方式等待实验跑完。实验完成后由人工通知，或者在下次启动时检查结果。**
