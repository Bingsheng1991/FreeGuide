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
python -m tdmpc2.train task=walker-walk steps=5000
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
# __init__ 中：
if cfg.freeguide.enabled:
    self._dynamics_ensemble = nn.ModuleList([
        nn.Sequential(
            nn.Linear(latent_dim + action_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.Mish(),
            nn.Linear(mlp_dim, latent_dim)
        ) for _ in range(cfg.freeguide.ensemble_K)
    ])

# 新方法：
def ensemble_dynamics(self, z, a):
    za = torch.cat([z, a], dim=-1)
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

确保 wandb（或 tensorboard）中记录以下量：
- `freeguide/beta`
- `freeguide/info_gain_edd`（raw，非归一化）
- `freeguide/info_gain_qev`（raw）
- `freeguide/info_gain_normalized`
- `freeguide/ensemble_loss`
- `freeguide/ig_running_mean`
- `freeguide/ig_running_std`

### 1.6 Commit

```bash
cd /home/miller/Desktop/FreeGuide/tdmpc2
git add -A && git commit -m "feat: FreeGuide implementation (ensemble dynamics + MPPI scoring + adaptive beta)"
```

### 1.7 快速验证

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

# Baseline Humanoid-Walk
python -m tdmpc2.train task=humanoid-walk steps=200000 seed=1 \
  freeguide.enabled=false \
  exp_name=validate_baseline_humanoid-walk

# FreeGuide Humanoid-Walk
python -m tdmpc2.train task=humanoid-walk steps=200000 seed=1 \
  freeguide.enabled=true \
  exp_name=validate_freeguide_humanoid-walk
```

跑完后，写一个 python 脚本画出对比 learning curve（2 个 subplot，每个任务一张），保存到 `logs/validation_curves.png`。

### 1.8 验证判定

读取 4 个实验的最终 return，判定：

- ✅ **通过**：FreeGuide 在 humanoid-walk 上的 200K 步 return ≥ baseline 的 95%，且训练中段（50K-150K）有明显领先 → 继续 Phase 2
- ⚠️ **边缘**：差距不明显 → 执行以下调试：
  1. 打印 `J_extrinsic` 和 `beta * J_epistemic` 的比值
  2. 如果 epistemic 项 < extrinsic 的 1%，尝试 fixed beta=0.3
  3. 如果归一化后 ig 方差过小，关闭归一化直接用 raw * 0.01
  4. 尝试 ensemble_K=5
  5. 换 dog-run 任务（38 DoF）试试
- ❌ **失败**：所有调试后仍无提升 → 写入 `logs/validation_failed.md` 记录所有尝试结果，等待人工决策

### 1.9 标记完成

```bash
echo "$(date)" > /home/miller/Desktop/FreeGuide/phase1.done
```

---

## Phase 2：DMControl 主实验

### 2.1 生成实验脚本

生成 `experiments/scripts/run_main.sh`：

```
方法 × 任务 × Seeds 矩阵：

方法（3种）：
  tdmpc2:        freeguide.enabled=false
  freeguide_qev: freeguide.enabled=true freeguide.use_edd=false freeguide.use_qev=true
  freeguide:     freeguide.enabled=true

任务（7个）：
  cheetah-run walker-walk walker-run humanoid-walk humanoid-run dog-walk dog-run

Seeds：1 2 3 4 5

每个实验 3M steps。
```

脚本要求：
- 用 `CUDA_VISIBLE_DEVICES=0` 跑（单卡 4090）
- 用 nohup 后台运行，日志输出到 `logs/{exp_name}.log`
- 每个实验之间不并行（4090 VRAM 不够并行两个 TD-MPC2）
- 加一个预估时间注释：每个实验约 20h，105 个实验总计约 2100h
- **优先级排序**：先跑 3 个关键任务（humanoid-walk, humanoid-run, dog-run）× 3 methods × 5 seeds = 45 个

同时生成 `experiments/scripts/run_priority.sh`：只包含优先级最高的 45 个实验。

### 2.2 执行优先实验

启动 `run_priority.sh` 中的第一个实验（串行模式）。

> 注意：完整实验需要人工在服务器上并行执行。这里在本地 4090 上只跑关键子集验证。
> 本地策略：先跑 humanoid-walk × 3 methods × 2 seeds = 6 个实验（约 5 天）。

```bash
conda activate freeguide
cd /home/miller/Desktop/FreeGuide/tdmpc2
bash /home/miller/Desktop/FreeGuide/experiments/scripts/run_priority.sh
```

### 2.3 标记完成

当优先实验跑完后：
```bash
echo "$(date)" > /home/miller/Desktop/FreeGuide/phase2.done
```

---

## Phase 3：消融实验

### 3.1 生成消融脚本

生成 `experiments/scripts/run_ablations.sh`：

```
任务：walker-run, humanoid-walk
Seeds：1 2 3
Steps：3M

消融变体（7种）：
  tdmpc2:        freeguide.enabled=false
  qev_only:      enabled=true use_edd=false use_qev=true use_adaptive_beta=true
  edd_only:      enabled=true use_edd=true use_qev=false use_adaptive_beta=true
  fixed_beta_01: enabled=true use_adaptive_beta=false beta_init=0.1
  fixed_beta_03: enabled=true use_adaptive_beta=false beta_init=0.3
  fixed_beta_05: enabled=true use_adaptive_beta=false beta_init=0.5
  freeguide:     enabled=true（完整版）

Ensemble K 消融（walker-run only）：
  K = 2, 3, 5  seeds=1,2,3
```

### 3.2 标记完成

```bash
echo "$(date)" > /home/miller/Desktop/FreeGuide/phase3.done
```

---

## Phase 4：分析与出图

### 4.1 画图配置

创建 `analysis/plot_config.py`：
- matplotlib 配置：serif 字体，10pt，300 DPI
- 配色 dict：TD-MPC2=#4C72B0, FreeGuide=#C44E52, FreeGuide-QEV=#DD8452, ...
- 通用函数：load_data, smooth, compute_ci, save_fig

### 4.2 生成所有图表

创建以下分析脚本（每个可独立运行）：

| 脚本 | 输出 | 内容 |
|------|------|------|
| `plot_main_results.py` | `paper/figures/fig2_main.pdf` | 7 任务 learning curve (2×4 grid) |
| `plot_sample_efficiency.py` | `paper/figures/fig3_efficiency.pdf` | 达到 80% baseline 的步数 bar chart |
| `plot_ablations.py` | `paper/figures/fig4_ablations.pdf` | 消融 learning curve (2×1) |
| `plot_ensemble_k.py` | `paper/figures/fig5_ensemble_k.pdf` | K 敏感性 bar chart |
| `plot_info_dynamics.py` | `paper/figures/fig6_dynamics.pdf` | info_gain + beta + loss 三面板 |
| `compute_tables.py` | `paper/tables/table1.tex`, `table2.tex` | 主结果表 + overhead 表 |
| `statistical_tests.py` | `paper/tables/stats.tex` | Welch t-test + Cohen's d |

创建 `analysis/generate_all.py` 一键调用所有脚本。

> 注意：如果实验数据还不完整（只有部分 seed 跑完），脚本应该能优雅处理——用已有数据画图，在标题中标注 "N seeds" 数量。

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
