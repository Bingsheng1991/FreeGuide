# FreeGuide Server Experiment Guide

> **⚠️ IMPORTANT CONSTRAINTS — READ FIRST ⚠️**
> 1. All training experiments use `nohup` to run in background. Start them and return immediately. **NEVER** poll/wait for completion.
> 2. **DO NOT** modify any source code under `tdmpc2/`. Only execute experiments.
> 3. If an experiment OOMs, restart it with `buffer_size=500000` added to the command.
> 4. If an experiment crashes with NaN, log it to `logs/failed_experiments.txt` and continue.
> 5. All commands must run inside `conda activate freeguide`.

---

## Part 1: Environment Deployment

### 1.1 Configure Chinese Mirror Sources

```bash
# Conda Tsinghua mirror
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes

# pip Aliyun mirror
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
```

### 1.2 Create Conda Environment

```bash
conda create -n freeguide python=3.10 -y
conda activate freeguide
```

### 1.3 Install Dependencies

```bash
# PyTorch (official cu118 source)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# If too slow, try: pip install torch torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu118

# Install TD-MPC2
cd /home/miller/FreeGuide/tdmpc2
pip install -e .

# Additional dependencies
pip install dm_control mujoco hydra-core omegaconf matplotlib pandas scipy tensorboard
pip install gymnasium==0.29.1 hydra-submitit-launcher imageio imageio-ffmpeg
```

### 1.4 Verify Environment

```bash
conda activate freeguide
cd /home/miller/FreeGuide/tdmpc2/tdmpc2
python train.py task=walker-run steps=5000 enable_wandb=false wandb_project=freeguide compile=false save_video=false save_agent=false
```

Must complete without errors. If it fails:
- `ModuleNotFoundError`: install the missing package
- CUDA errors: verify `torch.cuda.is_available()` returns True
- gym errors: ensure `gymnasium==0.29.1`

---

## Part 2: P1 Main Experiments (75 runs)

### Experiment Matrix

| Method | Config Flags | Tasks | Seeds | Steps |
|--------|-------------|-------|-------|-------|
| tdmpc2 | `freeguide.enabled=false rnd.enabled=false` | 5 | 1-5 | 3M |
| tdmpc2_rnd | `freeguide.enabled=false rnd.enabled=true` | 5 | 1-5 | 3M |
| freeguide | `freeguide.enabled=true rnd.enabled=false` | 5 | 1-5 | 3M |

**Tasks:** cheetah-run, walker-run, quadruped-run, humanoid-run, dog-run

### GPU Assignment (3× A800 80GB)

Each A800 can run **4 experiments in parallel** (each experiment uses ~8-12 GB).

**GPU assignment** (matches CLAUDE.md):
- GPU 0: humanoid-run + dog-run (high-dimensional tasks, most important)
- GPU 1: quadruped-run + walker-run
- GPU 2: cheetah-run + P2 ablation experiments

**Priority order** (start these first):
1. GPU 0: humanoid-run (all 15 runs) → then dog-run (all 15 runs)
2. GPU 1: quadruped-run (all 15 runs) → then walker-run (all 15 runs)
3. GPU 2: cheetah-run (all 15 runs) → then P2 ablations

### Common Flags

All experiments use these flags:
```
steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000
```

### Batch 1: humanoid-run on GPU 0 (15 runs, 4 at a time)

```bash
cd /home/miller/FreeGuide/tdmpc2/tdmpc2
mkdir -p /home/miller/FreeGuide/logs

# --- Batch 1a: seeds 1-4 (parallel) ---
CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=1 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_humanoid-run > /home/miller/FreeGuide/logs/tdmpc2_humanoid-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=2 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_humanoid-run > /home/miller/FreeGuide/logs/tdmpc2_humanoid-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=3 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_humanoid-run > /home/miller/FreeGuide/logs/tdmpc2_humanoid-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=4 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_humanoid-run > /home/miller/FreeGuide/logs/tdmpc2_humanoid-run_seed4.log 2>&1 &

# Wait for batch 1a to finish before starting 1b
wait

# --- Batch 1b: seed 5 + rnd seeds 1-3 ---
CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=5 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_humanoid-run > /home/miller/FreeGuide/logs/tdmpc2_humanoid-run_seed5.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=1 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_humanoid-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_humanoid-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=2 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_humanoid-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_humanoid-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=3 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_humanoid-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_humanoid-run_seed3.log 2>&1 &

wait

# --- Batch 1c: rnd seeds 4-5 + freeguide seeds 1-2 ---
CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=4 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_humanoid-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_humanoid-run_seed4.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=5 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_humanoid-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_humanoid-run_seed5.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=1 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_humanoid-run > /home/miller/FreeGuide/logs/freeguide_humanoid-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=2 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_humanoid-run > /home/miller/FreeGuide/logs/freeguide_humanoid-run_seed2.log 2>&1 &

wait

# --- Batch 1d: freeguide seeds 3-5 + 1 slot free ---
CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=3 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_humanoid-run > /home/miller/FreeGuide/logs/freeguide_humanoid-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=4 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_humanoid-run > /home/miller/FreeGuide/logs/freeguide_humanoid-run_seed4.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=5 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_humanoid-run > /home/miller/FreeGuide/logs/freeguide_humanoid-run_seed5.log 2>&1 &

wait
```

### Batch 2: dog-run on GPU 0 (after Batch 1 completes)

```bash
# --- Batch 2a: tdmpc2 seeds 1-4 ---
CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=1 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_dog-run > /home/miller/FreeGuide/logs/tdmpc2_dog-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=2 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_dog-run > /home/miller/FreeGuide/logs/tdmpc2_dog-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=3 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_dog-run > /home/miller/FreeGuide/logs/tdmpc2_dog-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=4 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_dog-run > /home/miller/FreeGuide/logs/tdmpc2_dog-run_seed4.log 2>&1 &

wait

# --- Batch 2b: tdmpc2 s5 + rnd s1-3 ---
CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=5 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_dog-run > /home/miller/FreeGuide/logs/tdmpc2_dog-run_seed5.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=1 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_dog-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_dog-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=2 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_dog-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_dog-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=3 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_dog-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_dog-run_seed3.log 2>&1 &

wait

# --- Batch 2c: rnd s4-5 + freeguide s1-2 ---
CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=4 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_dog-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_dog-run_seed4.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=5 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_dog-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_dog-run_seed5.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=1 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_dog-run > /home/miller/FreeGuide/logs/freeguide_dog-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=2 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_dog-run > /home/miller/FreeGuide/logs/freeguide_dog-run_seed2.log 2>&1 &

wait

# --- Batch 2d: freeguide s3-5 ---
CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=3 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_dog-run > /home/miller/FreeGuide/logs/freeguide_dog-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=4 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_dog-run > /home/miller/FreeGuide/logs/freeguide_dog-run_seed4.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=dog-run seed=5 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_dog-run > /home/miller/FreeGuide/logs/freeguide_dog-run_seed5.log 2>&1 &

wait
```

### Batch 3: quadruped-run on GPU 1 (15 runs, 4 at a time)

```bash
# --- Batch 3a: tdmpc2 seeds 1-4 ---
CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=1 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_quadruped-run > /home/miller/FreeGuide/logs/tdmpc2_quadruped-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=2 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_quadruped-run > /home/miller/FreeGuide/logs/tdmpc2_quadruped-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=3 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_quadruped-run > /home/miller/FreeGuide/logs/tdmpc2_quadruped-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=4 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_quadruped-run > /home/miller/FreeGuide/logs/tdmpc2_quadruped-run_seed4.log 2>&1 &

wait

# --- Batch 3b: tdmpc2 s5 + rnd s1-3 ---
CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=5 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_quadruped-run > /home/miller/FreeGuide/logs/tdmpc2_quadruped-run_seed5.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=1 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_quadruped-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_quadruped-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=2 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_quadruped-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_quadruped-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=3 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_quadruped-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_quadruped-run_seed3.log 2>&1 &

wait

# --- Batch 3c: rnd s4-5 + freeguide s1-2 ---
CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=4 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_quadruped-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_quadruped-run_seed4.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=5 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_quadruped-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_quadruped-run_seed5.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=1 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_quadruped-run > /home/miller/FreeGuide/logs/freeguide_quadruped-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=2 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_quadruped-run > /home/miller/FreeGuide/logs/freeguide_quadruped-run_seed2.log 2>&1 &

wait

# --- Batch 3d: freeguide s3-5 ---
CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=3 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_quadruped-run > /home/miller/FreeGuide/logs/freeguide_quadruped-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=4 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_quadruped-run > /home/miller/FreeGuide/logs/freeguide_quadruped-run_seed4.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=quadruped-run seed=5 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_quadruped-run > /home/miller/FreeGuide/logs/freeguide_quadruped-run_seed5.log 2>&1 &

wait
```

### Batch 4: walker-run on GPU 1 (after Batch 3 completes)

```bash
# --- Batch 4a: tdmpc2 seeds 1-4 ---
CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=1 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_walker-run > /home/miller/FreeGuide/logs/tdmpc2_walker-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=2 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_walker-run > /home/miller/FreeGuide/logs/tdmpc2_walker-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=3 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_walker-run > /home/miller/FreeGuide/logs/tdmpc2_walker-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=4 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_walker-run > /home/miller/FreeGuide/logs/tdmpc2_walker-run_seed4.log 2>&1 &

wait

# --- Batch 4b: tdmpc2 s5 + rnd s1-3 ---
CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=5 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_walker-run > /home/miller/FreeGuide/logs/tdmpc2_walker-run_seed5.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=1 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_walker-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_walker-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=2 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_walker-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_walker-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=3 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_walker-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_walker-run_seed3.log 2>&1 &

wait

# --- Batch 4c: rnd s4-5 + freeguide s1-2 ---
CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=4 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_walker-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_walker-run_seed4.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=5 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_walker-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_walker-run_seed5.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=1 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_walker-run > /home/miller/FreeGuide/logs/freeguide_walker-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=2 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_walker-run > /home/miller/FreeGuide/logs/freeguide_walker-run_seed2.log 2>&1 &

wait

# --- Batch 4d: freeguide s3-5 ---
CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=3 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_walker-run > /home/miller/FreeGuide/logs/freeguide_walker-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=4 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_walker-run > /home/miller/FreeGuide/logs/freeguide_walker-run_seed4.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=5 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_walker-run > /home/miller/FreeGuide/logs/freeguide_walker-run_seed5.log 2>&1 &

wait
```

### Batch 5: cheetah-run on GPU 2 (after Batch 2 completes)

```bash
# --- Batch 5a: tdmpc2 seeds 1-4 ---
CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=1 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_cheetah-run > /home/miller/FreeGuide/logs/tdmpc2_cheetah-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=2 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_cheetah-run > /home/miller/FreeGuide/logs/tdmpc2_cheetah-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=3 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_cheetah-run > /home/miller/FreeGuide/logs/tdmpc2_cheetah-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=4 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_cheetah-run > /home/miller/FreeGuide/logs/tdmpc2_cheetah-run_seed4.log 2>&1 &

wait

# --- Batch 5b: tdmpc2 s5 + rnd s1-3 ---
CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=5 freeguide.enabled=false rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_cheetah-run > /home/miller/FreeGuide/logs/tdmpc2_cheetah-run_seed5.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=1 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_cheetah-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_cheetah-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=2 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_cheetah-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_cheetah-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=3 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_cheetah-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_cheetah-run_seed3.log 2>&1 &

wait

# --- Batch 5c: rnd s4-5 + freeguide s1-2 ---
CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=4 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_cheetah-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_cheetah-run_seed4.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=5 freeguide.enabled=false rnd.enabled=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=tdmpc2_rnd_cheetah-run > /home/miller/FreeGuide/logs/tdmpc2_rnd_cheetah-run_seed5.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=1 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_cheetah-run > /home/miller/FreeGuide/logs/freeguide_cheetah-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=2 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_cheetah-run > /home/miller/FreeGuide/logs/freeguide_cheetah-run_seed2.log 2>&1 &

wait

# --- Batch 5d: freeguide s3-5 ---
CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=3 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_cheetah-run > /home/miller/FreeGuide/logs/freeguide_cheetah-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=4 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_cheetah-run > /home/miller/FreeGuide/logs/freeguide_cheetah-run_seed4.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=cheetah-run seed=5 freeguide.enabled=true rnd.enabled=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=freeguide_cheetah-run > /home/miller/FreeGuide/logs/freeguide_cheetah-run_seed5.log 2>&1 &

wait
```

---

## Part 3: P2 Ablation Experiments (24 runs)

Run these AFTER P1 completes. Use any available GPU.

**Note:** tdmpc2 and freeguide results for walker-run and humanoid-run can be reused from P1 (same exp_names, same seeds). Only the ablation-specific variants need to run.

### Component Ablation (18 new runs)

Tasks: walker-run, humanoid-run. Seeds: 1, 2, 3. 3M steps.

Variants that need running (3 new × 2 tasks × 3 seeds = 18):

```bash
# QEV-only (no EDD, no extra params)
CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=1 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_qev_only_walker-run > /home/miller/FreeGuide/logs/ablation_qev_only_walker-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=2 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_qev_only_walker-run > /home/miller/FreeGuide/logs/ablation_qev_only_walker-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=3 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_qev_only_walker-run > /home/miller/FreeGuide/logs/ablation_qev_only_walker-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=1 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_qev_only_humanoid-run > /home/miller/FreeGuide/logs/ablation_qev_only_humanoid-run_seed1.log 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=2 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_qev_only_humanoid-run > /home/miller/FreeGuide/logs/ablation_qev_only_humanoid-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=3 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_qev_only_humanoid-run > /home/miller/FreeGuide/logs/ablation_qev_only_humanoid-run_seed3.log 2>&1 &

# EDD-only (no QEV)
CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=1 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_edd_only_walker-run > /home/miller/FreeGuide/logs/ablation_edd_only_walker-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=2 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_edd_only_walker-run > /home/miller/FreeGuide/logs/ablation_edd_only_walker-run_seed2.log 2>&1 &

wait

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=walker-run seed=3 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_edd_only_walker-run > /home/miller/FreeGuide/logs/ablation_edd_only_walker-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=humanoid-run seed=1 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_edd_only_humanoid-run > /home/miller/FreeGuide/logs/ablation_edd_only_humanoid-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=humanoid-run seed=2 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_edd_only_humanoid-run > /home/miller/FreeGuide/logs/ablation_edd_only_humanoid-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=humanoid-run seed=3 freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_edd_only_humanoid-run > /home/miller/FreeGuide/logs/ablation_edd_only_humanoid-run_seed3.log 2>&1 &

wait

# Fixed beta=0.3 (no adaptive beta)
CUDA_VISIBLE_DEVICES=2 nohup python train.py task=walker-run seed=1 freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_fixed_beta03_walker-run > /home/miller/FreeGuide/logs/ablation_fixed_beta03_walker-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=walker-run seed=2 freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_fixed_beta03_walker-run > /home/miller/FreeGuide/logs/ablation_fixed_beta03_walker-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=walker-run seed=3 freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_fixed_beta03_walker-run > /home/miller/FreeGuide/logs/ablation_fixed_beta03_walker-run_seed3.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=humanoid-run seed=1 freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_fixed_beta03_humanoid-run > /home/miller/FreeGuide/logs/ablation_fixed_beta03_humanoid-run_seed1.log 2>&1 &

wait

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=humanoid-run seed=2 freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_fixed_beta03_humanoid-run > /home/miller/FreeGuide/logs/ablation_fixed_beta03_humanoid-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py task=humanoid-run seed=3 freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ablation_fixed_beta03_humanoid-run > /home/miller/FreeGuide/logs/ablation_fixed_beta03_humanoid-run_seed3.log 2>&1 &

wait
```

### Ensemble K Ablation (6 new runs)

Task: humanoid-run. Seeds: 1, 2, 3. K=3 results reuse P1's `freeguide_humanoid-run` (default ensemble_K=3).

```bash
# K=2
CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=1 freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=2 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ensemble_K2_humanoid-run > /home/miller/FreeGuide/logs/ensemble_K2_humanoid-run_seed1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=2 freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=2 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ensemble_K2_humanoid-run > /home/miller/FreeGuide/logs/ensemble_K2_humanoid-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py task=humanoid-run seed=3 freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=2 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ensemble_K2_humanoid-run > /home/miller/FreeGuide/logs/ensemble_K2_humanoid-run_seed3.log 2>&1 &

# K=5
CUDA_VISIBLE_DEVICES=1 nohup python train.py task=humanoid-run seed=1 freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=5 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ensemble_K5_humanoid-run > /home/miller/FreeGuide/logs/ensemble_K5_humanoid-run_seed1.log 2>&1 &

wait

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=humanoid-run seed=2 freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=5 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ensemble_K5_humanoid-run > /home/miller/FreeGuide/logs/ensemble_K5_humanoid-run_seed2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py task=humanoid-run seed=3 freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=5 steps=3000000 enable_wandb=false wandb_project=freeguide compile=true save_csv=true save_video=false eval_freq=50000 exp_name=ensemble_K5_humanoid-run > /home/miller/FreeGuide/logs/ensemble_K5_humanoid-run_seed3.log 2>&1 &

wait
```

---

## Part 4: Experiment Monitoring

### check_progress.sh

Save this as `/home/miller/FreeGuide/experiments/scripts/check_progress.sh`:

```bash
#!/bin/bash
# Check progress of all FreeGuide experiments

cd /home/miller/FreeGuide/tdmpc2/tdmpc2

COMPLETED=0
RUNNING=0
FAILED=0
TOTAL=0

printf "%-45s | %-10s | %-12s | %-10s\n" "Experiment" "Status" "Steps" "Final R"
printf "%s\n" "$(printf '%.0s-' {1..85})"

# Check all expected experiments
for METHOD in tdmpc2 tdmpc2_rnd freeguide; do
    for TASK in cheetah-run walker-run quadruped-run humanoid-run dog-run; do
        for SEED in 1 2 3 4 5; do
            EXP="${METHOD}_${TASK}"
            TOTAL=$((TOTAL + 1))

            # Find eval.csv
            EVAL_CSV="logs/${TASK}/${SEED}/${EXP}/eval.csv"

            if [ ! -f "$EVAL_CSV" ]; then
                # Check if process is running
                if pgrep -f "exp_name=${EXP}" > /dev/null 2>&1; then
                    STATUS="RUNNING"
                    RUNNING=$((RUNNING + 1))
                    printf "%-45s | %-10s | %-12s | %-10s\n" "$EXP" "RUNNING" "-" "-"
                else
                    STATUS="NOT_STARTED"
                    printf "%-45s | %-10s | %-12s | %-10s\n" "$EXP" "PENDING" "-" "-"
                fi
            else
                # Get last step and reward
                LAST_LINE=$(tail -1 "$EVAL_CSV")
                LAST_STEP=$(echo "$LAST_LINE" | cut -d',' -f1)
                LAST_REWARD=$(echo "$LAST_LINE" | cut -d',' -f2)

                if (( $(echo "$LAST_STEP >= 2900000" | bc -l 2>/dev/null || echo 0) )); then
                    STATUS="DONE"
                    COMPLETED=$((COMPLETED + 1))
                    printf "%-45s | \033[32m%-10s\033[0m | %-12s | %-10s\n" "$EXP" "DONE" "$LAST_STEP" "$LAST_REWARD"
                elif pgrep -f "exp_name=${EXP}" > /dev/null 2>&1; then
                    STATUS="RUNNING"
                    RUNNING=$((RUNNING + 1))
                    printf "%-45s | \033[33m%-10s\033[0m | %-12s | %-10s\n" "$EXP" "RUNNING" "$LAST_STEP" "$LAST_REWARD"
                else
                    STATUS="FAILED"
                    FAILED=$((FAILED + 1))
                    printf "%-45s | \033[31m%-10s\033[0m | %-12s | %-10s\n" "$EXP" "FAILED" "$LAST_STEP" "$LAST_REWARD"
                fi
            fi
        done
    done
done

# Check ablation experiments
for VARIANT in ablation_qev_only ablation_edd_only ablation_fixed_beta03; do
    for TASK in walker-run humanoid-run; do
        for SEED in 1 2 3; do
            EXP="${VARIANT}_${TASK}"
            TOTAL=$((TOTAL + 1))
            EVAL_CSV="logs/${TASK}/${SEED}/${EXP}/eval.csv"

            if [ ! -f "$EVAL_CSV" ]; then
                if pgrep -f "exp_name=${EXP}" > /dev/null 2>&1; then
                    RUNNING=$((RUNNING + 1))
                    printf "%-45s | %-10s | %-12s | %-10s\n" "$EXP" "RUNNING" "-" "-"
                else
                    printf "%-45s | %-10s | %-12s | %-10s\n" "$EXP" "PENDING" "-" "-"
                fi
            else
                LAST_LINE=$(tail -1 "$EVAL_CSV")
                LAST_STEP=$(echo "$LAST_LINE" | cut -d',' -f1)
                LAST_REWARD=$(echo "$LAST_LINE" | cut -d',' -f2)
                if (( $(echo "$LAST_STEP >= 2900000" | bc -l 2>/dev/null || echo 0) )); then
                    COMPLETED=$((COMPLETED + 1))
                    printf "%-45s | \033[32m%-10s\033[0m | %-12s | %-10s\n" "$EXP" "DONE" "$LAST_STEP" "$LAST_REWARD"
                elif pgrep -f "exp_name=${EXP}" > /dev/null 2>&1; then
                    RUNNING=$((RUNNING + 1))
                    printf "%-45s | \033[33m%-10s\033[0m | %-12s | %-10s\n" "$EXP" "RUNNING" "$LAST_STEP" "$LAST_REWARD"
                else
                    FAILED=$((FAILED + 1))
                    printf "%-45s | \033[31m%-10s\033[0m | %-12s | %-10s\n" "$EXP" "FAILED" "$LAST_STEP" "$LAST_REWARD"
                fi
            fi
        done
    done
done

# Ensemble K ablation
for K in 2 5; do
    for SEED in 1 2 3; do
        EXP="ensemble_K${K}_humanoid-run"
        TOTAL=$((TOTAL + 1))
        EVAL_CSV="logs/humanoid-run/${SEED}/${EXP}/eval.csv"
        if [ ! -f "$EVAL_CSV" ]; then
            if pgrep -f "exp_name=${EXP}" > /dev/null 2>&1; then
                RUNNING=$((RUNNING + 1))
                printf "%-45s | %-10s | %-12s | %-10s\n" "$EXP" "RUNNING" "-" "-"
            else
                printf "%-45s | %-10s | %-12s | %-10s\n" "$EXP" "PENDING" "-" "-"
            fi
        else
            LAST_LINE=$(tail -1 "$EVAL_CSV")
            LAST_STEP=$(echo "$LAST_LINE" | cut -d',' -f1)
            LAST_REWARD=$(echo "$LAST_LINE" | cut -d',' -f2)
            if (( $(echo "$LAST_STEP >= 2900000" | bc -l 2>/dev/null || echo 0) )); then
                COMPLETED=$((COMPLETED + 1))
                printf "%-45s | \033[32m%-10s\033[0m | %-12s | %-10s\n" "$EXP" "DONE" "$LAST_STEP" "$LAST_REWARD"
            elif pgrep -f "exp_name=${EXP}" > /dev/null 2>&1; then
                RUNNING=$((RUNNING + 1))
                printf "%-45s | \033[33m%-10s\033[0m | %-12s | %-10s\n" "$EXP" "RUNNING" "$LAST_STEP" "$LAST_REWARD"
            else
                FAILED=$((FAILED + 1))
                printf "%-45s | \033[31m%-10s\033[0m | %-12s | %-10s\n" "$EXP" "FAILED" "$LAST_STEP" "$LAST_REWARD"
            fi
        fi
    done
done

echo ""
echo "========================================="
echo "Summary: $COMPLETED completed / $RUNNING running / $FAILED failed / $TOTAL total"
echo "========================================="
```

### check_gpu.sh

Save this as `/home/miller/FreeGuide/experiments/scripts/check_gpu.sh`:

```bash
#!/bin/bash
# Check GPU usage and running experiments

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "=== Running Training Processes ==="
for GPU in 0 1 2; do
    echo ""
    echo "--- GPU $GPU ---"
    # Find python processes using this GPU
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU 2>/dev/null | tr -d ' ')
    if [ -z "$PIDS" ]; then
        echo "  (idle)"
    else
        for PID in $PIDS; do
            CMD=$(ps -p $PID -o args= 2>/dev/null | grep -o 'exp_name=[^ ]*' || echo "unknown")
            echo "  PID $PID: $CMD"
        done
    fi
done
```

---

## Part 5: Experiment Results Directory Structure

### Where results are stored

All experiment results are under:
```
/home/miller/FreeGuide/tdmpc2/tdmpc2/logs/{task}/{seed}/{exp_name}/
```

For example:
```
logs/humanoid-run/1/freeguide_humanoid-run/
├── eval.csv          # Evaluation metrics (step, episode_reward)
├── train.csv         # Training metrics (step, reward_loss, elapsed_time, freeguide/*, rnd/*)
├── latent_states/    # Latent state dumps every 100K steps (for Fig 5b)
│   ├── latent_100000.npz
│   ├── latent_200000.npz
│   └── ...
├── models/
│   └── final.pt      # Model checkpoint (saved at end)
└── eval_video/       # Evaluation videos (if save_video=true)
```

### Key files for Phase 4 analysis

| Paper Figure/Table | Data Source | File | Columns Needed |
|-------------------|-------------|------|----------------|
| Table 1 (main results) | P1 eval.csv | `logs/{task}/{seed}/{exp}/eval.csv` | step, episode_reward (last row) |
| Fig 2 (learning curves) | P1 eval.csv | same | step, episode_reward (all rows) |
| Fig 3 (sample efficiency) | P1 eval.csv | same | step, episode_reward |
| Fig 4 (DoF scaling) | P1 eval.csv | computed from Fig 3 data | |
| Fig 5a (reward prediction) | P1 train.csv | `logs/{task}/{seed}/{exp}/train.csv` | reward_loss |
| Fig 5b (latent coverage) | P1 latent dump | `logs/{task}/{seed}/{exp}/latent_states/latent_500000.npz` | z (PCA 2D projection) |
| Fig 6 (ablations) | P2 eval.csv | `logs/{task}/{seed}/{exp}/eval.csv` | step, episode_reward |
| Fig 7 (ensemble K) | P2 eval.csv | same | step, episode_reward |
| Fig 8 (info dynamics) | P1 train.csv (FreeGuide only) | `logs/{task}/{seed}/{exp}/train.csv` | freeguide/info_gain_edd, freeguide/beta, freeguide/ensemble_loss |
| Table 2 (overhead) | P1 train.csv | same | elapsed_time (last row) |

### Naming convention

```
{method}_{task}

Methods: tdmpc2, tdmpc2_rnd, freeguide
Ablations: ablation_qev_only, ablation_edd_only, ablation_fixed_beta03
Ensemble K: ensemble_K2, ensemble_K5 (K3 = freeguide default)

Note: seed is NOT part of exp_name. It is captured in the directory path: logs/{task}/{seed}/{exp_name}/
```

---

## Fault Handling

| Error | Fix |
|-------|-----|
| OOM | Add `buffer_size=500000` to command and retry |
| NaN crash | Log to `logs/failed_experiments.txt`: `echo "EXP_NAME: NaN at step X" >> logs/failed_experiments.txt`, then continue |
| CUDA error | Try `CUDA_VISIBLE_DEVICES=X` with a different GPU |
| dm_control error | Ensure `gymnasium==0.29.1` |
| Process killed | Check `dmesg` for OOM-killer, restart with smaller buffer |

### Time estimates

- Each 3M step experiment takes ~20-30 hours on A800
- 4 parallel experiments per GPU = ~4-8 days per batch of 15
- P1 total (75 runs): ~15-20 days with 3 GPUs
- P2 total (24 runs): ~5-7 days after P1
