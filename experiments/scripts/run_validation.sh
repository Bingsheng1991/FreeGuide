#!/bin/bash
set -e
cd /home/miller/Desktop/FreeGuide/tdmpc2/tdmpc2

echo "=== Experiment 1/4: Baseline Walker-Run ==="
python train.py task=walker-run steps=200000 seed=1 freeguide.enabled=false exp_name=validate_baseline_walker-run enable_wandb=false wandb_project=freeguide

echo "=== Experiment 2/4: FreeGuide Walker-Run ==="
python train.py task=walker-run steps=200000 seed=1 freeguide.enabled=true exp_name=validate_freeguide_walker-run enable_wandb=false wandb_project=freeguide

echo "=== Experiment 3/4: Baseline Humanoid-Run ==="
python train.py task=humanoid-run steps=200000 seed=1 freeguide.enabled=false exp_name=validate_baseline_humanoid-run enable_wandb=false wandb_project=freeguide

echo "=== Experiment 4/4: FreeGuide Humanoid-Run ==="
python train.py task=humanoid-run steps=200000 seed=1 freeguide.enabled=true exp_name=validate_freeguide_humanoid-run enable_wandb=false wandb_project=freeguide

echo "=== All validation experiments complete ==="
