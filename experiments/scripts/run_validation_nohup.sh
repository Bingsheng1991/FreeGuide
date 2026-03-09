#!/bin/bash
set -e
cd /home/miller/Desktop/FreeGuide/tdmpc2/tdmpc2

COMMON="enable_wandb=false save_video=false compile=false eval_freq=10000 save_csv=true wandb_project=freeguide"

echo "=== Experiment 1/4: Baseline Walker-Run ==="
python train.py task=walker-run steps=200000 seed=1 freeguide.enabled=false exp_name=validate_baseline_walker-run $COMMON

echo "=== Experiment 2/4: FreeGuide Walker-Run ==="
python train.py task=walker-run steps=200000 seed=1 freeguide.enabled=true exp_name=validate_freeguide_walker-run $COMMON

echo "=== Experiment 3/4: Baseline Humanoid-Walk ==="
python train.py task=humanoid-walk steps=200000 seed=1 freeguide.enabled=false exp_name=validate_baseline_humanoid-walk $COMMON

echo "=== Experiment 4/4: FreeGuide Humanoid-Walk ==="
python train.py task=humanoid-walk steps=200000 seed=1 freeguide.enabled=true exp_name=validate_freeguide_humanoid-walk $COMMON

echo "=== All validation experiments complete ==="
date > /home/miller/Desktop/FreeGuide/logs/validation_complete.marker
