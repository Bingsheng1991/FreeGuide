#!/bin/bash
set -e
cd /home/miller/Desktop/FreeGuide/tdmpc2/tdmpc2

COMMON="enable_wandb=false save_video=false compile=false eval_freq=10000 save_csv=true wandb_project=freeguide"

# Delete incomplete baseline data from previous run
rm -f /home/miller/Desktop/FreeGuide/tdmpc2/tdmpc2/logs/walker-run/1/validate_baseline_walker-run/eval.csv

echo "$(date) === Experiment 1/3: Baseline Walker-Run ==="
python train.py task=walker-run steps=200000 seed=1 freeguide.enabled=false exp_name=validate_baseline_walker-run $COMMON

echo "$(date) === Experiment 2/3: Baseline Humanoid-Walk ==="
python train.py task=humanoid-walk steps=200000 seed=1 freeguide.enabled=false exp_name=validate_baseline_humanoid-walk $COMMON

echo "$(date) === Experiment 3/3: FreeGuide Humanoid-Walk ==="
python train.py task=humanoid-walk steps=200000 seed=1 freeguide.enabled=true exp_name=validate_freeguide_humanoid-walk $COMMON

echo "$(date) === Remaining validation experiments complete ==="
date > /home/miller/Desktop/FreeGuide/logs/validation_complete.marker
