#!/bin/bash
# FreeGuide Main Experiments
# Methods x Tasks x Seeds = 3 x 5 x 5 = 75 experiments
# Each experiment: 3M steps, ~15h on single GPU
# Total: ~1125 GPU hours
#
# Run with: bash run_main.sh
# Uses CUDA_VISIBLE_DEVICES=0 (single 4090)
# Experiments run serially (4090 VRAM insufficient for parallel)

set -e
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate freeguide
cd /home/miller/Desktop/FreeGuide/tdmpc2/tdmpc2

STEPS=3000000
EVAL_FREQ=50000
LOGDIR="/home/miller/Desktop/FreeGuide/logs"

TASKS="cheetah-run walker-run quadruped-run humanoid-run dog-run"
SEEDS="1 2 3 4 5"

run_experiment() {
    local task=$1
    local method=$2
    local seed=$3
    local fg_args=$4
    local exp_name="${method}_${task}"

    echo "$(date): Starting ${exp_name} seed=${seed}"
    CUDA_VISIBLE_DEVICES=0 nohup python train.py \
        task=${task} steps=${STEPS} seed=${seed} \
        ${fg_args} \
        exp_name=${exp_name} \
        enable_wandb=false wandb_project=freeguide save_video=false compile=true \
        eval_freq=${EVAL_FREQ} save_csv=true \
        > "${LOGDIR}/${exp_name}_seed${seed}.log" 2>&1 &
    echo "  PID: $! -> ${exp_name} seed=${seed}"
}

for task in ${TASKS}; do
    for seed in ${SEEDS}; do
        # TD-MPC2 baseline
        run_experiment ${task} "tdmpc2" ${seed} "freeguide.enabled=false"

        # TD-MPC2 + RND baseline
        run_experiment ${task} "tdmpc2_rnd" ${seed} \
            "freeguide.enabled=false rnd.enabled=true"

        # FreeGuide (full: EDD + QEV + adaptive beta)
        run_experiment ${task} "freeguide" ${seed} \
            "freeguide.enabled=true"
    done
done

echo "=== All main experiments completed ==="
