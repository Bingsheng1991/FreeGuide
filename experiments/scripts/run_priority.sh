#!/bin/bash
# FreeGuide Priority Experiments
# Priority tasks: humanoid-run, dog-run (high-dimensional)
# 2 tasks x 3 methods x 5 seeds = 30 experiments
#
# Local 4090 strategy: humanoid-run x 3 methods x 2 seeds = 6 experiments (~5 days)
# Full priority set: run on server with multiple GPUs

set -e
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate freeguide
cd /home/miller/Desktop/FreeGuide/tdmpc2/tdmpc2

STEPS=3000000
EVAL_FREQ=50000
LOGDIR="/home/miller/Desktop/FreeGuide/logs"
mkdir -p ${LOGDIR}

PRIORITY_TASKS="humanoid-run dog-run"
SEEDS="1 2 3 4 5"

run_experiment() {
    local task=$1
    local method=$2
    local seed=$3
    local fg_args=$4
    local exp_name="${method}_${task}"

    echo "$(date): Starting ${exp_name} seed=${seed}"
    CUDA_VISIBLE_DEVICES=0 python train.py \
        task=${task} steps=${STEPS} seed=${seed} \
        ${fg_args} \
        exp_name=${exp_name} \
        enable_wandb=false save_video=false compile=true \
        eval_freq=${EVAL_FREQ} save_csv=true \
        2>&1 | tee "${LOGDIR}/${exp_name}_seed${seed}.log"
    echo "$(date): Finished ${exp_name} seed=${seed}"
}

# === LOCAL 4090 SUBSET (6 experiments, ~5 days) ===
echo "=== Running local 4090 subset: humanoid-run x 3 methods x 2 seeds ==="

for seed in 1 2; do
    run_experiment humanoid-run tdmpc2 ${seed} "freeguide.enabled=false"
    run_experiment humanoid-run tdmpc2_rnd ${seed} \
        "freeguide.enabled=false rnd.enabled=true"
    run_experiment humanoid-run freeguide ${seed} \
        "freeguide.enabled=true"
done

echo "=== Local subset completed ==="
echo "=== For full priority experiments, uncomment below and run on server ==="

# === FULL PRIORITY SET (45 experiments) ===
# Uncomment to run all priority experiments:
# for task in ${PRIORITY_TASKS}; do
#     for seed in ${SEEDS}; do
#         run_experiment ${task} tdmpc2 ${seed} "freeguide.enabled=false"
#         run_experiment ${task} tdmpc2_rnd ${seed} \
#             "freeguide.enabled=false rnd.enabled=true"
#         run_experiment ${task} freeguide ${seed} \
#             "freeguide.enabled=true"
#     done
# done
