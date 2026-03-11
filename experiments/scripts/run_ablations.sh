#!/bin/bash
# FreeGuide Ablation Experiments
# Tasks: walker-run, humanoid-run
# Seeds: 1 2 3
# Steps: 3M
#
# Ablation variants (7):
#   tdmpc2:        freeguide.enabled=false
#   qev_only:      enabled=true use_edd=false use_qev=true use_adaptive_beta=true
#   edd_only:      enabled=true use_edd=true use_qev=false use_adaptive_beta=true
#   fixed_beta_01: enabled=true use_adaptive_beta=false beta_init=0.1
#   fixed_beta_03: enabled=true use_adaptive_beta=false beta_init=0.3
#   fixed_beta_05: enabled=true use_adaptive_beta=false beta_init=0.5
#   freeguide:     enabled=true (full version)
#
# Ensemble K ablation (walker-run only): K=2,5 x seeds=1,2,3 (K=3 reused from P1)
#
# Ablation-only variants: 5 x 2 tasks x 3 seeds = 30 experiments
# + Ensemble K: 2 values x 3 seeds = 6 experiments
# Total new: 36 experiments
# Note: tdmpc2 baseline and full freeguide are reused from P1 main experiments
#       (exp_name=tdmpc2_{task} and freeguide_{task}, same seeds/hyperparams)

set -e
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate freeguide
cd /home/miller/Desktop/FreeGuide/tdmpc2/tdmpc2

STEPS=3000000
EVAL_FREQ=50000
LOGDIR="/home/miller/Desktop/FreeGuide/logs"
mkdir -p ${LOGDIR}

ABLATION_TASKS="walker-run humanoid-run"
SEEDS="1 2 3"

run_experiment() {
    local task=$1
    local method=$2
    local seed=$3
    local fg_args=$4
    local exp_name="ablation_${method}_${task}"

    echo "$(date): Starting ${exp_name} seed=${seed}"
    nohup python train.py \
        task=${task} steps=${STEPS} seed=${seed} \
        ${fg_args} \
        exp_name=${exp_name} \
        enable_wandb=false wandb_project=freeguide save_video=false compile=true \
        eval_freq=${EVAL_FREQ} save_csv=true \
        > "${LOGDIR}/${exp_name}_seed${seed}.log" 2>&1 &
    echo "  PID: $! -> ${exp_name} seed=${seed}"
}

echo "=== Component Ablations ==="
echo "(tdmpc2 baseline and full freeguide reused from P1: exp_name=tdmpc2_{task} / freeguide_{task})"
for task in ${ABLATION_TASKS}; do
    for seed in ${SEEDS}; do
        # QEV only
        run_experiment ${task} qev_only ${seed} \
            "freeguide.enabled=true freeguide.use_edd=false freeguide.use_qev=true freeguide.use_adaptive_beta=true"

        # EDD only
        run_experiment ${task} edd_only ${seed} \
            "freeguide.enabled=true freeguide.use_edd=true freeguide.use_qev=false freeguide.use_adaptive_beta=true"

        # Fixed beta=0.1
        run_experiment ${task} fixed_beta_01 ${seed} \
            "freeguide.enabled=true freeguide.use_adaptive_beta=false freeguide.beta_init=0.1"

        # Fixed beta=0.3
        run_experiment ${task} fixed_beta_03 ${seed} \
            "freeguide.enabled=true freeguide.use_adaptive_beta=false freeguide.beta_init=0.3"

        # Fixed beta=0.5
        run_experiment ${task} fixed_beta_05 ${seed} \
            "freeguide.enabled=true freeguide.use_adaptive_beta=false freeguide.beta_init=0.5"
    done
done

echo "=== Ensemble K Ablation (walker-run only) ==="
for seed in ${SEEDS}; do
    for K in 2 5; do  # K=3 reused from P1 freeguide results
        run_experiment walker-run "ensemble_K${K}" ${seed} \
            "freeguide.enabled=true freeguide.ensemble_K=${K}"
    done
done

echo "=== All ablation experiments completed ==="
