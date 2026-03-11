#!/bin/bash
# Check progress of all FreeGuide experiments
# Usage: bash check_progress.sh [PROJECT_ROOT]

PROJECT_ROOT="${1:-/home/miller/FreeGuide}"
LOGS_DIR="$PROJECT_ROOT/tdmpc2/tdmpc2/logs"

cd "$PROJECT_ROOT/tdmpc2/tdmpc2" 2>/dev/null || { echo "ERROR: Cannot cd to $PROJECT_ROOT/tdmpc2/tdmpc2"; exit 1; }

COMPLETED=0
RUNNING=0
FAILED=0
PENDING=0
TOTAL=0

printf "%-45s | %-10s | %-12s | %-10s\n" "Experiment" "Status" "Steps" "Final R"
printf "%s\n" "$(printf '%.0s-' {1..85})"

check_exp() {
    local EXP="$1"
    local TASK="$2"
    local SEED="$3"
    TOTAL=$((TOTAL + 1))

    EVAL_CSV="logs/${TASK}/${SEED}/${EXP}/eval.csv"

    if [ ! -f "$EVAL_CSV" ]; then
        if pgrep -f "exp_name=${EXP}" > /dev/null 2>&1; then
            RUNNING=$((RUNNING + 1))
            printf "%-45s | \033[33m%-10s\033[0m | %-12s | %-10s\n" "$EXP" "RUNNING" "-" "-"
        else
            PENDING=$((PENDING + 1))
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
}

echo "=== P1: Main Experiments (75 runs) ==="
for METHOD in tdmpc2 tdmpc2_rnd freeguide; do
    for TASK in cheetah-run walker-run quadruped-run humanoid-run dog-run; do
        for SEED in 1 2 3 4 5; do
            check_exp "${METHOD}_${TASK}" "$TASK" "$SEED"
        done
    done
done

echo ""
echo "=== P2: Component Ablations (30 runs) ==="
for VARIANT in ablation_qev_only ablation_edd_only ablation_fixed_beta_01 ablation_fixed_beta_03 ablation_fixed_beta_05; do
    for TASK in walker-run humanoid-run; do
        for SEED in 1 2 3; do
            check_exp "${VARIANT}_${TASK}" "$TASK" "$SEED"
        done
    done
done

echo ""
echo "=== P2: Ensemble K Ablations (6 runs) ==="
for K in 2 5; do
    for SEED in 1 2 3; do
        check_exp "ablation_ensemble_K${K}_walker-run" "walker-run" "$SEED"
    done
done

echo ""
echo "========================================="
echo "Summary: $COMPLETED done / $RUNNING running / $FAILED failed / $PENDING pending / $TOTAL total"
echo "========================================="
