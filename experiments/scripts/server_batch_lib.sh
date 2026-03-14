#!/bin/bash

set -u

PROJECT_ROOT="${PROJECT_ROOT:-/home/wbs/FreeGuide}"
TRAIN_ROOT="$PROJECT_ROOT/tdmpc2/tdmpc2"
RUN_LOG_DIR="$PROJECT_ROOT/logs"
FAILED_LOG="$RUN_LOG_DIR/failed_experiments.txt"
MONITOR_LOG="$RUN_LOG_DIR/orchestrator_monitor.log"
COMMON_FLAGS=(
  "steps=3000000"
  "enable_wandb=false"
  "wandb_project=freeguide"
  "compile=true"
  "save_csv=true"
  "save_video=false"
  "eval_freq=50000"
)

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate freeguide

mkdir -p "$RUN_LOG_DIR"
mkdir -p "$TRAIN_ROOT/logs"
touch "$FAILED_LOG" "$MONITOR_LOG"

cd "$TRAIN_ROOT" || exit 1

timestamp() {
  date "+%F %T"
}

monitor_snapshot() {
  {
    echo "[$(timestamp)] === check_progress ==="
    bash "$PROJECT_ROOT/experiments/scripts/check_progress.sh" "$PROJECT_ROOT" || true
    echo
    echo "[$(timestamp)] === check_gpu ==="
    bash "$PROJECT_ROOT/experiments/scripts/check_gpu.sh" || true
    echo
  } >> "$MONITOR_LOG" 2>&1
}

LAST_PID=""
LAST_STATUS=""

launch_train() {
  local gpu="$1"
  local task="$2"
  local seed="$3"
  local exp_name="$4"
  local extra_flags="$5"
  local log_file="$RUN_LOG_DIR/${exp_name}_seed${seed}.log"
  local eval_csv="$TRAIN_ROOT/logs/${task}/${seed}/${exp_name}/eval.csv"
  local existing_pid=""

  LAST_PID=""
  LAST_STATUS=""

  if [ -f "$eval_csv" ]; then
    local last_step
    last_step="$(tail -n 1 "$eval_csv" | cut -d',' -f1)"
    if awk "BEGIN {exit !($last_step >= 2900000)}"; then
      echo "[$(timestamp)] skip completed exp=${exp_name} seed=${seed}" >> "$MONITOR_LOG"
      LAST_STATUS="done"
      return 0
    fi
  fi

  existing_pid="$(pgrep -f "python train.py task=${task} seed=${seed} exp_name=${exp_name}" | head -n 1 || true)"
  if [ -n "$existing_pid" ]; then
    echo "[$(timestamp)] attach running gpu=${gpu} exp=${exp_name} seed=${seed} pid=${existing_pid}" >> "$MONITOR_LOG"
    LAST_PID="$existing_pid"
    LAST_STATUS="running"
    return 0
  fi

  echo "[$(timestamp)] launch gpu=${gpu} exp=${exp_name} seed=${seed}" >> "$MONITOR_LOG"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python train.py \
    task="$task" seed="$seed" exp_name="$exp_name" \
    "${COMMON_FLAGS[@]}" \
    $extra_flags \
    > "$log_file" 2>&1 &
  LAST_PID="$!"
  LAST_STATUS="launched"
}

check_and_recover() {
  local gpu="$1"
  local task="$2"
  local seed="$3"
  local exp_name="$4"
  local extra_flags="$5"
  local log_file="$RUN_LOG_DIR/${exp_name}_seed${seed}.log"
  local eval_csv="$TRAIN_ROOT/logs/${task}/${seed}/${exp_name}/eval.csv"

  if grep -Eqi "nan|not a number" "$log_file"; then
    echo "[$(timestamp)] ${exp_name} seed=${seed}: NaN crash" >> "$FAILED_LOG"
    return 0
  fi

  if grep -Eqi "out of memory|cuda out of memory|oom-killer|killed" "$log_file"; then
    echo "[$(timestamp)] retry ${exp_name} seed=${seed} with buffer_size=500000" >> "$MONITOR_LOG"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python train.py \
      task="$task" seed="$seed" exp_name="$exp_name" \
      "${COMMON_FLAGS[@]}" \
      $extra_flags buffer_size=500000 \
      >> "$log_file" 2>&1 &
    wait "$!"
  fi

  if grep -Eqi "nan|not a number" "$log_file"; then
    echo "[$(timestamp)] ${exp_name} seed=${seed}: NaN crash after retry" >> "$FAILED_LOG"
    return 0
  fi

  if [ -f "$eval_csv" ]; then
    local last_step
    last_step="$(tail -n 1 "$eval_csv" | cut -d',' -f1)"
    if awk "BEGIN {exit !($last_step >= 2900000)}"; then
      return 0
    fi
  fi

  if pgrep -f "python train.py task=${task} seed=${seed} exp_name=${exp_name}" > /dev/null 2>&1; then
    return 0
  fi

  echo "[$(timestamp)] ${exp_name} seed=${seed}: incomplete, inspect $log_file" >> "$FAILED_LOG"
}

run_batch() {
  local gpu="$1"
  shift
  local specs=("$@")
  local pids=()

  monitor_snapshot
  for spec in "${specs[@]}"; do
    IFS='|' read -r task seed exp_name extra_flags <<< "$spec"
    launch_train "$gpu" "$task" "$seed" "$exp_name" "$extra_flags"
    if [ -n "$LAST_PID" ]; then
      pids+=("$LAST_PID")
    fi
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  for spec in "${specs[@]}"; do
    IFS='|' read -r task seed exp_name extra_flags <<< "$spec"
    check_and_recover "$gpu" "$task" "$seed" "$exp_name" "$extra_flags"
  done
  monitor_snapshot
}
