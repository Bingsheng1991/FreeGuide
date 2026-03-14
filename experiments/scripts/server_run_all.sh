#!/bin/bash

set -e

PROJECT_ROOT="${PROJECT_ROOT:-/home/wbs/FreeGuide}"
RUN_LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$RUN_LOG_DIR"

echo "[$(date '+%F %T')] P1 start" >> "$RUN_LOG_DIR/orchestrator_monitor.log"
nohup bash "$PROJECT_ROOT/experiments/scripts/server_p1_gpu.sh" 0 > "$RUN_LOG_DIR/p1_gpu0.nohup.log" 2>&1 &
pid0=$!
nohup bash "$PROJECT_ROOT/experiments/scripts/server_p1_gpu.sh" 1 > "$RUN_LOG_DIR/p1_gpu1.nohup.log" 2>&1 &
pid1=$!
nohup bash "$PROJECT_ROOT/experiments/scripts/server_p1_gpu.sh" 2 > "$RUN_LOG_DIR/p1_gpu2.nohup.log" 2>&1 &
pid2=$!

wait "$pid0" "$pid1" "$pid2"

echo "[$(date '+%F %T')] P1 done; P2 start" >> "$RUN_LOG_DIR/orchestrator_monitor.log"
bash "$PROJECT_ROOT/experiments/scripts/server_p2.sh" > "$RUN_LOG_DIR/p2.nohup.log" 2>&1
echo "[$(date '+%F %T')] all orchestrated batches completed" >> "$RUN_LOG_DIR/orchestrator_monitor.log"
