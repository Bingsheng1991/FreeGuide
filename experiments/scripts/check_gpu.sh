#!/bin/bash
# Check GPU usage and running experiments

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || nvidia-smi

echo ""
echo "=== Running Training Processes ==="
for GPU in 0 1 2; do
    echo ""
    echo "--- GPU $GPU ---"
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
