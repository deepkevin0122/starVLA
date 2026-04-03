#!/usr/bin/env bash

# set -e

CHECKPOINT_DIR=$1
GPU_ID=${2:-0}
START_PORT=${3:-5694}

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: ./batch_eval.sh <checkpoint_dir> [gpu_id] [start_port]"
    exit 1
fi

TASKS=(
    "libero_goal"
    "libero_spatial"
    "libero_object"
    "libero_10"
)

# ===== 日志目录 =====
RUN_ID=$(date +"%Y%m%d_%H%M%S")

SUFFIX=${CHECKPOINT_DIR: -20:8}

LOG_DIR="logs/${RUN_ID}_${SUFFIX}"
mkdir -p "$LOG_DIR"

CURRENT_PORT=$START_PORT

echo "======================================"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "GPU: $GPU_ID"
echo "Log Dir: $LOG_DIR"
echo "======================================"

cleanup() {
    echo "Cleaning up background processes..."
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

for ckpt in "$CHECKPOINT_DIR"/*.pt; do

    [ -f "$ckpt" ] || continue

    CKPT_NAME=$(basename "$ckpt" .pt)

    echo "Processing: $CKPT_NAME"

    for task in "${TASKS[@]}"; do

        PORT=$CURRENT_PORT
        CURRENT_PORT=$((CURRENT_PORT+1))

        echo "Task: $task | Port: $PORT"

        # ===== 启动 server（子shell中激活环境）=====
        (
            eval "$(conda shell.bash hook)"
            conda activate starVLA
            bash /project/vonneumann1/zxr/code/starVLA/examples/LIBERO/eval_files/run_policy_server.sh $ckpt $GPU_ID $PORT
        ) > "${LOG_DIR}/server_${CKPT_NAME}_${task}.log" 2>&1 &

        SERVER_PID=$!
        echo "Server PID: $SERVER_PID"

        # ===== 等待端口 ready =====
        sleep 5
        echo "Server ready."
        sleep 3

        # ===== 启动 eval（子shell中激活另一个环境）=====
        (
            eval "$(conda shell.bash hook)"
            conda activate libero
            bash /project/vonneumann1/zxr/code/starVLA/examples/LIBERO/eval_files/eval_libero.sh $ckpt $task $PORT
        ) > "${LOG_DIR}/eval_${CKPT_NAME}_${task}.log" 2>&1 &

        EVAL_PID=$!
        echo "Eval PID: $EVAL_PID"

        wait $EVAL_PID
        echo "Eval finished."

        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true

        sleep 3
    done

done

echo "All evaluations completed."