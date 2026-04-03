#!/usr/bin/env bash

export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/home/zwanggk/.conda/envs/starVLA/bin/python

# ===== 接收参数 =====
your_ckpt=$1
gpu_id=${2:-0}
port=${3:-5694}

if [ -z "$your_ckpt" ]; then
    echo "Usage: run_policy_server.sh <ckpt_path> [gpu_id] [port]"
    exit 1
fi

echo "Starting policy server..."
echo "CKPT: $your_ckpt"
echo "GPU: $gpu_id"
echo "PORT: $port"

CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16