#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate libero

cd /project/vonneumann1/zxr/code/starVLA

export LIBERO_HOME=/project/vonneumann1/zxr/code/LIBERO
export LIBERO_DATASET_PATH=/project/vonneumann1/zxr/datasets/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/home/zwanggk/.conda/envs/libero/bin/python

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
unset DISPLAY

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}

# ===== 接收参数 =====
your_ckpt=$1
task_suite_name=$2
base_port=${3:-5694}

if [ -z "$your_ckpt" ] || [ -z "$task_suite_name" ]; then
    echo "Usage: eval_libero.sh <ckpt_path> <task_suite_name> [port]"
    exit 1
fi

host="127.0.0.1"
unnorm_key="franka"
num_trials_per_task=50

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

video_out_path="results/${task_suite_name}/${folder_name}"

echo "Evaluating:"
echo "CKPT: $your_ckpt"
echo "TASK: $task_suite_name"
echo "PORT: $base_port"

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"