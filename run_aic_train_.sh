#!/usr/bin/env bash
# source /opt/miniforge/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate starVLA

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
data_root=playground/Datasets/AIC
data_mix=aic
run_root_dir=./results/Checkpoints
# === End of environment variable configuration ===
###########################################################################################

export WANDB_API_KEY=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
wandb login

run_id="AIC_v0"
output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend pyav \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_AIC \
  --wandb_entity deepkevin0122-hong-kong-university-of-science-and-technology