#!/usr/bin/env bash
# source /opt/miniforge/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate starVLA

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)


# === Please modify the following paths according to your environment ===
###########################################################################################

Framework_name=QwenOFT
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
action_input_dim=2560
data_root=playground/Datasets/AIC
data_mix=aic
run_root_dir=./playground/Checkpoints
run_id=oft_v0_state
# === End of environment variable configuration ===
###########################################################################################

export WANDB_API_KEY=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
wandb login

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_hidden_dim ${action_input_dim} \
  --datasets.vla_data.data_root_dir ${data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 12 \
  --datasets.vla_data.video_backend pyav \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 20000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_AIC \
  --wandb_entity deepkevin0122-hong-kong-university-of-science-and-technology \
  # --is_debug True



###########################################################################################
####### multi-node launch example #######
###########################################################################################

# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port $MASTER_PORT \
#   --machine_rank $SLURM_PROCID \
#   --num_machines $SLURM_NNODES \
#   --num_processes=${TOTAL_GPUS} \
#   starVLA/training/train_starvla.py \
#   --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
#   --framework.framework_py QwenGR00T \
#   --framework.qwenvl.base_vlm microsoft/Florence-2-large \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --wandb_project your_project \
#   --wandb_entity your_name

