#!/bin/bash
set -x  
export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
export WANDB_PROJECT=nugget-matching-sft 
export WANDB_EXPERIMENT_BASE=nugget-matching-qwen3-4b-2507-instruct-sft-gemini-aug-optimized
export WANDB_DIR=/data/tensorboard/
export WANDB_MODE=offline
export USER_ID="name"
export JOB_ID=${JOB_ID:-manual}

timestamp=$(date +"%y%m%d%H%M")
export WANDB_EXPERIMENT=${WANDB_EXPERIMENT_BASE}-v${timestamp}-${WORLD_SIZE}xnode-${JOB_ID}

ulimit -n 65535  

EXPERIMENT_NAME=nugget-matching-qwen3-4b-dapo-gemini-aug-sft
DATA_DIR=/path/to/your/dataset
MODEL_DIR=/path/to/your/model
CHECKPOINT_DIR=/path/to/your/checkpoint/$EXPERIMENT_NAME


torchrun --nnodes=1 --nproc_per_node=8 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=${DATA_DIR}/train.parquet \
  data.val_files=${DATA_DIR}/val.parquet \
  data.train_batch_size=256 \
  data.micro_batch_size_per_gpu=2 \
  data.max_length=8192  \
  data.multiturn.enable=true \
  data.multiturn.messages_key="messages" \
  model.partial_pretrain=${MODEL_DIR} \
  model.trust_remote_code=true \
  optim.lr=1e-6 \
  optim.warmup_steps_ratio=0.2 \
  optim.weight_decay=0.1 \
  optim.clip_grad=1.0 \
  model.lora_rank=0 \
  trainer.default_local_dir=${CHECKPOINT_DIR} \
  trainer.project_name=nugget-matching-sft \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.logger=['console','wandb'] \
  trainer.test_freq=10 \
  trainer.save_freq=50 \
  trainer.n_gpus_per_node=8 \
  trainer.total_epochs=5 \
  hydra.run.dir=/path/to/your/logs/${EXPERIMENT_NAME} \
  $@

