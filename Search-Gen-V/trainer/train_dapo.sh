set -x  
export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0   
export WANDB_PROJECT=nugget-matching-sft 
export WANDB_EXPERIMENT_BASE=nugget-matching-qwen3-4b-dapo-gemini-aug-sft
export WANDB_DIR=/data/tensorboard/
export WANDB_MODE=offline
export USER_ID="name"
export JOB_ID=${JOB_ID:-manual}
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
timestamp=$(date +"%y%m%d%H%M")
export WANDB_EXPERIMENT=${WANDB_EXPERIMENT_BASE}-v${timestamp}-8xnode-${JOB_ID}

ulimit -n 65535  

EXPERIMENT_NAME=nugget-matching-qwen3-4b-dapo-gemini-aug-sft
DATA_DIR=/path/to/your/dataset
MODEL_DIR=/path/to/your/model
CHECKPOINT_DIR=/path/to/your/checkpoint/$EXPERIMENT_NAME
CONFIG_PATH=/path/to/your/config



python3 -m recipe.dapo.main_dapo \
  --config-path=${CONFIG_PATH} \
  --config-name='dapo_trainer' \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_batch_size=128 \
  data.gen_batch_size=256 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.truncation='left' \
  data.prompt_key=messages \
  data.train_files=${DATA_DIR}/train.parquet \
  data.val_files=${DATA_DIR}/val.parquet \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.max_num_gen_batches=5 \
  algorithm.filter_groups.metric="seq_final_reward" \
  actor_rollout_ref.model.path=${MODEL_DIR} \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_liger=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((1024 * 32)) \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.loss_agg_mode="token-mean" \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.rollout.temperature=1.1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  reward_model.reward_manager=dapo \
  reward_model.overlong_buffer.enable=True \
  reward_model.overlong_buffer.len=2048 \
  reward_model.overlong_buffer.penalty_factor=1.0 \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name=${WANDB_PROJECT} \
  trainer.experiment_name=${WANDB_EXPERIMENT} \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=10 \
  trainer.default_local_dir=${CHECKPOINT_DIR} \
  trainer.total_training_steps=800 \
  $@