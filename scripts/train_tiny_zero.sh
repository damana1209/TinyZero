ENTROPY_COEFF=1e-3

# Set global seed - you can change this value or make it an environment variable
GLOBAL_SEED=${GLOBAL_SEED:-42}

# Configure Ray Dashboard for remote access
export RAY_DASHBOARD_HOST=0.0.0.0
export RAY_DASHBOARD_PORT=8265

#? do not set the mini or the micro batch size to anything smaller than the total number of GPUs since 
#     config.actor_rollout_ref.actor.ppo_micro_batch_size //= dp_size <-- get a div by 0 err


python3 -m verl.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=128 \
data.val_batch_size=64 \
data.max_prompt_length=512 \
data.max_response_length=2048 \
data.truncation='right' \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=2e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_micro_batch_size=$((1 * $N_GPUS_PER_NODE * $N_NODE)) \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.actor.clip_ratio=0.2 \
actor_rollout_ref.actor.ppo_epochs=1 \
actor_rollout_ref.actor.shuffle=True \
actor_rollout_ref.actor.grad_clip=1.0 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=$(($N_GPUS_PER_NODE * $N_NODE > 4 ? $N_GPUS_PER_NODE * $N_NODE : 4))  \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=$(($N_GPUS_PER_NODE * $N_NODE > 4 ? $N_GPUS_PER_NODE * $N_NODE : 4)) \
critic.optim.lr=1e-5 \
critic.model.path="$BASE_MODEL" \
critic.ppo_mini_batch_size=64 \
critic.ppo_micro_batch_size=$(($N_GPUS_PER_NODE * $N_NODE > 4 ? $N_GPUS_PER_NODE * $N_NODE : 4)) \
critic.model.enable_gradient_checkpointing=True \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=['console','wandb'] \
trainer.default_local_dir=/work/nvme/betg/darora1/verifiers/TinyZero \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=2 \
trainer.save_freq=1000 \
trainer.test_freq=20 \
trainer.project_name="TinyZero" \
trainer.experiment_name="$EXPERIMENT_NAME" \
trainer.total_epochs=15 \
+description=\${oc.env:TRAINING_RUN_DESCRIPTION} \
+trainer.global_seed=$GLOBAL_SEED 2>&1 | tee verl_demo.log



#?changelog from Daman's last commit before mine
#* actor_rollout_ref.actor.ppo_micro_batch_size=2*$N_GPUS_PER_NODE \ becasue `self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)` was forcing ppo_micro_batch_size to 0
#* trainer.test_freq=20-->10 because I wanted to see metrics a bit more often for some testing thing \
#* data.max_response_length=1024->512 bc of illegal memory access \
#* +trainer.val_before_train=True-->False (for testing) \
#* actor_rollout_ref.actor.optim.lr=5e-7-->1e-5  \
#* critic.optim.lr=1e-5-->5e-5\
#* actor_rollout_ref.actor.ppo_mini_batch_size=256-->128 \
#* actor_rollout_ref.actor.ppo_micro_batch_size=$((2-->1 * $N_GPUS_PER_NODE)) \
#* decreased train_batch_size, val_batch_size, microbatchsize, logporbmicrobatchsize,log_prob_micro_batch_size by a factor of 2 
#* be at least as conservative as https://github.com/volcengine/verl/blob/72cae971d00e0dba60cbb191a9f13f5de3b5ae36/examples/ppo_trainer/run_deepseek7b_llm_pfppo.sh
#?nnodes for multinode training
