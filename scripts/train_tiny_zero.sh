ENTROPY_COEFF=1e-3


python3 -m verl.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=512 \
data.max_prompt_length=512 \
data.max_response_length=2048 \
data.truncation='right' \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=5e-7 \
actor_rollout_ref.actor.ppo_mini_batch_size=256 \
actor_rollout_ref.actor.ppo_micro_batch_size=$((2 * $N_GPUS)) \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.ppo_micro_batch_size=8 \
critic.model.enable_gradient_checkpointing=True \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=['console','wandb'] \
trainer.default_local_dir=/work/nvme/betg/darora1/verifiers/TinyZero \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=1000 \
trainer.test_freq=5 \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 2>&1 | tee verl_demo.log

#?changelog
#* actor_rollout_ref.actor.ppo_micro_batch_size=2*$N_GPUS \ becasue `self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)` was forcing ppo_micro_batch_size to 0
#* trainer.test_freq=20-->10 because I wanted to see metrics a bit more often for some testing thing \
#* data.max_response_length=1024->2048 bc of att-mask-all-1 and daman suggestion \
#* +trainer.val_before_train=True-->False (for testing) \


