ENTROPY_COEFF=1e-3
#?matan actor_rollout_ref.actor.ppo_micro_batch_size=4 and other large batch sizes becuase we are didviding by some self.ulysses_sequence_parallel_size

# Enable Ray Debugger
export RAY_DEBUG=1

# Configure Ray Dashboard for remote access
export RAY_DASHBOARD_HOST=0.0.0.0
export RAY_DASHBOARD_PORT=8265



python3 -m verl.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=512 \
data.max_prompt_length=512 \
data.max_response_length=1024 \
data.truncation='right' \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=5e-7 \
actor_rollout_ref.actor.ppo_mini_batch_size=256 \
actor_rollout_ref.actor.ppo_micro_batch_size=4 \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.actor.clip_ratio=0.2 \
actor_rollout_ref.actor.ppo_epochs=1 \
actor_rollout_ref.actor.shuffle=True \
actor_rollout_ref.actor.grad_clip=1.0 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.ppo_micro_batch_size=8 \
critic.ppo_mini_batch_size=256 \
critic.model.enable_gradient_checkpointing=True \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=['console','wandb'] \
trainer.default_local_dir=/work/nvme/betg/mshtepel/models/TinyZero \
+trainer.val_before_train=True \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=1000 \
trainer.test_freq=20 \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 \
2>&1 | tee verl_demo.log


#?changed to GRPO
#algorithm.adv_estimator=grpo \

#?I am trying the last 2 variables becuase of assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0 (main_task pid=1024716) ZeroDivisionError: integer division or modulo by zero