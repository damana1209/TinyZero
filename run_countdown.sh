#!/bin/bash

#SBATCH --mem=300g
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --account=betg-dtai-gh   # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=run
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00      # hh:mm:ss for the job
#SBATCH --exclude="gh066,gh015,gh089,gh016"
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out

echo "job is starting on `hostname`"

# DATASET="countdown"
# DATASET="countdown_idk"
# DATASET="lighteval/MATH"
DATASET="lighteval/MATH_idk"
export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS=2
export BASE_MODEL="/work/nvme/betg/darora1/verifiers/Qwen2.5-1.5B"
# export DATA_DIR="/work/nvme/betg/darora1/TinyZero/countdown_idk/"
export DATA_DIR="/work/nvme/betg/darora1/TinyZero/"$DATASET"/"
export ROLLOUT_TP_SIZE=2
# export EXPERIMENT_NAME=$DATASET"-qwen2.5-1.5b_entropy_coeff_1e-3"
export EXPERIMENT_NAME=$DATASET"-qwen2.5-1.5b_bsz_256_lr_2e-7_entropy_coeff_1e-3_idk_0.3"
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh