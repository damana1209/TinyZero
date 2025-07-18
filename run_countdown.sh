#!/bin/bash

#SBATCH --mem=300g
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=30
#SBATCH --account=betg-dtai-gh   # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=run
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00      # hh:mm:ss for the job
#SBATCH --exclude="gh066,gh015,gh089,gh016"
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out
#SBATCH --export=ALL,RAY_DEBUG_POST_MORTEM=0,RAY_DEBUGGER_POST_MORTEM=0,RAY_DEBUGGER_BREAKPOINT=0

echo "job is starting on `hostname`"

# DATASET="countdown"
# DATASET="countdown_idk"
# DATASET="lighteval/MATH"
DATASET="lighteval/MATH_idk"
# DATASET="gsm8k_idk"
# DATASET="gsm8k"
# DATASET="countdown_idk_and_answer"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4
export BASE_MODEL="/work/nvme/betg/mshtepel/models/Qwen/Qwen2.5-1.5B-Instruct"
# export BASE_MODEL="/work/nvme/betg/darora1/verifiers/Llama-3.2-1B"
# export BASE_MODEL="/work/nvme/betg/darora1/verifiers/OctoThinker-1B-Short-Base"
# export DATA_DIR="/work/nvme/betg/darora1/TinyZero/countdown_idk/"
export DATA_DIR="/u/mshtepel/data/math_idk"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=$DATASET"-Qwen2.5-1.5B"
# export EXPERIMENT_NAME=$DATASET"-qwen2.5-1.5b_4choice"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
#changed from XFORMERS

#? for debugging the mem fault 
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1


bash ./scripts/train_tiny_zero.sh