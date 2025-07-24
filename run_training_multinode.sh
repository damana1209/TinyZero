#!/bin/bash

#SBATCH --mem=500g
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=60
#SBATCH --account=betg-dtai-gh   # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=run
#SBATCH --partition=ghx4
#SBATCH --time=1:00:00      # hh:mm:ss for the job
#SBATCH --exclude="gh066,gh015,gh089,gh016"
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out


DESCRIPTION=""
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --desc)
      export DESCRIPTION="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      shift # past unrecognized argument
      ;;
  esac
done

echo "job is starting on `hostname` with \n DESC: ${DESCRIPTION}"


# DATASET="countdown"
# DATASET="countdown_idk"
# DATASET="lighteval/MATH"
export DATASET="lighteval/MATH_bestguess_and_uncertainty"
# DATASET="gsm8k_idk"
# DATASET="gsm8k"
# DATASET="countdown_idk_and_answer"
# For multi-node SLURM, each node should see local GPU indices (0,1,2,3)
# not global indices (0,1,2,3,4,5,6,7)
# export CUDA_VISIBLE_DEVICES=0,1,2,3 #?idk if this is supposed to be 0,...,7 or 0,...,3 so I'll try to have this be set automaticly
export N_NODE=2
export N_GPUS_PER_NODE=4
export VLLM_HOST_IP=172.28.81.248
export BASE_MODEL="/work/nvme/betg/mshtepel/models/Qwen2.5-7B-Instruct"
# export BASE_MODEL="/work/nvme/betg/mshtepel/models/Qwen2.5-7B-Instruct"
# export BASE_MODEL="/work/nvme/betg/darora1/verifiers/Llama-3.2-1B"
# export BASE_MODEL="/work/nvme/betg/darora1/verifiers/OctoThinker-1B-Short-Base"
# export DATA_DIR="/work/nvme/betg/darora1/TinyZero/countdown_idk/"
export DATA_DIR="/u/mshtepel/data/math/bestguess_and_uncertainty_est"
export ROLLOUT_TP_SIZE=4 
#? when running 7B on 4 GPUs, https://github.com/volcengine/verl/blob/72cae971d00e0dba60cbb191a9f13f5de3b5ae36/examples/ppo_trainer/run_deepseek7b_llm_pfppo.sh sets to 4
export EXPERIMENT_NAME=$DATASET"_$(basename $BASE_MODEL)_multinode"
# export EXPERIMENT_NAME=$DATASET"-qwen2.5-1.5b_4choice"
export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
#? FLASH_ATTN maybe caused memory errors? Daman guesses. Now installed XFORMERS. 
#!add idk base reward here? 
bash ./scripts/train_tiny_zero.sh
