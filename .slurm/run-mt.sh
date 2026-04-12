#!/bin/bash
#SBATCH --job-name=one_step_phenomena
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/pguerrero005/MT/informal-MT/.slurm/one_step.log
#SBATCH --error=/home/pguerrero005/MT/informal-MT/.slurm/one_step.err
#SBATCH --chdir=/home/pguerrero005/MT/informal-MT

source /home/pguerrero005/envs/informalMT/bin/activate

export HF_HOME="/home/pguerrero005/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/pguerrero005/.cache/huggingface"
export HF_HUB_CACHE="/home/pguerrero005/.cache/huggingface"

echo "Job started on $(hostname)"
echo "Date: $(date)"

srun python main.py \
    --approach one_step \
    --k 12 \
    --retrieval-strategy phenomena