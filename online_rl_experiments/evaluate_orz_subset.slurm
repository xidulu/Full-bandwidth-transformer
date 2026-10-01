#!/bin/bash
#SBATCH --job-name=std-orz-baseline
#SBATCH --partition=h100,a100,h200,l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=logs/orz-baseline-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
EVAL_SHARDS="${1:-1}"
srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node="$EVAL_SHARDS" evaluate_orz_subset.py \
  --output "results/orz-baseline-${SLURM_JOB_ID}" --shards "$EVAL_SHARDS"
srun --cpu-bind=none ../.venv/bin/python evaluate_orz_subset.py \
  --output "results/orz-baseline-${SLURM_JOB_ID}" --shards "$EVAL_SHARDS" --merge --wandb
