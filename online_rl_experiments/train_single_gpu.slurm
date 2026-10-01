#!/bin/bash
#SBATCH --job-name=std-dapo-online
#SBATCH --partition=a100,h100,h200,l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/online-rl-%j.log
set -euo pipefail
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "$EXPERIMENT_DIR"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$EXPERIMENT_DIR/.deps:$EXPERIMENT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
srun --cpu-bind=none ../.venv/bin/python -u main.py \
  --output "results/${SLURM_JOB_ID}" \
  --run-name "standard-4867-dapo17k-${SLURM_JOB_ID}" "$@"
