#!/bin/bash
#SBATCH --job-name=std-train-infer-check
#SBATCH --partition=h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --output=logs/train-inference-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
srun --cpu-bind=none ../.venv/bin/python -u check_train_inference.py \
  --output "results/train-inference-${SLURM_JOB_ID}" "$@"
