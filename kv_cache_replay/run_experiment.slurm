#!/bin/bash
#SBATCH --job-name=kv-replay-math500
#SBATCH --partition=h100,h200,a100,l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/kv-replay-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
unset SLURM_TRES_PER_TASK
srun --cpu-bind=none ../.venv/bin/python -u replay_experiment.py \
  --checkpoint "${1:?checkpoint required}" --output "results/${SLURM_JOB_ID}" \
  --examples 4 --max-new-tokens 1024 --passes 1 2 3 4 8 16 32
