#!/bin/bash
#SBATCH --job-name=bigmath-lf512-curriculum-lr10x-resume
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=3-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/bigmath-lf512-curriculum-lr10x-resume-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
srun --cpu-bind=none ../.venv/bin/python -u restart_curriculum_lr10x.py \
  --source "$PWD/results/942120/checkpoints/model_001500.pt" \
  --output-root "results/${SLURM_JOB_ID}" --target-step 1750
