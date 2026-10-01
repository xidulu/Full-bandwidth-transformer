#!/bin/bash
#SBATCH --job-name=bigmath-calibration
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=h100,a100,h200,l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/bigmath-calibration-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
VARIANT=standard
if (( SLURM_ARRAY_TASK_ID % 2 )); then VARIANT=three_pass; fi
srun --cpu-bind=none ../.venv/bin/python -u calibrate_big_math.py evaluate \
  --output "${1:?prepared output directory required}" --variant "$VARIANT" --rank "$((SLURM_ARRAY_TASK_ID / 2))"
