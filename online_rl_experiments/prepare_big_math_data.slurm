#!/bin/bash
#SBATCH --job-name=prepare-bigmath
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare-bigmath-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
test -f data/big_math_verified_source/data/train-00000-of-00001.parquet
srun --cpu-bind=none ../.venv/bin/python prepare_big_math_data.py --workers 8
