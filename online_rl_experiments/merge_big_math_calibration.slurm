#!/bin/bash
#SBATCH --job-name=bigmath-calibration-merge
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=interactive_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6000M
#SBATCH --time=00:20:00
#SBATCH --output=logs/bigmath-calibration-merge-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
srun --cpu-bind=none ../.venv/bin/python -u calibrate_big_math.py merge --output "${1:?output directory required}" --wandb
