#!/bin/bash
#SBATCH --job-name=lfsft-standard-math500-grade
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=interactive_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6000M
#SBATCH --time=00:20:00
#SBATCH --output=logs/math500-lfsft-standard-merge-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
EVAL_JOB="${1:?evaluation array required}"
srun --cpu-bind=none ../.venv/bin/python -m fbt_experiments.merge_math500_shards \
  results/math500-rl-"${EVAL_JOB}"-shards/shard_* \
  --output-dir "results/math500-rl-${EVAL_JOB}" --expected-count 500
srun --cpu-bind=none ../.venv/bin/python report_lfsft_decoding_math500.py \
  --standard "results/math500-rl-${EVAL_JOB}" \
  --output "results/math500-lfsft-decoding-1024-${EVAL_JOB}" --wandb
