#!/bin/bash
#SBATCH --job-name=lfsft-math500-1024-grade
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=interactive_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6000M
#SBATCH --time=00:20:00
#SBATCH --output=logs/math500-lfsft-1024-merge-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
EVAL_JOB="${1:?baseline evaluation array required}"
srun --cpu-bind=none ../.venv/bin/python -m fbt_experiments.merge_math500_shards \
  results/math500-soft-"${EVAL_JOB}"-shards/shard_* \
  --output-dir "results/math500-soft-${EVAL_JOB}" --expected-count 500
srun --cpu-bind=none ../.venv/bin/python report_large_batch_math500.py \
  --baseline "results/math500-soft-${EVAL_JOB}" \
  --uniform results/math500-soft-924201 --curriculum results/math500-soft-924202 \
  --max-new-tokens 1024 --output "results/math500-lfsft-1024-comparison-${EVAL_JOB}" --wandb
