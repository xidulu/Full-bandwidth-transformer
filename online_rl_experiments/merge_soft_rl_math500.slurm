#!/bin/bash
#SBATCH --job-name=soft-math500-grade
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=interactive_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6000M
#SBATCH --time=00:20:00
#SBATCH --output=logs/math500-soft-merge-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
THREE_PASS_JOB="${1:?three-pass evaluation job required}"
REPLAY_JOB="${2:?replay evaluation job required}"
for EVAL_JOB in "$THREE_PASS_JOB" "$REPLAY_JOB"; do
  srun --cpu-bind=none ../.venv/bin/python -m fbt_experiments.merge_math500_shards \
    results/math500-soft-"${EVAL_JOB}"-shards/shard_* \
    --output-dir "results/math500-soft-${EVAL_JOB}" --expected-count 500
done
srun --cpu-bind=none ../.venv/bin/python report_soft_math500.py \
  --three-pass "results/math500-soft-${THREE_PASS_JOB}" \
  --replay "results/math500-soft-${REPLAY_JOB}" \
  --output "results/math500-soft-comparison-${THREE_PASS_JOB}-${REPLAY_JOB}" --wandb
