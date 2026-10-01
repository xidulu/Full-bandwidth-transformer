#!/bin/bash
#SBATCH --job-name=large-batch-math500-grade
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=interactive_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6000M
#SBATCH --time=00:20:00
#SBATCH --output=logs/math500-large-batch-merge-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
UNIFORM_JOB="${1:?uniform evaluation array required}"
CURRICULUM_JOB="${2:?curriculum evaluation array required}"
MAX_NEW_TOKENS="${3:-512}"
for EVAL_JOB in "$UNIFORM_JOB" "$CURRICULUM_JOB"; do
  srun --cpu-bind=none ../.venv/bin/python -m fbt_experiments.merge_math500_shards \
    results/math500-soft-"${EVAL_JOB}"-shards/shard_* \
    --output-dir "results/math500-soft-${EVAL_JOB}" --expected-count 500
done
srun --cpu-bind=none ../.venv/bin/python report_large_batch_math500.py \
  --uniform "results/math500-soft-${UNIFORM_JOB}" \
  --curriculum "results/math500-soft-${CURRICULUM_JOB}" \
  --output "results/math500-large-batch-comparison-${UNIFORM_JOB}-${CURRICULUM_JOB}" \
  --max-new-tokens "$MAX_NEW_TOKENS" --wandb
