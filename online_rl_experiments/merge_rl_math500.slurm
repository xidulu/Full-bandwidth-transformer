#!/bin/bash
#SBATCH --job-name=math500-rl-grade
#SBATCH --partition=interactive_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/math500-rl-merge-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
EVAL_JOB="${1:?evaluation array job required}"
OUTPUT="results/math500-rl-${EVAL_JOB}"
srun --cpu-bind=none ../.venv/bin/python -m fbt_experiments.merge_math500_shards \
  results/math500-rl-"${EVAL_JOB}"-shards/shard_* --output-dir "$OUTPUT" --expected-count 500
srun --cpu-bind=none ../.venv/bin/python report_math500_post_rl.py "$OUTPUT" --wandb
