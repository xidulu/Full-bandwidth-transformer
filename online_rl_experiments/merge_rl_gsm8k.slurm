#!/bin/bash
#SBATCH --job-name=rl-gsm8k-grade
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=interactive_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6000M
#SBATCH --time=00:20:00
#SBATCH --output=logs/gsm8k-rl-merge-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
EVAL_JOB="${1:?evaluation array required}"
srun --cpu-bind=none ../.venv/bin/python merge_rl_gsm8k.py \
  results/gsm8k-rl-"${EVAL_JOB}"-shards/shard_* \
  --output-dir "results/gsm8k-rl-${EVAL_JOB}" --expected-count 1319
REPORT_ARGS=("results/gsm8k-rl-${EVAL_JOB}" --wandb)
if [[ $# -ge 2 ]]; then REPORT_ARGS+=(--compare-soft "$2"); fi
srun --cpu-bind=none ../.venv/bin/python report_rl_gsm8k.py "${REPORT_ARGS[@]}"
