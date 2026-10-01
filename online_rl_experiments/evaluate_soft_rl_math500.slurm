#!/bin/bash
#SBATCH --job-name=soft-rl-math500
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/math500-soft-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
CHECKPOINT="${1:?checkpoint required}"
MAX_NEW_TOKENS="${2:-512}"
SHARD_START=$((SLURM_ARRAY_TASK_ID * 500 / 8))
SHARD_END=$(((SLURM_ARRAY_TASK_ID + 1) * 500 / 8))
srun --cpu-bind=none ../.venv/bin/python -u evaluate_rl_math500.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "results/math500-soft-${SLURM_ARRAY_JOB_ID}-shards/shard_${SLURM_ARRAY_TASK_ID}" \
  --math500-start "$SHARD_START" --num-math500 "$((SHARD_END-SHARD_START))" \
  --max-new-tokens "$MAX_NEW_TOKENS" --decode-modes soft
