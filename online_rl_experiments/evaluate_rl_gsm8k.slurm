#!/bin/bash
#SBATCH --job-name=rl-gsm8k
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/gsm8k-rl-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
CHECKPOINT="${1:?checkpoint required}"
MAX_NEW_TOKENS="${2:-1024}"
DECODE_MODE="${3:-soft}"
case "$DECODE_MODE" in standard|soft) ;; *) echo 'Expected standard or soft decoding' >&2; exit 2;; esac
SHARD_START=$((SLURM_ARRAY_TASK_ID * 1319 / 8))
SHARD_END=$(((SLURM_ARRAY_TASK_ID + 1) * 1319 / 8))
srun --cpu-bind=none ../.venv/bin/python -u evaluate_rl_gsm8k.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "results/gsm8k-rl-${SLURM_ARRAY_JOB_ID}-shards/shard_${SLURM_ARRAY_TASK_ID}" \
  --gsm8k-start "$SHARD_START" --num-gsm8k "$((SHARD_END-SHARD_START))" \
  --gsm8k-shots 0 --gsm8k-prompt-format chat --skip-continuation \
  --max-new-tokens "$MAX_NEW_TOKENS" --decode-modes "$DECODE_MODE"
