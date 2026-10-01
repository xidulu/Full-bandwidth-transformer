#!/bin/bash
#SBATCH --job-name=nanochat-vllm-smoke
#SBATCH --partition=a100,h100,h200,l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=vllm_plugin/.slurm/nanochat-vllm-smoke-%j.out
#SBATCH --error=vllm_plugin/.slurm/nanochat-vllm-smoke-%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
unset SLURM_TRES_PER_TASK
unset VLLM_USE_V2_MODEL_RUNNER

CHECKPOINT="${CHECKPOINT:-${NANOCHAT_BASE_DIR}/chatsft_checkpoints/d20-standard-60k-openmath-train5m-k1-anygpu/model_004407.pt}"
DECODE_MODE="${DECODE_MODE:-standard}"
JOB_TMP="${SLURM_TMPDIR:-/tmp}/nanochat-vllm-${SLURM_JOB_ID}"
EXPORT_DIR="${JOB_TMP}/model"
RESULT="${JOB_TMP}/native.json"
mkdir -p "${JOB_TMP}"

srun --cpu-bind=none .venv/bin/nanochat-export-vllm \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EXPORT_DIR}" \
  --decode-mode "${DECODE_MODE}"

srun --cpu-bind=none .venv/bin/python vllm_plugin/validate_parity.py native \
  --checkpoint "${CHECKPOINT}" \
  --result "${RESULT}" \
  --decode-mode "${DECODE_MODE}"

srun --cpu-bind=none .venv/bin/python vllm_plugin/validate_parity.py vllm \
  --export-dir "${EXPORT_DIR}" \
  --result "${RESULT}" \
  --decode-mode "${DECODE_MODE}"
