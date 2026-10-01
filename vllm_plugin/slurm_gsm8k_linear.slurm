#!/bin/bash
#SBATCH --job-name=vllm-linear-gsm8k
#SBATCH --partition=a100,h100,h200,l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:05:00
#SBATCH --array=0-1
#SBATCH --output=vllm_plugin/.slurm/vllm-linear-gsm8k-%A_%a.out
#SBATCH --error=vllm_plugin/.slurm/vllm-linear-gsm8k-%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
unset SLURM_TRES_PER_TASK
unset VLLM_USE_V2_MODEL_RUNNER

MODES=(standard soft)
MODE="${MODES[${SLURM_ARRAY_TASK_ID}]}"
CHECKPOINT="${CHECKPOINT:-${NANOCHAT_BASE_DIR}/chatsft_checkpoints/d20-from40k-lf-k2-linear_addition-openmath-train5m-k3-anygpu/model_004407.pt}"
RESULT_DIR="${RESULT_DIR:-${PWD}/vllm_plugin/results/linear_addition_004407_gsm8k_0shot_chat_vllm}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
JOB_TMP="${SLURM_TMPDIR:-/tmp}/nanochat-vllm-gsm8k-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
MODEL_DIR="${JOB_TMP}/${MODE}-model"
mkdir -p "${JOB_TMP}" "${RESULT_DIR}"

srun --cpu-bind=none .venv/bin/nanochat-export-vllm \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${MODEL_DIR}" \
  --decode-mode "${MODE}"

srun --cpu-bind=none .venv/bin/python vllm_plugin/evaluate_gsm8k.py run \
  --checkpoint "${CHECKPOINT}" \
  --model-dir "${MODEL_DIR}" \
  --mode "${MODE}" \
  --output-dir "${RESULT_DIR}" \
  --nanochat-base-dir "${NANOCHAT_BASE_DIR}" \
  --start 0 \
  --count 1319 \
  --max-new-tokens 192 \
  --seed 42 \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization 0.8
