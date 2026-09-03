#!/bin/bash
#SBATCH --job-name=d20-lf-k2-modes
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=runs/d20-lf-k2-modes-%A_%a.log

set -euo pipefail

cd /weka/scratch/jhu/enalisn1/xiw/nanochat

MODES=(gate_product concat_projection linear_addition)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-d20-standard-60k}"
export SOURCE_STEP="${SOURCE_STEP:-40000}"
export RUN_PREFIX="${RUN_PREFIX:-d20-from40k-lf-k2}"
export WANDB_RUN="${WANDB_RUN:-${RUN_PREFIX}-${MODE}}"
export WANDB_ID="${WANDB_ID:-${RUN_PREFIX}-${MODE}}"
export MODEL_TAG="${MODEL_TAG:-${RUN_PREFIX}-${MODE}}"
unset SLURM_TRES_PER_TASK

SOURCE_CKPT="${NANOCHAT_BASE_DIR}/base_checkpoints/${SOURCE_MODEL_TAG}/model_$(printf '%06d' "${SOURCE_STEP}").pt"
if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "Missing source checkpoint: ${SOURCE_CKPT}" >&2
  exit 1
fi

source .venv/bin/activate

srun --cpu-bind=none .venv/bin/torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=20 \
  --num-forward-passes=2 \
  --latent-feedback-mode="${MODE}" \
  --feedback-start-fraction=0.0 \
  --num-iterations=60000 \
  --device-batch-size=16 \
  --total-batch-size=524288 \
  --eval-every=100 \
  --eval-tokens=1048576 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --run="${WANDB_RUN}" \
  --wandb-id="${WANDB_ID}" \
  --wandb-resume=allow \
  --model-tag="${MODEL_TAG}" \
  --resume-from-model-tag="${SOURCE_MODEL_TAG}" \
  --resume-from-step="${SOURCE_STEP}" \
  --weight-tying \
  --no-feedback-prefix-mixin \
  --save-every=10000
