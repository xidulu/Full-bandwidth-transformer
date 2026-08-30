#!/bin/bash
#SBATCH --job-name=eval-k3-sft-gsm8k
#SBATCH --partition=h100
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=runs/eval-final-openmath-gsm8k-%A_%a.log

set -euo pipefail

cd /weka/scratch/jhu/enalisn1/xiw/nanochat

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache
unset SLURM_TRES_PER_TASK

source .venv/bin/activate

CHECKPOINT=/home/jhu/xwang457/work/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train1m/model_000911.pt
RESULT_ROOT=fbt_experiments/results/d20_lf_k3_openmath_000911_gsm8k_5shot_full_shards
TOTAL=1319
SHARDS=8
BASE_COUNT=$((TOTAL / SHARDS))
REMAINDER=$((TOTAL % SHARDS))
SHARD=${SLURM_ARRAY_TASK_ID}

if [ "${SHARD}" -lt "${REMAINDER}" ]; then
  COUNT=$((BASE_COUNT + 1))
  START=$((SHARD * COUNT))
else
  COUNT=${BASE_COUNT}
  START=$((REMAINDER * (BASE_COUNT + 1) + (SHARD - REMAINDER) * BASE_COUNT))
fi

OUTPUT_DIR=$(printf "%s/shard_%02d" "${RESULT_ROOT}" "${SHARD}")

srun --cpu-bind=none .venv/bin/python fbt_experiments/evaluate_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --skip-continuation \
  --num-gsm8k "${COUNT}" \
  --gsm8k-start "${START}" \
  --gsm8k-shots 5 \
  --max-new-tokens 192
