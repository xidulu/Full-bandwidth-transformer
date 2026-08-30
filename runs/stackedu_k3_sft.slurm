#!/bin/bash
#SBATCH --job-name=d20-lf-k3-stackedu
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=10:00:00
#SBATCH --output=runs/stackedu-k3-sft-%j.log

set -euo pipefail

cd /weka/scratch/jhu/enalisn1/xiw/nanochat

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache
export WANDB_RUN="${WANDB_RUN:-d20-lf-k3-stackedu-python-budget}"
unset SLURM_TRES_PER_TASK

if [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi
export HF_TOKEN="${HF_TOKEN:-${HF_KEY:-}}"

source .venv/bin/activate

STACKEDU_PATH="${NANOCHAT_BASE_DIR}/task_data/HuggingFaceTB--stack-edu/materialized/Python/stackedu_python_budget.parquet"

if [[ ! -f "${STACKEDU_PATH}.done.json" ]]; then
  .venv/bin/python -m scripts.materialize_stackedu \
    --language Python \
    --output "${STACKEDU_PATH}" \
    --target-bytes 8000000000 \
    --val-docs 4096 \
    --min-length-bytes 64 \
    --max-length-bytes 65536 \
    --min-score 3.0 \
    --num-workers 64
fi

srun --cpu-bind=none .venv/bin/torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- \
  --model-tag d20-lf-k1-smoke \
  --model-step 50000 \
  --output-tag d20-lf-k3-stackedu-python-budget \
  --sft-dataset stackedu \
  --stackedu-path "${STACKEDU_PATH}" \
  --stackedu-val-examples 4096 \
  --stackedu-train-examples -1 \
  --num-iterations 4407 \
  --num-forward-passes 3 \
  --no-feedback-prefix-mixin \
  --feedback-jitter 0.02 \
  --max-seq-len 2048 \
  --device-batch-size 4 \
  --total-batch-size 524288 \
  --load-optimizer 0 \
  --init-lr-frac 0.2 \
  --warmup-ratio 0.03 \
  --warmdown-ratio 0.5 \
  --final-lr-frac 0.0 \
  --eval-every 250 \
  --eval-tokens 1048576 \
  --chatcore-every -1 \
  --save-every 500 \
  --run "$WANDB_RUN"
