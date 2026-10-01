#!/bin/bash
#SBATCH --job-name=std-vllm-b1024
#SBATCH --partition=h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --output=logs/online-rl-large-%j.log
set -euo pipefail
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "$EXPERIMENT_DIR"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$EXPERIMENT_DIR/.deps:$EXPERIMENT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK

# One policy update uses 128 prompt groups x 8 responses. Each rank generates
# 256 responses, scheduled at most 64 at a time, then accumulates 32 microbatches.
COMMON=(--rollout-engine vllm --microbatch-size 8 --gradient-accumulation-steps 32
        --samples-per-prompt 8 --prompts-per-step 128 --lr 1e-6
        --vllm-max-sequences 64 --vllm-kv-cache-gb 8 --vllm-max-batched-tokens 2048
        --project nanochat-online-rl)

# Validate real four-GPU training and every weight transfer before continuing.
# These are the first two updates of the experiment, not discarded warmup work.
srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node=4 main.py \
  "${COMMON[@]}" --steps 2 --vllm-verify-weights --final-eval-examples 0 \
  --output "results/${SLURM_JOB_ID}-preflight" \
  --run-name "standard-vllm-b1024-preflight-${SLURM_JOB_ID}"

test -f "results/${SLURM_JOB_ID}-preflight/completed.json"
srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node=4 main.py \
  "${COMMON[@]}" --steps 300 \
  --resume-from "results/${SLURM_JOB_ID}-preflight/checkpoints/model_000002.pt" \
  --output "results/${SLURM_JOB_ID}" \
  --run-name "standard-vllm-b1024-300steps-${SLURM_JOB_ID}"
