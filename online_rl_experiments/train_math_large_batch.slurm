#!/bin/bash
#SBATCH --job-name=std-math-b1024
#SBATCH --partition=h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --output=logs/online-rl-math-%j.log
set -euo pipefail
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "$EXPERIMENT_DIR"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$EXPERIMENT_DIR/.deps:$EXPERIMENT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK

COMMON=(--data "$EXPERIMENT_DIR/data/hendrycks-math-benchmark.train.jsonl"
        --rollout-engine vllm --microbatch-size 8 --gradient-accumulation-steps 32
        --samples-per-prompt 8 --prompts-per-step 128 --lr 1e-6
        --vllm-max-sequences 64 --vllm-kv-cache-gb 8 --vllm-max-batched-tokens 2048
        --project nanochat-online-rl)

# Optional positional checkpoint resumes an existing MATH experiment exactly.
RESUME_CHECKPOINT="${1:-}"
if [[ -z "$RESUME_CHECKPOINT" ]]; then
  # Fresh original SFT weights; first two updates verify all weight transfers.
  srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node=4 main.py \
    "${COMMON[@]}" --steps 2 --vllm-verify-weights --final-eval-examples 0 \
    --output "results/${SLURM_JOB_ID}-preflight" \
    --run-name "standard-math-vllm-b1024-preflight-${SLURM_JOB_ID}"
  test -f "results/${SLURM_JOB_ID}-preflight/completed.json"
  RESUME_CHECKPOINT="results/${SLURM_JOB_ID}-preflight/checkpoints/model_000002.pt"
fi

test -f "$RESUME_CHECKPOINT"
srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node=4 main.py \
  "${COMMON[@]}" --steps 300 \
  --resume-from "$RESUME_CHECKPOINT" \
  --output "results/${SLURM_JOB_ID}" \
  --run-name "standard-math-vllm-b1024-300steps-${SLURM_JOB_ID}"
