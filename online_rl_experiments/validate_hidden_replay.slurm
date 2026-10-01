#!/bin/bash
#SBATCH --job-name=soft-replay-check
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --output=logs/replay-validation-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
srun --cpu-bind=none ../.venv/bin/python validate_soft.py --hidden-replay --output "results/replay-validation-${SLURM_JOB_ID}"

# Exercise the complete trainer and resume path with a tiny real-data batch.
COMMON=(--decode-mode soft --soft-likelihood hidden_state_replay
        --checkpoint-dir "$NANOCHAT_BASE_DIR/chatsft_checkpoints/d20-from40k-lf-k2-gate_product-openmath-train5m-k3-anygpu"
        --checkpoint-step 4407 --data "$PWD/data/hendrycks-math-benchmark.train.jsonl"
        --rollout-engine vllm --microbatch-size 2 --gradient-accumulation-steps 2
        --samples-per-prompt 2 --max-new-tokens 128 --eval-examples 0 --final-eval-examples 0
        --vllm-verify-weights --project nanochat-online-rl)
srun --cpu-bind=none ../.venv/bin/python main.py "${COMMON[@]}" --steps 1 \
  --output "results/replay-smoke-${SLURM_JOB_ID}" --run-name "soft-replay-smoke-${SLURM_JOB_ID}"
srun --cpu-bind=none ../.venv/bin/python main.py "${COMMON[@]}" --steps 2 \
  --resume-from "results/replay-smoke-${SLURM_JOB_ID}/checkpoints/model_000001.pt" \
  --output "results/replay-resume-${SLURM_JOB_ID}" --run-name "soft-replay-resume-${SLURM_JOB_ID}"
