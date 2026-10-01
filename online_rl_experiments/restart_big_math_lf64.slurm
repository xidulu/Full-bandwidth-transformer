#!/bin/bash
#SBATCH --job-name=bigmath-lf64-resume
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bigmath-lf64-resume-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
SOURCE="$PWD/results/910519/checkpoints/model_000975.pt"
test -f "$SOURCE"
test -f "${SOURCE%/*}/meta_000975.json"
test -f "${SOURCE%/*}/optim_000975_rank0.pt"
COMMON=(--decode-mode soft --soft-likelihood three_pass_detached
        --data "$PWD/data/big-math-rl-verified.train.jsonl"
        --rollout-engine vllm --microbatch-size 8 --gradient-accumulation-steps 32
        --samples-per-prompt 8 --prompts-per-step 128 --lr 1e-6
        --vllm-max-sequences 64 --vllm-kv-cache-gb 16 --vllm-max-batched-tokens 2048
        --project nanochat-online-rl)
# Two real updates verify the larger cache/concurrency and weight transfers.
# Save and resume these updates rather than replaying them in the main stage.
srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node=4 main.py \
  "${COMMON[@]}" --resume-from "$SOURCE" --steps 977 --vllm-verify-weights \
  --eval-examples 0 --final-eval-examples 0 \
  --output "results/${SLURM_JOB_ID}-preflight" --run-name "bigmath-lf64-preflight-${SLURM_JOB_ID}"
test -f "results/${SLURM_JOB_ID}-preflight/completed.json"
srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node=4 main.py \
  "${COMMON[@]}" --resume-from "results/${SLURM_JOB_ID}-preflight/checkpoints/model_000977.pt" \
  --steps 1300 --output "results/${SLURM_JOB_ID}" --run-name "bigmath-lf64-resume-${SLURM_JOB_ID}"
