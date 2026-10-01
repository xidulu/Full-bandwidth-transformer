#!/bin/bash
#SBATCH --job-name=bigmath-continue
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --output=logs/bigmath-continue-%j.log
set -euo pipefail
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "$EXPERIMENT_DIR"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
VARIANT="${1:?standard or three_pass required}"
ADDITIONAL_STEPS="${2:-1000}"
case "$VARIANT" in
  standard) SOURCE="results/899298/checkpoints/model_000300.pt"; MODE=standard; CONCURRENT=64 ;;
  three_pass) SOURCE="results/906530/checkpoints/model_000300.pt"; MODE=soft; CONCURRENT=32 ;;
  *) echo 'Expected standard or three_pass' >&2; exit 2 ;;
esac
TARGET_STEP=$((300 + ADDITIONAL_STEPS))
((ADDITIONAL_STEPS >= 2))
test -f "$SOURCE"
test -f data/big-math-rl-verified.train.manifest.json
COMMON=(--decode-mode "$MODE" --soft-likelihood three_pass_detached
        --data "$PWD/data/big-math-rl-verified.train.jsonl" --allow-dataset-change
        --rollout-engine vllm --microbatch-size 8 --gradient-accumulation-steps 32
        --samples-per-prompt 8 --prompts-per-step 128 --lr 1e-6
        --vllm-max-sequences "$CONCURRENT" --vllm-kv-cache-gb 8 --vllm-max-batched-tokens 2048
        --project nanochat-online-rl)
srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node=4 main.py \
  "${COMMON[@]}" --resume-from "$SOURCE" --steps 302 --vllm-verify-weights --final-eval-examples 0 \
  --output "results/${SLURM_JOB_ID}-preflight" --run-name "bigmath-${VARIANT}-preflight-${SLURM_JOB_ID}"
test -f "results/${SLURM_JOB_ID}-preflight/completed.json"
if (( TARGET_STEP > 302 )); then
  srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node=4 main.py \
    "${COMMON[@]}" --resume-from "results/${SLURM_JOB_ID}-preflight/checkpoints/model_000302.pt" \
    --steps "$TARGET_STEP" --output "results/${SLURM_JOB_ID}" \
    --run-name "bigmath-${VARIANT}-${ADDITIONAL_STEPS}more-${SLURM_JOB_ID}"
fi
