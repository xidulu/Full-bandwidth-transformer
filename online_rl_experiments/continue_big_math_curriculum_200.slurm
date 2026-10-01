#!/bin/bash
#SBATCH --job-name=bigmath-curriculum-200more
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bigmath-curriculum-200more-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
SOURCE="$PWD/results/922491/checkpoints/model_001300.pt"
test -f "$SOURCE"
test -f "${SOURCE%/*}/meta_001300.json"
test -f "${SOURCE%/*}/optim_001300_rank0.pt"
srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node=4 main.py \
  --resume-from "$SOURCE" --steps 1500 \
  --decode-mode soft --soft-likelihood three_pass_detached \
  --data "$PWD/data/big-math-rl-verified.train.jsonl" \
  --curriculum-config "$PWD/data/bigmath-lf-curriculum.json" \
  --rollout-engine vllm --microbatch-size 8 --gradient-accumulation-steps 128 \
  --samples-per-prompt 8 --prompts-per-step 512 --lr 1e-6 \
  --vllm-max-sequences 64 --vllm-kv-cache-gb 16 --vllm-max-batched-tokens 2048 \
  --project nanochat-online-rl --output "results/${SLURM_JOB_ID}" \
  --run-name "bigmath-curriculum-200more-${SLURM_JOB_ID}"
