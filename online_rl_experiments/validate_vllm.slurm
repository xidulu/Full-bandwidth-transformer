#!/bin/bash
#SBATCH --job-name=std-vllm-validate
#SBATCH --partition=h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=00:15:00
#SBATCH --output=logs/vllm-validate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
srun --cpu-bind=none ../.venv/bin/python validate_vllm.py --output "results/vllm-validation-${SLURM_JOB_ID}"
srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node="${VALIDATE_NPROC:-4}" main.py \
  --output "results/vllm-smoke-${SLURM_JOB_ID}" --rollout-engine vllm \
  --steps 2 --eval-examples 0 --final-eval-examples 8 --vllm-verify-weights \
  --run-name "standard-vllm-smoke-${SLURM_JOB_ID}" \
  --microbatch-size 8 --gradient-accumulation-steps 4
