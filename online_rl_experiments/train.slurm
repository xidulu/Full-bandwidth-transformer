#!/bin/bash
#SBATCH --job-name=std-dapo-vllm-4gpu
#SBATCH --partition=h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --output=logs/online-rl-%j.log
set -euo pipefail
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "$EXPERIMENT_DIR"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$EXPERIMENT_DIR/.deps:$EXPERIMENT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
srun --cpu-bind=none ../.venv/bin/torchrun --standalone --nproc_per_node=4 main.py \
  --output "results/${SLURM_JOB_ID}" \
  --run-name "standard-4867-dapo17k-vllm-4gpu-ga4-${SLURM_JOB_ID}" \
  --rollout-engine vllm --microbatch-size 8 --gradient-accumulation-steps 4 "$@"
