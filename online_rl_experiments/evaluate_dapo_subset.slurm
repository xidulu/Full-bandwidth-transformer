#!/bin/bash
#SBATCH --job-name=rl-dapo500-eval
#SBATCH --account=enalisn1
#SBATCH --partition=h100,a100,h200,l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=logs/dapo500-eval-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
COMMON=(--output "results/dapo500-rl-${SLURM_JOB_ID}"
        --checkpoint "$PWD/results/899298/checkpoints/model_000300.pt"
        --data "$PWD/data/dapo-math-17k.unique.jsonl"
        --prompts 500 --seed 42 --max-tokens 1024 --shards 1
        --greedy-only --grade-truncated)
srun --cpu-bind=none ../.venv/bin/python evaluate_orz_subset.py "${COMMON[@]}"
srun --cpu-bind=none ../.venv/bin/python evaluate_orz_subset.py "${COMMON[@]}" --merge --wandb
