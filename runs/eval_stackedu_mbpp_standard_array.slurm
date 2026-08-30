#!/bin/bash
#SBATCH --job-name=eval-mbpp-std-arr
#SBATCH --partition=h100
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=runs/eval-mbpp-std-arr-%A_%a.log

set -euo pipefail

cd /weka/scratch/jhu/enalisn1/xiw/nanochat

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache
unset SLURM_TRES_PER_TASK

source .venv/bin/activate

OUT_DIR="fbt_experiments/results/d20_lf_k3_stackedu_004407_mbpp_standard_sharded"

srun --cpu-bind=none .venv/bin/torchrun --standalone --nproc_per_node=1 -m fbt_experiments.evaluate_code \
  --source sft \
  --model-tag d20-lf-k3-stackedu-python-budget \
  --step 4407 \
  --tasks mbpp \
  --mbpp-config full \
  --decode-modes standard \
  --max-new-tokens 512 \
  --temperature 0.0 \
  --top-k 50 \
  --execution-timeout 5.0 \
  --output-dir "${OUT_DIR}" \
  --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}"
