#!/bin/bash
#SBATCH --job-name=eval-stackedu-code
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=runs/eval-stackedu-code-%j.log

set -euo pipefail

cd /weka/scratch/jhu/enalisn1/xiw/nanochat

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache
unset SLURM_TRES_PER_TASK

source .venv/bin/activate

OUT_DIR="fbt_experiments/results/d20_lf_k3_stackedu_004407_code_eval"

srun --cpu-bind=none .venv/bin/torchrun --standalone --nproc_per_node=4 -m fbt_experiments.evaluate_code \
  --source sft \
  --model-tag d20-lf-k3-stackedu-python-budget \
  --step 4407 \
  --tasks humaneval,mbpp \
  --mbpp-config full \
  --decode-modes standard,soft,fused \
  --max-new-tokens 512 \
  --temperature 0.0 \
  --top-k 50 \
  --execution-timeout 5.0 \
  --output-dir "${OUT_DIR}"
