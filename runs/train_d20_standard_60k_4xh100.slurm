#!/bin/bash
#SBATCH --job-name=d20-standard-60k
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --output=runs/d20-standard-60k-%j.log

set -euo pipefail

cd /weka/scratch/jhu/enalisn1/xiw/nanochat

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache
export WANDB_RUN="${WANDB_RUN:-d20-standard-60k}"
export MODEL_TAG="${MODEL_TAG:-d20-standard-60k}"
unset SLURM_TRES_PER_TASK

source .venv/bin/activate

srun --cpu-bind=none .venv/bin/torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=20 \
  --num-forward-passes=2 \
  --feedback-start-fraction=1.0 \
  --num-iterations=60000 \
  --device-batch-size=16 \
  --total-batch-size=524288 \
  --eval-every=100 \
  --eval-tokens=1048576 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --run="${WANDB_RUN}" \
  --model-tag="${MODEL_TAG}" \
  --weight-tying \
  --save-every=10000
