#!/bin/bash
#SBATCH --job-name=stackedu-materialize
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=runs/stackedu-materialize-%j.log

set -euo pipefail

cd /weka/scratch/jhu/enalisn1/xiw/nanochat

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache

if [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi
export HF_TOKEN="${HF_TOKEN:-${HF_KEY:-}}"

source .venv/bin/activate

STACKEDU_PATH="${NANOCHAT_BASE_DIR}/task_data/HuggingFaceTB--stack-edu/materialized/Python/stackedu_python_budget.parquet"

.venv/bin/python -m scripts.materialize_stackedu \
  --language Python \
  --output "${STACKEDU_PATH}" \
  --target-bytes 8000000000 \
  --val-docs 4096 \
  --min-length-bytes 64 \
  --max-length-bytes 65536 \
  --min-score 3.0 \
  --num-workers 64
