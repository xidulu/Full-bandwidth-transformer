#!/bin/bash
#SBATCH --job-name=lr10x-math500-1024
#SBATCH --account=enalisn1_fall2026
#SBATCH --partition=a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --array=0-79%8
#SBATCH --output=logs/math500-lr10x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/jhu/xwang457/work/nanochat_cache}"
export PYTHONPATH="$PWD/.deps:$PWD:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_TRES_PER_TASK
MANIFEST="${1:?manifest required}"
read -r CHECKPOINT OUTPUT SHARD < <(../.venv/bin/python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
index, shard = divmod(int(sys.argv[2]), 8)
item = manifest['checkpoints'][index]
print(item['checkpoint'], item['output'], shard)
PY
)
START=$((SHARD * 500 / 8))
END=$(((SHARD + 1) * 500 / 8))
srun --cpu-bind=none ../.venv/bin/python -u evaluate_rl_math500.py \
  --checkpoint "$CHECKPOINT" --output-dir "${OUTPUT}-shards/shard_${SHARD}" \
  --math500-start "$START" --num-math500 "$((END-START))" \
  --max-new-tokens 1024 --decode-modes soft
