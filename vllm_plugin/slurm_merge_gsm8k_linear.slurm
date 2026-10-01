#!/bin/bash
#SBATCH --job-name=vllm-linear-merge
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=vllm_plugin/.slurm/vllm-linear-merge-%j.out
#SBATCH --error=vllm_plugin/.slurm/vllm-linear-merge-%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
unset SLURM_TRES_PER_TASK
RESULT_DIR="${RESULT_DIR:-${PWD}/vllm_plugin/results/linear_addition_004407_gsm8k_0shot_chat_vllm}"
NATIVE_REFERENCE="${NATIVE_REFERENCE:-${PWD}/fbt_experiments/results/d20_from40k_lf_k2_linear_addition_openmath_train5m_k3_anygpu_004407_gsm8k_0shot_chat_full/gsm8k_generations.jsonl}"

srun --cpu-bind=none .venv/bin/python vllm_plugin/evaluate_gsm8k.py merge \
  --standard "${RESULT_DIR}/standard_generations.jsonl" \
  --soft "${RESULT_DIR}/soft_generations.jsonl" \
  --output-dir "${RESULT_DIR}" \
  --native-reference "${NATIVE_REFERENCE}" \
  --expected-start 0 \
  --expected-count 1319
