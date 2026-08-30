#!/bin/bash
#SBATCH --job-name=merge-code-eval
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=runs/merge-code-eval-%j.log

set -euo pipefail

cd /weka/scratch/jhu/enalisn1/xiw/nanochat

source .venv/bin/activate

.venv/bin/python -m fbt_experiments.merge_code_eval \
  --output-dir fbt_experiments/results/d20_lf_k3_stackedu_004407_code_eval_standard_sharded
