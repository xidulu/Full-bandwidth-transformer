#!/bin/bash
#SBATCH --job-name=nanochat-vllm-soft
#SBATCH --partition=a100,h100,h200,l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:05:00
#SBATCH --output=vllm_plugin/.slurm/nanochat-vllm-soft-%j.out
#SBATCH --error=vllm_plugin/.slurm/nanochat-vllm-soft-%j.err

set -euo pipefail

export DECODE_MODE=soft
exec bash vllm_plugin/slurm_smoke.slurm
