---
name: latent-feedback-training
description: Launch or modify Nanochat latent-feedback pretraining/SFT jobs while preserving checkpoint modes, pass counts, and local Slurm conventions.
---

# Latent-feedback training jobs

Use this skill when the task is to start, resume, compare, or modify pretraining/SFT jobs for Full-Bandwidth Transformer / latent-feedback experiments in this repo.

## Core facts

- Source checkpoints are identified by both model path and matching metadata:
  - base: `${NANOCHAT_BASE_DIR}/base_checkpoints/<tag>/model_<step>.pt`
  - SFT: `${NANOCHAT_BASE_DIR}/chatsft_checkpoints/<tag>/model_<step>.pt`
- On the JHU cluster, `NANOCHAT_BASE_DIR` has been `/home/jhu/xwang457/work/nanochat_cache`.
- Preserve `meta_<step>.json:model_config.latent_feedback_mode`. The supported modes are defined in `nanochat/gpt.py` as `LATENT_FEEDBACK_MODES`.
- `scripts/chat_sft.py` does not accept a separate `--latent-feedback-mode`; it inherits the mode from the loaded checkpoint. Do not try to emulate a mode by changing the output tag only.
- For LF SFT jobs in this reproduction, use `--load-optimizer 0` unless the user explicitly wants to resume optimizer state from a matching SFT checkpoint.

## Current reproduction recipe

The established small-scale recipe is:

- pretraining: K=2 LF variants from a d20 standard checkpoint, often from step 40k to step 60k;
- mid-training/SFT: OpenMathInstruct-2 `train_5M` with K=3;
- `--no-feedback-prefix-mixin`;
- `--feedback-jitter 0.02`;
- `--total-batch-size 524288`;
- SFT `--device-batch-size 4` on 4 GPUs;
- SFT `--save-every -1` when only the final checkpoint should be retained.

Representative SFT scripts:

- `runs/openmath_standard_sft_train5m_a100.slurm`
- `runs/openmath_concat_projection_k3_sft_train5m_a100.slurm`
- `runs/openmath_gate_product_k3_sft_train5m_a100.slurm`
- `runs/openmath_linear_addition_k3_sft_train5m_a100.slurm`

## Before submitting a job

1. Inspect the source checkpoint metadata:

   ```bash
   jq '{step, model_config, user_config}' /path/to/meta_XXXXXX.json
   ```

2. Check that the output checkpoint directory does not already exist unless the task is explicitly to resume or overwrite.
3. Validate script syntax with `bash -n <script>`.
4. Confirm the script uses the intended source tag, source step, output tag, pass count, and dataset split.

## Slurm pattern

For d20 OpenMath SFT, the practical cluster spec has been:

```bash
#SBATCH --partition=a100,h100,h200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=12:00:00
```

Inside scripts:

```bash
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache
unset SLURM_TRES_PER_TASK
source .venv/bin/activate
```

Use `srun --cpu-bind=none .venv/bin/torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- ...`.

## Validation after changes

For latent-feedback training code changes, run:

```bash
.venv/bin/python -m pytest tests/test_latent_feedback.py tests/test_feedback_schedule.py tests/test_weight_tying.py -q
```

Add `tests/test_feedback_decoding.py` if the change touches shared feedback semantics used by inference.
