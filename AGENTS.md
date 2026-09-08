# Agent notes for this repository

This repo is a Nanochat fork used to reproduce small-scale Full-Bandwidth Transformer / latent-feedback experiments. Keep changes minimal and reproducible: prefer checked-in scripts, exact checkpoint tags, and machine-readable result artifacts over manual notes.

## Environment and data layout

- Use `uv` for dependencies: `uv sync --extra gpu --group dev` for CUDA development, then `source .venv/bin/activate`.
- The code expects `NANOCHAT_BASE_DIR` to point at the external cache/checkpoint root. On the JHU cluster used for these experiments this has been `/home/jhu/xwang457/work/nanochat_cache`.
- Checkpoints live outside the repo under:
  - `base_checkpoints/<tag>/model_<step>.pt` for pretraining.
  - `chatsft_checkpoints/<tag>/model_<step>.pt` for SFT.
  - matching `meta_<step>.json` files are required and should be treated as part of the checkpoint identity.
- Do not commit generated checkpoint files, Slurm logs, or large `fbt_experiments/results/*` directories unless the user explicitly asks. Prefer committing scripts/evaluators plus concise markdown summaries.

## Main code paths

- Model and latent feedback: `nanochat/gpt.py`
  - `LatentFeedback`
  - `LATENT_FEEDBACK_MODES`
  - `build_feedback_mask`
  - `GPT.forward(..., num_forward_passes=..., feedback_masks=..., feedback_jitter=...)`
- Decoding: `nanochat/engine.py`
  - `Engine.generate(..., decode_mode="standard"|"soft"|"fused")`
  - `standard`: ordinary prompt prefill and ordinary autoregressive decoding.
  - `soft`: ordinary prompt prefill, then recurrent latent-feedback decoding.
  - `fused`: ordinary prompt pass, fresh-cache fused prompt pass, then recurrent latent-feedback decoding.
- Training:
  - `scripts/base_train.py` for pretraining.
  - `scripts/chat_sft.py` for SFT.
  - `nanochat/loss_eval.py` for validation BPB and per-pass BPB.
- Experiment evaluators:
  - `fbt_experiments/evaluate_checkpoint.py` for continuation, CORE prefill, and GSM8K.
  - `fbt_experiments/evaluate_math500.py` for MATH-500 zero-shot chat eval.
  - `fbt_experiments/regrade_math500_math_verify.py` for symbolic MATH-500 regrading.
  - `fbt_experiments/evaluate_code.py` for HumanEval/MBPP style code eval.

## Latent-feedback invariants

- Preserve a checkpoint's `model_config.latent_feedback_mode`. SFT does not take a separate mode flag; it loads the mode from the source checkpoint.
- Do not compare LF checkpoint `standard` decoding to a separately trained standard model as if they had identical training objectives. In LF SFT/pretraining, later-pass losses backpropagate through first-pass hidden states.
- Current reproduction runs generally use:
  - pretraining K=2 variants from a standard checkpoint at step 40k;
  - SFT with `--num-forward-passes 3`;
  - `--no-feedback-prefix-mixin`;
  - `--feedback-jitter 0.02`;
  - `--save-every -1` for SFT when only the final checkpoint is wanted.
- `standard` and `soft` should share the ordinary prompt prefill. Their first generated token should match in greedy decoding; existing tests cover this invariant.
- `fused` prompt prefill must rebuild the KV cache from position zero. Do not append fused prefill onto the ordinary prefill cache.
- Weight tying is opt-in via `--weight-tying`; legacy checkpoints without `weight_tying` metadata are patched as untied.

## Evaluation conventions

- GSM8K full test has 1,319 examples. MATH-500 has 500 examples.
- For chat-template zero-shot math evals:
  - GSM8K: `--gsm8k-shots 0 --gsm8k-prompt-format chat`.
  - MATH-500: use `fbt_experiments.evaluate_math500`; it asks for final answers in `\boxed{}`.
- Greedy decoding means `temperature=0.0`; in these evaluators `top_k=None` for GSM8K/MATH-500 unless a task-specific script says otherwise.
- For MATH-500, prefer Math-Verify regrading when reporting final numbers. The lightweight evaluator is useful for quick signal but can miss symbolic equivalences.
- For paired modes on the same examples, report paired exact McNemar counts/p-values, not only independent binomial standard errors.

## Slurm conventions used here

- Use `srun --cpu-bind=none` under Slurm.
- Set `OMP_NUM_THREADS=1`.
- `unset SLURM_TRES_PER_TASK` before `torchrun`/single-GPU eval jobs to avoid GPU binding conflicts observed on this cluster.
- Eval jobs are single-GPU shards and have run on `a100,h100,h200,l40s` with `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=120G`.
- SFT jobs for d20 OpenMath train_5M have used 4 GPUs with `--gres=gpu:4`, `--cpus-per-task=32`, `--mem=360G`.
- Submit merge/regrade jobs with `--dependency=afterok:<eval_job>` so final result dirs are created only after all shards succeed.

## Validation commands

Run the narrow tests for the subsystem you touched. Useful sets:

```bash
.venv/bin/python -m pytest tests/test_latent_feedback.py tests/test_feedback_schedule.py tests/test_feedback_decoding.py tests/test_weight_tying.py -q
.venv/bin/python -m pytest tests/test_gsm8k_sharding.py tests/test_merge_gsm8k_shards.py -q
.venv/bin/python -m py_compile fbt_experiments/evaluate_checkpoint.py fbt_experiments/evaluate_math500.py fbt_experiments/merge_gsm8k_shards.py fbt_experiments/merge_math500_shards.py
```

Use `rg` first for repo search. Preserve unrelated dirty worktree changes; many experiment outputs may be intentionally untracked.
