---
name: latent-feedback-evaluation
description: Evaluate Nanochat latent-feedback checkpoints on GSM8K, MATH-500, CORE prefill, or code tasks with standard/soft/fused decoding and reproducible shard merging.
---

# Latent-feedback evaluation

Use this skill when the task is to evaluate a checkpoint, merge evaluation shards, regrade saved generations, or explain LF decoding results.

## Decoding modes

All mode names are passed via `--decode-modes` where supported.

- `standard`: ordinary prompt prefill and ordinary autoregressive decoding.
- `soft`: ordinary prompt prefill, then recurrent latent-feedback decoding.
- `fused`: ordinary prompt pass, fresh-cache fused prompt pass, then recurrent latent-feedback decoding.

Only latent-feedback checkpoints can run `soft` or `fused`. Standard checkpoints should be evaluated with `--decode-modes standard`.

## Main evaluators

- GSM8K and continuation/core-prefill probes: `fbt_experiments/evaluate_checkpoint.py`
- GSM8K shard merge: `fbt_experiments/merge_gsm8k_shards.py`
- MATH-500 zero-shot chat eval: `fbt_experiments/evaluate_math500.py`
- MATH-500 shard merge: `fbt_experiments/merge_math500_shards.py`
- MATH-500 symbolic regrade: `fbt_experiments/regrade_math500_math_verify.py`
- HumanEval/MBPP-style code eval: `fbt_experiments/evaluate_code.py`

## Math benchmark conventions

- GSM8K full test is 1,319 examples.
- MATH-500 full test is 500 examples from `HuggingFaceH4/MATH-500`, `default/test`.
- Zero-shot chat GSM8K requires:

  ```bash
  --gsm8k-shots 0 --gsm8k-prompt-format chat
  ```

- MATH-500 prompts are chat-template prompts with an instruction to put the final answer in `\boxed{}`.
- Use greedy decoding for comparable math results: `temperature=0.0`; the GSM8K/MATH-500 evaluators use `top_k=None`.
- Use Math-Verify for final MATH-500 reporting. If missing, install in the active `uv` venv with:

  ```bash
  uv pip install --python .venv/bin/python 'math-verify[antlr4_13_2]'
  ```

## Sharding pattern

For full math evals, shard by contiguous global example indices and merge after all shards finish. Existing scripts use 8 shards per model.

Single-GPU eval Slurm spec used here:

```bash
#SBATCH --partition=a100,h100,h200,l40s
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
```

Inside scripts:

```bash
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache
unset SLURM_TRES_PER_TASK
source .venv/bin/activate
```

Submit merges with `--dependency=afterok:<eval_job>`. For MATH-500, submit Math-Verify regrading after the merge job, not after individual shards.

Representative scripts:

- `runs/eval_openmath_train5m_anygpu_gsm8k_0shot_chat_array.slurm`
- `runs/eval_openmath_train5m_anygpu_math500_0shot_chat_array.slurm`
- `runs/eval_openmath_train5m_anygpu_linear_addition_gsm8k_0shot_chat.slurm`
- `runs/eval_openmath_train5m_anygpu_linear_addition_math500_0shot_chat.slurm`

## Reporting

Report:

- correct/total and accuracy for each mode;
- Wilson interval or binomial standard error for single-mode uncertainty;
- paired exact McNemar counts/p-values for mode comparisons on the same examples;
- prompt format, shot count, max tokens, and whether Math-Verify was used.

Do not interpret independent error bars as the uncertainty of paired differences. For `standard` vs `soft` vs `fused` on the same examples, paired McNemar evidence is the relevant comparison.

## Validation after evaluator changes

Run focused checks:

```bash
.venv/bin/python -m py_compile fbt_experiments/evaluate_checkpoint.py fbt_experiments/evaluate_math500.py fbt_experiments/merge_gsm8k_shards.py fbt_experiments/merge_math500_shards.py fbt_experiments/regrade_math500_math_verify.py
.venv/bin/python -m pytest tests/test_gsm8k_sharding.py tests/test_merge_gsm8k_shards.py tests/test_feedback_decoding.py -q
```

If changing decoding semantics, also run latent-feedback tests:

```bash
.venv/bin/python -m pytest tests/test_latent_feedback.py tests/test_feedback_decoding.py -q
```
