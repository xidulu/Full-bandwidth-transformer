# Latent-feedback decoding experiments

This directory contains a reproducible checkpoint-level evaluation of the
SOFT and FUSED decoding algorithms from [*Full-Bandwidth Transformer*](https://arxiv.org/abs/2608.08888).

`evaluate_checkpoint.py` runs three matched modes:

- `standard`: ordinary prompt prefill and ordinary autoregressive decoding.
- `soft`: ordinary prefill, followed by recurrent latent-feedback decoding.
- `fused`: ordinary prompt pass, a fresh-cache fully fused second prompt pass,
  then recurrent latent-feedback decoding.

The primary quantitative probe is sequential teacher-forced BPB on fixed
held-out documents. Greedy raw-text GSM8K probes supply paired behavioral
samples across the three decoding modes.

## Reproduce

From the nanochat repository root:

```bash
NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache \
  .venv/bin/python fbt_experiments/evaluate_checkpoint.py \
  --checkpoint /home/jhu/xwang457/work/nanochat_cache/base_checkpoints/d16-lf-k1-smoke/model_010000.pt \
  --output-dir fbt_experiments/results/model_010000 \
  --num-docs 32 \
  --prefix-tokens 64 \
  --continuation-tokens 64 \
  --num-gsm8k 20 \
  --gsm8k-shots 8 \
  --max-new-tokens 192
```

For the latest checkpoint in `d16-lf-k1-smoke`, evaluated on 200 five-shot
GSM8K problems only:

```bash
NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache \
  .venv/bin/python fbt_experiments/evaluate_checkpoint.py \
  --checkpoint /home/jhu/xwang457/work/nanochat_cache/base_checkpoints/d16-lf-k1-smoke/model_050000.pt \
  --output-dir fbt_experiments/results/model_050000_gsm8k_5shot_200 \
  --skip-continuation \
  --num-gsm8k 200 \
  --gsm8k-shots 5 \
  --max-new-tokens 192
```

The evaluator infers `NANOCHAT_BASE_DIR` from the checkpoint path, so the
environment assignment above is explicit documentation rather than a strict
requirement. Given a directory, it selects the highest step with a matching
model/meta pair; given an exact model path, it always pairs it with the metadata
at the same step. This matters because the custom-tag directory contains
multiple architectures, including an older non-feedback checkpoint.

Each result directory contains:

- `checkpoint_meta.json`: copied matching checkpoint metadata.
- `run_config.json` and `run.log`: provenance and complete run log.
- `continuation_details.jsonl`: per-document, per-mode likelihood/stability.
- `gsm8k_generations.jsonl`: prompts, completions, parsed answers, and latency.
- `metrics.json`: machine-readable aggregates.
- `summary.md`: concise human-readable results and limitations.

Full evaluations may be split with `--gsm8k-start` and merged with
`merge_gsm8k_shards.py`. A merged result also includes `merge_manifest.json`,
which records shard ranges and SHA-256 hashes; the original shard directories
retain their individual logs and configurations.

The completed step-10,000 run is in [`results/model_010000`](results/model_010000/summary.md).
On 2,048 held-out continuation tokens, FUSED is effectively neutral/slightly
favorable (-0.035% BPB, 18/32 document wins), while SOFT is worse (+0.323%,
14/32 wins). Both paths are stable and satisfy their decoding invariants, but
this small probe does not establish a statistically persuasive quality gain.

The earlier step-40,000 five-shot GSM8K run is in
[`results/model_040000_gsm8k_5shot_200`](results/model_040000_gsm8k_5shot_200/summary.md).
On the first 200 problems, STANDARD scored 3/200, SOFT 5/200, and FUSED 4/200;
the paired differences are not statistically significant.

The requested latest-directory evaluation selected step 50,000 and is in
[`results/model_050000_gsm8k_5shot_200`](results/model_050000_gsm8k_5shot_200/summary.md).
On the same five-shot, first-200 protocol, STANDARD scored 6/200, SOFT 3/200,
and FUSED 2/200; none of the paired differences is statistically significant.
Despite the directory/model tag, its metadata records a 20-layer, width-1,280
model (about 900M parameters), so it is not a continuation of the step-40,000
16-layer architecture.

The full 1,319-problem five-shot evaluation is in
[`results/model_050000_gsm8k_5shot_full`](results/model_050000_gsm8k_5shot_full/summary.md).
STANDARD scored 35/1,319 (2.65%), SOFT 30/1,319 (2.27%), and FUSED 29/1,319
(2.20%). The paired exact p-values were 0.568, 0.430, and 1.0 respectively;
this run shows no statistically significant accuracy difference or feedback-
decoding advantage. The eight validated source shards are retained in
`results/model_050000_gsm8k_5shot_full_shards/`.

The requested checkpoint used three-pass training only for its last 25% of
steps and disabled prefix mixin. Consequently, FUSED matches its trained
full-feedback prompt regime, while SOFT's switch from a plain prompt to fused
generation is out of distribution. Interpret a weak SOFT result accordingly.

## Paper-reproduction additions in this branch

The repo now includes the core pieces needed to reproduce small-scale
Full-Bandwidth Transformer experiments:

- `nanochat.gpt.LatentFeedback`, the latent-feedback fusion module.
- Multi-pass training objectives in base pretraining and chat SFT.
- Optional pass scheduling, e.g. one-pass warmup followed by three-pass
  feedback training.
- Training-time and validation-time logging for `L1`, `L2`, and `L3`.
- SOFT and FUSED decoding paths in `nanochat.engine.Engine.generate`.
- Optional input/output embedding weight tying.
- OpenMathInstruct-2 and Stack-Edu SFT task loaders and Slurm launch scripts.

The main experiment scripts are:

- `fbt_experiments/evaluate_checkpoint.py`: GSM8K and continuation/core-prefill
  probes across STANDARD, SOFT, and FUSED decoding.
- `fbt_experiments/evaluate_code.py`: raw Python continuation evaluation for
  HumanEval and MBPP.
- `fbt_experiments/merge_gsm8k_shards.py` and
  `fbt_experiments/merge_code_eval.py`: deterministic shard mergers.

The current Stack-Edu Python SFT checkpoint evaluation is summarized in
[`results/d20_lf_k3_stackedu_004407_code_eval_all_modes_final/summary.md`](results/d20_lf_k3_stackedu_004407_code_eval_all_modes_final/summary.md).
With greedy decoding, temperature 0, and `max_new_tokens=512`, the results were:

| benchmark/mode | pass@1 |
|---|---:|
| HumanEval / STANDARD | 25/164 (15.24%) |
| HumanEval / SOFT | 24/164 (14.63%) |
| HumanEval / FUSED | 26/164 (15.85%) |
| MBPP full / STANDARD | 56/500 (11.20%) |
| MBPP full / SOFT | 57/500 (11.40%) |
| MBPP full / FUSED | 81/500 (16.20%) |

For code-generation reproduction, use the checked-in Slurm scripts under
`runs/stackedu_*.slurm` and `runs/eval_stackedu_code*.slurm`, or invoke the
evaluator directly:

```bash
NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache \
  .venv/bin/python -m fbt_experiments.evaluate_code \
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
  --output-dir fbt_experiments/results/repro_code_eval
```
