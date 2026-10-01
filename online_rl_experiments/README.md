# Fully online RL on DAPO-Math-17k

Starting checkpoint: `d20-standard-60k-openmath-train5m-k1-anygpu/model_004407.pt`
and its matching `meta_004407.json`, under
`$NANOCHAT_BASE_DIR/chatsft_checkpoints`. The source evaluation records **642/1319
(48.6732%) GSM8K**, zero-shot chat, greedy, 192 new tokens, no calculator:
`../fbt_experiments/results/d20_standard_60k_openmath_train5m_k1_anygpu_004407_gsm8k_0shot_chat_full/metrics.json`.
Although this checkpoint contains dormant latent-feedback parameters, its SFT
objective used one pass. The trainer preserves its model configuration and tied
weights, freezes dormant feedback parameters, and uses standard one-pass decoding.

`main.py` defaults to **vLLM 0.14.0** for rollouts, using the existing Nanochat
adapter in `../vllm_plugin`. Each training rank owns an isolated, single-GPU vLLM
subprocess on its GPU. All four local prompts × eight responses are submitted
together for continuous batching. Every iteration synchronizes the current
weights, completes fresh rollouts, accumulates gradients, and makes **one** AdamW
update before the next rollout. There is no asynchronous sampler or replay buffer.
The subprocess acknowledges the policy version before generation; unsynchronized
generation is rejected. Persistent shared CPU buffers transfer weights without
writing checkpoints. The trainer retains FP32 master weights; vLLM uses BF16
inference weights and the adapter's FP32 scalars, with eager execution and prefix
caching disabled. Sampled terminal tokens are retained for the loss. Greedy
GSM8K evaluation continues to use the native Engine for continuity.
Temperature is 1 with no top-k filtering, matching the policy used for likelihoods.
Different sampling kernels and per-response seeds mean vLLM will not reproduce
the native engine's stochastic completions bit for bit. Use `--rollout-engine native`
to reproduce the previous rollout implementation. `--generation-batch-size` applies
only to native rollouts; vLLM schedules all local responses together.

For prompt group i, `A_i = (reward_i - mean(reward)) / max(std(reward), 1e-6)`.
The loss is `sum(response_token_NLL * A_i) / total_response_tokens_in_entire_batch`.
All accumulation microbatches on all ranks share the same global token denominator.
DDP averages gradients, so each rank scales its local loss by the world size.
All but the final backward pass use `no_sync()`; the final pass synchronizes gradients.
Each rank samples distinct complete prompt groups and uses the same updated policy
for the next batch. A zero-signal rank still participates in synchronization. Prompt and padding tokens
are excluded; sampled terminal tokens are included. This is DAPO-style token
normalization with a simple on-policy policy-gradient update, not the full DAPO
recipe (no dynamic resampling, clipping, reference KL, or overlength shaping).
Zero-variance groups contribute zero gradient; entirely zero-signal batches skip
the optimizer to avoid momentum-only updates. All sampled response tokens still
count in the denominator. Rewards use [Math-Verify](https://github.com/huggingface/Math-Verify)
0.9.0 through `verifier.py`, for both extraction and symbolic equivalence. Gold
answers are parsed as LaTeX; predictions use Math-Verify's LaTeX and expression
extractors with answer anchors required (boxed answers are supported). Thus
`84/2`, `sqrt(1764)`, and `42` can be equivalent answers. There is no custom regex
reward grader or raw-string fallback. Only the highest-priority first match is
parsed. Missing/unparseable predictions, verifier errors/timeouts, and truncated
training responses receive zero reward. Unparseable gold references fail the run
as a data error. `--parse-timeout` and `--verify-timeout` default to five seconds.
Package versions and parser configuration are recorded in run metadata; parsed
expressions and errors are stored with rollouts, and W&B logs the verifier error rate.

The official [DAPO dataset](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
is pinned to revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`.
Its parquet contains 1,791,700 repeated rows. `prepare_data.py` deduplicates exact
prompts and excludes seven prompts with inconsistent integer labels, leaving
17,391 unique training prompts. Original prompt text is preserved. The prepared
manifest records hashes and counts. The trainer filters prompts longer than 1,024
tokens, records the count, shuffles deterministically, and traverses without
replacement until the next epoch. The default 100-step run is an initial experiment,
not a full epoch over the dataset.

From this directory, using the parent's CUDA environment (`uv sync --extra gpu
--group dev` at repository setup):

```bash
# The existing W&B 0.21.3 does not support protobuf 7; use a local overlay.
uv pip install --python ../.venv/bin/python --target .deps 'protobuf==6.33.5' -r requirements-verifier.txt
# Already installed in the shared environment; install only when setting up a new one.
uv pip install --python ../.venv/bin/python -e ../vllm_plugin
mkdir -p data logs results
curl -fL --retry 3 \
  https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/65877096c24ffa7abc4e4fa5edb95cf3413a5674/data/dapo-math-17k.parquet \
  -o data/dapo-math-17k.parquet
../.venv/bin/python prepare_data.py
PYTHONPATH=.deps:.. ../.venv/bin/python -m pytest test_online_rl.py test_verifier.py test_vllm_rollout.py -q -o cache_dir="$PWD/.pytest_cache"
sbatch validate_vllm.slurm
sbatch train.slurm
```

Defaults: **4 H100 or A100 GPUs**, 100 rollout batches, **16 prompts × 8 responses
= 128 sequences per update**, up to 1,024 new tokens, context at most 2,048. Each
GPU uses **microbatch 8 × 4 gradient accumulation steps = 32 local sequences**.
`--prompts-per-step` is global and inferred from this layout; an inconsistent
explicit value is rejected. AdamW LR remains `1e-6`, with no weight
decay, gradient clipping at 1. Parameters and optimizer state use FP32; model
activations use the repository's selected compute dtype. Checkpoints every 25
steps and at the end contain model, optimizer, metadata, source hashes and config.
Use `--resume-from path/to/model_STEP.pt` to restore model, replicated AdamW
state, update counter, and deterministic shuffled data position. `--steps` is the
absolute final step, including already completed steps. Rollout seeds depend on
global step and prompt index. The saved data/batch/optimizer/verifier settings are
validated on resume; source-code fixes may change without changing those settings.
A resumed job gets a new W&B run with the original global step numbers, preserving
the failed run history.
Changing a resumed run's rollout engine requires `--allow-rollout-engine-change`;
it is recorded as an explicit experiment branch. Resume older runs with
`--rollout-engine native` for the original engine.

vLLM reserves 4 GiB of KV cache per GPU (`--vllm-kv-cache-gb`) and schedules at
most 2,048 tokens per forward (`--vllm-max-batched-tokens`). Its model and cache
remain resident alongside the trainer; it does not allocate extra GPUs. The
existing `perf/peak_gpu_gb` metric measures the trainer process only and is also
logged as `perf/trainer_peak_gpu_gb`. Weight synchronization, generation, and
verification are timed separately. Adapter sources and their hashes are saved
with every run. `validate_vllm.slurm` checks batched greedy parity and changed-weight
loading, then performs two full-length training updates on four GPUs with every
loaded parameter audited (`--vllm-verify-weights`). The adapter uses vLLM's
[worker RPC interface](https://docs.vllm.ai/en/v0.14.0/api/vllm/) inside its isolated process.

For a probability-level training/inference sanity check, run
`sbatch check_train_inference.slurm`. It checks the original SFT checkpoint and
the fixed RL step-250 checkpoint on 16 deterministic DAPO prompts × eight fresh
responses, with the normal 1,024-token cap and 32 concurrent local responses.
It compares vLLM's sampled-token log-probabilities against the actual autograd-enabled
trainer forward (microbatch eight, identical token histories/masks, FP32 master
weights), measures token likelihood ratios and argmax agreement, and compares
the original gradient against a fixed per-token importance-weighted diagnostic.
Native cached decoding on two groups serves as a control. No optimizer step is
taken. Per-token records, source hashes, and summaries are saved under
`results/train-inference-<job>/`. The nonnegative `k3_mean` is a sampled KL
estimator, not an exact full-vocabulary KL. The gradient comparison applies only
token corrections; it does not correct the distribution of preceding histories.

W&B project: `nanochat-online-rl`, using the current login's default entity unless
`--entity` is supplied. Online mode is the default; local metrics are also written
as JSONL. Logs include reward accuracy, sampled group pass@8, mixed-reward group
fraction, all-wrong/all-correct/mixed group counts and fractions, parse/truncation rates, response lengths, token-normalized loss, response
NLL (not exact entropy), gradient norm, learning rate, optimizer update count,
GPU memory, rollout/training duration, rollout tokens/sec, world size, global batch,
per-GPU microbatch, and accumulation steps. Metrics are aggregated across ranks
and only rank 0 creates a W&B run. All completions, rewards, parsed answers and
termination flags are saved in per-rank JSONL for reward auditing.

GSM8K checks use the first 128 test examples at step 0 and every 25 steps, then
all 1,319 examples after the final update. Evaluation is sharded across ranks
and merged by rank 0 with an exact index-coverage check. All use the baseline's 192-token greedy
zero-shot chat convention. Subset results are monitoring only and must not be
compared as full-test accuracy. GSM8K checks now also use Math-Verify; their saved
records identify the verifier. The historical 48.67% starting score and previous
run scores used the old numeric regex grader, so use the newly graded step-zero
baseline when comparing the revised verifier's scores. Benchmark grading checks
answer correctness at the token limit without requiring a terminal token.
GSM8K answers are never used for training.

Outputs are isolated in `results/<Slurm job ID>/`: `run_config.json`,
`wandb_run.json`, `metrics.jsonl`, `rollout_samples.rankNNNN.jsonl`, greedy evaluation JSONL,
`checkpoints/`, and `completed.json` only after successful completion. Large data,
checkpoints, local dependencies, W&B files and Slurm logs are git-ignored.

Overrides are forwarded by the launcher, for example:

```bash
sbatch --time=00:30:00 train.slurm --steps 2 \
  --eval-examples 8 --final-eval-examples 8
sbatch train.slurm --steps 500 --project nanochat-online-rl
```

The original single-GPU run `855770` completed 100 batches (52 nonzero-signal
updates), with 647/1319 GSM8K (49.05%). Its original source snapshot and outputs
remain under `results/855770/`; `run_receipt.json` identifies that run. The scaled
experiment restarts from the same step-4407 SFT checkpoint, preserving the starting
point and learning rate. At 100 batches it samples four times as many prompts and
responses as the original run, so the runs are not compute-matched.

`train_single_gpu.slurm` remains available. To reproduce the original batch layout
with the distributed-capable trainer, pass `--microbatch-size 1
--gradient-accumulation-steps 32`. The four-GPU launcher explicitly passes
`--microbatch-size 8 --gradient-accumulation-steps 4`.

The eight-times-larger vLLM experiment uses `sbatch train_large_batch.slurm`:
**128 prompts × 8 responses = 1,024 responses per update**, still on four H100/A100
GPUs. Each GPU uses microbatch 8 × **32 accumulation steps** = 256 responses.
This keeps activation memory near the previously tested microbatch size.
`--vllm-max-sequences 64` caps concurrent generation independently of the local
256-response batch, with 8 GiB of KV cache per GPU. All queued responses finish
before the optimizer update. LR stays `1e-6`; Math-Verify and token normalization
are unchanged. The run starts from the original step-4407 SFT checkpoint.

The launcher first trains steps 1–2 with all inference weights audited, saving
to `results/<job>-preflight/`. Only if that stage succeeds does it restore the
model, optimizer, and data position and continue steps 3–300 in `results/<job>/`.
The two stages have separate W&B runs; together they contain 300 rollout batches,
38,400 prompt draws and 307,200 responses. The Slurm time limit is 24 hours.

Four-GPU job `870835` was cancelled at the user's request after batch 28; its
step-25 checkpoint and original verifier/source snapshots remain preserved.
`run_receipt_4gpu.json` records cancellation. Replacement job `870904` uses
Math-Verify and the group-count metrics below; it starts from the original SFT
checkpoint with the same four-GPU batch configuration.

Per-batch group metrics are reduced across all four ranks: `reward/all_wrong_groups`
(no rewarded responses), `reward/all_correct_groups` (all eight responses rewarded),
and `reward/mixed_groups` (some but not all rewarded). They sum to the 16 prompts
in an update. Corresponding `reward/*_group_fraction` metrics sum to one. These
counts use final training rewards: truncations and verifier errors count as wrong.
Source snapshots of the trainer, verifier, and launcher are saved in each new run.

Resume the failed Math-Verify run from its last saved checkpoint and train through
step 300 (250 additional batches):

```bash
sbatch --job-name=std-dapo-resume300 train.slurm \
  --resume-from results/870904/checkpoints/model_000050.pt --steps 300 --rollout-engine native
```

The Math-Verify timeout handler now explicitly catches its `TimeoutException`,
which inherits from `BaseException`; the timeout is recorded and receives zero
reward. The regression suite includes the real signal-based timeout path.

Resume job `875331` restores step 50 of run `870904` and targets global step 300.
`run_receipt_resume_300.json` records the checkpoint/optimizer hashes and W&B link.


## ORZ dataset replacement

`prepare_orz_data.py` converts `Open-Reasoner-Zero/orz_math_72k_collection_extended`
(revision `b5c5890bcf04853531d4f2aeeef18fb7af6cabd1`) to the trainer JSONL
schema. It preserves question text and symbolic reference answers, appends the
MATH-500 boxed-answer instruction, deduplicates exact questions, excludes
conflicting labels, and validates each distinct reference with Math-Verify
(including a boxed-answer round trip). Unsupported references are excluded and
recorded in a rejection artifact. The manifest records counts and source/prepared
hashes. The trainer applies its existing 1,024-token prompt limit.

```bash
OMP_NUM_THREADS=1 ../.venv/bin/python prepare_orz_data.py --download --workers 8
sbatch train_orz_large_batch.slurm
```

The ORZ launcher starts fresh from original standard SFT checkpoint 004407 and
runs 300 total steps on four H100/A100 GPUs. It retains vLLM rollouts, 128 prompts
x 8 responses per update, per-GPU microbatch 8 and accumulation 32, learning rate
1e-6, Math-Verify, and DAPO-style token-normalized loss. The first two updates
verify all weight transfers; their model and optimizer are restored for steps
3–300. Both stages log to W&B project `nanochat-online-rl`.

The previous DAPO job 891102 was cancelled at the user's request after step 75;
its step-75 checkpoint and prior results are preserved.


### ORZ base-model subset evaluation

`sbatch evaluate_orz_subset.slurm` evaluates original SFT checkpoint 004407 on
512 eligible ORZ questions selected without replacement with seed 20260924.
Each question receives one greedy response and eight independent temperature-1
samples, using the training prompt, 1,024-token response cap, vLLM, and Math-Verify.
The evaluator saves all generations; merging checks exact question coverage and
logs numeric metrics to W&B. Confidence intervals use question-level variability
so repeated samples of one question are not counted as independent questions.
There are no optimizer updates. Results live under `results/orz-baseline-<job>`.

The ORZ preflight completed two updates in job 891614. Its continuation failed
because the original monitor created the trainer output directory too early.
The fixed monitor writes separately to `results/monitoring/<job>`, preserving the
trainer's overwrite protection. `train_orz_large_batch.slurm` accepts a saved ORZ
checkpoint as its first argument for exact model/optimizer/data-order resumption.


## Hendrycks MATH benchmark training run

`prepare_math_data.py` uses only the `train` split of
`nlile/hendrycks-MATH-benchmark`, pinned to revision
`465bcdb36f5962aa3512891498966df785fc3c18`. This repository defines a
12,000/500 train/test split; original `unique_id` prefixes do not determine the
current split. Prompts contain only the question and a boxed-answer instruction;
reference solutions are never included. Original subject, level, and IDs are
preserved. Preparation reuses exact-question deduplication and Math-Verify
reference validation from the ORZ preparation utilities.

Of 12,000 source rows, two have invalid/empty fields, one duplicates a question,
and 18 have unsupported references. This leaves 11,979 prepared rows, of which
11,960 fit the 1,024-token prompt budget. The source training split has no ID or
exact-question overlap with the existing 500-example MATH-500 evaluation.
Counts, hashes, exclusions, context checks, and overlap checks are saved in
`data/hendrycks-math-benchmark.train.*` artifacts.

```bash
OMP_NUM_THREADS=1 ../.venv/bin/python prepare_math_data.py --download
sbatch train_math_large_batch.slurm
```

Job 891864 starts fresh from original SFT checkpoint 004407, with four H100/A100
GPUs, vLLM, 128 prompts x eight responses/update, microbatch eight per GPU,
accumulation 32, Math-Verify, learning rate 1e-6, and 300 total steps. The first two
updates audit weight transfer and are retained when continuing through step 300.
The existing all-wrong/all-correct/mixed metrics and GSM8K evaluation schedule
remain enabled in W&B project `nanochat-online-rl`. ORZ continuation 891757 was
cancelled before it started at the user's request.


## MATH-500 after MATH RL step 300

Training job 899298 completed 300 optimizer updates. Evaluation array 904082
used `evaluate_rl_math500.py`, which adapts checkpoint-path resolution and keeps
the existing native evaluator's prompts and generation unchanged. Eight disjoint
single-GPU shards covered all 500 questions with greedy standard decoding and
512 response tokens. The dependent grading allocation 904083 was blocked by a
CPU quota; after verifying all shards succeeded, merge/grading ran in the existing
CPU allocation 904047 instead.

`report_math500_post_rl.py` applies the same Math-Verify grader to new generations
and the saved SFT baseline. RL accuracy is **200/500 (40.0%)**, versus
**173/500 (34.6%)** for SFT: **+5.4 percentage points**. Paired counts are 143
both-correct, 30 baseline-only, 57 RL-only, and 270 both-wrong; exact McNemar
p=0.0050136. Baseline generations are historical and GPU hardware differs.
Results, all generations, and hashes are in `results/math500-rl-904082` and
`run_receipt_math500_post_rl.json`. Numeric metrics were logged to
https://wandb.ai/xidulu-umass-amherst/nanochat-online-rl/runs/a8fjnjdm .

## Soft post-training with detached three-pass scores

`--decode-mode soft` uses the checkpoint's existing latent-feedback fusion for
both vLLM rollouts and teacher-forced training. `train_soft_math.slurm` selects
`d20-from40k-lf-k2-gate_product-openmath-train5m-k3-anygpu/model_004407.pt`
with its matching metadata, preserving `latent_feedback_mode=gate_product`.
This is the LF-trained SFT checkpoint, not the standard model's RL checkpoint.

`soft_likelihood.py` implements the requested score estimator:

1. Run the ordinary embedding inputs through the Transformer under `no_grad`.
2. Mix pass-1 hidden state at **t−1** with the embedding at **t**, then run the
   Transformer again under `no_grad`.
3. Mix detached pass-2 hidden state at **t−1** with a freshly computed embedding
   at **t**, and run the Transformer and output head with gradients enabled.

Only pass 3 produces logits and contributes a loss. Gradients reach its token
embeddings, active feedback matrices, Transformer, and output head. Passes 1–2
build no autograd graph. Prompt positions remain ordinary in every pass, and
feedback starts at the first generated *input* token. BOS and padded positions
are excluded. There is no feedback jitter or random prefix masking. Thus the
first response token uses the same ordinary prompt prefill as standard decoding.
The first three response scores match recurrent soft scores in exact arithmetic;
longer histories are a finite three-pass approximation to the recurrent rollout
policy. Detaching the earlier passes also deliberately changes the gradient.
This is the requested surrogate objective, not an exact recurrent policy gradient.

The existing Math-Verify rewards, group-standardized advantages, and DAPO-style
normalization over **all response tokens across all ranks and microbatches**
are retained. Active feedback weights are optimized and synchronized to vLLM;
inactive feedback matrices stay frozen. Checkpoints record the scoring mode and
reject resumes that silently change it. W&B additionally logs:

- `train/likelihood_forward_passes` (3), `train/gradient_forward_passes` (1).
- `policy/three_pass_minus_rollout_logp_mean` and
  `policy/three_pass_vs_rollout_logp_abs_mean`.
- `policy/token_ratio_outside_0p8_1p2`, the fraction of sampled tokens whose
  three-pass/rollout probability ratio lies outside [0.8, 1.2].
- Existing reward accuracy, all-wrong/all-correct/mixed group counts, response
  lengths, truncation, gradient norm, throughput, and GPU memory metrics.

The launcher uses the prepared `nlile/hendrycks-MATH-benchmark` **training split**,
4 H100/A100 GPUs, 128 prompts × 8 responses = 1,024 responses per update,
microbatch 8 per GPU and 32 accumulation steps, LR 1e-6, and 300 total steps.
The first two steps audit all weight transfers and then resume through step 300.
Soft concurrency is capped conservatively by full-context KV capacity because
the current adapter cannot replay recurrent feedback state after preemption.
The launcher requests 32 concurrent sequences per GPU with an 8 GiB KV cache;
this scheduling limit does not reduce the optimizer batch.

```bash
# CPU semantics, gradient isolation, and distributed accumulation tests:
OMP_NUM_THREADS=1 PYTHONPATH=.deps:.. ../.venv/bin/python -m pytest \
  test_soft_likelihood.py test_online_rl.py test_vllm_rollout.py -q
# Short GPU acceptance test; uses synthetic advantages, not an accuracy eval:
sbatch validate_soft.slurm
# Full experiment (separate submission):
sbatch train_soft_math.slurm
```

`validate_soft.py` checks a real LF checkpoint, fresh soft vLLM sampling, native
recurrent versus three-pass scores on identical token histories, backward,
a disposable optimizer update, and exact transfer of all changed weights.
It writes `metrics.json` and `tokens.json` under `results/soft-validation-JOBID`.

Validation completed in job **904477** on an H100 NVL: both active feedback
matrices received nonzero gradients and changed after the disposable update;
vLLM verified all transferred weights. Across two 64-token continuations,
native recurrent and vLLM sampled-token log-probabilities matched exactly.
Three-pass versus recurrent mean absolute log-probability error was **0.010566
nats/token**, with 0/128 ratios outside [0.8, 1.2]; the first three scores matched.
This small acceptance sample is not a dataset-wide mismatch estimate. Detailed
results and checkpoint/source hashes: `results/soft-validation-904477/metrics.json`.
CPU validation: 70 selected unit/regression tests plus the separate two-process
DDP accumulation test passed.

## Replaying rollout hidden states

Use `--decode-mode soft --rollout-engine vllm --soft-likelihood hidden_state_replay`
to replace the three-pass estimate with a single differentiable forward using
recorded recurrent feedback. `train_soft_replay_math.slurm` has the same LF source,
MATH training data, 4 GPUs, batch 1,024, 300-step horizon and
`enalisn1_fall2026` account as the three-pass launcher. It is a separate experiment;
the default remains `three_pass_detached`, including job 904517.

The vLLM API is opt-in:

```python
engine = VLLMRollout(..., decode_mode='soft', hidden_replay=True)
groups = engine.generate_groups(
    prompts, seeds, n, max_tokens, policy_version,
    return_hidden_states=True,
)
# Each group is (suffixes, ended, scores).
# scores[i]['hidden_states']: detached CPU tensor [response_length, hidden_size]
# scores[i]['logprobs']: sampled-token log-probabilities, including terminal tokens.
```

Hidden row `j` is the final normalized Transformer state that predicted response
token `j`. Row 0 comes from the last ordinary prompt position. To score response
token `j+1`, `pack_replay` supplies row `j` as feedback when processing input token
`j`. Prompt inputs remain ordinary. The last returned state is not required as
feedback for this teacher-forced forward, but is included so states and sampled
tokens have the same length. A one-token response needs no feedback.

The local `replay_worker.py` extends the existing Nanochat vLLM runner. Capture
tracks absolute positions and exact internal request IDs, handles chunked prompt
prefill and changing active-batch order, and retains completed requests until the
batch is returned. Alignment mismatches fail the rollout. States are copied in
batches to CPU and discarded after each rollout batch; only the current scoring
microbatch is transferred to the trainer GPU. No recurrence graph or vLLM KV
cache is transferred. At 256 responses/GPU × 1,024 tokens × 1,280 hidden dimensions,
BF16 replay tensors occupy about **640 MiB of CPU memory per rank** (excluding
transient copies). Capturing states adds device-to-host transfer overhead.

With unchanged model weights, this recreates the recurrent forward inputs in
one training pass, up to numerical/backend differences. It does **not** recreate
full backpropagation through recurrent states: those states are detached.
Gradients still reach the embeddings, active fusion matrices, Transformer and
output head through the differentiable training pass. Keep one update per fresh
rollout batch; replaying states after additional parameter updates would make
the feedback stale. The policy-version checks and resume-estimator checks remain
active. This feature supplies training feedback; it does not enable vLLM cache
preemption/recomputation, so the conservative concurrency cap remains necessary.

W&B records `train/likelihood_forward_passes=1`,
`policy/replay_minus_rollout_logp_mean`,
`policy/replay_vs_rollout_logp_abs_mean`,
`policy/token_ratio_outside_0p8_1p2`, and
`rollout/hidden_replay_bytes_per_rank`, alongside existing RL metrics.

Validate with `test_hidden_replay.py` and the distributed tests in
`test_soft_likelihood.py`; `sbatch validate_hidden_replay.slurm` checks the real
checkpoint's vLLM capture, score agreement, backward, and updated-weight transfer.

Replay validation **904856** completed on an H100 80GB. All 78 selected CPU
unit/regression tests and both two-process DDP tests passed. Captured states
matched native recurrent states exactly on 128 sampled tokens; replay used one
differentiable trunk pass and updated both active feedback matrices. On identical
histories, mean absolute token log-probability errors were 0.010316 for replay and
0.010069 for the three-pass control; neither had ratios outside [0.8, 1.2]. This
small check demonstrates correct state capture, not a reduction in BF16/backend
numerical differences. Artifacts: `results/replay-validation-904856/metrics.json`
and `rollout_hidden_states.pt`.

A separate tiny real-data trainer check completed step 1, saved a checkpoint,
restored optimizer/data state, and completed step 2. Its short responses were
all truncated, so both batches correctly skipped zero-signal optimizer updates;
the standalone synthetic-advantage check verified nonzero updates and weight
transfer. W&B runs:
[initial step](https://wandb.ai/xidulu-umass-amherst/nanochat-online-rl/runs/k2punqdl),
[resumed step](https://wandb.ai/xidulu-umass-amherst/nanochat-online-rl/runs/0bczg8my).
No full replay training run was submitted.

## Continue standard and three-pass LF training on Big-Math Verified

`prepare_big_math_data.py` pins `SynthLabsAI/Big-Math-RL-Verified` to revision
`c75d2f117cddfecb6bd08756e61e508e59732b21` and reads only its training parquet.
It deduplicates exact questions, excludes conflicting references and exact
normalized overlap with MATH-500/GSM8K test questions, validates reference answers
with Math-Verify, and records source/data/verifier hashes and filtering counts.
The model sees only the problem plus the boxed-answer instruction. Reference
answers and source solve rates are never included in prompts. Long questions are
excluded by the trainer's existing 1,024-token prompt limit. The verified dataset
has gated Hugging Face access; download with an authorized account before running
`prepare_big_math_data.slurm`. No credentials are stored in experiment artifacts.

`--resume-from ... --allow-dataset-change` continues model weights, AdamW state,
optimizer-update count, and global step while starting a changed dataset's
shuffle at epoch/cursor zero. Other resume compatibility checks remain active.
The checkpoint records `data_start_step`; future resumes on the same dataset
restore its actual cursor instead of resetting it. `train/dataset_step` counts
steps since switching to this dataset. The transition and both dataset hashes
are recorded in W&B configuration.

The requested continuation adds **1,000 steps**, from global step 300 to **1300**:

```bash
# After the pinned source parquet has been downloaded:
sbatch --account=enalisn1_fall2026 prepare_big_math_data.slurm
# After successful preparation:
sbatch --account=enalisn1_fall2026 train_big_math_continue.slurm standard 1000
sbatch --account=enalisn1_fall2026 train_big_math_continue.slurm three_pass 1000
```

The standard run starts at `results/899298/checkpoints/model_000300.pt`; the
three-pass LF run starts at `results/906530/checkpoints/model_000300.pt`, each with
its matching metadata and optimizer. Both retain 4 H100/A100 GPUs, 128 prompts ×
8 responses = 1,024 responses/update, microbatch 8/GPU, 32 accumulation steps,
LR 1e-6, Math-Verify rewards, and DAPO-style token normalization. The LF run keeps
soft vLLM rollouts and detached three-pass likelihoods. The first two additional
steps audit all weight transfers and checkpoint at 302; the main stage restores
that checkpoint and continues through 1300. The 48-hour job limit accommodates
slower A100 allocations. W&B project: `nanochat-online-rl`.

## Big-Math difficulty calibration

`calibrate_big_math.py` evaluates the fixed standard (899298) and three-pass LF
(906530) step-300 checkpoints on the same 2,048 questions, with no optimizer
updates. Sampling is balanced across source × Llama solve-rate bands, capped by
available questions. The pinned prepared dataset, eligible stratum populations,
subset tokens/IDs, source snapshots, and checkpoint hashes are recorded.
Each question gets eight vLLM responses at temperature 1, top-p 1, no top-k,
and a 1,024-token response limit. LF uses recurrent soft decoding. Math-Verify
assigns zero reward to truncated responses, matching the training objective.

```bash
PYTHONPATH=.deps:.. ../.venv/bin/python calibrate_big_math.py prepare \
  --output results/bigmath-calibration-20260927
sbatch calibrate_big_math.slurm results/bigmath-calibration-20260927
# Substitute the successful array's job ID:
sbatch --dependency=afterok:ARRAY_JOB_ID merge_big_math_calibration.slurm \
  results/bigmath-calibration-20260927
```

The array contains four single-GPU shards per model, with at most four GPUs
running concurrently. The dependent CPU merge validates coverage and logs
accuracy, all-wrong/mixed/all-correct groups, parsing, and truncation to W&B.
Per-source and per-stratum summaries retain the calibration sampling mixture;
the separately reported population-weighted accuracy estimates accuracy over
the eligible dataset mixture. This is a training-pool calibration, not a held-out
benchmark. Existing training jobs are not changed by these scripts.

## LF continuation with 512 questions per update

`restart_big_math_lf512.slurm` resumes the model, optimizer, and exact data cursor
from `results/919724/checkpoints/model_001250.pt`, targeting step 1300. It uses
512 unique prompts × 8 responses = 4,096 responses/update on four GPUs:
microbatch 8/GPU and 128 accumulation steps. Rollout concurrency remains 64/GPU
with a 16 GiB KV cache. Two preflight updates save step 1252, then the main stage
continues automatically. The LR and three-pass scoring objective are unchanged.

The explicit `--allow-batch-size-change` flag permits only prompts/update and
accumulation changes in the resume checks. Checkpoints record a batch-boundary
step and absolute consumed-prompt count so subsequent resumes validate and
restore the stream across batch changes and epoch boundaries. Dataset origin
and optimizer update counts remain intact. This branch retains the original
training horizon; it does not add another 1,000 updates.

## Matched LF curriculum branch

`prepare_curriculum.py` constructs `data/bigmath-lf-curriculum.json` from the
step-300 LF calibration, with pinned calibration/data identities. The launcher
`train_big_math_lf512_curriculum.slurm` repeats the uniform large-batch branch
from the same step-1250 model/optimizer through step 1300, with the same 512 × 8
batch, four GPUs, accumulation 128, LR 1e-6, rollout concurrency 64, and 16 GiB
KV cache. Fixed GSM8K evaluations and W&B logging match the control run.

Each batch draws 154 Big-Math-Reformulated, 102 MATH, 102 cn_k12, 77 Orca-Math,
51 selected competition, and 26 broad-exploration questions. The JSON lists
the precise Llama solve-rate bands. Initial draws are population-proportional
within each pool. At step 1275, observed mixed-group and truncation rates are
smoothed with 20 calibration-prior groups per stratum. Steps 1276–1300 weight
stratum populations by `max(0.1, mixed_fraction * (1-truncation_rate))`, retaining
pool quotas and uniform broad exploration. This refresh changes weights within
the specified bands; it does not automatically expand the difficulty bands.

`--allow-curriculum-change` explicitly branches a uniform checkpoint. The
curriculum configuration hash must match on later resumes; sampler positions,
statistics, refresh state, and origin step are checkpointed. Stratum decks
shuffle deterministically and avoid repeat questions until each deck cycles;
all 512 questions in an update are distinct even when pools overlap. All ranks
select the same global questions and reduce observed group statistics before
updating the sampler. The old uniform cursor is frozen at the branch point;
`curriculum_state` controls subsequent draws. No responses are replayed or
filtered from the policy-gradient update.

This is a different training distribution. Compare fixed evaluations, not raw
training rewards alone. The existing uniform job is retained as the control.

`continue_big_math_curriculum_200.slurm` continues the curriculum model from
`results/922491/checkpoints/model_001300.pt` through step 1500 (200 additional
updates). It restores optimizer moments, sampler deck positions, accumulated
statistics, and the phase-two weights learned at step 1275. Those weights remain
fixed, matching the existing curriculum; the warm-up and refresh are not reset.
The 512 × 8 batch, four GPUs, accumulation 128, LR 1e-6, 1,024-token response
limit, concurrency 64/GPU, and 16 GiB KV cache/GPU remain unchanged. Checkpoints
are saved every 25 steps, with training and curriculum metrics logged to W&B.
