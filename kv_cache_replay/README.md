# KV cache replay experiment

Generate greedy MATH-500 rollouts with Nanochat's native `Engine` in `soft` mode,
capture the recurrent KV cache, and reconstruct it from the fixed token sequence
using repeated parallel forward passes. The data comes from the cached
`HuggingFaceH4/MATH-500` test set and uses the existing zero-shot chat prompt.

## Result: checkpoint 1675

**On four MATH-500 soft rollouts, three parallel passes reconstruct the response
KV cache from tokens with about 1.2% relative L2 error and 99.88% next-token
argmax agreement. Improvement largely levels off after four passes, near the
numerical discrepancy measured by replaying the recorded hidden states.**

The checkpoint is `lf512-curriculum-lr10x-945294`, step **1675**.
Job **948880** completed on an H100 NVL with BF16 and Flash Attention 3.
All four rollouts ended naturally, producing 438, 742, 327, and 163 sampled
tokens (1,670 total, including one terminal token per rollout; average 417.5).
None reached the 1,024-token generation limit.

| Total passes | Response K relative L2 | Response V relative L2 | Next-token argmax agreement |
|---|---:|---:|---:|
| 1 | 21.93% | 27.73% | 98.74% |
| 2 | 2.66% | 3.17% | 99.76% |
| 3 | 1.17% | 1.22% | 99.88% |
| 4 | 1.08% | 1.09% | 99.94% |
| 32 | 1.07% | 1.08% | 99.88% |
| Recorded-hidden control | 0.99% | 0.95% | 99.94% |

Three passes reconstruct the response cache closely on these examples. Mean
next-token KL is **0.000165 nats** at pass 3; K/V cosine similarities are both
approximately **0.99993**. Most improvement occurs by pass 3–4. Later passes
approach the recorded-hidden control's numerical discrepancy, with little
further improvement. This control supports attributing much of the residual
error to parallel versus incremental BF16 execution; it does not prove that
every remaining error is numerical.

Relative L2 is `||replay − true||₂ / ||true||₂`, pooled over response positions
and layers. Prompt positions are excluded from cache aggregates. Logit metrics
cover all sampled tokens. These results describe four fixed greedy trajectories;
they do not establish behavior on other prompts or sampled trajectories.

### What the recorded-hidden control measures

During the original sequential soft rollout, we save the actual hidden state
at every input position. The control then processes the fixed token sequence
in one parallel forward pass, giving each response token the recorded hidden
state from the preceding position as feedback. Prompt inputs remain ordinary.

This supplies the correct feedback directly. In the multi-pass reconstruction,
each pass estimates that feedback using the previous pass's hidden states.
The control's remaining **0.99% K / 0.95% V** error measures numerical differences
between parallel and incremental execution with BF16. It provides a reference
for the three-pass result of **1.17% K / 1.22% V**; it is not a strict lower bound
or an error term that can simply be subtracted.

The control requires saved hidden states. The multi-pass reconstruction uses
only the checkpoint, rollout tokens, and prompt boundary.

Artifacts: [full report](results/948880/report.md),
[convergence plot](results/948880/convergence.png),
[rollout and cache directory](results/948880/), and
[compact machine-readable summary](summary.json).
The saved tensors occupy about 2.0 GiB. Validation: 13 tests passed, including
the existing feedback-decoding suite.

## Method

- Pin a checkpoint and its matching metadata. The first run uses run `945294`,
  step `1675`, with `gate_product` feedback and tied embeddings.
- Generate four examples (test rows 0–3), up to 1,024 tokens each, with
  temperature zero, no top-k restriction, and no calculator.
- Save the native generation cache and hidden states. Cache positions correspond
  to `prompt + generated[:-1]`; the final sampled token has not been consumed.
  Terminal tokens are retained in token IDs and excluded from displayed text.
- Run 1, 2, 3, 4, 8, 16, and 32 full-sequence passes. Pass 1 is ordinary.
  Later passes fuse the previous pass's hidden state at position `t−1` with
  the token at `t`, only for response inputs. Keep prompt inputs ordinary.
  Rebuild every cache from position zero. Use evaluation mode and no jitter.
- Run a recorded-hidden control: one parallel pass using the actual recurrent
  hidden states. This measures numerical differences caused by processing the
  sequence in parallel versus incrementally.
- Compare response K and V separately using relative L2 error, cosine similarity,
  RMSE, and maximum absolute error. Also report prompt, layer, and position metrics.
  Compare next-token KL divergence, argmax agreement, and sampled-token log probabilities.

K tensors include RoPE, QK normalization, and the model's 1.2 scale; V tensors
include value-embedding residuals. Cache layout is
`[layer, batch, input_position, kv_head, head_dim]`.

This reproduces the forward recurrence used by `SoftThreePassLikelihood` for
three passes. That objective is a finite parallel approximation to sequential
soft decoding. In exact arithmetic, each additional pass recovers one more
response-input position; whether fewer passes approximate the remaining cache
well is the empirical question.

## Run

From this directory, using the repository's existing uv environment:

```bash
mkdir -p logs
sbatch --parsable --account=enalisn1 --qos=scavenger run_experiment.slurm \
  ../online_rl_experiments/results/945294/checkpoints/model_001675.pt
../.venv/bin/python report_results.py results/JOB_ID
```

The Slurm script requests one GPU for at most 30 minutes. Use an available account
and QoS on your cluster. It does not modify the checkpoint or the training run.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=..:. ../.venv/bin/python -m pytest \
  test_replay_experiment.py -q -p no:cacheprovider
```

Tests cover cache alignment, recorded-hidden replay, causal convergence, and
agreement with the training code's third-pass logits for all three feedback modes.

## Artifacts

`results/JOB_ID/manifest.json` pins checkpoint, metadata, source, and dataset hashes
and records the hardware, backend, dtype, and command arguments.

Each `example_NNN/` contains:

- `rollout.json`: problem, reference answer, prompt, completion, token IDs, and stop reason.
- `true_cache.pt`: native recurrent K/V tensors.
- `true_states.pt`: recurrent hidden states, logits for every sampled token, and cache input IDs.
- `pass_NN_cache.pt`: reconstructed K/V at each requested pass count.
- `oracle_hidden_cache.pt`: recorded-hidden control K/V.
- `metrics.json`: full prompt/response, per-layer, per-position, and logit diagnostics.

`results.json` contains all records and pooled response metrics. The report script
writes `report.md`, `convergence.png`, `convergence.pdf`, and `position_errors.png`.
Generated results and logs are ignored by Git. These four examples assess cache
reconstruction; they do not estimate MATH-500 accuracy.
