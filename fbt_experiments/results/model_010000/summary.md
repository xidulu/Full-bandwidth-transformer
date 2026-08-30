# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/base_checkpoints/d16-lf-k1-smoke/model_010000.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## Held-out sequential continuation

| mode | BPB | delta vs standard | relative delta | document wins | finite | hidden RMS range |
|---|---:|---:|---:|---:|:---:|---:|
| standard | 0.839388 | +0.000000 | +0.000% | — | True | 0.998091–1.002000 |
| soft | 0.842097 | +0.002709 | +0.323% | 14/32 | True | 0.998033–1.002324 |
| fused | 0.839091 | -0.000297 | -0.035% | 18/32 | True | 0.998035–1.002193 |

STANDARD and SOFT share the ordinary prompt prefill; their first-target NLL max difference was `0`.

## Small raw-text GSM8K probe

| mode | exact | accuracy | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|
| standard | 0/20 | 0.0% | 100.0% | 1463 | 82.12 |
| soft | 1/20 | 5.0% | 100.0% | 1584 | 81.24 |
| fused | 2/20 | 10.0% | 100.0% | 1196 | 79.91 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

## Verdict

Both decoding algorithms operate correctly and remain numerically stable on this checkpoint. Quality is different: SOFT changes BPB by +0.323%, while FUSED changes it by -0.035%. For this run, FUSED is effectively neutral/slightly favorable and SOFT is worse than standard decoding. The exact shared-first-token invariant provides an additional implementation check.

These likelihood deltas are small and mixed across documents, so they do not establish a statistically persuasive quality gain. The tiny GSM8K result is also retained as a behavioral smoke test only.

## Interpretation limits

- This is a base checkpoint trained for about 5.24B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction 0.75 of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K slice is deterministic and paired but small; use continuation BPB and exact recurrence tests as the primary evidence.

Paper: https://arxiv.org/abs/2608.08888
