# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/base_checkpoints/d16-lf-k1-smoke/model_040000.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (200 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 3/200 | 1.5% | 0.5%–4.3% | 99.5% | 12375 | 83.19 |
| soft | 5/200 | 2.5% | 1.1%–5.7% | 99.0% | 13146 | 82.22 |
| fused | 4/200 | 2.0% | 0.8%–5.0% | 99.0% | 10904 | 80.79 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.6875`, STANDARD↔FUSED `1`, SOFT↔FUSED `1`.

## Verdict

SOFT and FUSED are numerically above STANDARD on this subset, but only by two and one correct answers, respectively. The confidence intervals overlap heavily and all paired exact tests are non-significant, so this run does not establish a GSM8K accuracy advantage for either feedback decoder.

## Interpretation limits

- This is a base checkpoint trained for about 20.97B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction 0.75 of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
