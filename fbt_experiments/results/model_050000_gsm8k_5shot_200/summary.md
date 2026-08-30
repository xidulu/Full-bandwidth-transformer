# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/base_checkpoints/d16-lf-k1-smoke/model_050000.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (200 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 6/200 | 3.0% | 1.4%–6.4% | 100.0% | 12606 | 68.22 |
| soft | 3/200 | 1.5% | 0.5%–4.3% | 100.0% | 14852 | 67.62 |
| fused | 2/200 | 1.0% | 0.3%–3.6% | 100.0% | 11207 | 66.38 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.4531`, STANDARD↔FUSED `0.2188`, SOFT↔FUSED `1`.

## Verdict

STANDARD scored 6/200; SOFT scored 3/200 (-3 versus STANDARD), and FUSED scored 2/200 (-4 versus STANDARD). None of the paired exact tests is significant at 0.05 (smallest p=0.2188), so this run does not establish an accuracy difference among the decoding modes.

## Interpretation limits

- This is a base checkpoint trained for about 26.21B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction 0.75 of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
