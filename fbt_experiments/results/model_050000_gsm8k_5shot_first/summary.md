# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/base_checkpoints/d20-lf-k1-smoke/model_050000.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (1 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 0/1 | 0.0% | 0.0%–79.3% | 100.0% | 47 | 123.45 |
| soft | 0/1 | 0.0% | 0.0%–79.3% | 100.0% | 41 | 171.06 |
| fused | 0/1 | 0.0% | 0.0%–79.3% | 100.0% | 40 | 165.02 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `1`, STANDARD↔FUSED `1`, SOFT↔FUSED `1`.

## Verdict

STANDARD scored 0/1; SOFT scored 0/1 (+0 versus STANDARD), and FUSED scored 0/1 (+0 versus STANDARD). None of the paired exact tests is significant at 0.05 (smallest p=1), so this run does not establish an accuracy difference among the decoding modes.

## Interpretation limits

- This is a base checkpoint trained for about 26.21B raw tokens, far smaller than the paper's main runs.
- It used K=2 only after fraction 0.6 of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
