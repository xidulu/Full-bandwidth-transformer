# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train1m/model_000911.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (164 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 26/164 | 15.9% | 11.1%–22.2% | 100.0% | 13353 | 90.62 |
| soft | 29/164 | 17.7% | 12.6%–24.2% | 100.0% | 15568 | 90.17 |
| fused | 29/164 | 17.7% | 12.6%–24.2% | 100.0% | 15804 | 89.09 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.6476`, STANDARD↔FUSED `0.6291`, SOFT↔FUSED `1`.

## Verdict

STANDARD scored 26/164; SOFT scored 29/164 (+3 versus STANDARD), and FUSED scored 29/164 (+3 versus STANDARD). None of the paired exact tests is significant at 0.05 (smallest p=0.6291), so this run does not establish an accuracy difference among the decoding modes.

## Interpretation limits

- This is a base checkpoint trained for about 0.00B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction None of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
