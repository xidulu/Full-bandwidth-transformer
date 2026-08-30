# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train1m/model_000911.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 31/165 | 18.8% | 13.6%–25.4% | 100.0% | 13927 | 88.81 |
| soft | 33/165 | 20.0% | 14.6%–26.8% | 100.0% | 15722 | 88.52 |
| fused | 37/165 | 22.4% | 16.7%–29.4% | 100.0% | 14303 | 87.47 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.8145`, STANDARD↔FUSED `0.1796`, SOFT↔FUSED `0.4545`.

## Verdict

STANDARD scored 31/165; SOFT scored 33/165 (+2 versus STANDARD), and FUSED scored 37/165 (+6 versus STANDARD). None of the paired exact tests is significant at 0.05 (smallest p=0.1796), so this run does not establish an accuracy difference among the decoding modes.

## Interpretation limits

- This is a base checkpoint trained for about 0.00B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction None of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
