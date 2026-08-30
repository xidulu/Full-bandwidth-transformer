# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train1m/model_000911.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 26/165 | 15.8% | 11.0%–22.1% | 100.0% | 13633 | 93.05 |
| soft | 34/165 | 20.6% | 15.1%–27.4% | 100.0% | 15425 | 92.70 |
| fused | 32/165 | 19.4% | 14.1%–26.1% | 100.0% | 15441 | 91.59 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.1516`, STANDARD↔FUSED `0.3915`, SOFT↔FUSED `0.8601`.

## Verdict

STANDARD scored 26/165; SOFT scored 34/165 (+8 versus STANDARD), and FUSED scored 32/165 (+6 versus STANDARD). None of the paired exact tests is significant at 0.05 (smallest p=0.1516), so this run does not establish an accuracy difference among the decoding modes.

## Interpretation limits

- This is a base checkpoint trained for about 0.00B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction None of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
