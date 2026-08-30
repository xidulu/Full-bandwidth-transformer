# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/base_checkpoints/d16-lf-k1-smoke/model_050000.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 5/165 | 3.0% | 1.3%–6.9% | 100.0% | 10950 | 70.39 |
| soft | 6/165 | 3.6% | 1.7%–7.7% | 100.0% | 13427 | 69.72 |
| fused | 6/165 | 3.6% | 1.7%–7.7% | 100.0% | 10507 | 68.60 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `1`, STANDARD↔FUSED `1`, SOFT↔FUSED `1`.

## Verdict

STANDARD scored 5/165; SOFT scored 6/165 (+1 versus STANDARD), and FUSED scored 6/165 (+1 versus STANDARD). None of the paired exact tests is significant at 0.05 (smallest p=1), so this run does not establish an accuracy difference among the decoding modes.

## Interpretation limits

- This is a base checkpoint trained for about 26.21B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction 0.75 of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
