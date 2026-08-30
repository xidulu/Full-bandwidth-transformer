# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/base_checkpoints/d16-lf-k1-smoke/model_050000.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 4/165 | 2.4% | 0.9%–6.1% | 100.0% | 10189 | 70.65 |
| soft | 2/165 | 1.2% | 0.3%–4.3% | 100.0% | 11791 | 70.04 |
| fused | 1/165 | 0.6% | 0.1%–3.4% | 100.0% | 9333 | 68.75 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.6875`, STANDARD↔FUSED `0.375`, SOFT↔FUSED `1`.

## Verdict

STANDARD scored 4/165; SOFT scored 2/165 (-2 versus STANDARD), and FUSED scored 1/165 (-3 versus STANDARD). None of the paired exact tests is significant at 0.05 (smallest p=0.375), so this run does not establish an accuracy difference among the decoding modes.

## Interpretation limits

- This is a base checkpoint trained for about 26.21B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction 0.75 of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
