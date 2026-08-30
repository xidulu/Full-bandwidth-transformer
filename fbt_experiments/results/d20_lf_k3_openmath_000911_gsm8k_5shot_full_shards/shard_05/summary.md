# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train1m/model_000911.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 19/165 | 11.5% | 7.5%–17.3% | 100.0% | 13607 | 90.23 |
| soft | 28/165 | 17.0% | 12.0%–23.4% | 100.0% | 15800 | 89.79 |
| fused | 31/165 | 18.8% | 13.6%–25.4% | 100.0% | 15195 | 88.71 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.04904`, STANDARD↔FUSED `0.02266`, SOFT↔FUSED `0.6072`.

## Verdict

STANDARD scored 19/165; SOFT scored 28/165 (+9 versus STANDARD), and FUSED scored 31/165 (+12 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=0.02266); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is a base checkpoint trained for about 0.00B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction None of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
