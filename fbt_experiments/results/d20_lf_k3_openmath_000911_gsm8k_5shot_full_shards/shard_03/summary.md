# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train1m/model_000911.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 16/165 | 9.7% | 6.1%–15.2% | 100.0% | 13443 | 89.95 |
| soft | 31/165 | 18.8% | 13.6%–25.4% | 100.0% | 15451 | 89.47 |
| fused | 31/165 | 18.8% | 13.6%–25.4% | 100.0% | 15356 | 88.42 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.004077`, STANDARD↔FUSED `0.002599`, SOFT↔FUSED `1`.

## Verdict

STANDARD scored 16/165; SOFT scored 31/165 (+15 versus STANDARD), and FUSED scored 31/165 (+15 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=0.002599); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is a base checkpoint trained for about 0.00B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction None of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
