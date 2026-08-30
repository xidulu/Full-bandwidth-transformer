# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train1m/model_000911.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 22/165 | 13.3% | 9.0%–19.4% | 100.0% | 13665 | 90.46 |
| soft | 30/165 | 18.2% | 13.0%–24.8% | 99.4% | 15897 | 90.08 |
| fused | 35/165 | 21.2% | 15.7%–28.1% | 100.0% | 15501 | 89.02 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.1153`, STANDARD↔FUSED `0.01916`, SOFT↔FUSED `0.4244`.

## Verdict

STANDARD scored 22/165; SOFT scored 30/165 (+8 versus STANDARD), and FUSED scored 35/165 (+13 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=0.01916); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is a base checkpoint trained for about 0.00B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction None of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
