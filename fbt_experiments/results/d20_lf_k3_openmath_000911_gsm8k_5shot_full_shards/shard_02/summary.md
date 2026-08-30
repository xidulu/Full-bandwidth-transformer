# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train1m/model_000911.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 16/165 | 9.7% | 6.1%–15.2% | 100.0% | 14000 | 91.03 |
| soft | 25/165 | 15.2% | 10.5%–21.4% | 100.0% | 15258 | 90.55 |
| fused | 35/165 | 21.2% | 15.7%–28.1% | 100.0% | 13866 | 89.40 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.02246`, STANDARD↔FUSED `0.0003107`, SOFT↔FUSED `0.08716`.

## Verdict

STANDARD scored 16/165; SOFT scored 25/165 (+9 versus STANDARD), and FUSED scored 35/165 (+19 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=0.0003107); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is a base checkpoint trained for about 0.00B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction None of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
