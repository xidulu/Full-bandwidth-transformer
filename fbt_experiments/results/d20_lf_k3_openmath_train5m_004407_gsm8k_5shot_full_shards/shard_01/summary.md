# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train5m/model_004407.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 41/165 | 24.8% | 18.9%–32.0% | 100.0% | 14375 | 90.15 |
| soft | 50/165 | 30.3% | 23.8%–37.7% | 99.4% | 16592 | 89.72 |
| fused | 62/165 | 37.6% | 30.5%–45.2% | 100.0% | 16051 | 88.71 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.1078`, STANDARD↔FUSED `0.0001922`, SOFT↔FUSED `0.04277`.

## Verdict

STANDARD scored 41/165; SOFT scored 50/165 (+9 versus STANDARD), and FUSED scored 62/165 (+21 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=0.0001922); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is an SFT checkpoint at step 4407, initialized from `d20-lf-k1-smoke` step `50000`.
- SFT data: `openmath` split `train_5M`; checkpoint validation bpb was `0.1497`.
- Evaluation used greedy decoding (`temperature=0.0`) with K=3 latent-feedback training metadata.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
