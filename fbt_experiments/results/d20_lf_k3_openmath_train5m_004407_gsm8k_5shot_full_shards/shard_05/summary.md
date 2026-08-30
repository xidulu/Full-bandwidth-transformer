# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train5m/model_004407.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 36/165 | 21.8% | 16.2%–28.7% | 100.0% | 14717 | 91.61 |
| soft | 40/165 | 24.2% | 18.3%–31.3% | 99.4% | 15407 | 91.24 |
| fused | 52/165 | 31.5% | 24.9%–39.0% | 100.0% | 15472 | 90.15 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.5716`, STANDARD↔FUSED `0.005223`, SOFT↔FUSED `0.0501`.

## Verdict

STANDARD scored 36/165; SOFT scored 40/165 (+4 versus STANDARD), and FUSED scored 52/165 (+16 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=0.005223); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is an SFT checkpoint at step 4407, initialized from `d20-lf-k1-smoke` step `50000`.
- SFT data: `openmath` split `train_5M`; checkpoint validation bpb was `0.1497`.
- Evaluation used greedy decoding (`temperature=0.0`) with K=3 latent-feedback training metadata.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
