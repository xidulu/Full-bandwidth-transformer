# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train5m/model_004407.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 28/165 | 17.0% | 12.0%–23.4% | 100.0% | 14066 | 91.54 |
| soft | 44/165 | 26.7% | 20.5%–33.9% | 100.0% | 15405 | 91.21 |
| fused | 56/165 | 33.9% | 27.2%–41.5% | 100.0% | 15188 | 90.07 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.005223`, STANDARD↔FUSED `4.056e-05`, SOFT↔FUSED `0.04277`.

## Verdict

STANDARD scored 28/165; SOFT scored 44/165 (+16 versus STANDARD), and FUSED scored 56/165 (+28 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=4.056e-05); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is an SFT checkpoint at step 4407, initialized from `d20-lf-k1-smoke` step `50000`.
- SFT data: `openmath` split `train_5M`; checkpoint validation bpb was `0.1497`.
- Evaluation used greedy decoding (`temperature=0.0`) with K=3 latent-feedback training metadata.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
