# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train5m/model_004407.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 35/165 | 21.2% | 15.7%–28.1% | 100.0% | 14758 | 93.52 |
| soft | 45/165 | 27.3% | 21.1%–34.5% | 100.0% | 15790 | 93.11 |
| fused | 48/165 | 29.1% | 22.7%–36.4% | 99.4% | 16001 | 92.09 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.1214`, STANDARD↔FUSED `0.04096`, SOFT↔FUSED `0.7111`.

## Verdict

STANDARD scored 35/165; SOFT scored 45/165 (+10 versus STANDARD), and FUSED scored 48/165 (+13 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=0.04096); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is an SFT checkpoint at step 4407, initialized from `d20-lf-k1-smoke` step `50000`.
- SFT data: `openmath` split `train_5M`; checkpoint validation bpb was `0.1497`.
- Evaluation used greedy decoding (`temperature=0.0`) with K=3 latent-feedback training metadata.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
