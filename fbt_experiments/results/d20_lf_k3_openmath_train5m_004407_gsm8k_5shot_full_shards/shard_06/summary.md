# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train5m/model_004407.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 32/165 | 19.4% | 14.1%–26.1% | 99.4% | 14853 | 94.19 |
| soft | 46/165 | 27.9% | 21.6%–35.2% | 100.0% | 16376 | 93.75 |
| fused | 57/165 | 34.5% | 27.7%–42.1% | 100.0% | 16750 | 92.67 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.01254`, STANDARD↔FUSED `0.0001122`, SOFT↔FUSED `0.08014`.

## Verdict

STANDARD scored 32/165; SOFT scored 46/165 (+14 versus STANDARD), and FUSED scored 57/165 (+25 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=0.0001122); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is an SFT checkpoint at step 4407, initialized from `d20-lf-k1-smoke` step `50000`.
- SFT data: `openmath` split `train_5M`; checkpoint validation bpb was `0.1497`.
- Evaluation used greedy decoding (`temperature=0.0`) with K=3 latent-feedback training metadata.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
