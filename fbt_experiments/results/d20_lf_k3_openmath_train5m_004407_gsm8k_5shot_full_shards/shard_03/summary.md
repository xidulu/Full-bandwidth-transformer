# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train5m/model_004407.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 31/165 | 18.8% | 13.6%–25.4% | 100.0% | 14542 | 91.41 |
| soft | 46/165 | 27.9% | 21.6%–35.2% | 100.0% | 15210 | 91.01 |
| fused | 61/165 | 37.0% | 30.0%–44.6% | 100.0% | 16240 | 89.95 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.002599`, STANDARD↔FUSED `2.272e-07`, SOFT↔FUSED `0.01353`.

## Verdict

STANDARD scored 31/165; SOFT scored 46/165 (+15 versus STANDARD), and FUSED scored 61/165 (+30 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=2.272e-07); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is an SFT checkpoint at step 4407, initialized from `d20-lf-k1-smoke` step `50000`.
- SFT data: `openmath` split `train_5M`; checkpoint validation bpb was `0.1497`.
- Evaluation used greedy decoding (`temperature=0.0`) with K=3 latent-feedback training metadata.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
