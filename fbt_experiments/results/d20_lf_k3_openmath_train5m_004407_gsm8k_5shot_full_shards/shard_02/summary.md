# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train5m/model_004407.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (165 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 44/165 | 26.7% | 20.5%–33.9% | 99.4% | 14029 | 91.14 |
| soft | 48/165 | 29.1% | 22.7%–36.4% | 100.0% | 15008 | 90.74 |
| fused | 55/165 | 33.3% | 26.6%–40.8% | 99.4% | 15626 | 89.64 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.5716`, STANDARD↔FUSED `0.08014`, SOFT↔FUSED `0.281`.

## Verdict

STANDARD scored 44/165; SOFT scored 48/165 (+4 versus STANDARD), and FUSED scored 55/165 (+11 versus STANDARD). None of the paired exact tests is significant at 0.05 (smallest p=0.08014), so this run does not establish an accuracy difference among the decoding modes.

## Interpretation limits

- This is an SFT checkpoint at step 4407, initialized from `d20-lf-k1-smoke` step `50000`.
- SFT data: `openmath` split `train_5M`; checkpoint validation bpb was `0.1497`.
- Evaluation used greedy decoding (`temperature=0.0`) with K=3 latent-feedback training metadata.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
