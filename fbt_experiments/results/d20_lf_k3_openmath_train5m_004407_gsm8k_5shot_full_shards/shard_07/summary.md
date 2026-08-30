# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train5m/model_004407.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (164 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 31/164 | 18.9% | 13.6%–25.6% | 99.4% | 14337 | 91.98 |
| soft | 41/164 | 25.0% | 19.0%–32.1% | 99.4% | 15668 | 91.62 |
| fused | 57/164 | 34.8% | 27.9%–42.3% | 100.0% | 16501 | 90.49 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.08716`, STANDARD↔FUSED `4.228e-05`, SOFT↔FUSED `0.002494`.

## Verdict

STANDARD scored 31/164; SOFT scored 41/164 (+10 versus STANDARD), and FUSED scored 57/164 (+26 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=4.228e-05); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is an SFT checkpoint at step 4407, initialized from `d20-lf-k1-smoke` step `50000`.
- SFT data: `openmath` split `train_5M`; checkpoint validation bpb was `0.1497`.
- Evaluation used greedy decoding (`temperature=0.0`) with K=3 latent-feedback training metadata.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
