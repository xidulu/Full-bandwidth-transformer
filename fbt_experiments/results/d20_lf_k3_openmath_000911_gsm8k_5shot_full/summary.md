# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train1m/model_000911.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (1319 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 176/1319 | 13.3% | 11.6%–15.3% | 100.0% | 109686 | 90.38 |
| soft | 238/1319 | 18.0% | 16.1%–20.2% | 99.9% | 125660 | 89.94 |
| fused | 256/1319 | 19.4% | 17.4%–21.6% | 100.0% | 120215 | 88.90 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `7.532e-07`, STANDARD↔FUSED `4.026e-09`, SOFT↔FUSED `0.2075`.

## Verdict

STANDARD scored 176/1319; SOFT scored 238/1319 (+62 versus STANDARD), and FUSED scored 256/1319 (+80 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=4.026e-09); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is an SFT checkpoint at step 911, initialized from `d20-lf-k1-smoke` step `50000`.
- SFT data: `openmath` split `train_1M`; checkpoint validation bpb was `0.1924`.
- Evaluation used greedy decoding (`temperature=0.0`) with K=3 latent-feedback training metadata.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888

## Merge provenance

This report was recomputed from 8 validated, disjoint shards covering global examples 0–1318. The per-mode `seconds` totals are summed GPU-seconds, not parallel wall time. The three reported McNemar p-values are unadjusted for multiple comparisons. See `merge_manifest.json` for source paths and SHA-256 hashes.
