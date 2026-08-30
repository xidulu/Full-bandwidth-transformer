# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/chatsft_checkpoints/d20-lf-k3-openmath-train5m/model_004407.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (1319 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 278/1319 | 21.1% | 19.0%–23.4% | 99.8% | 115677 | 91.94 |
| soft | 360/1319 | 27.3% | 25.0%–29.8% | 99.8% | 125456 | 91.54 |
| fused | 448/1319 | 34.0% | 31.5%–36.6% | 99.8% | 127829 | 90.47 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `4.453e-08`, STANDARD↔FUSED `1.916e-24`, SOFT↔FUSED `1.807e-08`.

## Verdict

STANDARD scored 278/1319; SOFT scored 360/1319 (+82 versus STANDARD), and FUSED scored 448/1319 (+170 versus STANDARD). At least one paired exact test is below 0.05 (smallest p=1.916e-24); inspect the paired counts above before drawing a conclusion.

## Interpretation limits

- This is an SFT checkpoint at step 4407, initialized from `d20-lf-k1-smoke` step `50000`.
- SFT data: `openmath` split `train_5M`; checkpoint validation bpb was `0.1497`.
- Evaluation used greedy decoding (`temperature=0.0`) with K=3 latent-feedback training metadata.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888

## Merge provenance

This report was recomputed from 8 validated, disjoint shards covering global examples 0–1318. The per-mode `seconds` totals are summed GPU-seconds, not parallel wall time. The three reported McNemar p-values are unadjusted for multiple comparisons. See `merge_manifest.json` for source paths and SHA-256 hashes.
