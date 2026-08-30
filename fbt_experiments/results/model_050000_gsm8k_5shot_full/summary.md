# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/base_checkpoints/d16-lf-k1-smoke/model_050000.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## 5-shot GSM8K (1319 problems)

| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| standard | 35/1319 | 2.7% | 1.9%–3.7% | 100.0% | 85032 | 68.95 |
| soft | 30/1319 | 2.3% | 1.6%–3.2% | 99.8% | 100990 | 68.35 |
| fused | 29/1319 | 2.2% | 1.5%–3.1% | 100.0% | 81779 | 67.24 |

STANDARD/SOFT first generated token match rate: 100.0% (expected 100%).

Paired exact McNemar p-values: STANDARD↔SOFT `0.5682`, STANDARD↔FUSED `0.4296`, SOFT↔FUSED `1`.

## Verdict

STANDARD scored 35/1319; SOFT scored 30/1319 (-5 versus STANDARD), and FUSED scored 29/1319 (-6 versus STANDARD). None of the paired exact tests is significant at 0.05 (smallest p=0.4296), so this run does not establish an accuracy difference among the decoding modes.

## Interpretation limits

- This is a base checkpoint trained for about 26.21B raw tokens, far smaller than the paper's main runs.
- It used K=3 only after fraction 0.75 of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888

## Merge provenance

This report was recomputed from 8 validated, disjoint shards covering global examples 0–1318. The per-mode `seconds` totals are summed GPU-seconds, not parallel wall time. The three reported McNemar p-values are unadjusted for multiple comparisons. See `merge_manifest.json` for source paths and SHA-256 hashes.
