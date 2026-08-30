# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/base_checkpoints/d20-lf-k1-smoke/model_050000.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## CORE Prefill Likelihood (1 examples per task)

| prefill mode | correct | raw accuracy | CORE metric | seconds |
|---|---:|---:|---:|---:|
| standard_prefill | 10/22 | 45.5% | 0.3477 | 0.4 |
| fused_prefill | 9/22 | 40.9% | 0.3023 | 0.4 |

Paired exact McNemar: standard-only `1`, fused-only `0`, delta `-4.545%`, p=`1`.

## Interpretation limits

- This is a base checkpoint trained for about 26.21B raw tokens, far smaller than the paper's main runs.
- It used K=2 only after fraction 0.6 of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The GSM8K subset is deterministic and paired across all decoding modes.

Paper: https://arxiv.org/abs/2608.08888
