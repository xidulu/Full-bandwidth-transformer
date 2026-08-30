# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/base_checkpoints/d20-lf-k1-smoke/model_050000.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## CORE Prefill Likelihood (up to 100 examples per task)

| prefill mode | correct | raw accuracy | CORE metric | seconds |
|---|---:|---:|---:|---:|
| standard_prefill | 968/2132 | 45.4% | 0.3025 | 23.0 |
| fused_prefill | 969/2132 | 45.5% | 0.3013 | 39.2 |

Paired exact McNemar: standard-only `67`, fused-only `68`, delta `+0.047%`, p=`1`.

## Interpretation limits

- This is a base checkpoint trained for about 26.21B raw tokens, far smaller than the paper's main runs.
- It used K=2 only after fraction 0.6 of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The CORE subset is deterministically shuffled and paired across both prefill modes.

Paper: https://arxiv.org/abs/2608.08888
