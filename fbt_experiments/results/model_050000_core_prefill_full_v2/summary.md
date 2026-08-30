# SOFT/FUSED checkpoint evaluation

Checkpoint: `/weka/scratch/jhu/enalisn1/xiw/nanochat_cache/base_checkpoints/d20-lf-k1-smoke/model_050000.pt`

The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.

## CORE Prefill Likelihood (full task set)

| prefill mode | correct | raw accuracy | CORE metric | seconds |
|---|---:|---:|---:|---:|
| standard_prefill | 42638/91037 | 46.8% | 0.2945 | 989.1 |
| fused_prefill | 42903/91037 | 47.1% | 0.2956 | 1903.3 |

Paired exact McNemar: standard-only `2659`, fused-only `2924`, delta `+0.291%`, p=`0.0004096`.

## Interpretation limits

- This is a base checkpoint trained for about 26.21B raw tokens, far smaller than the paper's main runs.
- It used K=2 only after fraction 0.6 of training.
- `feedback_prefix_mixin=False`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.
- The CORE examples are deterministically ordered and paired across both prefill modes.

Paper: https://arxiv.org/abs/2608.08888
