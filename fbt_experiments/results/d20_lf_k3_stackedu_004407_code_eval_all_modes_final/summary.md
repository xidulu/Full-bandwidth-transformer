# d20-lf-k3 Stack-Edu code eval all modes

Checkpoint: `/home/jhu/xwang457/work/nanochat_cache/chatsft_checkpoints/d20-lf-k3-stackedu-python-budget/model_004407.pt`

| benchmark/mode | pass@1 |
|---|---:|
| humaneval/standard | 25/164 (15.24%) |
| humaneval/soft | 24/164 (14.63%) |
| humaneval/fused | 26/164 (15.85%) |
| mbpp_full/standard | 56/500 (11.20%) |
| mbpp_full/soft | 57/500 (11.40%) |
| mbpp_full/fused | 81/500 (16.20%) |

Settings: greedy decoding, temperature 0, max_new_tokens 512.

HumanEval uses raw prompt continuation. MBPP uses the full split with task text and public tests in comments, then evaluates against the provided test list.
