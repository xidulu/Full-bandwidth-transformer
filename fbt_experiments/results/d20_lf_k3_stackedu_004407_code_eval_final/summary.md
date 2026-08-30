# d20-lf-k3 Stack-Edu code eval final

Checkpoint: `/home/jhu/xwang457/work/nanochat_cache/chatsft_checkpoints/d20-lf-k3-stackedu-python-budget/model_004407.pt`

| benchmark/mode | pass@1 |
|---|---:|
| humaneval/standard | 25/164 (15.24%) |
| mbpp_full/standard | 56/500 (11.20%) |

Settings: standard greedy decoding, temperature 0, max_new_tokens 512.

HumanEval uses raw prompt continuation. MBPP uses the full split with task text and public tests in comments, then evaluates against the provided test list.
