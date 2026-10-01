# Nanochat vLLM rollout adapter

This directory is an isolated, out-of-tree vLLM plugin for the `standard` and
`soft` decode paths of this Nanochat fork. It targets **vLLM 0.14.0**, whose
CUDA build uses the same PyTorch 2.9.1 ABI pinned by the parent project.

The adapter preserves the model's custom QK normalization/scaling, negative-
angle RoPE convention, alternating value embeddings, ReLU-squared MLP,
residual/x0 scalars, smear, backout, tied-weight input scaling, sliding-window
pattern, and logit soft cap. In `soft` mode it performs ordinary prompt prefill,
then carries each request's normalized top-layer hidden state into the next
generated-token input through the checkpoint's configured latent-feedback
fusion. `fused` prompt decoding is not implemented.

## Supported scope

- One GPU: tensor, pipeline, and data parallel sizes must all be 1.
- Eager execution (`enforce_eager=True`).
- Token-ID prompts with vLLM tokenizer initialization skipped.
- Ordinary and chunked prefill plus continuous batching.
- Prefix caching is supported for `standard` and rejected for `soft`, because
  vLLM's prefix cache does not retain Nanochat's recurrent top-layer state.
- No speculative decoding, async scheduling, prompt embeddings, LoRA, or
  quantization yet.
- vLLM's default 0.14 model runner; `VLLM_USE_V2_MODEL_RUNNER` must be unset.

These restrictions are checked at startup instead of producing approximate
outputs.

## Install

From the repository root, after creating the normal CUDA environment:

```bash
uv sync --extra gpu --group dev
uv pip install -e ./vllm_plugin
```

Installing the package also registers `NanochatForCausalLM` through vLLM's
general-plugin entry point.

## Export a checkpoint

The exporter requires the matching `meta_*.json` beside the native checkpoint:

```bash
source .venv/bin/activate
nanochat-export-vllm \
  --checkpoint /home/jhu/xwang457/work/nanochat_cache/chatsft_checkpoints/TAG/model_004407.pt \
  --output-dir /tmp/nanochat-vllm-TAG \
  --decode-mode soft
```

Use `--decode-mode standard` (the default) for ordinary decoding. Soft export
requires `model_config.latent_feedback=true` and all latent-feedback weights.
The decode mode is recorded in both `config.json` and the export manifest.

The output contains `config.json`, `model.safetensors`, the complete original
metadata, and a machine-readable export manifest with hashes. The source
checkpoint's vocabulary padding is removed during export; vLLM restores its
own internal padding.

For environments without `safetensors`, `--weight-format pytorch` writes
`pytorch_model.bin` instead.

## Offline rollout

Keep tokenization in Nanochat and pass token IDs to vLLM:

```python
from vllm import LLM, SamplingParams

from nanochat.tokenizer import get_tokenizer
from nanochat_vllm.worker import NanochatWorker

tokenizer = get_tokenizer()
prompt_ids = tokenizer.encode("What is 2 + 2?", prepend=tokenizer.get_bos_token_id())

llm = LLM(
    model="/tmp/nanochat-vllm-TAG",
    skip_tokenizer_init=True,
    worker_cls="nanochat_vllm.worker.NanochatWorker",
    enforce_eager=True,
    tensor_parallel_size=1,
    enable_prefix_caching=False,
    async_scheduling=False,
)
params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
    stop_token_ids=[
        tokenizer.encode_special("<|assistant_end|>"),
        tokenizer.get_bos_token_id(),
    ],
)
outputs = llm.generate([{"prompt_token_ids": prompt_ids}], params)
completion_ids = outputs[0].outputs[0].token_ids
print(tokenizer.decode(completion_ids))
```

Importing `NanochatWorker` is optional in the client process; the string passed
to `worker_cls` is what makes every vLLM worker use the smear/feedback-aware
runner. The exported model directory selects standard versus soft behavior.

## Validation

Run the exporter tests without a GPU:

```bash
PYTHONPATH=vllm_plugin/src .venv/bin/python -m pytest vllm_plugin/tests -q
```

Before using a checkpoint for rollouts, compare greedy output against
`nanochat.engine.Engine(...)` with the matching decode mode. Exact token agreement is
the acceptance criterion; small raw-logit differences can occur because vLLM
uses different fused attention kernels.

On the JHU cluster, the checked-in single-GPU smoke job performs that comparison
in separate native and vLLM processes:

```bash
sbatch vllm_plugin/slurm_smoke.slurm
sbatch vllm_plugin/slurm_soft_smoke.slurm
```

## Full linear-addition GSM8K evaluation

The checked-in array job evaluates the `linear_addition` FBT checkpoint in
`standard` and `soft` modes on all 1,319 GSM8K test examples. It uses the
zero-shot Nanochat chat prompt, greedy decoding, and a 192-token limit:

```bash
eval_job=$(sbatch --parsable vllm_plugin/slurm_gsm8k_linear.slurm)
sbatch --dependency="afterok:${eval_job}" \
  vllm_plugin/slurm_merge_gsm8k_linear.slurm
```

Set `CHECKPOINT` and `RESULT_DIR` in the submission environment to override the
defaults. `MAX_NUM_SEQS` defaults to 256; set it to 1319 when checking exact
first-token agreement across modes with all requests admitted together:

```bash
result_dir="${PWD}/vllm_plugin/results/linear_addition_004407_gsm8k_0shot_chat_vllm_maxseq1319"
eval_job=$(sbatch --parsable \
  --export="ALL,RESULT_DIR=${result_dir},MAX_NUM_SEQS=1319" \
  vllm_plugin/slurm_gsm8k_linear.slurm)
sbatch --dependency="afterok:${eval_job}" \
  --export="ALL,RESULT_DIR=${result_dir}" \
  vllm_plugin/slurm_merge_gsm8k_linear.slurm
```

The merge writes JSONL generations, `metrics.json`, `run_config.json`, and a
short `summary.md`. It reports paired standard/soft outcomes and an exact
McNemar p-value, plus agreement with the native Nanochat result when the native
reference file is available. Generated results and Slurm logs are ignored by
Git.
