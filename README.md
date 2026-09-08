# An open sourced reproduction of Full-bandwidth transformer https://arxiv.org/abs/2608.08888 (based on Nanochat)

(Work in progress)

## Introduction

(*No Microsoft resources or assets are used for the reproduction*)

An attempt (fully agentic implementation) to reproduce Full-bandwidth transformer using the Nanochat codebase.

The current math reproduction uses a pretrain + mid-train pipeline:

- Standard baseline: d20 Nanochat model trained to step 60k with one-pass behavior.
- FBT variants: initialized from the standard d20 checkpoint at step 40k, then trained to step 60k with K=2 latent-feedback pretraining.
- SFT / mid-training: OpenMathInstruct-2 `train_5M`, one epoch after reserving 4096 validation examples.
- Standard SFT uses one forward pass. FBT SFT uses K=3, `--no-feedback-prefix-mixin`, and `--feedback-jitter 0.02`.
- Reported math evals are zero-shot chat-template greedy decoding. FBT checkpoints are evaluated with `standard`, `soft`, and `fused` decoding; the standard checkpoint is evaluated with `standard` decoding only.

The main completed comparison covers the standard baseline plus two FBT fusion variants, `concat_projection` and `gate_product`. `linear_addition` uses the same recipe but its eval artifacts are not included in the tables below until its queued eval jobs finish.

### GSM8K zero-shot chat-template results

Full GSM8K test set, 1,319 problems. Exact-match grading uses the GSM8K numeric answer parser in `fbt_experiments/evaluate_checkpoint.py`.

| model | decode | correct | accuracy |
|---|---:|---:|---:|
| standard | standard | 642/1319 | 48.67% |
| concat_projection | standard | 651/1319 | 49.36% |
| concat_projection | soft | 681/1319 | 51.63% |
| concat_projection | fused | 691/1319 | 52.39% |
| gate_product | standard | 632/1319 | 47.92% |
| gate_product | soft | 650/1319 | 49.28% |
| gate_product | fused | 692/1319 | 52.46% |

Best GSM8K result: `gate_product + fused`, 692/1319 = 52.46%.

### MATH-500 zero-shot chat-template results

Full MATH-500 test set, 500 problems. Final results below use Hugging Face Math-Verify symbolic grading from `fbt_experiments/regrade_math500_math_verify.py`.

| model | decode | correct | accuracy |
|---|---:|---:|---:|
| standard | standard | 173/500 | 34.6% |
| concat_projection | standard | 176/500 | 35.2% |
| concat_projection | soft | 193/500 | 38.6% |
| concat_projection | fused | 198/500 | 39.6% |
| gate_product | standard | 171/500 | 34.2% |
| gate_product | soft | 183/500 | 36.6% |
| gate_product | fused | 198/500 | 39.6% |

Best MATH-500 result: tie between `concat_projection + fused` and `gate_product + fused`, both 198/500 = 39.6%.

Across both benchmarks, the clearest signal is that `fused` decoding improves the FBT checkpoints relative to their own `standard` decoding. For paired comparisons on the same examples, use the McNemar counts/p-values in the saved `metrics.json` or `metrics_math_verify.json` files rather than independent binomial error bars.


## Reproduce the standard-vs-FBT math pipeline

This section documents the exact script flow used for the d20 standard baseline and the latent-feedback / FBT variants, from pretraining through OpenMath SFT and zero-shot math evaluation.

The checked-in Slurm scripts are cluster-specific. Before running on a new clone, edit the `cd /weka/scratch/jhu/enalisn1/xiw/nanochat` line in the scripts if your repo lives elsewhere, and set `NANOCHAT_BASE_DIR` to a large external cache/checkpoint directory. The paths below assume:

```bash
export NANOCHAT_BASE_DIR=/home/jhu/xwang457/work/nanochat_cache
```

Install the GPU environment and the symbolic verifier used for final MATH-500 grading:

```bash
uv sync --extra gpu --group dev
uv pip install --python .venv/bin/python 'math-verify[antlr4_13_2]'
source .venv/bin/activate
```

### 1. Pretrain the standard d20 baseline

Run:

```bash
std_pretrain=$(sbatch --parsable runs/train_d20_standard_60k_4xh100.slurm)
echo "${std_pretrain}"
```

This trains the standard baseline tag:

```text
base_checkpoints/d20-standard-60k/model_060000.pt
```

The script is intentionally configured with `--num-forward-passes=2 --feedback-start-fraction=1.0`, which keeps the whole run in the one-pass regime while preserving the same training script surface. It also saves intermediate checkpoints every 10k steps; the FBT runs below resume from step 40k.

### 2. Pretrain the FBT variants from standard step 40k

Run after the standard pretraining job succeeds:

```bash
fbt_pretrain=$(sbatch --parsable --dependency=afterok:${std_pretrain} runs/train_d20_lf_k2_modes_from40k_a100.slurm)
echo "${fbt_pretrain}"
```

This is a three-element Slurm array:

```text
array task 0: gate_product
array task 1: concat_projection
array task 2: linear_addition
```

It resumes from:

```text
base_checkpoints/d20-standard-60k/model_040000.pt
```

and writes:

```text
base_checkpoints/d20-from40k-lf-k2-gate_product/model_060000.pt
base_checkpoints/d20-from40k-lf-k2-concat_projection/model_060000.pt
base_checkpoints/d20-from40k-lf-k2-linear_addition/model_060000.pt
```

The FBT pretraining recipe uses K=2 from the start of this resumed phase:

```text
--num-forward-passes=2
--feedback-start-fraction=0.0
--no-feedback-prefix-mixin
--feedback-jitter=0.02
--weight-tying
```

### 3. SFT on OpenMathInstruct-2 train_5M

Submit standard SFT after standard pretraining, and the FBT SFT jobs after the FBT pretraining array:

```bash
sft_std=$(sbatch --parsable --dependency=afterok:${std_pretrain} runs/openmath_standard_sft_train5m_a100.slurm)
sft_concat=$(sbatch --parsable --dependency=afterok:${fbt_pretrain} runs/openmath_concat_projection_k3_sft_train5m_a100.slurm)
sft_gate=$(sbatch --parsable --dependency=afterok:${fbt_pretrain} runs/openmath_gate_product_k3_sft_train5m_a100.slurm)
sft_linear=$(sbatch --parsable --dependency=afterok:${fbt_pretrain} runs/openmath_linear_addition_k3_sft_train5m_a100.slurm)
echo "${sft_std} ${sft_concat} ${sft_gate} ${sft_linear}"
```

The standard SFT job uses one forward pass:

```text
--num-forward-passes 1
```

The FBT SFT jobs use three passes and inherit their `latent_feedback_mode` from the source checkpoint metadata:

```text
--num-forward-passes 3
--no-feedback-prefix-mixin
--feedback-jitter 0.02
```

All four scripts use OpenMathInstruct-2 `train_5M`, reserve 4096 validation examples, train over the remaining examples once, and save only the final checkpoint:

```text
--sft-dataset openmath
--openmath-split train_5M
--openmath-val-examples 4096
--openmath-train-examples -1
--save-every -1
```

Expected SFT outputs:

```text
chatsft_checkpoints/d20-standard-60k-openmath-train5m-k1-anygpu/model_004407.pt
chatsft_checkpoints/d20-from40k-lf-k2-concat_projection-openmath-train5m-k3-anygpu/model_004407.pt
chatsft_checkpoints/d20-from40k-lf-k2-gate_product-openmath-train5m-k3-anygpu/model_004407.pt
chatsft_checkpoints/d20-from40k-lf-k2-linear_addition-openmath-train5m-k3-anygpu/model_004407.pt
```

### 4. Run zero-shot chat-template GSM8K

The main three-model eval array covers the standard, concat-projection, and gate-product SFT checkpoints:

```bash
main_gsm8k=$(sbatch --parsable --dependency=afterok:${sft_std}:${sft_concat}:${sft_gate} runs/eval_openmath_train5m_anygpu_gsm8k_0shot_chat_array.slurm)
main_gsm8k_merge_std=$(sbatch --parsable --dependency=afterok:${main_gsm8k} runs/merge_openmath_train5m_anygpu_standard_gsm8k_0shot_chat.slurm)
main_gsm8k_merge_concat=$(sbatch --parsable --dependency=afterok:${main_gsm8k} runs/merge_openmath_train5m_anygpu_concat_gsm8k_0shot_chat.slurm)
main_gsm8k_merge_gate=$(sbatch --parsable --dependency=afterok:${main_gsm8k} runs/merge_openmath_train5m_anygpu_gate_gsm8k_0shot_chat.slurm)
echo "${main_gsm8k} ${main_gsm8k_merge_std} ${main_gsm8k_merge_concat} ${main_gsm8k_merge_gate}"
```

This evaluates all 1,319 GSM8K test examples with:

```text
--gsm8k-shots 0
--gsm8k-prompt-format chat
--max-new-tokens 192
--decode-modes standard                  # standard model
--decode-modes standard,soft,fused       # FBT models
```

To also evaluate `linear_addition` on GSM8K:

```bash
linear_gsm8k=$(sbatch --parsable --dependency=afterok:${sft_linear} runs/eval_openmath_train5m_anygpu_linear_addition_gsm8k_0shot_chat.slurm)
linear_gsm8k_merge=$(sbatch --parsable --dependency=afterok:${linear_gsm8k} runs/merge_openmath_train5m_anygpu_linear_addition_gsm8k_0shot_chat.slurm)
echo "${linear_gsm8k} ${linear_gsm8k_merge}"
```

Final GSM8K result directories:

```text
fbt_experiments/results/d20_standard_60k_openmath_train5m_k1_anygpu_004407_gsm8k_0shot_chat_full
fbt_experiments/results/d20_from40k_lf_k2_concat_projection_openmath_train5m_k3_anygpu_004407_gsm8k_0shot_chat_full
fbt_experiments/results/d20_from40k_lf_k2_gate_product_openmath_train5m_k3_anygpu_004407_gsm8k_0shot_chat_full
fbt_experiments/results/d20_from40k_lf_k2_linear_addition_openmath_train5m_k3_anygpu_004407_gsm8k_0shot_chat_full
```

### 5. Run zero-shot chat-template MATH-500 and Math-Verify regrading

The main three-model MATH-500 eval array covers the same standard, concat-projection, and gate-product checkpoints:

```bash
main_math500=$(sbatch --parsable --dependency=afterok:${sft_std}:${sft_concat}:${sft_gate} runs/eval_openmath_train5m_anygpu_math500_0shot_chat_array.slurm)
main_math500_merge_std=$(sbatch --parsable --dependency=afterok:${main_math500} runs/merge_openmath_train5m_anygpu_standard_math500_0shot_chat.slurm)
main_math500_merge_concat=$(sbatch --parsable --dependency=afterok:${main_math500} runs/merge_openmath_train5m_anygpu_concat_math500_0shot_chat.slurm)
main_math500_merge_gate=$(sbatch --parsable --dependency=afterok:${main_math500} runs/merge_openmath_train5m_anygpu_gate_math500_0shot_chat.slurm)
main_math500_regrade=$(sbatch --parsable --dependency=afterok:${main_math500_merge_std}:${main_math500_merge_concat}:${main_math500_merge_gate} runs/regrade_openmath_train5m_anygpu_math500_0shot_chat_math_verify.slurm)
echo "${main_math500} ${main_math500_merge_std} ${main_math500_merge_concat} ${main_math500_merge_gate} ${main_math500_regrade}"
```

This evaluates all 500 MATH-500 examples with zero-shot chat prompts, `--max-new-tokens 512`, and the same decode-mode policy as GSM8K. The merge creates lightweight exact-match metrics; the regrade job adds Math-Verify symbolic metrics:

```text
metrics_math_verify.json
summary_math_verify.md
math500_generations_math_verify.jsonl
```

To also evaluate and Math-Verify regrade `linear_addition`:

```bash
linear_math500=$(sbatch --parsable --dependency=afterok:${sft_linear} runs/eval_openmath_train5m_anygpu_linear_addition_math500_0shot_chat.slurm)
linear_math500_merge=$(sbatch --parsable --dependency=afterok:${linear_math500} runs/merge_openmath_train5m_anygpu_linear_addition_math500_0shot_chat.slurm)
linear_math500_regrade=$(sbatch --parsable --dependency=afterok:${linear_math500_merge} runs/regrade_openmath_train5m_anygpu_linear_addition_math500_0shot_chat_math_verify.slurm)
echo "${linear_math500} ${linear_math500_merge} ${linear_math500_regrade}"
```

Final MATH-500 result directories:

```text
fbt_experiments/results/d20_standard_60k_openmath_train5m_k1_anygpu_004407_math500_0shot_chat_full
fbt_experiments/results/d20_from40k_lf_k2_concat_projection_openmath_train5m_k3_anygpu_004407_math500_0shot_chat_full
fbt_experiments/results/d20_from40k_lf_k2_gate_product_openmath_train5m_k3_anygpu_004407_math500_0shot_chat_full
fbt_experiments/results/d20_from40k_lf_k2_linear_addition_openmath_train5m_k3_anygpu_004407_math500_0shot_chat_full
```

### 6. Inspect final metrics

GSM8K:

```bash
for d in \
  fbt_experiments/results/d20_standard_60k_openmath_train5m_k1_anygpu_004407_gsm8k_0shot_chat_full \
  fbt_experiments/results/d20_from40k_lf_k2_concat_projection_openmath_train5m_k3_anygpu_004407_gsm8k_0shot_chat_full \
  fbt_experiments/results/d20_from40k_lf_k2_gate_product_openmath_train5m_k3_anygpu_004407_gsm8k_0shot_chat_full \
  fbt_experiments/results/d20_from40k_lf_k2_linear_addition_openmath_train5m_k3_anygpu_004407_gsm8k_0shot_chat_full
do
  echo "${d}"
  jq '.gsm8k | {standard, soft, fused, paired_accuracy}' "${d}/metrics.json"
done
```

MATH-500, using Math-Verify:

```bash
for d in \
  fbt_experiments/results/d20_standard_60k_openmath_train5m_k1_anygpu_004407_math500_0shot_chat_full \
  fbt_experiments/results/d20_from40k_lf_k2_concat_projection_openmath_train5m_k3_anygpu_004407_math500_0shot_chat_full \
  fbt_experiments/results/d20_from40k_lf_k2_gate_product_openmath_train5m_k3_anygpu_004407_math500_0shot_chat_full \
  fbt_experiments/results/d20_from40k_lf_k2_linear_addition_openmath_train5m_k3_anygpu_004407_math500_0shot_chat_full
do
  echo "${d}"
  jq '.math500_math_verify | {standard, soft, fused, paired_accuracy}' "${d}/metrics_math_verify.json"
done
```

For mode comparisons on the same examples, prefer the paired exact McNemar counts and p-values in `paired_accuracy` over independent binomial error bars.


## Getting started

### Setup

nanochat uses [uv](https://docs.astral.sh/uv/) for dependency management. To install:

```bash
uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) Use for CPU-only / MPS
source .venv/bin/activate
```

For development (adds pytest, matplotlib, ipykernel, transformers, etc.):

```bash
uv sync --extra gpu --group dev
```

### Reproduce and talk to GPT-2

The most fun you can have is to train your own GPT-2 and talk to it. The entire pipeline to do so is contained in the single file [runs/speedrun.sh](runs/speedrun.sh), which is designed to be run on an 8XH100 GPU node. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash runs/speedrun.sh
```

You may wish to do so in a screen session as this will take ~1.5 hours to run. Once it's done, you can talk to your model over the CLI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and chat:

```bash
python -m scripts.chat_cli
```

Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :). An example conversation with a speedrun model:

> **User:** Hello whats up?
>
> **Assistant:** Hello! How can I help you today?
>
> **User:** Why is the sky blue?
>
> **Assistant:** The sky is blue due to an optical illusion called the Rayleigh Scattering of Light, which occurs when sunlight passes through tiny particles in the air. These particles are made up of tiny blue and violet particles that cause the light to bend in a particular way.
>
> **User:** That's cool. Write a poem about it.
>
> **Assistant:** In the twilight's glow, the sky is blue,<br>
> A canvas of shimmering blue, where skies unfold,<br>
> A world of blue, where blue and more blue,<br>
> Are woven together in a tapestry so fine,<br>
> Where every hue seems to sing a story.<br>
> ...

A few more notes:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device-batch-size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't personally exercised all of these code paths so there might be sharp edges.

## Research

If you are a researcher and wish to help improve nanochat, two scripts of interest are [runs/scaling_laws.sh](runs/scaling_laws.sh) and [runs/miniseries.sh](runs/miniseries.sh). See [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) for related documentation. For quick experimentation (~5 min pretraining runs) my favorite scale is to train a 12-layer model (GPT-1 sized), e.g. like this:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This uses wandb (run name "d12"), only runs the CORE metric on last step, and it doesn't sample and save intermediate checkpoints. I like to change something in the code, re-run a d12 (or a d16 etc) and see if it helped, in an iteration loop. To see if a run helps, I like to monitor the wandb plots for:

1. `val_bpb` (validation loss in vocab-size-invariant units of bits per byte) as a function of `step`, `total_training_time` and `total_training_flops`.
2. `core_metric` (the DCLM CORE score)
3. VRAM utilization, `train/mfu` (Model FLOPS utilization), `train/tok_per_sec` (training throughput)

See an example [here](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044).

The important thing to note is that nanochat is written and configured around one single dial of complexity - the depth of the transformer. This single integer automatically determines all other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) so that the trained model comes out compute optimal. The idea is that the user doesn't have to think about or set any of this, they are simply asking for a smaller or bigger model using `--depth`, and everything "just works". By sweeping out the depth, you achieve the nanochat miniseries of compute optimal models at various sizes. GPT-2 capability model (which is of most interest at the moment) happens to be somewhere around d24-d26 range with the current code. But any candidate changes to the repo have to be principled enough that they work for all settings of depth.

## Running on CPU / MPS

The script [runs/runcpu.sh](runs/runcpu.sh) shows a very simple example of running on CPU or Apple Silicon. It dramatically shrinks the LLM that is being trained to make things fit into a reasonable time interval of a few ten minutes of training. You will not get strong results in this way.

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`). By default this is auto-detected based on your hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | Safe default. On recent macOS, MPS also runs `NANOCHAT_DTYPE=bfloat16` fine (~25% less memory, similar speed) |

You can override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

How it works: model weights are stored in fp32 (for optimizer precision), but our custom `Linear` layer casts them to `COMPUTE_DTYPE` during the forward pass. Untied, lookup-only embeddings are stored directly in `COMPUTE_DTYPE` to save memory. Models trained with `--weight-tying` keep the shared embedding/output matrix in fp32 because it is also a dense output projection. This gives us the same mixed-precision benefit as autocast but with full explicit control over what runs in which precision.

Note: `float16` training automatically enables a `GradScaler` in `base_train.py` to prevent gradient underflow. SFT supports this too but RL currently does not. Inference in fp16 works fine everywhere.

## Guides

I've published a number of guides that might contain helpful information, most recent to least recent:

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) documents the first nanochat miniseries of models.
- To add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) introducing nanochat, though now it contains some deprecated information and the model is a lot older (with worse results) than current master.

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── nanochat.png
│   └── repackage_data_reference.py # Pretraining data shard generation
├── nanochat
│   ├── __init__.py                 # empty
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── optim.py                    # AdamW + Muon optimizer, 1GPU and distributed
│   └── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries training script
│   ├── runcpu.sh                   # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh             # Scaling laws experiments
│   └── speedrun.sh                 # Train the ~$100 nanochat d20
├── scripts
│   ├── base_eval.py                # Base model: CORE score, bits per byte, samples
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model: talk to over CLI
│   ├── chat_eval.py                # Chat model: eval tasks
│   ├── chat_rl.py                  # Chat model: reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── infer_bench.py              # Inference: latency/throughput/VRAM bench
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   └── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
├── tests
│   ├── test_attention_fallback.py  # FA3/SDPA attention fallback
│   ├── test_engine.py              # Inference engine, KV cache
│   ├── test_execution.py           # Sandboxed code execution
│   ├── test_optim.py               # MuonAdamW optimizer (needs GPU)
│   ├── test_tasks.py               # Task slicing, mixtures, HubDataset
│   └── test_tokenizer.py           # BPE round-trips, chat rendering
└── uv.lock
```

## Contributing

The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to. Currently, the most interesting part personally is speeding up the latency to GPT-2 (i.e. getting a CORE score above 0.256525). Currently this takes ~1.5 hours (down from 3h), but by improving the pretraining stage we can improve this further.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
