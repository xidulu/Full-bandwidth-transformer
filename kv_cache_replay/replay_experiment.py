#!/usr/bin/env python3
"""Measure parallel latent-feedback KV reconstruction on fixed soft rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import Engine, KVCache


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def make_cache(model, length):
    c = model.config
    return KVCache(1, c.n_kv_head, length, c.n_embd // c.n_head,
                   c.n_layer, model.get_device(), COMPUTE_DTYPE)


class CaptureModel:
    """Observe the unchanged native Engine's forward calls."""
    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.latent_feedback = model.latent_feedback
        self.hidden = []
        self.logits = []
        self.cache = None

    def get_device(self):
        return self.model.get_device()

    def forward(self, *args, **kwargs):
        logits, hidden = self.model.forward(*args, **kwargs)
        self.hidden.append(hidden.cpu())
        self.logits.append(logits[:, -1].cpu())
        self.cache = kwargs['kv_cache']
        return logits, hidden


@torch.inference_mode()
def capture_rollout(model, tokenizer, prompt, max_new_tokens, seed):
    observed = CaptureModel(model)
    stream = Engine(observed, tokenizer).generate(
        prompt, max_tokens=max_new_tokens, temperature=0.0, top_k=None,
        seed=seed, decode_mode='soft', use_calculator=False)
    stop_ids = {tokenizer.get_bos_token_id(), tokenizer.encode_special('<|assistant_end|>')}
    generated = []
    # Stop at the last yield, before Engine consumes the final sampled token.
    # Thus cache inputs = prompt + generated[:-1], matching likelihood training.
    try:
        for column, _ in stream:
            generated.append(column[0])
            if column[0] in stop_ids or len(generated) == max_new_tokens:
                break
    finally:
        stream.close()
    length = len(prompt) + len(generated) - 1
    assert observed.cache.get_pos() == length
    hidden = torch.cat(observed.hidden, dim=1)
    logits = torch.cat(observed.logits, dim=0)
    assert hidden.shape[1] == length and logits.shape[0] == len(generated)
    return generated, observed.cache, hidden, logits


@torch.inference_mode()
def parallel_pass(model, ids, prompt_length, previous_hidden=None):
    """A fresh full-sequence pass; feedback is shifted and only on response inputs."""
    cache = make_cache(model, ids.shape[1])
    embeddings = model._embed_tokens(ids)
    ordinary = model._prepare_token_inputs(embeddings, cache)
    inputs = ordinary
    if previous_hidden is not None:
        fused = model.latent_feedback(previous_hidden[:, :-1], embeddings[:, 1:])
        fused = torch.cat((ordinary[:, :1], fused), dim=1)
        mask = torch.arange(ids.shape[1], device=ids.device)[None, :] >= prompt_length
        inputs = torch.where(mask[..., None], fused, ordinary)
    hidden = model._run_trunk(ids, inputs, cache)
    return cache, hidden


def tensor_metrics(estimate, truth):
    """Float64 reductions of fp32 differences; tensors can be any shape."""
    x, y = estimate.float(), truth.float()
    delta = x - y
    error2 = delta.square().sum(dtype=torch.float64).item()
    truth2 = y.square().sum(dtype=torch.float64).item()
    estimate2 = x.square().sum(dtype=torch.float64).item()
    dot = (x * y).sum(dtype=torch.float64).item()
    count = y.numel()
    return dict(relative_l2=(error2 / max(truth2, 1e-30)) ** 0.5,
                cosine=dot / max((truth2 * estimate2) ** 0.5, 1e-30),
                rmse=(error2 / max(count, 1)) ** 0.5,
                max_abs=delta.abs().max().item() if count else 0.0,
                error2=error2, truth2=truth2, estimate2=estimate2, dot=dot, count=count)


def cache_metrics(estimate, truth, prompt_length, length):
    result = {}
    for name in ('k', 'v'):
        x = getattr(estimate, name + '_cache')[:, :, :length]
        y = getattr(truth, name + '_cache')[:, :, :length]
        result[name] = {}
        for region, start, end in (('prompt', 0, prompt_length),
                                   ('response', prompt_length, length)):
            if end <= start:
                continue
            xx, yy = x[:, :, start:end], y[:, :, start:end]
            metrics = tensor_metrics(xx, yy)
            metrics['per_layer'] = [tensor_metrics(a, b) for a, b in zip(xx, yy)]
            # Reduce layer/batch/head/channel axes, preserving response position.
            error2 = (xx.float() - yy.float()).square().sum(dim=(0, 1, 3, 4))
            truth2 = yy.float().square().sum(dim=(0, 1, 3, 4))
            metrics['relative_l2_by_position'] = (error2 / truth2.clamp_min(1e-30)).sqrt().tolist()
            result[name][region] = metrics
    return result


def logit_metrics(estimate, truth, generated):
    estimate, truth = estimate.float(), truth.to(estimate.device).float()
    p, q = F.log_softmax(truth, -1), F.log_softmax(estimate, -1)
    target = torch.tensor(generated, device=estimate.device)[:, None]
    difference = (q - p).gather(-1, target).squeeze(-1)
    kl = (p.exp() * (p - q)).sum(-1)
    agree = estimate.argmax(-1).eq(truth.argmax(-1))
    return dict(kl_true_to_replay_mean=kl.mean().item(),
                top1_agreement=agree.float().mean().item(),
                sampled_logp_abs_error_mean=difference.abs().mean().item(),
                sampled_logp_signed_error_mean=difference.mean().item(),
                kl_by_position=kl.tolist(), top1_agreement_by_position=agree.tolist(),
                sampled_logp_error_by_position=difference.tolist(), count=len(generated))


def save_cache(path, cache, length):
    torch.save({name: getattr(cache, name + '_cache')[:, :, :length].cpu().contiguous()
                for name in ('k', 'v')}, path)


def aggregate(records):
    result = {}
    for label in records[0]['comparisons']:
        values = [r['comparisons'][label] for r in records]
        item = {}
        for name in ('k', 'v'):
            parts = [v[name]['response'] for v in values if 'response' in v[name]]
            e2 = sum(p['error2'] for p in parts)
            y2 = sum(p['truth2'] for p in parts)
            x2 = sum(p['estimate2'] for p in parts)
            dot = sum(p['dot'] for p in parts)
            item[name] = dict(relative_l2=(e2 / max(y2, 1e-30)) ** 0.5,
                              cosine=dot / max((x2 * y2) ** 0.5, 1e-30))
        total = sum(v['logits']['count'] for v in values)
        item['logits'] = {key: sum(v['logits'][key] * v['logits']['count'] for v in values) / total
                          for key in ('kl_true_to_replay_mean', 'top1_agreement',
                                      'sampled_logp_abs_error_mean')}
        result[label] = item
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--examples', type=int, default=4)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--passes', type=int, nargs='+', default=[1, 2, 3, 4, 8, 16, 32])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.passes) < 1 or args.max_new_tokens < 2:
        parser.error('Pass counts must be positive; max-new-tokens must be >=2')
    args.output.mkdir(parents=True, exist_ok=False)
    from nanochat.checkpoint_manager import build_model
    from nanochat import flash_attention
    from fbt_experiments.evaluate_math500 import load_math500_rows, build_math500_chat_prompt_ids
    checkpoint = args.checkpoint.resolve()
    step = int(checkpoint.stem.split('_')[-1])
    meta_path = checkpoint.with_name(f'meta_{step:06d}.json')
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    model, tokenizer, metadata = build_model(str(checkpoint.parent), step, device, 'eval')
    if model.latent_feedback is None:
        raise ValueError('Checkpoint must have latent feedback')
    examples = load_math500_rows(args.examples, args.start)
    dataset_dir = Path(os.environ['NANOCHAT_BASE_DIR']) / 'task_data/HuggingFaceH4--MATH-500/default/test'
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    metadata_sha256=sha256(meta_path), model_config=metadata['model_config'],
                    checkpoint_step=step, arguments={k: str(v) if isinstance(v, Path) else v
                                                     for k, v in vars(args).items()},
                    dataset='HuggingFaceH4/MATH-500', split='test',
                    dataset_sha256={p.name: sha256(p) for p in dataset_dir.glob('*.parquet')},
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                    source_sha256={str(p.relative_to(REPO)): sha256(p) for p in
                                   [Path(__file__), REPO/'nanochat/gpt.py', REPO/'nanochat/engine.py',
                                    REPO/'nanochat/flash_attention.py', REPO/'online_rl_experiments/soft_likelihood.py']},
                    torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                    compute_dtype=str(COMPUTE_DTYPE), attention_backend='fa3' if flash_attention.USE_FA3 else 'sdpa',
                    generation_backend='nanochat.Engine', decode_mode='soft', temperature=0.0,
                    top_k=None, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    cache_layout='[layer,batch,input_position,kv_head,head_dim]',
                    cache_alignment='prompt + generated[:-1]; final sampled token has not been consumed',
                    key_definition='post-RoPE, post-QK-normalization',
                    value_definition='includes value-embedding residual',
                    pass_definition='1=ordinary; later passes fuse shifted previous hidden only on response inputs',
                    oracle_definition='one parallel pass using recorded true recurrent hidden states')
    dump(args.output/'manifest.json', manifest)
    dump(args.output/'checkpoint_metadata.json', metadata)
    records = []
    for offset, example in enumerate(examples):
        started = time.monotonic()
        index = args.start + offset
        folder = args.output/f'example_{index:03d}'
        folder.mkdir()
        prompt, prompt_text = build_math500_chat_prompt_ids(tokenizer, example['problem'])
        if len(prompt) + args.max_new_tokens > model.config.sequence_len:
            raise ValueError(f'Example {index} exceeds model context')
        generated, true_cache, true_hidden, true_logits = capture_rollout(
            model, tokenizer, prompt, args.max_new_tokens, args.seed + index)
        length = len(prompt) + len(generated) - 1
        ids = torch.tensor([prompt + generated[:-1]], device=device)
        terminal = generated[-1] in {tokenizer.get_bos_token_id(), tokenizer.encode_special('<|assistant_end|>')}
        record = dict(index=index, dataset_row=example, prompt=prompt_text, prompt_token_ids=prompt,
                      generated_token_ids=generated, prompt_tokens=len(prompt), generated_tokens=len(generated),
                      cached_response_tokens=len(generated)-1,
                      completion=tokenizer.decode(generated[:-1] if terminal else generated),
                      stop_reason='terminal_token' if terminal else 'max_new_tokens', comparisons={})
        dump(folder/'rollout.json', {k: v for k, v in record.items() if k != 'comparisons'})
        save_cache(folder/'true_cache.pt', true_cache, length)
        torch.save(dict(hidden=true_hidden, logits=true_logits, input_ids=ids.cpu()), folder/'true_states.pt')
        print(f'example={index} generated={len(generated)} stop={record["stop_reason"]}', flush=True)

        def compare(label, cache, hidden):
            metrics = cache_metrics(cache, true_cache, len(prompt), length)
            logits = model._project_and_loss(hidden[:, len(prompt)-1:], None, 'mean')[0]
            metrics['logits'] = logit_metrics(logits, true_logits, generated)
            record['comparisons'][label] = metrics
            save_cache(folder/f'{label}_cache.pt', cache, length)
            print(f'  {label}: K={metrics["k"].get("response", {}).get("relative_l2", 0):.6f} '
                  f'V={metrics["v"].get("response", {}).get("relative_l2", 0):.6f} '
                  f'KL={metrics["logits"]["kl_true_to_replay_mean"]:.6f}', flush=True)

        oracle_cache, oracle_hidden = parallel_pass(model, ids, len(prompt), true_hidden.to(device))
        compare('oracle_hidden', oracle_cache, oracle_hidden)
        del oracle_cache, oracle_hidden
        previous = None
        for number in range(1, max(args.passes) + 1):
            cache, previous = parallel_pass(model, ids, len(prompt), previous)
            if number in args.passes:
                compare(f'pass_{number:02d}', cache, previous)
            del cache
        del previous, true_cache
        record['seconds'] = time.monotonic() - started
        dump(folder/'metrics.json', record['comparisons'])
        records.append(record)
        dump(args.output/'results.json', dict(completed_examples=len(records), records=records,
                                              aggregate=aggregate(records)))
    print(json.dumps(aggregate(records), indent=2), flush=True)


if __name__ == '__main__':
    main()
