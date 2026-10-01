"""Measure rollout/training likelihood mismatch on identical DAPO token histories.

No optimizer updates: use the real FP32-master/BF16 trainer forward with autograd,
microbatch eight, response masks, and Math-Verify advantages. Compare its gradient
with a fixed token-importance-weighted diagnostic (not a new training objective).
"""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from main import TAG, group_advantages, pack, restore_data_order, token_loss
from prepare_data import sha256
from verifier import MathVerifier
from vllm_rollout import VLLMRollout, prepare_model_config


def distribution(values):
    x = torch.as_tensor(values, dtype=torch.float64)
    return dict(count=x.numel(), mean=x.mean().item(), std=x.std(unbiased=False).item(),
                min=x.min().item(), max=x.max().item(),
                **{f'p{q:g}': torch.quantile(x, q / 100).item() for q in (1, 5, 50, 95, 99, 99.9)})


def mismatch(train_logp, rollout_logp):
    delta = torch.as_tensor(train_logp, dtype=torch.float64) - torch.as_tensor(rollout_logp, dtype=torch.float64)
    ratio = delta.exp()
    return dict(log_ratio=distribution(delta), absolute_logprob_error=distribution(delta.abs()),
                token_probability_ratio=distribution(ratio),
                fraction_relative_error_over_1pct=((ratio - 1).abs() > .01).double().mean().item(),
                fraction_relative_error_over_5pct=((ratio - 1).abs() > .05).double().mean().item(),
                fraction_relative_error_over_10pct=((ratio - 1).abs() > .1).double().mean().item(),
                fraction_outside_08_12=((ratio < .8) | (ratio > 1.2)).double().mean().item(),
                k3_mean=(torch.expm1(delta) - delta).mean().item())


def native_cached_scores(model, samples, pad):
    """Teacher-force the same continuations through native Engine's cache path."""
    from nanochat.engine import KVCache
    from nanochat.common import COMPUTE_DTYPE
    config, device = model.config, model.get_device()
    prompt = samples[0]['prompt']
    assert all(sample['prompt'] == prompt for sample in samples)
    def cache(n, length):
        return KVCache(n, config.n_kv_head, length, config.n_embd // config.n_head,
                       config.n_layer, device, COMPUTE_DTYPE)
    result = [[] for _ in samples]
    with torch.no_grad():
        prefill = cache(1, len(prompt))
        logits = model(torch.tensor([prompt], device=device), kv_cache=prefill)[:, -1].expand(len(samples), -1)
        steps = max(len(sample['suffix']) for sample in samples)
        decode = cache(len(samples), len(prompt) + steps)
        decode.prefill(prefill)
        del prefill
        for pos in range(steps):
            tokens = [sample['suffix'][pos] if pos < len(sample['suffix']) else pad for sample in samples]
            ids = torch.tensor(tokens, device=device)
            logps = logits.log_softmax(-1).gather(1, ids[:, None]).squeeze(1).tolist()
            for i, sample in enumerate(samples):
                if pos < len(sample['suffix']):
                    result[i].append(logps[i])
            if pos + 1 < steps:
                logits = model(ids[:, None], kv_cache=decode)[:, -1]
    return result


def compare_gradients(model, original):
    norm0 = norm1 = dot = diff = 0.0
    for name, parameter in model.named_parameters():
        if name not in original:
            continue
        a, b = original[name].double(), parameter.grad.detach().cpu().double()
        norm0 += a.square().sum().item()
        norm1 += b.square().sum().item()
        dot += (a * b).sum().item()
        diff += (b - a).square().sum().item()
    return dict(original_norm=norm0**.5, token_is_norm=norm1**.5,
                cosine=dot / (norm0 * norm1)**.5 if norm0 * norm1 else None,
                relative_l2_difference=(diff / norm0)**.5 if norm0 else None,
                explanation='Fixed per-token p_train/p_vllm weighting; diagnostic only, no optimizer step. '
                            'This does not correct the distribution of preceding histories.')


def run_checkpoint(checkpoint, args, label):
    from nanochat.checkpoint_manager import build_model
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.flash_attention import USE_FA3
    step = int(checkpoint.stem.removeprefix('model_'))
    output = args.output / label
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, tokenizer, _ = build_model(str(checkpoint.parent), step, torch.device('cuda'), 'train')
    model.float()
    model.tie_weights()
    model.cos, model.sin = model.cos.to(COMPUTE_DTYPE), model.sin.to(COMPUTE_DTYPE)
    if model.latent_feedback is not None:
        model.latent_feedback.requires_grad_(False)
    pad = tokenizer.encode_special('<|assistant_end|>')
    eligible = []
    for line in args.data.read_text().splitlines():
        row = json.loads(line)
        prompt = tokenizer.render_for_completion({'messages': row['messages'] + [{'role': 'assistant', 'content': ''}]})
        if len(prompt) <= 1024 and len(prompt) + args.max_tokens <= model.config.sequence_len:
            eligible.append(dict(row, tokens=prompt))
    _, order, _, _ = restore_data_order(len(eligible), args.seed)
    rows = [eligible[i] for i in order[:args.prompts]]
    prepare_model_config(vars(model.config), output / 'model')
    engine = VLLMRollout(model, tokenizer, output / 'model', 0, 32,
                        seed=args.seed, verify_weights=True)
    samples = []
    try:
        engine.sync_weights(model, 0)
        verifier = MathVerifier()
        # Four groups per call: identical local concurrency to four-GPU training.
        # Interleave global prompt groups exactly as each DDP rank receives them.
        groups = {}
        rank_groups = [list(range(rank, args.prompts, 4)) for rank in range(4)]
        for indices in rank_groups:
            for start in range(0, len(indices), 4):
                batch = indices[start:start+4]
                result = engine.generate_groups(
                    [rows[i]['tokens'] for i in batch],
                    [args.seed + (args.prompts + i) * 8 for i in batch],
                    8, args.max_tokens, 0, return_scores=True)
                for group, (suffixes, ended, scores) in zip(batch, result):
                    row = rows[group]
                    texts = [tokenizer.decode(s[:-1] if done else s) for s, done in zip(suffixes, ended)]
                    grades = [verifier.grade(text, row['answer'], completed=done) for text, done in zip(texts, ended)]
                    rewards = [float(grade['correct']) for grade in grades]
                    adv = group_advantages(rewards).tolist()
                    groups[group] = [dict(prompt=row['tokens'], suffix=suffix, advantage=adv[i],
                        group=group, id=row['id'], answer=row['answer'], reward=rewards[i], ended=ended[i],
                        rollout_logp=scores[i]['logprobs'], rollout_top1=scores[i]['top1'])
                        for i, suffix in enumerate(suffixes)]
                print(f'{label}: sampled {len(groups)} / {args.prompts} groups', flush=True)
        samples = [sample for group in sorted(groups) for sample in groups[group]]
        total_tokens = sum(len(sample['suffix']) for sample in samples)
        model.train()
        model.zero_grad(set_to_none=True)
        # Hook captures argmax without changing the actual target/loss forward.
        top1 = []
        def capture_top1(module, inputs, logits):
            top1.append(logits.detach()[..., :model.config.vocab_size].argmax(-1).cpu())
        handle = model.lm_head.register_forward_hook(capture_top1)
        for start in range(0, len(samples), 8):
            batch = samples[start:start+8]
            inputs, targets, adv = pack(batch, pad, model.get_device())
            nll = model(inputs, targets, loss_reduction='none', num_forward_passes=1).view_as(targets)
            valid = targets.ne(-1)
            ids = top1.pop()
            for i, sample in enumerate(batch):
                mask = valid[i].cpu()
                sample['train_logp'] = (-nll[i].detach().cpu()[mask]).tolist()
                sample['train_top1'] = ids[i][mask].tolist()
            token_loss(nll, adv, valid, total_tokens).backward()
        handle.remove()
        original = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        # How much would per-token likelihood-ratio correction change this gradient?
        for start in range(0, len(samples), 8):
            batch = samples[start:start+8]
            inputs, targets, adv = pack(batch, pad, model.get_device())
            nll = model(inputs, targets, loss_reduction='none', num_forward_passes=1).view_as(targets)
            valid = targets.ne(-1)
            ratio = torch.ones_like(nll)
            for i, sample in enumerate(batch):
                delta = torch.tensor(sample['train_logp'], device=nll.device) - torch.tensor(sample['rollout_logp'], device=nll.device)
                ratio[i, valid[i]] = delta.exp()
            token_loss(nll * ratio, adv, valid, total_tokens).backward()
        gradient = compare_gradients(model, original)
        del original
        model.zero_grad(set_to_none=True)
        print(f'{label}: gradient comparison {gradient}', flush=True)
        model.eval()
        for group in (0, args.prompts // 2):
            cached = native_cached_scores(model, groups[group], pad)
            for sample, logp in zip(groups[group], cached):
                sample['native_cached_logp'] = logp
        (output / 'tokens.jsonl').write_text(''.join(json.dumps(sample) + '\n' for sample in samples))
        train = torch.tensor([x for s in samples for x in s['train_logp']], dtype=torch.float64)
        rollout = torch.tensor([x for s in samples for x in s['rollout_logp']], dtype=torch.float64)
        positions = torch.tensor([j for s in samples for j in range(len(s['suffix']))])
        active = torch.tensor([s['advantage'] != 0 for s in samples for _ in s['suffix']])
        train_top = torch.tensor([x for s in samples for x in s['train_top1']])
        rollout_top = torch.tensor([x for s in samples for x in s['rollout_top1']])
        seq_delta = torch.tensor([sum(s['train_logp']) - sum(s['rollout_logp']) for s in samples], dtype=torch.float64)
        cached_samples = [s for s in samples if 'native_cached_logp' in s]
        cached = [x for s in cached_samples for x in s['native_cached_logp']]
        cached_train = [x for s in cached_samples for x in s['train_logp']]
        cached_rollout = [x for s in cached_samples for x in s['rollout_logp']]
        summary = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
            gpu=torch.cuda.get_device_name(), native_fa3=USE_FA3, torch_version=torch.__version__,
            vllm_version='0.14.0', prompts=len(rows), responses=len(samples), tokens=total_tokens,
            response_lengths=distribution([len(s['suffix']) for s in samples]),
            prompt_lengths=distribution([len(row['tokens']) for row in rows]),
            reward_mean=sum(s['reward'] for s in samples)/len(samples),
            mixed_groups=sum(any(s['advantage'] != 0 for s in group) for group in groups.values()),
            same_weights_verified=True, rollout_temperature=1.0, rollout_top_k=-1,
            train_vs_vllm=mismatch(train, rollout),
            top1_agreement=(train_top == rollout_top).double().mean().item(),
            active_tokens_train_vs_vllm=mismatch(train[active], rollout[active]) if active.any() else None,
            by_response_position={f'{lo}-{hi}': mismatch(train[(positions >= lo) & (positions <= hi)],
                rollout[(positions >= lo) & (positions <= hi)]) for lo, hi in [(0,127),(128,511),(512,1023)]
                if ((positions >= lo) & (positions <= hi)).any()},
            sequence_log_ratio=distribution(seq_delta),
            sequence_importance_ess_fraction=(1 / seq_delta.softmax(0).square().sum() / len(samples)).item(),
            native_cached_control=dict(responses=len(cached_samples), tokens=len(cached),
                train_vs_native_cached=mismatch(cached_train, cached),
                native_cached_vs_vllm=mismatch(cached, cached_rollout)),
            gradient_comparison=gradient, elapsed_seconds=time.perf_counter()-started)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        print(json.dumps(summary, allow_nan=False), flush=True)
        return summary
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append')
    parser.add_argument('--data', type=Path, default=Path('data/dapo-math-17k.unique.jsonl'))
    parser.add_argument('--prompts', type=int, default=16)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoints = args.checkpoint or [
        Path(f'/home/jhu/xwang457/work/nanochat_cache/chatsft_checkpoints/{TAG}/model_004407.pt'),
        Path('results/875331/checkpoints/model_000250.pt')]
    config = {key:str(value) if isinstance(value, Path) else value for key,value in vars(args).items() if key != 'checkpoint'}
    config.update(checkpoints=[str(path.resolve()) for path in checkpoints],
                  sources={name:sha256(Path(name)) for name in ('check_train_inference.py','vllm_rollout.py','main.py','verifier.py')})
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    for name in ('check_train_inference.py','vllm_rollout.py'):
        (args.output/f'source_{name}').write_bytes(Path(name).read_bytes())
    summaries = []
    for i, checkpoint in enumerate(checkpoints):
        summaries.append(run_checkpoint(checkpoint, args, f'checkpoint_{i}'))
        gc.collect()
        torch.cuda.empty_cache()
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
