"""Synchronous, distributed online RL with DAPO token normalization.

Each current-policy batch is used for exactly one optimizer update. At this
on-policy point the standard unclipped policy-gradient and PPO gradients agree.
Soft mode uses detached three-pass estimates or replayed recurrent hidden states.
Both soft estimators omit gradients through the rollout recurrence.
This implements DAPO's token reduction, not its full dynamic-sampling recipe.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from prepare_data import sha256
from verifier import MathVerifier

ROOT = Path(__file__).resolve().parent
TAG = 'd20-standard-60k-openmath-train5m-k1-anygpu'


def group_advantages(rewards):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)


def group_outcome_metrics(total_groups, successful_groups, mixed_groups):
    """Global prompt counts from binary rewards; successful means any correct."""
    total_groups, successful_groups, mixed_groups = map(
        int, (total_groups, successful_groups, mixed_groups))
    if not 0 <= mixed_groups <= successful_groups <= total_groups or total_groups <= 0:
        raise ValueError('Invalid global group counts')
    counts = {
        'all_wrong': total_groups - successful_groups,
        'all_correct': successful_groups - mixed_groups,
        'mixed': mixed_groups,
    }
    metrics = {f'reward/{name}_groups': count for name, count in counts.items()}
    metrics.update({f'reward/{name}_group_fraction': count / total_groups
                    for name, count in counts.items()})
    return metrics


def token_loss(nll, advantages, valid, total_tokens):
    """One denominator for ALL microbatches/prompts in the optimizer update."""
    return (nll * advantages[:, None] * valid).sum() / total_tokens


def batch_layout(microbatch_size, accumulation_steps, world_size, samples_per_prompt,
                 prompts_per_step=None):
    local_sequences = microbatch_size * accumulation_steps
    if min(microbatch_size, accumulation_steps, world_size, samples_per_prompt) <= 0:
        raise ValueError('Batch dimensions must be positive')
    if local_sequences % samples_per_prompt:
        raise ValueError('Each rank must hold complete response groups: microbatch * accumulation '
                         'must be divisible by samples per prompt')
    global_prompts = local_sequences * world_size // samples_per_prompt
    if prompts_per_step is not None and prompts_per_step != global_prompts:
        raise ValueError(f'--prompts-per-step must equal {global_prompts} for this batch layout')
    return global_prompts, local_sequences


def reduce_values(values, device, op=dist.ReduceOp.SUM):
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor.tolist()


def sync_context(model, microbatch_index, accumulation_steps):
    if isinstance(model, DDP) and microbatch_index + 1 < accumulation_steps:
        return model.no_sync()
    return nullcontext()


def restore_data_order(size, seed, epoch=0, cursor=0):
    """Rebuild the deterministic shuffle and its RNG at a checkpoint boundary."""
    if size <= 0 or epoch < 0 or not 0 <= cursor <= size:
        raise ValueError('Invalid saved dataset position')
    rng = random.Random(seed)
    order = list(range(size))
    for _ in range(epoch + 1):
        rng.shuffle(order)
    return rng, order, cursor, epoch


def validate_resume_config(saved, current):
    # Changing the horizon, output or logging is allowed; changing the sampled
    # data, batch layout or optimizer would no longer resume this experiment.
    if saved.get('curriculum_sha256') != current.get('curriculum_sha256'):
        if not (current.get('allow_curriculum_change') and not saved.get('curriculum_sha256')
                and current.get('curriculum_sha256')):
            raise ValueError('Resume curriculum mismatch; explicitly enable a new curriculum branch')
    keys = ('data_sha256', 'seed', 'world_size', 'prompts_per_step',
            'samples_per_prompt', 'generation_batch_size', 'microbatch_size',
            'gradient_accumulation_steps', 'max_new_tokens', 'max_prompt_tokens',
            'lr', 'grad_clip', 'temperature', 'top_k', 'objective', 'verifier')
    for key in keys:
        if saved.get(key) != current.get(key):
            if (key == 'data_sha256' and current.get('allow_dataset_change')
                    and saved.get(key) and current.get(key)):
                continue
            if (key in ('prompts_per_step', 'gradient_accumulation_steps')
                    and current.get('allow_batch_size_change')):
                continue
            if key == 'lr' and current.get('allow_learning_rate_change'):
                continue
            raise ValueError(f'Resume configuration mismatch for {key}')
    # Legacy checkpoints are standard-policy runs. A policy/scorer change must
    # start a new optimizer/data stream, never silently resume an old one.
    for key, default in (('decode_mode', 'standard'), ('likelihood_estimator', 'standard'),
                         ('likelihood_forward_passes', 1), ('gradient_forward_passes', 1)):
        if saved.get(key, default) != current.get(key, default):
            raise ValueError(f'Resume configuration mismatch for {key}')
    if saved.get('rollout_engine', 'native') != current.get('rollout_engine', 'native'):
        if not current.get('allow_rollout_engine_change', False):
            raise ValueError('Resume rollout engine changed; pass --allow-rollout-engine-change '
                             'to explicitly branch this experiment')


def restore_optimizer(optimizer, path, *, learning_rate=None):
    # DDP optimizer states are replicated, so every rank restores rank 0's file.
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    optimizer.load_state_dict(state)
    if learning_rate is not None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('Resume learning rate must be finite and positive')
        # load_state_dict restores the old rate as well as the Adam moments.
        for group in optimizer.param_groups:
            group['lr'] = learning_rate


def resumed_data_state(source_meta, size, seed, prompts_per_step, dataset_changed=False):
    """Keep global training steps while starting a new dataset at position zero."""
    step = source_meta['step']
    origin = step if dataset_changed else source_meta.get('data_start_step', 0)
    epoch = 0 if dataset_changed else source_meta['data_epoch']
    cursor = 0 if dataset_changed else source_meta['data_cursor']
    # Batch-size branches preserve the absolute data position. Legacy checkpoints
    # used one prompts/update value since their dataset origin.
    anchor_step = step if dataset_changed else source_meta.get('data_batch_start_step', origin)
    anchor_position = 0 if dataset_changed else source_meta.get('data_batch_start_position', 0)
    previous_batch = source_meta.get('user_config', {}).get('prompts_per_step', prompts_per_step)
    expected_position = anchor_position + (step-anchor_step) * previous_batch
    if (origin < 0 or not origin <= anchor_step <= step or anchor_position < 0
            or epoch * size + cursor != expected_position):
        raise ValueError('Saved dataset position does not match completed steps')
    return (*restore_data_order(size, seed, epoch, cursor), origin)


def collect(engine, prompt, n, max_tokens, seed, temperature=1.0, decode_mode='standard'):
    """Retain sampled EOS for training; discard post-EOS padding from the engine."""
    stops = {engine.tokenizer.get_bos_token_id(), engine.tokenizer.encode_special('<|assistant_end|>')}
    suffixes, ended = [[] for _ in range(n)], [False] * n
    stream = engine.generate(prompt, num_samples=n, max_tokens=max_tokens,
                             temperature=temperature, top_k=None, seed=seed,
                             decode_mode=decode_mode, use_calculator=False)
    try:
        for column, masks in stream:
            for i, token in enumerate(column):
                if not ended[i]:
                    assert masks[i] == 1, 'Unexpected forced token'
                    suffixes[i].append(token)
                    ended[i] = token in stops
            if all(ended):
                break
    finally:
        stream.close()
    return suffixes, ended


def pack(samples, pad, device):
    width = max(len(s['prompt']) + len(s['suffix']) for s in samples)
    ids = torch.full((len(samples), width), pad, dtype=torch.long, device=device)
    targets = torch.full((len(samples), width - 1), -1, dtype=torch.long, device=device)
    for i, sample in enumerate(samples):
        seq = sample['prompt'] + sample['suffix']
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        start = len(sample['prompt']) - 1
        targets[i, start:len(seq)-1] = ids[i, start+1:len(seq)]
    adv = torch.tensor([s['advantage'] for s in samples], device=device)
    return ids[:, :-1].contiguous(), targets, adv


def evaluate(model, engine, tokenizer, count, output, rank=0, world_size=1, verifier=None,
             decode_mode='standard'):
    from fbt_experiments.evaluate_checkpoint import (
        load_gsm8k_rows, build_gsm8k_chat_prompt_ids)
    verifier = verifier if verifier is not None else MathVerifier()
    path = Path(os.environ['NANOCHAT_BASE_DIR']) / 'eval_bundle/eval_data/symbolic_problem_solving/gsm8k_prepended_8shot.jsonl'
    rows = load_gsm8k_rows(path, count)
    correct = 0
    model.eval()
    shard = output.with_name(f'{output.stem}.rank{rank:04d}.jsonl') if world_size > 1 else output
    with shard.open('w') as f, torch.no_grad():
        for i in range(rank, len(rows), world_size):
            row = rows[i]
            prompt, _ = build_gsm8k_chat_prompt_ids(tokenizer, row['context'])
            suffixes, ended = collect(engine, prompt, 1, 192, 42, temperature=0.0, decode_mode=decode_mode)
            text = tokenizer.decode(suffixes[0][:-1] if ended[0] else suffixes[0])
            # Benchmark grading checks the answer even at the fixed token limit.
            grade = verifier.grade(text, str(row['answer']))
            ok = grade['correct']
            correct += ok
            f.write(json.dumps(dict(index=i, correct=ok, completion=text, verifier='math-verify',
                                    parsed_answer=grade['prediction'], verifier_error=grade['error'])) + '\n')
            f.flush()
    correct = int(reduce_values([correct], model.get_device())[0])
    if world_size > 1 and rank == 0:
        records = []
        for r in range(world_size):
            path = output.with_name(f'{output.stem}.rank{r:04d}.jsonl')
            records.extend(json.loads(line) for line in path.read_text().splitlines())
        assert sorted(row['index'] for row in records) == list(range(count))
        output.write_text(''.join(json.dumps(row) + '\n' for row in sorted(records, key=lambda x: x['index'])))
    return {'eval/gsm8k_accuracy': correct / count, 'eval/gsm8k_correct': correct,
            'eval/gsm8k_examples': count, 'eval/gsm8k_max_new_tokens': 192}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir', default=f'/home/jhu/xwang457/work/nanochat_cache/chatsft_checkpoints/{TAG}')
    p.add_argument('--checkpoint-step', type=int, default=4407)
    p.add_argument('--data', type=Path, default=ROOT / 'data/dapo-math-17k.unique.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100, help='Total target step, including resumed steps')
    p.add_argument('--resume-from', type=Path, default=None, help='Exact RL model_STEP.pt checkpoint; restore optimizer and data position')
    p.add_argument('--allow-dataset-change', action='store_true',
                   help='Explicit dataset branch: retain model/optimizer/global step, reset data position only if its hash changed')
    p.add_argument('--allow-batch-size-change', action='store_true',
                   help='Explicitly change prompts/update and accumulation while preserving optimizer and data position')
    p.add_argument('--allow-learning-rate-change', action='store_true',
                   help='Explicitly override the restored optimizer learning rate while preserving its moments and step')
    p.add_argument('--curriculum-config', type=Path, default=None)
    p.add_argument('--allow-curriculum-change', action='store_true',
                   help='Start a curriculum branch from a uniform-sampling checkpoint')
    p.add_argument('--prompts-per-step', type=int, default=None, help='Global prompts, inferred from batch layout')
    p.add_argument('--samples-per-prompt', type=int, default=8)
    p.add_argument('--generation-batch-size', type=int, default=8)
    p.add_argument('--rollout-engine', choices=['native', 'vllm'], default='vllm')
    p.add_argument('--decode-mode', choices=['standard', 'soft'], default='standard',
                   help='Soft rollouts use checkpoint latent feedback; choose a scorer with --soft-likelihood')
    p.add_argument('--soft-likelihood', choices=['three_pass_detached', 'hidden_state_replay'],
                   default='three_pass_detached', help='Soft scoring estimator; replay requires vLLM')
    p.add_argument('--allow-rollout-engine-change', action='store_true',
                   help='Explicitly branch a resumed experiment onto another rollout engine')
    p.add_argument('--vllm-kv-cache-gb', type=float, default=4.0, help='Reserved KV cache GiB per GPU')
    p.add_argument('--vllm-max-batched-tokens', type=int, default=2048)
    p.add_argument('--vllm-max-sequences', type=int, default=None,
                   help='Maximum concurrently scheduled rollout responses per GPU; defaults to local batch size')
    p.add_argument('--vllm-verify-weights', action='store_true', help='Audit every loaded parameter after each sync')
    p.add_argument('--microbatch-size', type=int, default=8, help='Sequences per GPU per backward pass')
    p.add_argument('--gradient-accumulation-steps', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--max-prompt-tokens', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--parse-timeout', type=int, default=5)
    p.add_argument('--verify-timeout', type=int, default=5)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--eval-examples', type=int, default=128)
    p.add_argument('--final-eval-examples', type=int, default=1319)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--project', default='nanochat-online-rl')
    p.add_argument('--entity', default=None)
    p.add_argument('--run-name', default='standard-4867-dapo17k-online')
    p.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    return p.parse_args()


def main():
    args = arguments()
    if args.allow_dataset_change and not args.resume_from:
        raise ValueError('--allow-dataset-change requires --resume-from')
    if args.allow_batch_size_change and not args.resume_from:
        raise ValueError('--allow-batch-size-change requires --resume-from')
    if args.allow_learning_rate_change and not args.resume_from:
        raise ValueError('--allow-learning-rate-change requires --resume-from')
    if args.allow_curriculum_change and (not args.resume_from or not args.curriculum_config):
        raise ValueError('--allow-curriculum-change requires a resumed curriculum branch')
    if args.curriculum_config and args.allow_dataset_change:
        raise ValueError('Curriculum requires the unchanged prepared dataset')
    replay_enabled = args.soft_likelihood == 'hidden_state_replay'
    if replay_enabled and (args.decode_mode != 'soft' or args.rollout_engine != 'vllm'):
        raise ValueError('Hidden state replay requires --decode-mode soft --rollout-engine vllm')
    verifier = MathVerifier(args.parse_timeout, args.verify_timeout)
    assert args.steps > 0 and args.samples_per_prompt >= 2
    assert min(args.generation_batch_size, args.microbatch_size,
               args.max_new_tokens, args.max_prompt_tokens, args.save_every, args.eval_every) > 0
    assert 0 <= args.eval_examples <= 1319 and 0 <= args.final_eval_examples <= 1319
    assert args.lr > 0 and args.grad_clip > 0
    assert args.vllm_kv_cache_gb > 0 and args.vllm_max_batched_tokens > 0
    assert args.vllm_max_sequences is None or args.vllm_max_sequences > 0
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    args.prompts_per_step, local_sequences = batch_layout(
        args.microbatch_size, args.gradient_accumulation_steps, world_size,
        args.samples_per_prompt, args.prompts_per_step)
    if not torch.cuda.is_available():
        raise RuntimeError('Run this trainer in a GPU allocation.')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world_size > 1:
        dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=45))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()
    os.environ.setdefault('NANOCHAT_BASE_DIR', '/home/jhu/xwang457/work/nanochat_cache')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from nanochat.checkpoint_manager import build_model, save_checkpoint
    from nanochat.engine import Engine
    import wandb
    checkpoint = (args.resume_from.resolve() if args.resume_from else
                  Path(args.checkpoint_dir) / f'model_{args.checkpoint_step:06d}.pt')
    if args.resume_from:
        if not checkpoint.name.startswith('model_') or checkpoint.suffix != '.pt':
            raise ValueError('--resume-from must name model_STEP.pt')
        args.checkpoint_dir = str(checkpoint.parent)
        args.checkpoint_step = int(checkpoint.stem.removeprefix('model_'))
    metadata = checkpoint.with_name(f'meta_{args.checkpoint_step:06d}.json')
    source_meta = json.loads(metadata.read_text())
    if args.decode_mode == 'standard' and source_meta.get('num_forward_passes') != 1:
        raise ValueError('Expected a one-pass checkpoint')
    if args.decode_mode == 'soft' and not source_meta['model_config'].get('latent_feedback'):
        raise ValueError('Soft post-training requires a latent-feedback checkpoint')
    start_step = source_meta['step'] if args.resume_from else 0
    optimizer_path = checkpoint.with_name(f'optim_{start_step:06d}_rank0.pt')
    if args.resume_from:
        if start_step != args.checkpoint_step or args.steps <= start_step:
            raise ValueError('Resume step must match metadata and precede --steps')
        if not optimizer_path.is_file():
            raise FileNotFoundError(optimizer_path)
        for key in ('optimizer_updates', 'data_epoch', 'data_cursor', 'user_config'):
            if key not in source_meta:
                raise ValueError(f'Resume checkpoint missing {key}')
    run = None
    config = None
    if rank == 0:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config['curriculum_sha256'] = sha256(args.curriculum_config) if args.curriculum_config else None
        if args.curriculum_config:
            curriculum_spec = json.loads(args.curriculum_config.read_text())
            if curriculum_spec['data_sha256'] != sha256(args.data):
                raise ValueError('Curriculum dataset identity mismatch')
            if sum(p['questions'] for p in curriculum_spec['pools']) != args.prompts_per_step:
                raise ValueError('Curriculum question count differs from batch')
            config['curriculum'] = curriculum_spec
            (args.output/'curriculum_config.json').write_text(json.dumps(curriculum_spec,indent=2)+'\n')
        config.update(checkpoint_sha256=sha256(checkpoint), metadata_sha256=sha256(metadata),
                      data_sha256=sha256(args.data), source_gsm8k_correct=642, source_gsm8k_examples=1319,
                      objective='on-policy group-standardized REINFORCE, batch token mean',
                      temperature=1.0, top_k=None, updates_per_batch=1, kl_coefficient=0.0,
                      torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                      trainer_sha256=sha256(Path(__file__)), world_size=world_size,
                      global_sequences_per_update=local_sequences * world_size,
                      local_sequences_per_update=local_sequences,
                      verifier=verifier.metadata(), verifier_sha256=sha256(ROOT / 'verifier.py'),
                      source_gsm8k_verifier='legacy numeric regex', start_step=start_step)
        config['evaluation_engine'] = 'native'
        config.update(likelihood_estimator=args.soft_likelihood if args.decode_mode == 'soft' else 'standard',
                      likelihood_forward_passes=3 if args.decode_mode == 'soft' and not replay_enabled else 1,
                      gradient_forward_passes=1,
                      soft_likelihood_sha256=sha256(ROOT / 'soft_likelihood.py'),
                      hidden_replay_sha256=sha256(ROOT / 'hidden_replay.py'),
                      replay_worker_sha256=sha256(ROOT / 'replay_worker.py'))
        if args.decode_mode == 'soft':
            config.update(objective='soft rollouts; detached three-pass group-relative surrogate, batch token mean',
                          feedback_scope='generated input positions only; hidden states shifted by one token',
                          feedback_jitter=0.0, likelihood_is_recurrent_policy_approximation=True,
                          source_gsm8k_correct=None, source_gsm8k_examples=None, source_gsm8k_verifier=None,
                          source_forward_passes=source_meta.get('num_forward_passes'),
                          latent_feedback_mode=source_meta['model_config']['latent_feedback_mode'])
        if replay_enabled:
            config.update(objective='soft rollouts; detached hidden replay group-relative surrogate, batch token mean',
                          likelihood_is_recurrent_policy_approximation=False,
                          gradient_through_rollout_hidden_states=False)
        if args.rollout_engine == 'vllm':
            import importlib.metadata
            import nanochat_vllm
            adapter_root = Path(nanochat_vllm.__file__).parent
            adapter_sources = {path.name: sha256(path) for path in sorted(adapter_root.glob('*.py'))}
            config['vllm'] = dict(version=importlib.metadata.version('vllm'),
                                  adapter_sources=adapter_sources, dtype='bfloat16',
                                  prefix_caching=False, tensor_parallel_size=1,
                                  rollout_source_sha256=sha256(ROOT / 'vllm_rollout.py'))
            from vllm_rollout import prepare_model_config
            from nanochat_vllm.export_checkpoint import _patched_model_config
            prepare_model_config(_patched_model_config(source_meta), args.output / 'vllm_model', args.decode_mode)
            for path in adapter_root.glob('*.py'):
                (args.output / f'source_nanochat_vllm_{path.name}').write_bytes(path.read_bytes())
        if args.resume_from:
            validate_resume_config(source_meta['user_config'], config)
            config['resume'] = {
                'checkpoint': str(checkpoint), 'completed_step': start_step,
                'optimizer_path': str(optimizer_path), 'optimizer_sha256': sha256(optimizer_path),
                'optimizer_updates': source_meta['optimizer_updates'],
                'data_epoch': source_meta['data_epoch'], 'data_cursor': source_meta['data_cursor'],
                'parent_output': source_meta['user_config']['output'],
            }
        config['dataset_changed'] = bool(args.resume_from and
            source_meta['user_config']['data_sha256'] != config['data_sha256'])
        config['batch_size_changed'] = bool(args.resume_from and
            source_meta['user_config']['prompts_per_step'] != args.prompts_per_step)
        config['learning_rate_changed'] = bool(args.resume_from and
            source_meta['user_config']['lr'] != args.lr)
        if config['learning_rate_changed']:
            config['learning_rate_transition'] = dict(at_step=start_step,
                previous_lr=source_meta['user_config']['lr'], new_lr=args.lr,
                optimizer_moments_restored=True)
        if config['batch_size_changed']:
            config['batch_transition'] = dict(at_step=start_step,
                previous_prompts_per_step=source_meta['user_config']['prompts_per_step'],
                new_prompts_per_step=args.prompts_per_step,
                previous_gradient_accumulation_steps=source_meta['user_config']['gradient_accumulation_steps'],
                new_gradient_accumulation_steps=args.gradient_accumulation_steps,
                optimizer_restored=True, data_position_preserved=not config['dataset_changed'])
        config['data_start_step'] = (start_step if config['dataset_changed'] else
                                    source_meta.get('data_start_step', 0) if args.resume_from else 0)
        if config['dataset_changed']:
            config['dataset_transition'] = dict(at_step=start_step,
                previous_data_sha256=source_meta['user_config']['data_sha256'],
                previous_dataset=source_meta['user_config'].get('dataset'),
                new_data_sha256=config['data_sha256'], optimizer_restored=True,
                reset_data_epoch=0, reset_data_cursor=0)
            config.update(source_gsm8k_correct=None, source_gsm8k_examples=None, source_gsm8k_verifier=None)
        config['dataset'] = json.loads(args.data.with_suffix('.manifest.json').read_text())
        if config['data_sha256'] != config['dataset']['prepared_sha256']:
            raise ValueError('Prepared dataset hash differs from manifest')
        (args.output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
        for source in ('main.py', 'verifier.py', 'vllm_rollout.py', 'train.slurm', 'train_large_batch.slurm',
                       'train_orz_large_batch.slurm', 'prepare_orz_data.py',
                       'train_math_large_batch.slurm', 'prepare_math_data.py',
                       'soft_likelihood.py', 'hidden_replay.py', 'replay_worker.py',
                       'train_soft_math.slurm', 'train_soft_replay_math.slurm',
                       'prepare_big_math_data.py', 'train_big_math_continue.slurm',
                       'restart_big_math_lf512.slurm', 'curriculum.py', 'prepare_curriculum.py',
                       'train_big_math_lf512_curriculum.slurm',
                       'continue_big_math_curriculum_200.slurm',
                       'train_big_math_lf512_curriculum_lr10x.slurm',
                       'restart_curriculum_lr10x.py', 'restart_curriculum_lr10x.slurm'):
            (args.output / f'source_{source}').write_bytes((ROOT / source).read_bytes())
        run = wandb.init(project=args.project, entity=args.entity, name=args.run_name,
                         config=config, mode=args.wandb_mode, dir=str(args.output))
        (args.output / 'wandb_run.json').write_text(json.dumps(dict(id=run.id, url=run.url), indent=2))
        run.define_metric('step')
        run.define_metric('*', step_metric='step')
    if world_size > 1:
        config_list = [config]
        dist.broadcast_object_list(config_list, src=0, device=device)
        config = config_list[0]
    model, tokenizer, _ = build_model(str(checkpoint.parent), args.checkpoint_step, device, 'train')
    # FP32 master weights, including embeddings, for small AdamW updates.
    model.float()
    model.tie_weights()
    from nanochat.common import COMPUTE_DTYPE
    model.cos = model.cos.to(COMPUTE_DTYPE)
    model.sin = model.sin.to(COMPUTE_DTYPE)
    from soft_likelihood import configure_feedback_gradients, SoftThreePassLikelihood, SoftReplayLikelihood
    configure_feedback_gradients(model, args.decode_mode)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    if args.resume_from:
        restore_optimizer(optimizer, optimizer_path,
                          learning_rate=args.lr if args.allow_learning_rate_change else None)
        if any(group['lr'] != args.lr for group in optimizer.param_groups):
            raise ValueError('Restored optimizer learning rate differs from configured lr')
        if source_meta['optimizer_updates'] and not optimizer.state:
            raise ValueError('Resume optimizer state is empty')
        if rank == 0:
            run.summary.update({'resume/start_step': start_step,
                                'resume/optimizer_updates': source_meta['optimizer_updates'],
                                'resume/optimizer_restored': True,
                                'resume/effective_lr': optimizer.param_groups[0]['lr']})
            print(f'Restored optimizer at step {start_step}; target step {args.steps}; '
                  f"effective lr {optimizer.param_groups[0]['lr']}", flush=True)
    scorer_type = SoftReplayLikelihood if replay_enabled else SoftThreePassLikelihood
    scorer = scorer_type(model, tokenizer.get_bos_token_id()) if args.decode_mode == 'soft' else model
    train_model = DDP(scorer, device_ids=[local_rank], broadcast_buffers=False,
                      gradient_as_bucket_view=True) if world_size > 1 else scorer
    engine = Engine(model, tokenizer)
    rollout_engine = None
    if args.rollout_engine == 'vllm':
        from vllm_rollout import VLLMRollout
        rollout_engine = VLLMRollout(
            model, tokenizer, args.output / 'vllm_model', local_rank,
            min(local_sequences, args.vllm_max_sequences or local_sequences),
            kv_cache_gb=args.vllm_kv_cache_gb, max_batched_tokens=args.vllm_max_batched_tokens,
            seed=args.seed + rank, verify_weights=args.vllm_verify_weights, decode_mode=args.decode_mode,
            hidden_replay=replay_enabled)
        if rank == 0:
            run.summary['rollout/max_concurrent_sequences_per_gpu'] = rollout_engine.max_sequences
    pad = tokenizer.encode_special('<|assistant_end|>')
    data, skipped = [], 0
    for line in args.data.read_text().splitlines():
        row = json.loads(line)
        prompt = tokenizer.render_for_completion({'messages': row['messages'] + [{'role': 'assistant', 'content': ''}]})
        if len(prompt) > args.max_prompt_tokens or len(prompt) + args.max_new_tokens > model.config.sequence_len:
            skipped += 1
            continue
        data.append(dict(row, tokens=prompt))
    if not data:
        raise ValueError('No prompts fit the configured context budget')
    if rank == 0:
        run.summary.update({'data/eligible_prompts': len(data), 'data/skipped_long_prompts': skipped})
    curriculum = None
    if args.resume_from and source_meta.get('curriculum_state'):
        if not args.curriculum_config:
            raise ValueError('Curriculum checkpoint requires its sampler configuration')
        rng, order, cursor, epoch = restore_data_order(len(data), args.seed,
            source_meta['data_epoch'], source_meta['data_cursor'])
        data_start_step = source_meta['data_start_step']
    elif args.resume_from:
        rng, order, cursor, epoch, data_start_step = resumed_data_state(
            source_meta, len(data), args.seed, args.prompts_per_step, config['dataset_changed'])
    else:
        rng, order, cursor, epoch = restore_data_order(len(data), args.seed)
        data_start_step = 0
    if args.resume_from and not config['dataset_changed'] and not config['batch_size_changed']:
        data_batch_start_step = source_meta.get('data_batch_start_step', data_start_step)
        data_batch_start_position = source_meta.get('data_batch_start_position', 0)
    else:
        data_batch_start_step = start_step
        data_batch_start_position = epoch * len(data) + cursor
    if args.curriculum_config:
        from curriculum import CurriculumSampler
        if source_meta.get('user_config', {}).get('curriculum_sha256') and not source_meta.get('curriculum_state'):
            raise ValueError('Missing saved curriculum sampler state')
        curriculum = CurriculumSampler(data, config['curriculum'], args.seed, start_step,
                                       source_meta.get('curriculum_state'))
    metrics_file = (args.output / 'metrics.jsonl').open('w', buffering=1) if rank == 0 else None
    samples_file = (args.output / f'rollout_samples.rank{rank:04d}.jsonl').open('w', buffering=1)

    def log(step, metrics):
        if rank != 0:
            return
        record = dict(step=step, **metrics)
        metrics_file.write(json.dumps(record, allow_nan=False) + '\n')
        run.log(record)
        print(json.dumps(record, allow_nan=False), flush=True)

    if args.eval_examples:
        log(start_step, evaluate(model, engine, tokenizer, args.eval_examples, args.output / f'gsm8k_{start_step:06d}.jsonl', rank, world_size, verifier=verifier, decode_mode=args.decode_mode))
    updates = source_meta['optimizer_updates'] if args.resume_from else 0
    for step in range(start_step + 1, args.steps + 1):
        started = time.perf_counter()
        model.eval()
        policy_version = updates
        samples, group_success, group_mixed = [], [], []
        torch.cuda.reset_peak_memory_stats()
        # Finish all fresh rollouts BEFORE any gradient or optimizer step.
        local_groups = []
        curriculum_indices = curriculum.draw(step) if curriculum else None
        curriculum_totals = ([[0.0]*6 for _ in curriculum.keys] if curriculum else None)
        for group in range(args.prompts_per_step):
            if curriculum:
                row = data[curriculum_indices[group]]
            else:
                if cursor == len(order):
                    rng.shuffle(order)
                    cursor, epoch = 0, epoch + 1
                row = data[order[cursor]]
                cursor += 1
            if group % world_size != rank:
                continue
            seed = args.seed + (step * args.prompts_per_step + group) * args.samples_per_prompt
            local_groups.append((group, row, seed))
        sync_seconds = rollout_engine.sync_weights(model, policy_version) if rollout_engine else 0.0
        generation_started = time.perf_counter()
        scored_rollouts = args.decode_mode == 'soft' and rollout_engine is not None
        if rollout_engine:
            generated = rollout_engine.generate_groups(
                [row['tokens'] for _, row, _ in local_groups],
                [seed for _, _, seed in local_groups], args.samples_per_prompt,
                args.max_new_tokens, policy_version, return_scores=scored_rollouts,
                return_hidden_states=replay_enabled)
        else:
            generated = []
            for _, row, seed in local_groups:
                suffixes, ended = [], []
                for offset in range(0, args.samples_per_prompt, args.generation_batch_size):
                    n = min(args.generation_batch_size, args.samples_per_prompt - offset)
                    seq, done = collect(engine, row['tokens'], n, args.max_new_tokens, seed + offset,
                                        decode_mode=args.decode_mode)
                    suffixes.extend(seq)
                    ended.extend(done)
                generated.append((suffixes, ended))
        generation_seconds = time.perf_counter() - generation_started
        verification_started = time.perf_counter()
        for (group, row, _), result in zip(local_groups, generated):
            suffixes, ended = result[:2]
            scores = result[2] if scored_rollouts else None
            texts = [tokenizer.decode(s[:-1] if done else s) for s, done in zip(suffixes, ended)]
            # Math-Verify owns extraction and symbolic comparison; no regex reward path.
            grades = [verifier.grade(text, row['answer'], completed=done)
                      for text, done in zip(texts, ended)]
            answers = [grade['prediction'] for grade in grades]
            rewards = [float(grade['correct']) for grade in grades]
            if curriculum:
                key = curriculum.index_keys[curriculum_indices[group]]
                values = [1, int(0 < sum(rewards) < len(rewards)), sum(rewards)/len(rewards),
                          int(not any(rewards)), int(all(rewards)), sum(not x for x in ended)/len(ended)]
                index = curriculum.key_indices[key]
                curriculum_totals[index] = [a+b for a,b in zip(curriculum_totals[index],values)]
            advantages = group_advantages(rewards).tolist()
            group_success.append(float(any(rewards)))
            group_mixed.append(float(0 < sum(rewards) < len(rewards)))
            for i, suffix in enumerate(suffixes):
                samples.append(dict(prompt=row['tokens'], suffix=suffix, advantage=advantages[i],
                                    reward=rewards[i], ended=ended[i], parsed=grades[i]['parsed'], verifier_error=grades[i]['error']))
                if scores is not None:
                    if len(scores[i]['logprobs']) != len(suffix):
                        raise RuntimeError('Rollout log-probabilities are not aligned with response tokens')
                    samples[-1]['rollout_logprobs'] = scores[i]['logprobs']
                    if replay_enabled:
                        samples[-1]['rollout_hidden_states'] = scores[i]['hidden_states']
            samples_file.write(json.dumps(dict(step=step, group=group, id=row['id'], answer=row['answer'],
                                               rewards=rewards, completions=texts, ended=ended, parsed_answers=answers,
                                               verifier_errors=[grade['error'] for grade in grades])) + '\n')
        verification_seconds = time.perf_counter() - verification_started
        rollout_seconds = time.perf_counter() - started
        assert len(samples) == local_sequences
        local_tokens = sum(len(s['suffix']) for s in samples)
        total_tokens = int(reduce_values([local_tokens], device)[0])
        if total_tokens == 0:
            raise RuntimeError('Empty rollout batch')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, nll_sum = 0.0, 0.0
        likelihood_stats = torch.zeros(3, device=device, dtype=torch.float64)
        train_started = time.perf_counter()
        # DDP averages rank gradients. Multiply by world size so the result is
        # sum(all token losses) / sum(all response tokens), with no extra /GAS.
        for microbatch_index in range(args.gradient_accumulation_steps):
            offset = microbatch_index * args.microbatch_size
            microbatch = samples[offset:offset + args.microbatch_size]
            inputs, targets, adv = pack(microbatch, pad, device)
            with sync_context(train_model, microbatch_index, args.gradient_accumulation_steps):
                score_kwargs = ({'prompt_lengths':[len(s['prompt']) for s in microbatch]}
                                if args.decode_mode == 'soft' else {'num_forward_passes':1})
                if replay_enabled:
                    from hidden_replay import pack_replay
                    score_kwargs['replay_hidden'] = pack_replay(microbatch, inputs, model.config.n_embd, COMPUTE_DTYPE)
                nll = train_model(inputs, targets, loss_reduction='none', **score_kwargs).view_as(targets)
                valid = targets.ne(-1)
                loss = token_loss(nll, adv, valid, total_tokens) * world_size
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite policy loss')
                loss.backward()
            loss_sum += loss.detach().item() / world_size
            nll_sum += (nll.detach() * valid).sum().item()
            if scored_rollouts:
                rollout_logp = torch.tensor([p for s in microbatch for p in s['rollout_logprobs']], device=device)
                delta = -nll.detach()[valid] - rollout_logp
                likelihood_stats += torch.stack((delta.sum(), delta.abs().sum(),
                    ((delta < math.log(0.8)) | (delta > math.log(1.2))).sum())).double()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True).item()
        # Avoid momentum-only updates when every prompt group has zero advantage.
        has_signal = bool(reduce_values([int(any(group_mixed))], device)[0])
        if has_signal:
            optimizer.step()
            updates += 1
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_started
        sums = reduce_values([loss_sum, nll_sum, sum(s['reward'] for s in samples),
                              sum(group_success), sum(group_mixed), sum(s['parsed'] for s in samples),
                              sum(not s['ended'] for s in samples),
                              sum(s['verifier_error'] is not None for s in samples),
                              *likelihood_stats.tolist()], device)
        maxima = reduce_values([max(len(s['suffix']) for s in samples), rollout_seconds,
                                train_seconds, time.perf_counter() - started,
                                torch.cuda.max_memory_allocated() / 1e9,
                                sync_seconds, generation_seconds, verification_seconds], device, dist.ReduceOp.MAX)
        global_sequences = local_sequences * world_size
        curriculum_metrics = {}
        if curriculum:
            totals = torch.tensor(curriculum_totals, device=device, dtype=torch.float64)
            if world_size > 1:
                dist.all_reduce(totals)
            curriculum_metrics = curriculum.observe(step, totals.cpu().tolist())
        score_metrics = {'train/likelihood_forward_passes':config['likelihood_forward_passes'],
                         'train/gradient_forward_passes':1}
        if scored_rollouts:
            prefix = 'replay' if replay_enabled else 'three_pass'
            score_metrics.update({
                f'policy/{prefix}_minus_rollout_logp_mean':sums[8]/total_tokens,
                f'policy/{prefix}_vs_rollout_logp_abs_mean':sums[9]/total_tokens,
                'policy/token_ratio_outside_0p8_1p2':sums[10]/total_tokens})
        if replay_enabled:
            score_metrics['rollout/hidden_replay_bytes_per_rank'] = sum(
                s['rollout_hidden_states'].numel() * s['rollout_hidden_states'].element_size() for s in samples)
        log(step, {'train/loss': sums[0], 'train/response_nll': sums[1] / total_tokens,
                   **score_metrics, **curriculum_metrics,
                   'train/grad_norm': grad_norm, 'train/lr': args.lr, 'train/optimizer_updates': updates,
                   'train/dataset_step': step - data_start_step,
                   'train/skipped_zero_signal': int(not has_signal), 'train/epoch': epoch,
                   'batch/global_sequences': global_sequences, 'batch/global_prompts': args.prompts_per_step,
                   'batch/world_size': world_size, 'batch/microbatch_size_per_gpu': args.microbatch_size,
                   'batch/gradient_accumulation_steps': args.gradient_accumulation_steps,
                   'reward/mean': sums[2] / global_sequences,
                   'reward/verifier_error_rate': sums[7] / global_sequences,
                   'reward/group_pass_at_k': sums[3] / args.prompts_per_step,
                   **group_outcome_metrics(args.prompts_per_step, sums[3], sums[4]),
                   'rollout/parse_rate': sums[5] / global_sequences,
                   'rollout/truncation_rate': sums[6] / global_sequences,
                   'rollout/response_tokens_mean': total_tokens / global_sequences,
                   'rollout/response_tokens_max': maxima[0],
                   'rollout/tokens': total_tokens, 'rollout/policy_version': policy_version,
                   'time/rollout_seconds': maxima[1], 'time/train_seconds': maxima[2],
                   'time/weight_sync_seconds': maxima[5], 'time/generation_seconds': maxima[6],
                   'time/verification_seconds': maxima[7],
                   'time/step_seconds': maxima[3], 'perf/rollout_tokens_per_second': total_tokens / maxima[1],
                   'perf/peak_gpu_gb': maxima[4], 'perf/trainer_peak_gpu_gb': maxima[4]})
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            save_checkpoint(str(args.output / 'checkpoints'), step, model.state_dict(), optimizer.state_dict(),
                            dict(step=step, model_config=vars(model.config), num_forward_passes=config['likelihood_forward_passes'],
                                 decode_mode=args.decode_mode, likelihood_estimator=config['likelihood_estimator'],
                                 gradient_forward_passes=1,
                                 user_config=config, optimizer_updates=updates, data_epoch=epoch, data_cursor=cursor,
                                 data_start_step=data_start_step,
                                 data_batch_start_step=data_batch_start_step,
                                 data_batch_start_position=data_batch_start_position,
                                 curriculum_state=curriculum.state_dict() if curriculum else None))
        if world_size > 1:
            dist.barrier()
        count = args.final_eval_examples if step == args.steps else args.eval_examples
        if count and (step % args.eval_every == 0 or step == args.steps):
            log(step, evaluate(model, engine, tokenizer, count, args.output / f'gsm8k_{step:06d}.jsonl', rank, world_size, verifier=verifier, decode_mode=args.decode_mode))
    samples_file.close()
    if rollout_engine:
        rollout_engine.close()
    if rank == 0:
        (args.output / 'completed.json').write_text(json.dumps(dict(steps=args.steps, optimizer_updates=updates, wandb_url=run.url)))
        metrics_file.close()
        run.finish()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
