"""Fixed-checkpoint Big-Math calibration with source/difficulty stratification."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random

from prepare_data import sha256
from evaluate_orz_subset import summarize

CHECKPOINTS = {'standard': 'results/899298/checkpoints/model_000300.pt',
               'three_pass': 'results/906530/checkpoints/model_000300.pt'}


def difficulty(value):
    if value is None or not math.isfinite(value):
        return 'missing'
    if not 0 <= value <= 1:
        raise ValueError('Solve rate outside [0,1]')
    for upper, label in [(0.125, '0_to_0.125'), (0.25, '0.125_to_0.25'),
                         (0.5, '0.25_to_0.5'), (0.75, '0.5_to_0.75'),
                         (0.9, '0.75_to_0.9')]:
        if value < upper:
            return label
    return '0.9_to_1'


def allocation(populations, count):
    """Equal allocation across nonempty strata, capped by their populations."""
    if count > sum(populations.values()) or count < len(populations):
        raise ValueError('Sample must cover every stratum and fit the population')
    result = dict.fromkeys(sorted(populations), 0)
    while sum(result.values()) < count:
        for key in result:
            if result[key] < populations[key]:
                result[key] += 1
                if sum(result.values()) == count:
                    break
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    manifest = json.loads(args.data.with_suffix('.manifest.json').read_text())
    if sha256(args.data) != manifest['prepared_sha256']:
        raise ValueError('Prepared data hash mismatch')
    strata = defaultdict(list)
    for line in args.data.open():
        row = json.loads(line)
        tokens = tokenizer.render_for_completion({'messages': row['messages'] +
                                                  [{'role': 'assistant', 'content': ''}]})
        if len(tokens) > 1024:
            continue
        row['stratum'] = row['source'] + '/' + difficulty(row['llama8b_solve_rate'])
        row['tokens'] = tokens
        strata[row['stratum']].append(row)
    populations = {key: len(rows) for key, rows in strata.items()}
    counts = allocation(populations, args.prompts)
    rng = random.Random(args.seed)
    subset = []
    for key in sorted(strata):
        subset.extend(rng.sample(strata[key], counts[key]))
    rng.shuffle(subset)
    if len({r['id'] for r in subset}) != args.prompts:
        raise ValueError('Duplicate subset IDs')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'subset.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in subset))
    checkpoints = {}
    for variant, filename in CHECKPOINTS.items():
        path = Path(filename).resolve()
        meta = path.with_name('meta_000300.json')
        checkpoints[variant] = dict(path=str(path), sha256=sha256(path),
                                   metadata_sha256=sha256(meta))
    sources = ['calibrate_big_math.py', 'calibrate_big_math.slurm', 'merge_big_math_calibration.slurm',
               'evaluate_orz_subset.py', 'verifier.py', 'vllm_rollout.py']
    write_json(args.output/'config.json', dict(dataset=manifest, checkpoints=checkpoints,
        subset_sha256=sha256(args.output/'subset.jsonl'), seed=args.seed,
        prompts=args.prompts, shards=4, samples_per_prompt=8, temperature=1.0,
        top_p=1.0, top_k=None, max_new_tokens=1024, max_prompt_tokens=1024,
        populations=populations, sample_counts=counts,
        sampling='Equal allocation across source x Llama solve-rate strata, capped by population',
        optimizer_updates=0, source_sha256={name: sha256(Path(name)) for name in sources}))
    for source in sources:
        (args.output/('source_'+source)).write_bytes(Path(source).read_bytes())
    print(f'Prepared {len(subset)} questions in {len(strata)} strata', flush=True)


def evaluate(args):
    import torch
    from nanochat.checkpoint_manager import build_model
    from nanochat.common import COMPUTE_DTYPE
    from verifier import MathVerifier
    from vllm_rollout import VLLMRollout, prepare_model_config
    config = json.loads((args.output/'config.json').read_text())
    if sha256(args.output/'subset.jsonl') != config['subset_sha256']:
        raise ValueError('Subset hash mismatch')
    source = config['checkpoints'][args.variant]
    path = Path(source['path'])
    if sha256(path) != source['sha256'] or sha256(path.with_name('meta_000300.json')) != source['metadata_sha256']:
        raise ValueError('Checkpoint identity mismatch')
    torch.cuda.set_device(0)
    torch.manual_seed(config['seed'])
    model, tokenizer, meta = build_model(str(path.parent), 300, torch.device('cuda'), 'train')
    mode = 'standard' if args.variant == 'standard' else 'soft'
    if mode == 'standard' and meta.get('num_forward_passes') != 1:
        raise ValueError('Expected one-pass standard checkpoint')
    if mode == 'soft' and not model.config.latent_feedback:
        raise ValueError('Expected LF checkpoint')
    model.float()
    model.tie_weights()
    model.cos, model.sin = model.cos.to(COMPUTE_DTYPE), model.sin.to(COMPUTE_DTYPE)
    model.eval().requires_grad_(False)
    subset = [json.loads(line) for line in (args.output/'subset.jsonl').read_text().splitlines()]
    if any(len(row['tokens']) + 1024 > model.config.sequence_len for row in subset):
        raise ValueError('Calibration exceeds model context')
    directory = args.output/args.variant/f'rank{args.rank:04d}'
    directory.mkdir(parents=True, exist_ok=False)
    verifier = MathVerifier()
    write_json(directory/'config.json', dict(variant=args.variant, rank=args.rank,
        subset_sha256=config['subset_sha256'], checkpoint=source, decode_mode=mode,
        verifier=verifier.metadata(), gpu=torch.cuda.get_device_name(), optimizer_updates=0))
    prepare_model_config(vars(model.config), directory/'model', decode_mode=mode)
    engine = VLLMRollout(model, tokenizer, directory/'model', 0, 64 if mode == 'standard' else 32,
        kv_cache_gb=8, max_batched_tokens=2048, seed=config['seed']+args.rank,
        verify_weights=True, decode_mode=mode)
    indices = list(range(args.rank, len(subset), config['shards']))
    try:
        engine.sync_weights(model, 300)
        with (directory/'generations.jsonl').open('w', buffering=1) as handle:
            for start in range(0, len(indices), 16):
                batch = indices[start:start+16]
                outputs = engine.generate_groups([subset[i]['tokens'] for i in batch],
                    [config['seed']+100000+i*8 for i in batch], 8, 1024, 300, temperature=1.0)
                for index, (suffixes, ended) in zip(batch, outputs):
                    row = subset[index]
                    texts = [tokenizer.decode(tokens[:-1] if done else tokens)
                             for tokens, done in zip(suffixes, ended)]
                    grades = [verifier.grade(text, row['answer'], completed=done)
                              for text, done in zip(texts, ended)]
                    handle.write(json.dumps(dict(subset_index=index, id=row['id'],
                        source=row['source'], domain=row['domain'], stratum=row['stratum'],
                        llama8b_solve_rate=row['llama8b_solve_rate'], answer=row['answer'],
                        mode='sampled', completions=texts, ended=ended, grades=grades,
                        rewards=[int(g['correct']) for g in grades],
                        response_tokens=[len(tokens) for tokens in suffixes]))+'\n')
                print(f'{args.variant} rank {args.rank}: {start+len(batch)}/{len(indices)} prompts', flush=True)
    finally:
        engine.close()
    write_json(directory/'completed.json', dict(prompts=len(indices), optimizer_updates=0))


def merge(args):
    config = json.loads((args.output/'config.json').read_text())
    subset = [json.loads(line) for line in (args.output/'subset.jsonl').read_text().splitlines()]
    for variant in CHECKPOINTS:
        records = []
        for rank in range(config['shards']):
            directory = args.output/variant/f'rank{rank:04d}'
            if not (directory/'completed.json').exists():
                raise ValueError(f'Incomplete shard {directory}')
            records.extend(json.loads(line) for line in (directory/'generations.jsonl').read_text().splitlines())
        records.sort(key=lambda r:r['subset_index'])
        if [r['subset_index'] for r in records] != list(range(config['prompts'])):
            raise ValueError('Incomplete or duplicate subset coverage')
        if any(r['id'] != s['id'] or len(r['rewards']) != 8 for r,s in zip(records,subset)):
            raise ValueError('Question identity or response count mismatch')
        buckets = defaultdict(list)
        sources = defaultdict(list)
        for row in records:
            buckets[row['stratum']].append(row)
            sources[row['source']].append(row)
        summary = dict(stratified_sample=summarize(records, 'sampled'),
            by_stratum={k:summarize(v, 'sampled') for k,v in buckets.items()},
            by_source={k:summarize(v, 'sampled') for k,v in sources.items()})
        population = sum(config['populations'].values())
        summary['population_weighted_accuracy'] = sum(config['populations'][k]/population *
            v['accuracy'] for k,v in summary['by_stratum'].items())
        summary['note'] = 'Sample/source summaries reflect the stratified calibration mixture; weighted accuracy estimates the eligible dataset mixture. Truncated responses earn zero, matching training.'
        write_json(args.output/variant/'summary.json', summary)
        if args.wandb:
            import wandb
            run = wandb.init(project='nanochat-online-rl', name=f'bigmath-calibration-{variant}-{args.output.name}',
                job_type='calibration', config=dict(config, variant=variant), dir=str(args.output/variant),
                settings=wandb.Settings(disable_git=True, console='off'))
            run.log({f'calibration/{k}':v for k,v in summary['stratified_sample'].items() if isinstance(v,(float,int))} |
                    {'calibration/population_weighted_accuracy':summary['population_weighted_accuracy']})
            table = wandb.Table(columns=['stratum','population','prompts','accuracy','all_wrong_groups','mixed_groups','all_correct_groups','truncation_rate'])
            for key, value in sorted(summary['by_stratum'].items()):
                table.add_data(key, config['populations'][key], *[value[k] for k in table.columns[2:]])
            run.log({'calibration/strata': table})
            write_json(args.output/variant/'wandb_run.json', dict(id=run.id,url=run.url))
            run.finish()
        print(variant, json.dumps(summary['stratified_sample']), flush=True)
    write_json(args.output/'completed.json', dict(variants=list(CHECKPOINTS), optimizer_updates=0))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['prepare','evaluate','merge'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/big-math-rl-verified.train.jsonl'))
    p.add_argument('--prompts', type=int, default=2048)
    p.add_argument('--seed', type=int, default=20260927)
    p.add_argument('--variant', choices=list(CHECKPOINTS), default='standard')
    p.add_argument('--rank', type=int, choices=range(4), default=0)
    p.add_argument('--wandb', action='store_true')
    args = p.parse_args()
    {'prepare':prepare, 'evaluate':evaluate, 'merge':merge}[args.action](args)


if __name__ == '__main__':
    main()
