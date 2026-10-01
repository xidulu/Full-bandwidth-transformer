"""Evaluate a reproducible prepared math subset without optimizer updates."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

from prepare_data import sha256


def summarize(records, mode):
    groups = [row for row in records if row['mode'] == mode]
    rates = [sum(row['rewards']) / len(row['rewards']) for row in groups]
    responses = sum(len(row['rewards']) for row in groups)
    correct = sum(sum(row['rewards']) for row in groups)
    # Uncertainty uses independent questions, not correlated samples of a question.
    se = statistics.stdev(rates) / math.sqrt(len(rates)) if len(rates) > 1 else 0.0
    accuracy = correct / responses
    return dict(prompts=len(groups), responses=responses, correct=correct, accuracy=accuracy,
                prompt_cluster_standard_error=se,
                accuracy_normal_95=[max(0, accuracy-1.96*se), min(1, accuracy+1.96*se)],
                any_correct_fraction=sum(any(row['rewards']) for row in groups)/len(groups),
                all_wrong_groups=sum(not any(row['rewards']) for row in groups),
                all_correct_groups=sum(all(row['rewards']) for row in groups),
                mixed_groups=sum(any(row['rewards']) and not all(row['rewards']) for row in groups),
                truncation_rate=sum(not end for row in groups for end in row['ended'])/responses,
                parse_rate=sum(g['parsed'] for row in groups for g in row['grades'])/responses,
                verifier_errors=sum(bool(g['error']) for row in groups for g in row['grades']),
                completed_only_correct=sum(g['correct'] and end for row in groups for g,end in zip(row['grades'],row['ended'])),
                completed_only_accuracy=sum(g['correct'] and end for row in groups for g,end in zip(row['grades'],row['ended']))/responses)


def evaluation_modes(args):
    return [('greedy',0.0,1)] if args.greedy_only else [('greedy',0.0,1),('sampled',1.0,8)]


def merge(args):
    records = []
    for rank in range(args.shards):
        directory = args.output/f'rank{rank:04d}'
        if not (directory/'completed.json').exists():
            raise ValueError(f'Shard {rank} did not complete')
        records.extend(json.loads(line) for line in (directory/'generations.jsonl').read_text().splitlines())
    for mode, _, count in evaluation_modes(args):
        groups = [row for row in records if row['mode'] == mode]
        if sorted(row['subset_index'] for row in groups) != list(range(args.prompts)):
            raise ValueError(f'Incomplete/duplicate subset coverage for {mode}')
        if any(len(row['rewards']) != count for row in groups):
            raise ValueError(f'Incorrect response count for {mode}')
    summary = {mode:summarize(records, mode) for mode, _, _ in evaluation_modes(args)}
    config = json.loads((args.output/'config.json').read_text())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if args.wandb:
        import wandb
        run = wandb.init(project='nanochat-online-rl', name=args.output.name,
                         job_type='math-subset-evaluation', dir=str(args.output), save_code=False,
                         settings=wandb.Settings(disable_git=True, console='off'),
                         config=config)
        metrics = {f'{mode}/{key}':value for mode, item in summary.items()
                   for key,value in item.items() if isinstance(value, (int,float))}
        run.log(metrics)
        (args.output/'wandb_run.json').write_text(json.dumps(dict(id=run.id,url=run.url),indent=2)+'\n')
        run.finish()
    (args.output/'completed.json').write_text(json.dumps({'prompts':args.prompts,'optimizer_updates':0})+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def evaluate(args):
    import torch
    from nanochat.checkpoint_manager import build_model
    from nanochat.common import COMPUTE_DTYPE
    from verifier import MathVerifier
    from vllm_rollout import VLLMRollout, prepare_model_config
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    assert world == args.shards
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)
    directory = args.output/f'rank{rank:04d}'
    directory.mkdir(parents=True, exist_ok=False)
    step = int(args.checkpoint.stem.removeprefix('model_'))
    metadata = args.checkpoint.with_name(f'meta_{step:06d}.json')
    if json.loads(metadata.read_text()).get('num_forward_passes') != 1:
        raise ValueError('This evaluator requires a standard one-pass checkpoint')
    model, tokenizer, _ = build_model(str(args.checkpoint.parent), step, torch.device('cuda',local_rank), 'train')
    model.float()
    model.tie_weights()
    model.cos, model.sin = model.cos.to(COMPUTE_DTYPE), model.sin.to(COMPUTE_DTYPE)
    model.eval()
    model.requires_grad_(False)
    eligible = []
    for line in args.data.read_text().splitlines():
        row = json.loads(line)
        tokens = tokenizer.render_for_completion({'messages':row['messages']+[{'role':'assistant','content':''}]})
        if len(tokens) <= 1024 and len(tokens)+args.max_tokens <= model.config.sequence_len:
            eligible.append(dict(row, tokens=tokens))
    order = list(range(len(eligible)))
    random.Random(args.seed).shuffle(order)
    assert args.prompts <= len(order)
    subset = [eligible[i] for i in order[:args.prompts]]
    if len({row['id'] for row in subset}) != args.prompts:
        raise ValueError('Subset contains duplicate question IDs')
    if rank == 0:
        metadata = args.checkpoint.with_name(f'meta_{step:06d}.json')
        config = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                      metadata_sha256=sha256(metadata), data_sha256=sha256(args.data),
                      dataset=json.loads(args.data.with_suffix('.manifest.json').read_text()),
                      seed=args.seed, prompts=args.prompts, eligible_prompts=len(eligible),
                      max_new_tokens=args.max_tokens, temperatures=[t for _,t,_ in evaluation_modes(args)], top_p=1.0, top_k=None,
                      decode_mode='standard', checkpoint_step=step,
                      excluded_long_prompts=len(args.data.read_text().splitlines())-len(eligible),
                      grade_truncated_answers=args.grade_truncated,
                      samples_per_prompt={mode:n for mode,_,n in evaluation_modes(args)}, verifier=MathVerifier().metadata(),
                      gpu=torch.cuda.get_device_name(), world_size=world, optimizer_updates=0)
        if config['data_sha256'] != config['dataset']['prepared_sha256']:
            raise ValueError('Dataset hash does not match the preparation manifest')
        (args.output/'subset.jsonl').write_text(''.join(json.dumps({k:v for k,v in row.items() if k!='tokens'})+'\n' for row in subset))
        config['subset_sha256'] = sha256(args.output/'subset.jsonl')
        (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        for source in ['evaluate_orz_subset.py','evaluate_orz_subset.slurm','evaluate_dapo_subset.slurm','verifier.py','vllm_rollout.py']:
            (args.output/f'source_{source}').write_bytes(Path(source).read_bytes())
    prepare_model_config(vars(model.config), directory/'model')
    engine = VLLMRollout(model, tokenizer, directory/'model', local_rank, 64,
                        kv_cache_gb=8, max_batched_tokens=2048, seed=args.seed+rank, verify_weights=True)
    started = time.perf_counter()
    try:
        engine.sync_weights(model, 0)
        verifier = MathVerifier()
        indices = list(range(rank, args.prompts, world))
        with (directory/'generations.jsonl').open('w', buffering=1) as handle:
            for mode, temperature, n in evaluation_modes(args):
                for start in range(0,len(indices),32):
                    batch = indices[start:start+32]
                    outputs = engine.generate_groups([subset[i]['tokens'] for i in batch],
                        [args.seed+100000+i*8 for i in batch], n,args.max_tokens,0,temperature=temperature)
                    for index,(suffixes,ended) in zip(batch,outputs):
                        row = subset[index]
                        texts = [tokenizer.decode(tokens[:-1] if done else tokens) for tokens,done in zip(suffixes,ended)]
                        grades = [verifier.grade(text,row['answer'],completed=done or args.grade_truncated) for text,done in zip(texts,ended)]
                        record = dict(subset_index=index,id=row['id'],answer=row['answer'],mode=mode,
                                      completions=texts,ended=ended,grades=grades,
                                      rewards=[int(g['correct']) for g in grades],
                                      response_tokens=[len(tokens) for tokens in suffixes])
                        handle.write(json.dumps(record)+'\n')
                    print(f'rank {rank}: {mode} {start+len(batch)}/{len(indices)} prompts',flush=True)
    finally:
        engine.close()
    (directory/'completed.json').write_text(json.dumps({'seconds':time.perf_counter()-started,'optimizer_updates':0})+'\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=Path('data/orz-math-72k-extended.unique.jsonl'))
    parser.add_argument('--checkpoint',type=Path,default=Path('/home/jhu/xwang457/work/nanochat_cache/chatsft_checkpoints/d20-standard-60k-openmath-train5m-k1-anygpu/model_004407.pt'))
    parser.add_argument('--prompts',type=int,default=512)
    parser.add_argument('--seed',type=int,default=20260924)
    parser.add_argument('--max-tokens',type=int,default=1024)
    parser.add_argument('--shards',type=int,default=4)
    parser.add_argument('--greedy-only',action='store_true')
    parser.add_argument('--grade-truncated',action='store_true',help='Grade answers present at the token limit; also report completed-only accuracy')
    parser.add_argument('--merge',action='store_true')
    parser.add_argument('--wandb',action='store_true')
    args = parser.parse_args()
    merge(args) if args.merge else evaluate(args)


if __name__ == '__main__':
    main()
