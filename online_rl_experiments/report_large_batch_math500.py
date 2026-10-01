"""Matched Math-Verify comparison of uniform and curriculum LF checkpoints."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace

from fbt_experiments.regrade_math500_math_verify import regrade_dir
from prepare_data import sha256
from report_soft_math500 import records, paired


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--uniform', type=Path, required=True)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=None)
    parser.add_argument('--initial-rl', type=Path, default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, graded, identities = {}, {}, {}
    directories = [('uniform',args.uniform), ('curriculum',args.curriculum)]
    if args.initial_rl:
        directories.insert(0, ('initial_rl', args.initial_rl))
    if args.baseline:
        directories.insert(0, ('lf_sft', args.baseline))
    for name, directory in directories:
        config = json.loads((directory/'run_config.json').read_text())
        if config['modes'] != ['soft'] or config['max_new_tokens'] != args.max_new_tokens or config['num_math500'] != 500:
            raise ValueError(f'Expected 500 soft-decoded examples with a {args.max_new_tokens}-token response limit')
        cached = directory/'metrics_math_verify.json'
        if cached.exists():
            metrics = json.loads(cached.read_text())
            expected_verifier = dict(package='math-verify',parse_timeout=8,verify_timeout=8,
                gold_extraction='LatexExtractionConfig on $reference_answer$',
                prediction_extraction='default parse(completion)')
            if metrics['math_verify'] != expected_verifier:
                raise ValueError('Cached grading used different verifier settings')
            raw = {row['unique_id']:row for row in map(json.loads,(directory/'math500_generations.jsonl').read_text().splitlines())}
            cached_rows = records(directory)
            if set(raw) != set(cached_rows):
                raise ValueError('Cached grading question IDs differ from raw generations')
            for key, row in raw.items():
                other = cached_rows[key]
                if any(row[k] != other[k] for k in ['problem','reference_answer']) or row['modes']['soft']['completion'] != other['modes']['soft']['completion']:
                    raise ValueError('Cached grading differs from raw generations')
            if sum(row['modes']['soft']['math_verify_correct'] for row in cached_rows.values()) != metrics['math500_math_verify']['soft']['correct']:
                raise ValueError('Cached score differs from graded records')
        else:
            metrics = regrade_dir(directory, SimpleNamespace(parse_timeout=8,verify_timeout=8,force=False))
        expected_step = 4407 if name=='lf_sft' else 300 if name=='initial_rl' else 1300
        if metrics['step'] != expected_step:
            raise ValueError(f'Expected checkpoint step {expected_step}')
        graded[name] = records(directory)
        summaries[name] = dict(metrics['math500_math_verify']['soft'],
            truncation_rate=sum(r['modes']['soft']['stop_reason']=='max_new_tokens' for r in graded[name].values())/500,
            verifier_errors=sum(bool(r['modes']['soft']['math_verify_error']) for r in graded[name].values()))
        checkpoint = Path(metrics['checkpoint'])
        meta = json.loads((directory/'checkpoint_meta.json').read_text())
        identities[name] = dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
            metadata_sha256=sha256(directory/'checkpoint_meta.json'),
            step=metrics['step'],model_config=meta['model_config'])
        if name in ('lf_sft', 'initial_rl'):
            original_key = 'lf_sft' if name=='lf_sft' else 'three_pass'
            original = json.loads(Path('results/math500-soft-comparison-910153-910154/comparison.json').read_text())['checkpoints'][original_key]
            expected = Path(original['checkpoint'])
            if identities[name]['checkpoint_sha256'] != original['checkpoint_sha256'] or identities[name]['metadata_sha256'] != original['metadata_sha256']:
                raise ValueError(f'{name} differs from the original checkpoint identity')
        else:
            expected = Path('results')/('920647' if name=='uniform' else '922491')/'checkpoints/model_001300.pt'
        if checkpoint.resolve() != expected.resolve():
            raise ValueError('Unexpected checkpoint identity')
    comparison = paired(graded['uniform'],graded['curriculum'])
    baseline_comparisons = ({name:paired(graded['lf_sft'],graded[name]) for name in graded if name!='lf_sft'}
                            if args.baseline else {})
    initial_comparisons = ({name:paired(graded['initial_rl'],graded[name]) for name in ['uniform','curriculum']}
                          if args.initial_rl else {})
    result = dict(examples=500,decode_mode='soft',temperature=0.,max_new_tokens=args.max_new_tokens,
        verifier='Math-Verify',math_verify_version=version('math-verify'),
        checkpoints=identities,models=summaries,paired_uniform_vs_curriculum=comparison,
        paired_baseline_vs_post_training=baseline_comparisons,
        paired_initial_rl_vs_later_training=initial_comparisons)
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[f'MATH-500: zero-shot chat, greedy soft decoding, {args.max_new_tokens}-token response limit, Math-Verify.','',
           '| Model | Correct | Accuracy | Truncated |','|---|---:|---:|---:|']
    for name,row in summaries.items():
        lines.append(f"| {name} | {row['correct']}/500 | {row['accuracy']:.1%} | {row['truncation_rate']:.1%} |")
    lines.extend(['',f'Paired uniform versus curriculum: {comparison}'])
    lines.extend(f'Paired LF-SFT versus {name}: {value}' for name,value in baseline_comparisons.items())
    lines.extend(f'Paired initial RL versus {name}: {value}' for name,value in initial_comparisons.items())
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    if args.wandb:
        import wandb
        run = wandb.init(project='nanochat-online-rl',name=args.output.name,job_type='math500-curriculum-comparison',
            dir=str(args.output),save_code=False,settings=wandb.Settings(disable_git=True,console='off'),
            config=dict(training_jobs=([906530] if args.initial_rl else [])+[920647,922491],checkpoint_steps={name:row['step'] for name,row in identities.items()},examples=500,
                        decode_mode='soft',temperature=0.,max_new_tokens=args.max_new_tokens,checkpoints=identities))
        run.log({f'math500/{name}/{key}':value for name,row in summaries.items() for key,value in row.items()
                 if isinstance(value,(int,float))} |
                {f'paired/{key}':value for key,value in comparison.items()} |
                {f'paired/baseline_vs_{name}/{key}':value for name,row in baseline_comparisons.items() for key,value in row.items()} |
                {f'paired/initial_rl_vs_{name}/{key}':value for name,row in initial_comparisons.items() for key,value in row.items()})
        (args.output/'wandb_run.json').write_text(json.dumps(dict(id=run.id,url=run.url),indent=2)+'\n')
        run.finish()
    (args.output/'completed.json').write_text(json.dumps(dict(examples=500,models=list(summaries)))+'\n')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
