"""Compare standard and soft decoding of the same LF checkpoint."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace

from fbt_experiments.evaluate_checkpoint import exact_mcnemar_p
from fbt_experiments.regrade_math500_math_verify import regrade_dir
from prepare_data import sha256
from report_soft_math500 import records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--standard', type=Path, required=True)
    parser.add_argument('--soft', type=Path, default=Path('results/math500-soft-924333'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-key', choices=['lf_sft', 'curriculum'], default='lf_sft')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    original = json.loads(Path('results/math500-initial-rl-1024-comparison-933611/comparison.json').read_text())['checkpoints'][args.checkpoint_key]
    summaries, graded, configs = {}, {}, {}
    for mode, directory in [('standard', args.standard), ('soft', args.soft)]:
        config = json.loads((directory/'run_config.json').read_text())
        if config['modes'] != [mode] or config['max_new_tokens'] != 1024 or config['num_math500'] != 500 or config['step'] != original['step']:
            raise ValueError('Expected the selected checkpoint step, 500 questions, and a 1024-token budget')
        if Path(config['checkpoint']).resolve() != Path(original['checkpoint']).resolve():
            raise ValueError('Unexpected checkpoint')
        if sha256(directory/'checkpoint_meta.json') != original['metadata_sha256']:
            raise ValueError('Checkpoint metadata differs from the soft baseline')
        configs[mode] = config
        if mode == 'standard':
            if sha256(Path(config['checkpoint'])) != original['checkpoint_sha256']:
                raise ValueError('Checkpoint weights differ from the soft baseline')
            metrics = regrade_dir(directory, SimpleNamespace(parse_timeout=8, verify_timeout=8, force=False))
        else:
            metrics = json.loads((directory/'metrics_math_verify.json').read_text())
        expected = dict(package='math-verify', parse_timeout=8, verify_timeout=8,
                        gold_extraction='LatexExtractionConfig on $reference_answer$',
                        prediction_extraction='default parse(completion)')
        if metrics['math_verify'] != expected:
            raise ValueError('Verifier settings differ')
        graded[mode] = records(directory)
        raw = [json.loads(s) for s in (directory/'math500_generations.jsonl').read_text().splitlines()]
        if len(raw) != 500 or {r['unique_id'] for r in raw} != set(graded[mode]):
            raise ValueError('Unexpected raw question IDs')
        for row in raw:
            other = graded[mode][row['unique_id']]
            if any(row[k] != other[k] for k in ['problem', 'reference_answer']) or row['modes'][mode]['completion'] != other['modes'][mode]['completion']:
                raise ValueError('Graded records differ from saved generations')
        summary = metrics['math500_math_verify'][mode]
        if sum(r['modes'][mode]['math_verify_correct'] for r in graded[mode].values()) != summary['correct']:
            raise ValueError('Score does not match graded records')
        summaries[mode] = dict(summary,
            truncation_rate=sum(r['modes'][mode]['stop_reason']=='max_new_tokens' for r in graded[mode].values())/500,
            verifier_errors=sum(bool(r['modes'][mode]['math_verify_error']) for r in graded[mode].values()))
    for key in ['seed', 'math500_start', 'math500_prompt_format', 'math500_shots']:
        if configs['standard'][key] != configs['soft'][key]:
            raise ValueError(f'Evaluation settings differ: {key}')
    if set(graded['standard']) != set(graded['soft']):
        raise ValueError('Question IDs differ')
    counts = dict(both_correct=0, standard_only_correct=0, soft_only_correct=0, both_wrong=0)
    for key, left in graded['standard'].items():
        right = graded['soft'][key]
        if any(left[k] != right[k] for k in ['problem', 'reference_answer']):
            raise ValueError('Question content differs')
        a, b = left['modes']['standard']['math_verify_correct'], right['modes']['soft']['math_verify_correct']
        label = 'both_correct' if a and b else 'standard_only_correct' if a else 'soft_only_correct' if b else 'both_wrong'
        counts[label] += 1
    comparison = dict(**counts,
        accuracy_delta_soft_minus_standard=(counts['soft_only_correct']-counts['standard_only_correct'])/500,
        exact_mcnemar_p=exact_mcnemar_p(counts['standard_only_correct'], counts['soft_only_correct']))
    hardware = {}
    for mode, config in configs.items():
        sources = config.get('source_shards', [])
        hardware[mode] = sorted({json.loads((Path(p)/'run_config.json').read_text())['cuda_device'] for p in sources}) if sources else [config['cuda_device']]
    result = dict(examples=500, checkpoint_key=args.checkpoint_key, checkpoint=original, max_new_tokens=1024, temperature=0.,
                  generation_hardware=hardware,
                  math_verify_version=version('math-verify'), models=summaries, paired=comparison,
                  generation_sources=dict(standard=str(args.standard.resolve()), soft=str(args.soft.resolve())))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    if args.wandb:
        import wandb
        run = wandb.init(project='nanochat-online-rl', name=args.output.name, job_type='math500-lf-decoding',
            dir=str(args.output), save_code=False, settings=wandb.Settings(disable_git=True, console='off'),
            config={k:v for k,v in result.items() if k not in ['models','paired']})
        run.log({f'math500/{mode}/{key}':v for mode,row in summaries.items() for key,v in row.items() if isinstance(v,(int,float))}
                | {f'paired/{key}':v for key,v in comparison.items()})
        (args.output/'wandb_run.json').write_text(json.dumps(dict(id=run.id, url=run.url), indent=2)+'\n')
        run.finish()
    (args.output/'completed.json').write_text(json.dumps(dict(examples=500, modes=list(summaries)))+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
