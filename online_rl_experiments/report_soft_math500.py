"""Shared Math-Verify grading and paired MATH-500 comparisons for soft RL."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from fbt_experiments.evaluate_checkpoint import exact_mcnemar_p
from fbt_experiments.regrade_math500_math_verify import regrade_dir
from prepare_data import sha256

BASELINE = Path('../fbt_experiments/results/d20_from40k_lf_k2_gate_product_openmath_train5m_k3_anygpu_004407_math500_0shot_chat_full')


def records(directory):
    rows = [json.loads(line) for line in (directory/'math500_generations_math_verify.jsonl').read_text().splitlines()]
    result = {row['unique_id']:row for row in rows}
    if len(rows) != 500 or len(result) != 500:
        raise ValueError('Expected exactly 500 unique MATH-500 questions')
    return result


def paired(left, right):
    if set(left) != set(right):
        raise ValueError('Evaluation question IDs differ')
    counts = dict(both_correct=0,left_only_correct=0,right_only_correct=0,both_wrong=0)
    for key in left:
        if any(left[key][k] != right[key][k] for k in ('problem','reference_answer')):
            raise ValueError(f'Question content differs: {key}')
        a=left[key]['modes']['soft']['math_verify_correct']
        b=right[key]['modes']['soft']['math_verify_correct']
        label='both_correct' if a and b else 'left_only_correct' if a else 'right_only_correct' if b else 'both_wrong'
        counts[label]+=1
    return dict(**counts, accuracy_delta_right_minus_left=(counts['right_only_correct']-counts['left_only_correct'])/500,
                exact_mcnemar_p=exact_mcnemar_p(counts['left_only_correct'],counts['right_only_correct']))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--three-pass',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=BASELINE)
    p.add_argument('--wandb',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    baseline=args.output/'baseline_regraded'
    baseline.mkdir()
    for name in ['run_config.json','metrics.json','math500_generations.jsonl','checkpoint_meta.json']:
        shutil.copyfile(args.baseline/name,baseline/name)
    # Grade only the matched SOFT baseline, preserving the original files.
    config=json.loads((baseline/'run_config.json').read_text())
    config['modes']=['soft']
    (baseline/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    source_rows=[json.loads(line) for line in (baseline/'math500_generations.jsonl').read_text().splitlines()]
    for row in source_rows:row['modes']={'soft':row['modes']['soft']}
    (baseline/'math500_generations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in source_rows))
    dirs={'lf_sft':baseline,'three_pass':args.three_pass,'hidden_replay':args.replay}
    summaries={};graded={};identities={}
    for name,directory in dirs.items():
        config=json.loads((directory/'run_config.json').read_text())
        if config['modes'] != ['soft'] or config['max_new_tokens'] != 512 or config['num_math500'] != 500:
            raise ValueError('Expected 500 SOFT examples with a 512-token cap')
        metrics=regrade_dir(directory,SimpleNamespace(parse_timeout=8,verify_timeout=8,force=False))
        graded[name]=records(directory)
        summaries[name]=dict(metrics['math500_math_verify']['soft'],
            truncation_rate=sum(r['modes']['soft']['stop_reason']=='max_new_tokens' for r in graded[name].values())/500,
            verifier_errors=sum(bool(r['modes']['soft']['math_verify_error']) for r in graded[name].values()))
        checkpoint=Path(metrics['checkpoint'])
        meta=json.loads((directory/'checkpoint_meta.json').read_text())
        identities[name]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                             metadata_sha256=sha256(directory/'checkpoint_meta.json'),
                             step=metrics['step'],likelihood_estimator=meta.get('likelihood_estimator'),
                             model_config=meta['model_config'])
    comparisons={f'{a}_vs_{b}':paired(graded[a],graded[b]) for a,b in
                 [('lf_sft','three_pass'),('lf_sft','hidden_replay'),('three_pass','hidden_replay')]}
    result=dict(examples=500,decode_mode='soft',temperature=0.,max_new_tokens=512,
                math_verify_version=version('math-verify'),baseline_generation_source=str(args.baseline.resolve()),
                checkpoints=identities,models=summaries,paired_comparisons=comparisons)
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['MATH-500: zero-shot chat, greedy SOFT decoding, 512 response tokens, Math-Verify.','',
           '| Model | Correct | Accuracy | Truncated |','|---|---:|---:|---:|']
    for name,row in summaries.items():
        lines.append(f"| {name} | {row['correct']}/500 | {row['accuracy']:.1%} | {row['truncation_rate']:.1%} |")
    lines.extend(['','Paired comparisons (right minus left):'])
    for name,counts in comparisons.items():
        lines.append(f"- {name}: {counts['accuracy_delta_right_minus_left']*100:+.1f} pp; exact McNemar p={counts['exact_mcnemar_p']:.6g}; counts={counts}.")
    lines.extend(['','LF-SFT baseline generations are reused; all three are graded with the same verifier. GPU hardware may differ.'])
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    if args.wandb:
        import wandb
        run=wandb.init(project='nanochat-online-rl',name=args.output.name,job_type='math500-soft-post-rl',
                       dir=str(args.output),save_code=False,settings=wandb.Settings(disable_git=True,console='off'),
                       config=dict(training_jobs=[906530,906531],checkpoint_step=300,eval_examples=500,
                                   decode_mode='soft',temperature=0.,max_new_tokens=512,checkpoints=identities,
                                   math_verify_version=version('math-verify')))
        logged={f'math500/{name}/{k}':v for name,row in summaries.items() for k,v in row.items() if isinstance(v,(int,float))}
        logged.update({f'paired/{name}/{k}':v for name,row in comparisons.items() for k,v in row.items()})
        run.log(logged)
        (args.output/'wandb_run.json').write_text(json.dumps(dict(id=run.id,url=run.url),indent=2)+'\n')
        run.finish()
    (args.output/'completed.json').write_text(json.dumps(dict(examples=500,models=list(dirs)))+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
