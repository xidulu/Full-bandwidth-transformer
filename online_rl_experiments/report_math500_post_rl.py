"""Regrade MATH-500 with a shared verifier and compare paired baseline outputs."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from fbt_experiments.evaluate_checkpoint import exact_mcnemar_p
from fbt_experiments.regrade_math500_math_verify import regrade_dir
from prepare_data import sha256


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--baseline',type=Path,default=Path('../fbt_experiments/results/d20_standard_60k_openmath_train5m_k1_anygpu_004407_math500_0shot_chat_full'))
    parser.add_argument('--wandb',action='store_true')
    args = parser.parse_args()
    grade_args = SimpleNamespace(parse_timeout=8,verify_timeout=8,force=False)
    rl_metrics = regrade_dir(args.output,grade_args)
    baseline = args.output/'baseline_regraded'
    baseline.mkdir(exist_ok=False)
    for name in ['run_config.json','metrics.json','math500_generations.jsonl','checkpoint_meta.json']:
        shutil.copyfile(args.baseline/name,baseline/name)
    base_metrics = regrade_dir(baseline,grade_args)
    def records(directory):
        rows = [json.loads(line) for line in (directory/'math500_generations_math_verify.jsonl').read_text().splitlines()]
        result = {row['unique_id']:row for row in rows}
        if len(rows)!=500 or len(result)!=500:
            raise ValueError('Expected exactly 500 unique evaluation examples')
        return result
    left,right = records(baseline),records(args.output)
    if set(left)!=set(right):
        raise ValueError('Baseline and RL examples differ')
    paired = dict(both_correct=0,baseline_only_correct=0,rl_only_correct=0,both_wrong=0)
    for key in left:
        if any(left[key][k]!=right[key][k] for k in ('problem','reference_answer')):
            raise ValueError(f'Example content mismatch: {key}')
        a=left[key]['modes']['standard']['math_verify_correct']
        b=right[key]['modes']['standard']['math_verify_correct']
        kind = 'both_correct' if a and b else 'baseline_only_correct' if a else 'rl_only_correct' if b else 'both_wrong'
        paired[kind]+=1
    base=base_metrics['math500_math_verify']['standard'];rl=rl_metrics['math500_math_verify']['standard']
    checkpoint=Path(rl_metrics['checkpoint'])
    comparison=dict(examples=500,baseline=base,rl=rl,paired=paired,
        accuracy_delta=rl['accuracy']-base['accuracy'],
        exact_mcnemar_p=exact_mcnemar_p(paired['baseline_only_correct'],paired['rl_only_correct']),
        checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        metadata_sha256=sha256(checkpoint.with_name('meta_000300.json')),
        math_verify_version=version('math-verify'),max_new_tokens=512,temperature=0.0,
        baseline_generation_source=str(args.baseline.resolve()),
        baseline_truncation_rate=sum(r['modes']['standard']['stop_reason']=='max_new_tokens' for r in left.values())/500,
        rl_truncation_rate=sum(r['modes']['standard']['stop_reason']=='max_new_tokens' for r in right.values())/500,
        verifier_errors=sum(bool(r['modes']['standard']['math_verify_error']) for r in right.values()))
    (args.output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    text=(f"# MATH-500 after 300 RL updates\n\n"
          f"Original SFT: {base['correct']}/500 ({base['accuracy']:.2%}).\n\n"
          f"RL step 300: {rl['correct']}/500 ({rl['accuracy']:.2%}); delta {comparison['accuracy_delta']*100:+.2f} percentage points.\n\n"
          f"Paired counts: {paired}. Exact McNemar p={comparison['exact_mcnemar_p']:.6g}.\n\n"
          "Both use zero-shot chat, greedy native decoding, 512 response tokens, and the same Math-Verify grader. "
          "Baseline generations are reused from the original evaluation; GPU hardware may differ.\n")
    (args.output/'comparison.md').write_text(text)
    if args.wandb:
        import wandb
        run=wandb.init(project='nanochat-online-rl',name=args.output.name,job_type='math500-post-rl',
            dir=str(args.output),save_code=False,settings=wandb.Settings(disable_git=True,console='off'),
            config={'training_job':899298,'checkpoint_step':300,'eval_examples':500,'max_new_tokens':512,
                    'temperature':0.0,'math_verify_version':version('math-verify')})
        run.log({'math500/accuracy':rl['accuracy'],'math500/correct':rl['correct'],
                 'math500/baseline_accuracy':base['accuracy'],'math500/accuracy_delta':comparison['accuracy_delta'],
                 'math500/exact_mcnemar_p':comparison['exact_mcnemar_p'],
                 'math500/truncation_rate':comparison['rl_truncation_rate'],
                 **{f'math500/paired/{k}':v for k,v in paired.items()}})
        (args.output/'wandb_run.json').write_text(json.dumps({'id':run.id,'url':run.url},indent=2)+'\n')
        run.finish()
    print(text,flush=True)


if __name__ == '__main__':
    main()
