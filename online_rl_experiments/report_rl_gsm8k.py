"""Math-Verify grading and W&B reporting of a full GSM8K evaluation."""
import argparse
import json
from pathlib import Path

from fbt_experiments.evaluate_checkpoint import wilson_interval, exact_mcnemar_p
from prepare_data import sha256
from verifier import MathVerifier


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--compare-soft', type=Path)
    args = parser.parse_args()
    directory = args.directory
    config = json.loads((directory/'run_config.json').read_text())
    if config['num_gsm8k'] != 1319 or config['gsm8k_start'] != 0 or config['modes'] not in [['soft'], ['standard']] or config['gsm8k_shots'] != 0 or config['gsm8k_prompt_format'] != 'chat':
        raise ValueError('Expected full-test, zero-shot chat, single-mode GSM8K evaluation')
    decode_mode = config['modes'][0]
    rows = [json.loads(s) for s in (directory/'gsm8k_generations.jsonl').read_text().splitlines()]
    if [r['example_index'] for r in rows] != list(range(1319)):
        raise ValueError('Expected all 1319 unique test questions in order')
    checkpoint = Path(config['checkpoint'])
    checkpoint_meta = checkpoint.with_name(f"meta_{config['step']:06d}.json")
    identity = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    metadata_sha256=sha256(checkpoint_meta),
                    evaluation_metadata_sha256=sha256(directory/'checkpoint_meta.json'), step=config['step'])
    # The repository merger reserializes JSON; compare its content, not whitespace.
    if json.loads(checkpoint_meta.read_text()) != json.loads((directory/'checkpoint_meta.json').read_text()):
        raise ValueError('Saved evaluation metadata differs from checkpoint metadata')
    verifier = MathVerifier()
    output = directory/'gsm8k_generations_math_verify.jsonl'
    with output.open('x') as f:
        for row in rows:
            mode = row['modes'][decode_mode]
            # Grade answers even if generation reached the response token limit.
            grade = verifier.grade(mode['completion'], row['reference_answer'])
            mode.update(math_verify_correct=grade['correct'], math_verify_prediction=grade['prediction'],
                        math_verify_parsed=grade['parsed'], math_verify_error=grade['error'])
            f.write(json.dumps(row)+'\n')
    correct = sum(r['modes'][decode_mode]['math_verify_correct'] for r in rows)
    truncated = sum(r['modes'][decode_mode]['stop_reason']=='max_new_tokens' for r in rows)
    summary = dict(examples=len(rows), correct=correct, accuracy=correct/len(rows),
        accuracy_wilson_95=wilson_interval(correct,len(rows)), truncated=truncated,
        truncation_rate=truncated/len(rows),
        mean_completion_tokens=sum(r['modes'][decode_mode]['completion_tokens'] for r in rows)/len(rows),
        verifier_errors=sum(bool(r['modes'][decode_mode]['math_verify_error']) for r in rows),
        answer_parse_rate=sum(r['modes'][decode_mode]['math_verify_parsed'] for r in rows)/len(rows))
    result = dict(dataset='GSM8K full test', checkpoint=identity, decode_mode=decode_mode, temperature=0.,
                  max_new_tokens=config['max_new_tokens'], math_verify=verifier.metadata(), metrics=summary)
    if args.compare_soft:
        if decode_mode != 'standard':
            raise ValueError('--compare-soft requires standard decoding')
        other = json.loads((args.compare_soft/'metrics_math_verify.json').read_text())
        other_config = json.loads((args.compare_soft/'run_config.json').read_text())
        for key in ['checkpoint_sha256', 'metadata_sha256', 'step']:
            if identity[key] != other['checkpoint'][key]:
                raise ValueError(f'Comparison checkpoint mismatch: {key}')
        for key in ['seed', 'max_new_tokens', 'gsm8k_prompt_format', 'gsm8k_shots']:
            if config[key] != other_config[key]:
                raise ValueError(f'Comparison protocol mismatch: {key}')
        if other['decode_mode'] != 'soft' or other['math_verify'] != result['math_verify']:
            raise ValueError('Comparison mode or verifier mismatch')
        soft_rows = [json.loads(s) for s in (args.compare_soft/'gsm8k_generations_math_verify.jsonl').read_text().splitlines()]
        if [r['example_index'] for r in soft_rows] != list(range(1319)):
            raise ValueError('Comparison question IDs differ')
        counts = dict(both_correct=0, standard_only_correct=0, soft_only_correct=0, both_wrong=0)
        for left, right in zip(rows, soft_rows):
            if any(left[k] != right[k] for k in ['prompt', 'reference_answer']):
                raise ValueError('Comparison question content differs')
            a, b = left['modes']['standard']['math_verify_correct'], right['modes']['soft']['math_verify_correct']
            key = 'both_correct' if a and b else 'standard_only_correct' if a else 'soft_only_correct' if b else 'both_wrong'
            counts[key] += 1
        if counts['both_correct']+counts['soft_only_correct'] != other['metrics']['correct']:
            raise ValueError('Soft comparison metrics differ from graded rows')
        result['soft_comparison'] = dict(source=str(args.compare_soft.resolve()), metrics=other['metrics'])
        result['paired_standard_vs_soft'] = dict(**counts,
            accuracy_delta_soft_minus_standard=(counts['soft_only_correct']-counts['standard_only_correct'])/1319,
            exact_mcnemar_p=exact_mcnemar_p(counts['standard_only_correct'],counts['soft_only_correct']))
    (directory/'metrics_math_verify.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.wandb:
        import wandb
        run = wandb.init(project='nanochat-online-rl', name=directory.name, job_type='gsm8k-post-rl',
            dir=str(directory), save_code=False, settings=wandb.Settings(disable_git=True,console='off'),
            config={k:v for k,v in result.items() if k!='metrics'})
        run.log({f'gsm8k/{k}':v for k,v in summary.items() if isinstance(v,(int,float))}
                | {f'paired/{k}':v for k,v in result.get('paired_standard_vs_soft',{}).items()})
        (directory/'wandb_run.json').write_text(json.dumps(dict(id=run.id,url=run.url),indent=2)+'\n')
        run.finish()
    (directory/'completed_math_verify.json').write_text(json.dumps(dict(examples=len(rows)))+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
