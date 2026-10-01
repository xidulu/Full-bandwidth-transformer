"""Summarize full-test GSM8K results while preserving budget and verifier labels."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--standard', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    base = Path('../fbt_experiments/results')
    for label, name, modes in [
        ('Standard SFT', 'd20_standard_60k_openmath_train5m_k1_anygpu_004407_gsm8k_0shot_chat_full', ['standard']),
        ('LF-SFT', 'd20_from40k_lf_k2_gate_product_openmath_train5m_k3_anygpu_004407_gsm8k_0shot_chat_full', ['standard','soft'])]:
        directory = base/name
        config = json.loads((directory/'run_config.json').read_text())
        metrics = json.loads((directory/'metrics.json').read_text())
        assert config['num_gsm8k']==1319 and config['gsm8k_shots']==0 and config['gsm8k_prompt_format']=='chat'
        for mode in modes:
            m = metrics['gsm8k'][mode]
            rows.append(dict(model=label,rl_step=0,decode_mode=mode,max_new_tokens=config['max_new_tokens'],
                correct=m['correct'],examples=1319,accuracy=m['accuracy'],verifier='Legacy numeric extractor',source=str(directory.resolve())))
    for label, job, step, mode in [
        ('Standard initial RL','899298',300,'standard'),
        ('LF initial RL, three-pass','906530',300,'soft'),
        ('LF initial RL, hidden replay','906531',300,'soft'),
        ('Standard Big-Math continuation','910518',1300,'standard'),
        ('LF large batch, uniform','920647',1300,'soft'),
        ('LF large batch, curriculum','922491',1300,'soft')]:
        directory = Path('results')/job
        generations = directory/f'gsm8k_{step:06d}.jsonl'
        records = [json.loads(s) for s in generations.read_text().splitlines()]
        assert [r['index'] for r in records]==list(range(1319))
        assert all(r['verifier']=='math-verify' for r in records)
        metric_rows = [json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()]
        metric = next(r for r in reversed(metric_rows) if r.get('step')==step and r.get('eval/gsm8k_examples')==1319)
        correct = sum(r['correct'] for r in records)
        assert metric['eval/gsm8k_correct']==correct and metric['eval/gsm8k_max_new_tokens']==192
        rows.append(dict(model=label,rl_step=step,decode_mode=mode,max_new_tokens=192,correct=correct,
                         examples=1319,accuracy=correct/1319,verifier='Math-Verify',source=str(generations.resolve())))
    for directory in [Path('results/gsm8k-rl-936300'),args.standard]:
        assert (directory/'completed_math_verify.json').exists()
        m = json.loads((directory/'metrics_math_verify.json').read_text())
        rows.append(dict(model='LF large batch, curriculum',rl_step=m['checkpoint']['step'],decode_mode=m['decode_mode'],
            max_new_tokens=m['max_new_tokens'],verifier='Math-Verify',source=str(directory.resolve()),**m['metrics']))
    paired = json.loads((args.standard/'metrics_math_verify.json').read_text())['paired_standard_vs_soft']
    prior = json.loads(Path('results/gsm8k-large-batch-comparison-920647-922491/comparison.json').read_text())
    result = dict(dataset='GSM8K full test',examples=1319,shots=0,prompt_format='chat',temperature=0.,results=rows,
        paired_curriculum_standard_vs_soft_1024=paired,
        paired_uniform_vs_curriculum_soft_192=dict(**prior['paired_counts'],exact_mcnemar_p=prior['exact_mcnemar_p']),
        note='Token budgets and legacy versus Math-Verify grading differ across historical runs; compare matching protocols.')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['GSM8K full test: 1,319 questions, zero-shot chat, greedy decoding.','',
        '| Model | RL step | Decoding | Token cap | Correct | Accuracy | Grader |',
        '|---|---:|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['rl_step']} | {r['decode_mode']} | {r['max_new_tokens']} | {r['correct']}/1319 | {r['accuracy']:.2%} | {r['verifier']} |")
    lines.extend(['',result['note'],'',f'Paired curriculum standard versus soft at 1024 tokens: {paired}'])
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
