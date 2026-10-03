"""Merge a checkpoint sweep and compare matched 1024-token MATH-500 results."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from fbt_experiments.regrade_math500_math_verify import regrade_dir
from prepare_data import sha256
from report_soft_math500 import paired, records


def load_result(directory):
    config = json.loads((directory / 'run_config.json').read_text())
    expected = dict(modes=['soft'], max_new_tokens=1024, num_math500=500,
                    math500_start=0, math500_prompt_format='chat', math500_shots=0, seed=42)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'{directory}: expected {key}={value}')
    cached = directory / 'metrics_math_verify.json'
    metrics = (json.loads(cached.read_text()) if cached.exists() else
               regrade_dir(directory, SimpleNamespace(parse_timeout=8, verify_timeout=8, force=False)))
    if metrics['math_verify'] != dict(package='math-verify', parse_timeout=8, verify_timeout=8,
            gold_extraction='LatexExtractionConfig on $reference_answer$',
            prediction_extraction='default parse(completion)'):
        raise ValueError(f'{directory}: incompatible verifier settings')
    graded = records(directory)
    raw = [json.loads(line) for line in (directory / 'math500_generations.jsonl').read_text().splitlines()]
    if len(raw) != 500 or {r['unique_id'] for r in raw} != set(graded):
        raise ValueError(f'{directory}: invalid question coverage')
    for row in raw:
        other = graded[row['unique_id']]
        if any(row[k] != other[k] for k in ('problem', 'reference_answer', 'prompt', 'prompt_tokens')):
            raise ValueError(f'{directory}: cached grading content differs')
        if row['modes']['soft']['completion'] != other['modes']['soft']['completion']:
            raise ValueError(f'{directory}: cached completion differs')
    summary = dict(metrics['math500_math_verify']['soft'])
    if sum(r['modes']['soft']['math_verify_correct'] for r in graded.values()) != summary['correct']:
        raise ValueError(f'{directory}: cached score differs')
    summary.update(truncation_rate=sum(r['modes']['soft']['stop_reason'] == 'max_new_tokens'
                                      for r in graded.values()) / 500,
                   verifier_errors=sum(bool(r['modes']['soft']['math_verify_error']) for r in graded.values()),
                   mean_completion_tokens=summary['completion_tokens'] / 500)
    return config, graded, summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    summaries, identities, graded = {}, {}, {}
    for item in manifest['baselines'] + manifest['checkpoints']:
        name, directory = item['name'], Path(item['output'])
        if item in manifest['checkpoints'] and not (directory / 'metrics.json').exists():
            shards = [str(Path(f'{directory}-shards') / f'shard_{i}') for i in range(8)]
            subprocess.run([sys.executable, '-m', 'fbt_experiments.merge_math500_shards',
                            *shards, '--output-dir', str(directory), '--expected-count', '500'], check=True)
        config, rows, summary = load_result(directory)
        checkpoint = Path(config['checkpoint'])
        identity = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                        metadata_sha256=sha256(directory / 'checkpoint_meta.json'), step=config['step'])
        for key in ('checkpoint_sha256', 'metadata_sha256', 'step'):
            if identity[key] != item[key]:
                raise ValueError(f'{name}: checkpoint identity differs: {key}')
        identities[name], graded[name], summaries[name] = identity, rows, summary
    comparisons = {}
    for baseline in manifest['baselines']:
        left = baseline['name']
        comparisons[left] = {}
        for item in manifest['checkpoints']:
            right = item['name']
            for key in graded[left]:
                if any(graded[left][key][field] != graded[right][key][field]
                       for field in ('prompt', 'prompt_tokens')):
                    raise ValueError(f'{left} / {right}: prompts differ')
            comparisons[left][right] = paired(graded[left], graded[right])
    result = dict(training_job=manifest['training_job'], examples=500, decode_mode='soft',
                  temperature=0.0, top_k=None, max_new_tokens=1024, verifier='Math-Verify',
                  math_verify_version=version('math-verify'), checkpoints=identities, models=summaries,
                  paired_baselines_vs_checkpoints=comparisons,
                  p_value_note='Two-sided exact McNemar p-values are unadjusted for multiple comparisons.',
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    output = Path(manifest['comparison_output'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['MATH-500: zero-shot chat, greedy soft decoding, 1024-token limit, Math-Verify.', '',
             '| Model | Correct | Accuracy | Truncated | Mean tokens |', '|---|---:|---:|---:|---:|']
    for name, row in summaries.items():
        lines.append(f"| {name} | {row['correct']}/500 | {row['accuracy']:.1%} | "
                     f"{row['truncation_rate']:.1%} | {row['mean_completion_tokens']:.1f} |")
    lines += ['', '| Baseline | Checkpoint | Delta (pp) | Both correct | Old only | New only | Both wrong | Exact p |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for left, values in comparisons.items():
        for right, row in values.items():
            lines.append(f"| {left} | {right} | {100*row['accuracy_delta_right_minus_left']:+.1f} | "
                         f"{row['both_correct']} | {row['left_only_correct']} | {row['right_only_correct']} | "
                         f"{row['both_wrong']} | {row['exact_mcnemar_p']:.4g} |")
    lines += ['', result['p_value_note']]
    (output / 'comparison.txt').write_text('\n'.join(lines) + '\n')
    (output / 'completed.json').write_text(json.dumps(dict(examples=500, models=list(summaries))) + '\n')
    if manifest.get('receipt'):
        receipt_path = Path(manifest['receipt'])
        receipt = json.loads(receipt_path.read_text())
        receipt.update(status='completed', completed_at_utc=result['completed_at_utc'],
                       results=summaries, comparison_json=str(output / 'comparison.json'))
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
