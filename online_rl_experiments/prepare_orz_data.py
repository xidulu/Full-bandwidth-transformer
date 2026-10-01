"""Prepare pinned ORZ math data for online RL with Math-Verify references."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from prepare_data import sha256

DATASET = 'Open-Reasoner-Zero/orz_math_72k_collection_extended'
REVISION = 'b5c5890bcf04853531d4f2aeeef18fb7af6cabd1'
FILENAME = 'orz_math_72k_collection_extended.json'
ANSWER_INSTRUCTION = '\n\nPut your final answer in \\boxed{}.'


def convert_rows(rows):
    unique, conflicts = {}, set()
    counts = Counter(source_rows=len(rows))
    rejected = []
    for index, conversation in enumerate(rows):
        try:
            if (len(conversation) != 2 or conversation[0]['from'] != 'human'
                    or conversation[1]['from'] != 'assistant'):
                raise ValueError('Expected one human question and one assistant reference')
            prompt = conversation[0]['value']
            answer = conversation[1]['ground_truth']['value']
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError('Empty/non-text question')
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError('Empty/non-text reference')
            answer = answer.strip()
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            counts['invalid_schema_rows'] += 1
            rejected.append({'source_index': index, 'reason': str(exc)})
            continue
        if prompt in unique:
            counts['duplicate_prompt_rows'] += 1
            if unique[prompt]['answer'] != answer:
                conflicts.add(prompt)
        else:
            unique[prompt] = {'id': f'orz-{index:06d}', 'source_index': index,
                             'messages': [{'role': 'user', 'content': prompt + ANSWER_INSTRUCTION}],
                             'answer': answer}
    for prompt in conflicts:
        row = unique.pop(prompt)
        rejected.append({'source_index': row['source_index'], 'reason': 'Conflicting references for exact prompt'})
    counts['conflicting_prompts_dropped'] = len(conflicts)
    counts['unique_prompts_before_reference_validation'] = len(unique)
    return list(unique.values()), counts, rejected


def validate_reference(answer):
    from math_verify.errors import TimeoutException
    from verifier import MathVerifier
    try:
        result = MathVerifier().grade('\\boxed{' + answer + '}', answer)
        if result['correct']:
            return answer, None
        return answer, result['error'] or 'Reference does not pass boxed-answer round trip'
    except (Exception, TimeoutException) as exc:
        return answer, f'{type(exc).__name__}: {exc}'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).parent / 'data/orz_math_72k_extended_source')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'data/orz-math-72k-extended.unique.jsonl')
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import hf_hub_download
        hf_hub_download(DATASET, FILENAME, repo_type='dataset', revision=REVISION, local_dir=args.source_dir)
    source = args.source_dir / FILENAME
    rows, counts, rejected = convert_rows(json.loads(source.read_text()))
    answers = sorted({row['answer'] for row in rows})
    print(f'Validating {len(answers)} distinct references from {len(rows)} unique prompts', flush=True)
    failures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (answer, error) in enumerate(pool.map(validate_reference, answers, chunksize=32), 1):
            if error:
                failures[answer] = error
            if index % 2000 == 0:
                print(f'Validated {index}/{len(answers)} references; unsupported={len(failures)}', flush=True)
    kept = []
    for row in rows:
        if row['answer'] in failures:
            rejected.append({'source_index': row['source_index'], 'answer': row['answer'],
                             'reason': failures[row['answer']]})
        else:
            kept.append(row)
    counts['unverifiable_reference_rows_dropped'] = len(rows) - len(kept)
    counts['unique_rows'] = len(kept)
    if not kept:
        raise ValueError('No verifiable prompts remain')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    rejection_path = args.output.with_suffix('.rejected.jsonl')
    with rejection_path.open('w') as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    from verifier import MathVerifier
    manifest = dict(dataset=DATASET, revision=REVISION, source_file=FILENAME, **counts,
                    source_sha256=sha256(source), prepared_sha256=sha256(args.output),
                    rejected_sha256=sha256(rejection_path), preparation_sha256=sha256(__file__),
                    verifier_sha256=sha256(Path(__file__).with_name('verifier.py')),
                    deduplication='exact original question; exclude conflicting stripped reference strings',
                    reference_validation='Math-Verify gold parsing and boxed-answer self-verification',
                    verifier=MathVerifier().metadata(), prompt_suffix=ANSWER_INSTRUCTION)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
