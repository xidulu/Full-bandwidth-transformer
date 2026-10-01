"""Prepare only the pinned nlile/hendrycks-MATH-benchmark training split."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from prepare_data import sha256
from prepare_orz_data import ANSWER_INSTRUCTION, convert_rows, validate_reference

DATASET = 'nlile/hendrycks-MATH-benchmark'
REVISION = '465bcdb36f5962aa3512891498966df785fc3c18'
FILENAME = 'data/train-00000-of-00001.parquet'


def convert_math_rows(source):
    # Reference solutions are deliberately never included in model inputs.
    conversations = [[{'from': 'human', 'value': row.get('problem')},
                      {'from': 'assistant', 'ground_truth': {'value': row.get('answer')}}]
                     for row in source]
    rows, counts, rejected = convert_rows(conversations)
    for row in rows:
        original = source[row['source_index']]
        row.update(id=original['unique_id'], subject=original['subject'], level=original['level'], split='train')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Non-unique source IDs')
    return rows, counts, rejected


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).parent/'data/hendrycks_math_benchmark_source')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'data/hendrycks-math-benchmark.train.jsonl')
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import hf_hub_download
        hf_hub_download(DATASET, FILENAME, repo_type='dataset', revision=REVISION, local_dir=args.source_dir)
    import pyarrow.parquet as pq
    source = args.source_dir/FILENAME
    rows, counts, rejected = convert_math_rows(pq.read_table(source).to_pylist())
    answers = sorted({row['answer'] for row in rows})
    print(f'Validating {len(answers)} distinct references from {len(rows)} train prompts', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        failures = {answer:error for answer,error in pool.map(validate_reference,answers,chunksize=32) if error}
    kept = []
    for row in rows:
        if row['answer'] in failures:
            rejected.append({'source_index':row['source_index'], 'id':row['id'],
                             'answer':row['answer'], 'reason':failures[row['answer']]})
        else:
            kept.append(row)
    counts['unverifiable_reference_rows_dropped'] = len(rows)-len(kept)
    counts['unique_rows'] = len(kept)
    if not kept:
        raise ValueError('No verifiable training prompts remain')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in kept))
    rejection_path = args.output.with_suffix('.rejected.jsonl')
    rejection_path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rejected))
    from verifier import MathVerifier
    manifest = dict(dataset=DATASET, revision=REVISION, split='train', source_file=FILENAME,
                    **counts, source_sha256=sha256(source), prepared_sha256=sha256(args.output),
                    rejected_sha256=sha256(rejection_path), preparation_sha256=sha256(__file__),
                    preparation_dependency_sha256=sha256(Path(__file__).with_name('prepare_orz_data.py')),
                    verifier_sha256=sha256(Path(__file__).with_name('verifier.py')),
                    deduplication='exact question; exclude conflicting stripped reference strings',
                    reference_validation='Math-Verify gold parsing and boxed-answer self-verification',
                    verifier=MathVerifier().metadata(), prompt_suffix=ANSWER_INSTRUCTION,
                    solution_used_in_prompt=False)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
