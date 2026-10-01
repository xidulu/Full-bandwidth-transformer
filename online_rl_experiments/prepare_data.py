"""Materialize unique DAPO prompts, preserving the official prompt and answer."""
import hashlib
import json
import re
from pathlib import Path

DATASET = 'BytedTsinghua-SIA/DAPO-Math-17k'
REVISION = '65877096c24ffa7abc4e4fa5edb95cf3413a5674'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import pyarrow.parquet as pq
    source = Path(__file__).parent / 'data/dapo-math-17k.parquet'
    target = source.with_name('dapo-math-17k.unique.jsonl')
    unique = {}
    conflicts = set()
    count = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            count += 1
            prompt = row['prompt']
            key = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
            answer = row['reward_model']['ground_truth'].strip()
            assert re.fullmatch(r'[+-]?\d+', answer), f'Unexpected noninteger ground truth: {answer}'
            if key in unique:
                if int(unique[key]['answer']) != int(answer):
                    conflicts.add(key)
            else:
                unique[key] = {'id': row['extra_info']['index'], 'messages': prompt, 'answer': answer}
    for key in conflicts:
        del unique[key]
    with target.open('w') as f:
        for row in unique.values():
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = dict(dataset=DATASET, revision=REVISION, source_rows=count,
                    unique_rows=len(unique), conflicting_prompts_dropped=len(conflicts), source_sha256=sha256(source),
                    prepared_sha256=sha256(target), deduplication='exact prompt; exclude prompts with conflicting integer answers')
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
