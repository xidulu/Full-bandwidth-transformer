"""Prepare pinned Big-Math Verified with verifiable references and eval exclusions."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import unicodedata

from prepare_data import sha256
from prepare_orz_data import ANSWER_INSTRUCTION, convert_rows, validate_reference

DATASET = 'SynthLabsAI/Big-Math-RL-Verified'
REVISION = 'c75d2f117cddfecb6bd08756e61e508e59732b21'
FILENAME = 'data/train-00000-of-00001.parquet'


def normalize_question(text):
    return ' '.join(unicodedata.normalize('NFKC',text).split()).casefold()


def evaluation_questions():
    from fbt_experiments.evaluate_math500 import load_math500_rows
    from fbt_experiments.evaluate_checkpoint import load_gsm8k_rows, extract_gsm8k_question
    base=Path(os.environ['NANOCHAT_BASE_DIR'])
    gsm_path=base/'eval_bundle/eval_data/symbolic_problem_solving/gsm8k_prepended_8shot.jsonl'
    rows=load_gsm8k_rows(gsm_path,1319)
    excluded={normalize_question(extract_gsm8k_question(row['context'])):'gsm8k_test' for row in rows}
    math_rows=load_math500_rows(500)
    excluded.update({normalize_question(row['problem']):'math500' for row in math_rows})
    evidence=dict(gsm8k_file_sha256=sha256(gsm_path),gsm8k_count=1319,math500_count=500,
        math500_rows_sha256=hashlib.sha256(json.dumps(math_rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        normalization='Unicode NFKC, whitespace collapse, casefold; exact normalized question match only')
    return excluded,evidence


def convert_big_math_rows(source, excluded):
    conversations=[[{'from':'human','value':row.get('problem')},
                    {'from':'assistant','ground_truth':{'value':row.get('answer')}}] for row in source]
    rows,counts,rejected=convert_rows(conversations)
    kept=[]
    for row in rows:
        original=source[row['source_index']]
        prompt=original['problem']
        overlap=excluded.get(normalize_question(prompt))
        if overlap:
            counts[f'{overlap}_overlap_rows_dropped']+=1
            rejected.append(dict(source_index=row['source_index'],reason=f'Exact normalized overlap: {overlap}'))
            continue
        digest=hashlib.sha256(prompt.encode()).hexdigest()
        row.update(id=f'bigmath-{digest}',source=original.get('source'),domain=original.get('domain'),
                   llama8b_solve_rate=original.get('llama8b_solve_rate'),split='train')
        kept.append(row)
    if len({row['id'] for row in kept}) != len(kept):
        raise ValueError('Non-unique prepared IDs')
    return kept,counts,rejected


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--download',action='store_true')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--source-dir',type=Path,default=Path('data/big_math_verified_source'))
    p.add_argument('--output',type=Path,default=Path('data/big-math-rl-verified.train.jsonl'))
    args=p.parse_args()
    os.environ.setdefault('NANOCHAT_BASE_DIR','/home/jhu/xwang457/work/nanochat_cache')
    if args.download:
        from huggingface_hub import hf_hub_download
        hf_hub_download(DATASET,FILENAME,repo_type='dataset',revision=REVISION,local_dir=args.source_dir)
    import pyarrow.parquet as pq
    source_path=args.source_dir/FILENAME
    source=pq.read_table(source_path).to_pylist()
    exclusions,eval_evidence=evaluation_questions()
    rows,counts,rejected=convert_big_math_rows(source,exclusions)
    del source
    answers=sorted({row['answer'] for row in rows})
    print(f'Validating {len(answers)} distinct references from {len(rows)} prompts',flush=True)
    failures={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(answer,error) in enumerate(pool.map(validate_reference,answers,chunksize=16),1):
            if error:failures[answer]=error
            if i%2000==0:print(f'Validated {i}/{len(answers)}; unsupported={len(failures)}',flush=True)
    from nanochat.tokenizer import get_tokenizer
    tokenizer=get_tokenizer()
    from verifier import MathVerifier
    verifier=MathVerifier()
    kept=[];sources=Counter();eligible=0
    for row in rows:
        if row['answer'] in failures:
            rejected.append(dict(source_index=row['source_index'],id=row['id'],answer=row['answer'],reason=failures[row['answer']]))
            continue
        prompt=tokenizer.render_for_completion({'messages':row['messages']+[{'role':'assistant','content':''}]})
        eligible+=len(prompt)<=1024
        sources[row['source']]+=1
        kept.append(row)
    counts['unverifiable_reference_rows_dropped']=len(rows)-len(kept)
    counts['unique_rows']=len(kept)
    if not kept or not eligible:raise ValueError('No eligible verifiable questions remain')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w') as f:
        for row in kept:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    rejected_path=args.output.with_suffix('.rejected.jsonl')
    with rejected_path.open('w') as f:
        for row in rejected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest=dict(dataset=DATASET,revision=REVISION,split='train',source_file=FILENAME,**counts,
        source_sha256=sha256(source_path),prepared_sha256=sha256(args.output),rejected_sha256=sha256(rejected_path),
        preparation_sha256=sha256(__file__),preparation_dependency_sha256=sha256('prepare_orz_data.py'),
        verifier_sha256=sha256('verifier.py'),verifier=verifier.metadata(),prompt_suffix=ANSWER_INSTRUCTION,
        solution_used_in_prompt=False,deduplication='Exact original question, remove conflicting references',
        source_counts=dict(sources),eval_overlap_exclusions=eval_evidence,
        context=dict(max_prompt_tokens=1024,max_new_tokens=1024,eligible_prompts=eligible,excluded_long_prompts=len(kept)-eligible),
        reference_validation='Math-Verify parsing and boxed-answer self-verification')
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
