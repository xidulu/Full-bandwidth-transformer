import json
from types import SimpleNamespace

import pytest

from calibrate_big_math import allocation, difficulty, merge, write_json


def test_strata_boundaries_and_capped_allocation():
    assert [difficulty(x) for x in [0,.125,.25,.5,.75,.9,1,None]] == [
        '0_to_0.125','0.125_to_0.25','0.25_to_0.5','0.5_to_0.75',
        '0.75_to_0.9','0.9_to_1','0.9_to_1','missing']
    assert allocation({'a':1,'b':100,'c':100}, 10) == {'a':1,'b':5,'c':4}
    with pytest.raises(ValueError):
        allocation({'a':1,'b':1},3)


def fixture_output(tmp_path):
    write_json(tmp_path/'config.json',dict(shards=1,prompts=2,populations={'easy':90,'hard':10}))
    (tmp_path/'subset.jsonl').write_text('{"id":"a"}\n{"id":"b"}\n')
    for variant in ['standard','three_pass']:
        directory=tmp_path/variant/'rank0000'
        directory.mkdir(parents=True)
        write_json(directory/'completed.json',{})
        rows=[]
        for i,(key,reward) in enumerate([('easy',1),('hard',0)]):
            rows.append(dict(subset_index=i,id='ab'[i],stratum=key,source='source',
                mode='sampled',rewards=[reward]*8,ended=[True]*8,
                grades=[dict(correct=bool(reward),parsed=True,error=None)]*8))
        (directory/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_merge_distinguishes_sample_and_population_accuracy(tmp_path):
    fixture_output(tmp_path)
    merge(SimpleNamespace(output=tmp_path,wandb=False))
    summary=json.loads((tmp_path/'standard/summary.json').read_text())
    assert summary['stratified_sample']['accuracy'] == .5
    assert summary['population_weighted_accuracy'] == .9
    assert summary['stratified_sample']['all_wrong_groups'] == 1
    assert summary['stratified_sample']['all_correct_groups'] == 1


def test_merge_rejects_duplicate_coverage(tmp_path):
    fixture_output(tmp_path)
    path=tmp_path/'standard/rank0000/generations.jsonl'
    row=path.read_text().splitlines()[0]
    path.write_text(row+'\n'+row+'\n')
    with pytest.raises(ValueError,match='coverage'):
        merge(SimpleNamespace(output=tmp_path,wandb=False))
