from copy import deepcopy
import json
import pytest
from curriculum import CurriculumSampler
from main import validate_resume_config


def example():
    data=[dict(id=str(i),source='a' if i<20 else 'b',llama8b_solve_rate=.3) for i in range(40)]
    strata={k:dict(population=20,mixed_fraction=.5,truncation_rate=0) for k in ['a/0.25_to_0.5','b/0.25_to_0.5']}
    spec=dict(strata=strata,refresh_after_updates=3,prior_groups=2,
        pools=[dict(name='a',questions=8,strata=['a/0.25_to_0.5']),dict(name='exploration',questions=2,strata=list(strata))])
    return data,spec


def observe(sampler, step, indices):
    totals=[[0.0]*6 for k in sampler.keys]
    for i in indices:
        j=sampler.key_indices[sampler.index_keys[i]]
        values=[1,1,.5,0,0,0] if j==0 else [1,0,0,1,0,0]
        totals[j]=[a+b for a,b in zip(totals[j],values)]
    return sampler.observe(step,totals)


def test_distinct_sampling_and_resume_across_refresh_and_deck_wrap():
    data,spec=example(); sampler=CurriculumSampler(data,spec,42,1250)
    for step in range(1251,1260):
        restored=CurriculumSampler(data,spec,42,step-1,json.loads(json.dumps(sampler.state_dict())))
        indices=sampler.draw(step)
        assert indices==restored.draw(step)
        assert len(set(indices))==10
        assert sum(i<20 for i in indices)>=8
        assert observe(sampler,step,indices)==observe(restored,step,indices)
        assert sampler.state_dict()==restored.state_dict()
        if step>=1253:
            assert sampler.frozen['a/0.25_to_0.5']>sampler.frozen['b/0.25_to_0.5']


def test_curriculum_identity_guard():
    old=dict(curriculum_sha256=None)
    new=dict(curriculum_sha256='a')
    with pytest.raises(ValueError,match='curriculum'):validate_resume_config(old,new)
    validate_resume_config(old,dict(new,allow_curriculum_change=True))
    validate_resume_config(new,new)
    with pytest.raises(ValueError,match='curriculum'):validate_resume_config(new,old)
    with pytest.raises(ValueError,match='curriculum'):
        validate_resume_config(new,dict(curriculum_sha256='b',allow_curriculum_change=True))


def test_wrong_population_or_step_rejected():
    data,spec=example(); sampler=CurriculumSampler(data,spec,42,1250)
    with pytest.raises(ValueError,match='step'):
        CurriculumSampler(data,spec,42,1251,sampler.state_dict())
    with pytest.raises(ValueError,match='population'):
        CurriculumSampler(data[:-1],spec,42,1250)
    sampler.draw(1251)
    with pytest.raises(ValueError,match='every prompt'):
        sampler.observe(1251,[[0]*6,[0]*6])
