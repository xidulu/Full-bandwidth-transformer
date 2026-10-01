from copy import deepcopy
import pytest

from main import validate_resume_config,resumed_data_state,restore_data_order
from prepare_big_math_data import convert_big_math_rows,normalize_question


def test_dataset_change_is_explicit_and_does_not_allow_other_changes():
    old=dict(data_sha256='old',world_size=4,decode_mode='soft',likelihood_estimator='three_pass_detached')
    new=dict(old,data_sha256='new')
    with pytest.raises(ValueError,match='data_sha256'):validate_resume_config(old,new)
    new['allow_dataset_change']=True
    validate_resume_config(old,new)
    for key,value in [('world_size',1),('decode_mode','standard'),('likelihood_estimator','hidden_state_replay')]:
        with pytest.raises(ValueError,match=key):validate_resume_config(old,dict(new,**{key:value}))


def test_dataset_branch_preserves_global_step_and_resumes_new_stream():
    old=dict(step=300,data_epoch=3,data_cursor=2520)
    rng,order,cursor,epoch,origin=resumed_data_state(old,1000,42,128,True)
    assert (cursor,epoch,origin)==(0,0,300)
    assert order==restore_data_order(1000,42)[1]
    # Two new rollout batches, checkpoint at 302, then same-data resume.
    saved=dict(step=302,data_epoch=0,data_cursor=256,data_start_step=origin)
    rr,ro,rc,re,start=resumed_data_state(saved,1000,42,128)
    assert (ro,rc,re,start)==(order,256,0,300)
    assert rr.getstate()==rng.getstate()
    with pytest.raises(ValueError,match='dataset position'):
        resumed_data_state(dict(saved,data_cursor=0),1000,42,128)
    with pytest.raises(ValueError,match='dataset position'):
        resumed_data_state(dict(saved,data_start_step=303),1000,42,128)


def test_legacy_resume_uses_zero_data_origin():
    meta=dict(step=3,data_epoch=0,data_cursor=384)
    assert resumed_data_state(meta,1000,42,128)[2:]==(384,0,0)


def test_batch_branch_requires_opt_in_and_preserves_other_guards():
    old=dict(prompts_per_step=128,gradient_accumulation_steps=32,
             samples_per_prompt=8,microbatch_size=8,lr=1e-6,world_size=4)
    new=dict(old,prompts_per_step=512,gradient_accumulation_steps=128)
    with pytest.raises(ValueError,match='prompts_per_step'):
        validate_resume_config(old,new)
    new['allow_batch_size_change']=True
    validate_resume_config(old,new)
    for key,value in [('samples_per_prompt',4),('microbatch_size',16),('lr',2e-6),('world_size',8)]:
        with pytest.raises(ValueError,match=key):
            validate_resume_config(old,dict(new,**{key:value}))


def test_batch_branch_preserves_cursor_and_restarts_after_epoch_wrap():
    old=dict(step=1250,data_start_step=300,data_epoch=0,data_cursor=121600,
             user_config=dict(prompts_per_step=128))
    # A different current batch does not reinterpret already consumed prompts.
    assert resumed_data_state(old,232263,42,512)[2:]==(121600,0,300)
    saved=dict(old,step=1252,data_cursor=122624,
               data_batch_start_step=1250,data_batch_start_position=121600,
               user_config=dict(prompts_per_step=512))
    assert resumed_data_state(saved,232263,42,512)[2:]==(122624,0,300)
    total=121600+300*512
    wrapped=dict(saved,step=1550,data_epoch=total//232263,data_cursor=total%232263)
    restored=resumed_data_state(wrapped,232263,42,512)
    assert restored[2:]==(total%232263,total//232263,300)
    assert restored[1]==restore_data_order(232263,42,total//232263)[1]
    with pytest.raises(ValueError,match='dataset position'):
        resumed_data_state(dict(saved,data_cursor=121856),232263,42,512)
    # An explicit dataset change still resets the stream, even after a batch change.
    assert resumed_data_state(saved,1000,42,512,True)[2:]==(0,0,1252)


def test_conversion_never_exposes_answer_and_excludes_overlap_conflicts():
    source=[dict(problem='Compute 2+2.',answer='4',source='a',domain=['arithmetic']),
            dict(problem='Compute 2+2.',answer='4',source='a'),
            dict(problem='Test   question',answer='3',source='b'),
            dict(problem='Conflict',answer='1',source='b'),
            dict(problem='Conflict',answer='2',source='b'),
            dict(problem='',answer='2',source='c')]
    rows,counts,rejected=convert_big_math_rows(source,{normalize_question('TEST question'):'math500'})
    assert len(rows)==1
    assert rows[0]['messages']==[{'role':'user','content':'Compute 2+2.\n\nPut your final answer in \\boxed{}.'}]
    assert rows[0]['answer']=='4' and rows[0]['source']=='a'
    assert counts['math500_overlap_rows_dropped']==1
    assert counts['conflicting_prompts_dropped']==1
    assert counts['invalid_schema_rows']==1
