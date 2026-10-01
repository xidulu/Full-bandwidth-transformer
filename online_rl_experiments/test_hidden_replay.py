import pytest
import torch

from hidden_replay import HiddenReplayBuffer, pack_replay
from soft_likelihood import SoftReplayLikelihood
from test_soft_likelihood import tiny_model, cpu_attention


def test_buffer_chunked_prompt_reordering_terminal_and_reset():
    buffer = HiddenReplayBuffer(); buffer.begin()
    # Partial prompt must not create a predictor state.
    buffer.capture([('a',1)],torch.tensor([0,1]),torch.randn(2,4),{'a':4})
    assert buffer.rows == {}
    hidden = torch.arange(12.).reshape(3,4).requires_grad_()
    buffer.capture([('a',1),('b',2)],torch.tensor([2,3,1]),hidden,{'a':4,'b':2})
    buffer.capture([('b',0),('a',1)],torch.tensor([2,4]),hidden[:2],{'a':4,'b':2})
    # Complete requests can disappear from the active batch; retained until drain.
    result = buffer.finish([('b',2,2),('a',4,2)])
    torch.testing.assert_close(result[0],torch.stack([hidden[2],hidden[0]]))
    torch.testing.assert_close(result[1],hidden[1].expand(2,-1))
    assert all(not h.requires_grad and h.device.type == 'cpu' for h in result)
    buffer.begin(); assert buffer.rows == {}
    with pytest.raises(RuntimeError,match='mismatch'):
        buffer.finish([('a',4,1)])
    assert buffer.rows is None


def test_buffer_rejects_preemption_replay_and_wrong_lifecycle():
    buffer=HiddenReplayBuffer();buffer.begin()
    with pytest.raises(RuntimeError,match='already active'):buffer.begin()
    buffer.capture([('a',0)],torch.tensor([2]),torch.ones(1,4),{'a':3})
    with pytest.raises(RuntimeError,match='Non-contiguous'):
        buffer.capture([('a',0)],torch.tensor([2]),torch.ones(1,4),{'a':3})
    buffer.finish([('a',3,1)])
    with pytest.raises(RuntimeError,match='not active'):buffer.finish([])


def test_pack_replay_shift_variable_lengths_and_one_token_response():
    samples=[dict(prompt=[0,2,3],suffix=[4,5,1],rollout_hidden_states=torch.arange(12.).reshape(3,4)),
             dict(prompt=[0,2],suffix=[1],rollout_hidden_states=torch.ones(1,4))]
    inputs=torch.zeros((2,5),dtype=torch.long)
    packed=pack_replay(samples,inputs,4,torch.float32)
    torch.testing.assert_close(packed[0,3:],samples[0]['rollout_hidden_states'][:-1])
    assert not packed[0,:3].any() and not packed[1].any()
    samples[0]['rollout_hidden_states']=torch.zeros(2,4)
    with pytest.raises(ValueError,match='align'):pack_replay(samples,inputs,4,torch.float32)


@pytest.mark.parametrize('mode',['gate_product','concat_projection','linear_addition'])
def test_replay_matches_entire_recurrence_and_has_only_one_grad_pass(mode,monkeypatch):
    from nanochat.engine import KVCache
    from nanochat.common import COMPUTE_DTYPE
    from main import pack
    model=tiny_model(mode)
    prompt=[0,2,3];suffix=[4,5,6,7,8,1]
    cache=KVCache(1,2,16,16,2,torch.device('cpu'),COMPUTE_DTYPE)
    outputs=[];states=[]
    with torch.no_grad():
        logits,hidden=model(torch.tensor([prompt]),kv_cache=cache,return_hidden=True)
        previous=hidden[:,-1:]
        for i,token in enumerate(suffix):
            outputs.append(logits[:,-1:]);states.append(previous[0,0].clone())
            if i+1<len(suffix):
                logits,previous=model(torch.tensor([[token]]),kv_cache=cache,feedback_hidden=previous,return_hidden=True)
    sample=dict(prompt=prompt,suffix=suffix,advantage=1.,rollout_hidden_states=torch.stack(states))
    ids,targets,_=pack([sample],1,torch.device('cpu'))
    replay=pack_replay([sample],ids,32,COMPUTE_DTYPE).requires_grad_()
    calls=[];original=model._run_trunk
    def trunk(*a):
        calls.append(torch.is_grad_enabled());return original(*a)
    monkeypatch.setattr(model,'_run_trunk',trunk)
    scorer=SoftReplayLikelihood(model,0)
    actual=scorer(ids,prompt_lengths=[len(prompt)],replay_hidden=replay)
    torch.testing.assert_close(actual[:,2:],torch.cat(outputs,dim=1),rtol=2e-5,atol=2e-6)
    assert calls==[True]
    actual.square().mean().backward()
    assert replay.grad is None
    assert model.transformer.wte.weight.grad.abs().sum()>0
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.latent_feedback.parameters() if p.requires_grad)


def test_exact_request_id_capture_and_cleanup_on_error():
    from types import SimpleNamespace
    from hidden_replay import capture_request_ids
    class Processor:
        @staticmethod
        def assign_request_id(request):
            request.external_req_id=request.request_id
            request.request_id='opaque-internal'
    processor=Processor()
    with pytest.raises(ValueError):
        with capture_request_ids(processor) as mapping:
            processor.assign_request_id(SimpleNamespace(request_id='external-id'))
            assert mapping=={'external-id':'opaque-internal'}
            raise ValueError('failed generation')
    assert 'assign_request_id' not in vars(processor)
