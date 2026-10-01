from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from soft_likelihood import SoftThreePassLikelihood, configure_feedback_gradients


def tiny_model(mode='gate_product'):
    import nanochat.flash_attention as flash
    from nanochat.gpt import GPT, GPTConfig
    flash.USE_FA3 = False
    config = GPTConfig(sequence_len=32,vocab_size=32,n_layer=2,n_head=2,n_kv_head=2,
                       n_embd=32,window_pattern='L',latent_feedback=True,latent_feedback_mode=mode)
    with torch.device('meta'):
        model = GPT(config,pad_vocab_size_to=1)
    model.to_empty(device='cpu')
    torch.manual_seed(19)
    model.init_weights()
    configure_feedback_gradients(model,'soft')
    return model


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    import nanochat.flash_attention as flash
    monkeypatch.setattr(flash,'USE_FA3',False)
    torch.set_num_threads(1)


@pytest.mark.parametrize('mode',['gate_product','concat_projection','linear_addition'])
def test_score_matches_existing_third_pass_only(mode):
    model=tiny_model(mode)
    scorer=SoftThreePassLikelihood(model,0)
    ids=torch.tensor([[0,2,3,4,5,6],[0,3,4,5,6,7]])
    targets=torch.tensor([[-1,-1,4,5,6,1],[-1,-1,-1,6,7,1]])
    mask=torch.tensor([[False,False,False,True,True,True],
                       [False,False,False,False,True,True]])
    _,components=model(ids,targets,loss_reduction='none',num_forward_passes=3,
                       feedback_masks=mask[None].expand(2,-1,-1),feedback_jitter=0,
                       return_loss_components=True)
    actual=scorer(ids,targets,prompt_lengths=[3,4],loss_reduction='none')
    torch.testing.assert_close(actual,components[2])


def test_only_final_pass_builds_graph_and_embedding_and_fusion_receive_gradients(monkeypatch):
    model=tiny_model();scorer=SoftThreePassLikelihood(model,0)
    calls=[];embeds=[];heads=[]
    original=model._run_trunk
    def trunk(*args):
        hidden=original(*args)
        calls.append((torch.is_grad_enabled(),hidden.requires_grad,hidden.grad_fn))
        return hidden
    monkeypatch.setattr(model,'_run_trunk',trunk)
    hook=model.transformer.wte.register_forward_hook(lambda m,x,y:embeds.append(torch.is_grad_enabled()))
    head=model.lm_head.register_forward_hook(lambda *args:heads.append(True))
    ids=torch.tensor([[0,2,3,4,5,6]])
    loss=scorer(ids,torch.tensor([[-1,-1,4,5,6,1]]),prompt_lengths=[3])
    loss.backward();hook.remove();head.remove()
    assert [(enabled,requires) for enabled,requires,_ in calls]==[(False,False),(False,False),(True,True)]
    assert calls[0][2] is None and calls[1][2] is None
    assert embeds==[False,True] and len(heads)==1
    assert model.transformer.wte.weight.grad.abs().sum()>0
    assert model.lm_head.weight.grad.abs().sum()>0
    for name,param in model.latent_feedback.named_parameters():
        if name in model.latent_feedback.active_parameter_names():
            assert param.grad is not None and param.grad.abs().sum()>0
        else:
            assert not param.requires_grad and param.grad is None


def test_prompt_padding_and_bos_do_not_receive_feedback():
    scorer=SoftThreePassLikelihood(tiny_model(),0)
    ids=torch.tensor([[0,2,3,4,5,1,1],[0,2,3,4,0,6,7]])
    targets=torch.tensor([[-1,-1,4,5,1,-1,-1],[-1,3,4,0,6,7,1]])
    assert scorer.feedback_mask(ids,[3,2],targets).tolist()==[
        [False,False,False,True,True,False,False],
        [False,False,True,True,False,True,True]]


def test_causal_and_prompt_logits_match_standard():
    model=tiny_model().eval();scorer=SoftThreePassLikelihood(model,0)
    ids=torch.tensor([[0,2,3,4,5,6,7,8]])
    with torch.no_grad():
        actual=scorer(ids,prompt_lengths=[3]);standard=model(ids)
        changed=ids.clone();changed[:,6:]=torch.tensor([[9,10]])
        later=scorer(changed,prompt_lengths=[3])
    torch.testing.assert_close(actual[:,:3],standard[:,:3])
    torch.testing.assert_close(actual[:,:6],later[:,:6])


def test_matches_first_three_recurrent_scores_but_is_not_exact_long_recurrence():
    from nanochat.engine import KVCache
    from nanochat.common import COMPUTE_DTYPE
    model=tiny_model().eval();scorer=SoftThreePassLikelihood(model,0)
    ids=torch.tensor([[0,2,3,4,5,6,7,8]])
    cache=KVCache(batch_size=1,num_heads=2,seq_len=16,head_dim=16,num_layers=2,
                  device=torch.device('cpu'),dtype=COMPUTE_DTYPE)
    with torch.no_grad():
        approximation=scorer(ids,prompt_lengths=[3])[:,2:]
        logits,hidden=model(ids[:,:3],kv_cache=cache,return_hidden=True)
        outputs=[logits[:,-1:]];previous=hidden[:,-1:]
        for position in range(3,ids.shape[1]):
            logits,previous=model(ids[:,position:position+1],kv_cache=cache,
                                  feedback_hidden=previous,return_hidden=True)
            outputs.append(logits)
        recurrent=torch.cat(outputs,dim=1)
    torch.testing.assert_close(approximation[:,:3],recurrent[:,:3],rtol=2e-5,atol=2e-6)
    assert (approximation[:,3:]-recurrent[:,3:]).abs().max()>1e-5


def _ddp_worker(rank,world,rendezvous,replay_mode=False):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from main import sync_context,token_loss
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'file://{rendezvous}',rank=rank,world_size=world)
    try:
        model=tiny_model();reference=deepcopy(model)
        from soft_likelihood import SoftReplayLikelihood
        cls = SoftReplayLikelihood if replay_mode else SoftThreePassLikelihood
        distributed=DDP(cls(model,0))
        full=cls(reference,0)
        ids=torch.tensor([[0,2,3,4,5],[0,3,4,5,6],[0,4,5,6,7],[0,5,6,7,8]])
        targets=torch.tensor([[-1,3,4,5,1],[-1,-1,5,6,1],[-1,5,6,7,1],[-1,-1,7,8,1]])
        lengths=[2,3,2,3];advantages=torch.tensor([1.,-1.,-1.,1.]);valid=targets.ne(-1)
        total=int(valid.sum())
        replay=torch.randn(4,5,32)
        extra={'replay_hidden':replay} if replay_mode else {}
        optim=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
        ref_optim=torch.optim.SGD([p for p in reference.parameters() if p.requires_grad],lr=.01)
        # Two iterations expose DDP unused-parameter/reduction problems.
        for _ in range(2):
            nll=full(ids,targets,prompt_lengths=lengths,loss_reduction='none',**extra).view_as(targets)
            token_loss(nll,advantages,valid,total).backward()
            for micro,i in enumerate(range(rank*2,rank*2+2)):
                with sync_context(distributed,micro,2):
                    local_extra={'replay_hidden':replay[i:i+1]} if replay_mode else {}
                    nll=distributed(ids[i:i+1],targets[i:i+1],prompt_lengths=lengths[i:i+1],loss_reduction='none',**local_extra).view(1,-1)
                    (world*token_loss(nll,advantages[i:i+1],valid[i:i+1],total)).backward()
            for (name,p),(other,q) in zip(model.named_parameters(),reference.named_parameters()):
                assert name==other
                if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=3e-6)
            optim.step();ref_optim.step();optim.zero_grad();ref_optim.zero_grad()
    finally:
        dist.destroy_process_group()


def test_distributed_three_pass_accumulation_matches_global_token_loss(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker,args=(2,str(tmp_path/'rendezvous')),nprocs=2,join=True)


def test_resume_rejects_silent_policy_changes():
    from main import validate_resume_config
    with pytest.raises(ValueError,match='decode_mode'):
        validate_resume_config({}, {'decode_mode':'soft'})
    with pytest.raises(ValueError,match='likelihood_estimator'):
        validate_resume_config({'decode_mode':'soft'}, {'decode_mode':'soft','likelihood_estimator':'three_pass_detached'})


def test_distributed_hidden_replay_matches_global_token_loss(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker,args=(2,str(tmp_path/'rendezvous'),True),nprocs=2,join=True)
