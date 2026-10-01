"""Real-checkpoint soft rollout/score/backward and weight-transfer acceptance test.

Synthetic advantages test optimizer plumbing; this is not an accuracy evaluation.
"""
import argparse
import json
import os
from pathlib import Path

import torch

from check_train_inference import mismatch
from main import pack, token_loss
from prepare_data import sha256
from soft_likelihood import SoftThreePassLikelihood, SoftReplayLikelihood, configure_feedback_gradients
from vllm_rollout import VLLMRollout, prepare_model_config

TAG = 'd20-from40k-lf-k2-gate_product-openmath-train5m-k3-anygpu'


def recurrent_scores(model, prompt, suffix, return_hidden=False):
    from nanochat.engine import KVCache
    from nanochat.common import COMPUTE_DTYPE
    c = model.config
    cache = KVCache(1, c.n_kv_head, len(prompt)+len(suffix), c.n_embd//c.n_head,
                    c.n_layer, model.get_device(), COMPUTE_DTYPE)
    with torch.no_grad():
        logits, hidden = model(torch.tensor([prompt], device='cuda'), kv_cache=cache, return_hidden=True)
        result, states = [], []
        previous = hidden[:, -1:]
        for i, token in enumerate(suffix):
            states.append(previous[0,0].cpu().clone())
            result.append(logits[0, -1].log_softmax(-1)[token].item())
            if i+1 < len(suffix):
                logits, previous = model(torch.tensor([[token]], device='cuda'), kv_cache=cache,
                                         feedback_hidden=previous, return_hidden=True)
    return (result, torch.stack(states)) if return_hidden else result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hidden-replay', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from nanochat.checkpoint_manager import build_model
    from nanochat.common import COMPUTE_DTYPE
    directory = Path(os.environ['NANOCHAT_BASE_DIR'])/'chatsft_checkpoints'/TAG
    model, tokenizer, _ = build_model(str(directory), 4407, torch.device('cuda'), 'train')
    model.float().eval(); model.tie_weights()
    model.cos, model.sin = model.cos.to(COMPUTE_DTYPE), model.sin.to(COMPUTE_DTYPE)
    configure_feedback_gradients(model, 'soft')
    scorer_type = SoftReplayLikelihood if args.hidden_replay else SoftThreePassLikelihood
    scorer = scorer_type(model, tokenizer.get_bos_token_id())
    prompt = tokenizer.render_for_completion({'messages': [
        {'role':'user','content':'Compute 17 times 23. Explain your reasoning and put the answer in \\boxed{}.'},
        {'role':'assistant','content':''}]})
    prepare_model_config(vars(model.config), args.output/'model', 'soft')
    engine = VLLMRollout(model, tokenizer, args.output/'model', 0, 2,
                        verify_weights=True, decode_mode='soft', hidden_replay=args.hidden_replay,
                        max_batched_tokens=64 if args.hidden_replay else 2048)
    try:
        engine.sync_weights(model, 0)
        suffixes, ended, scores = engine.generate_groups([prompt], [42], 2, 64, 0, return_scores=True, return_hidden_states=args.hidden_replay)[0]
        samples = [dict(prompt=prompt, suffix=s, advantage=a) for s,a in zip(suffixes, [1.,-1.])]
        if args.hidden_replay:
            for sample, score in zip(samples, scores):
                sample['rollout_hidden_states'] = score['hidden_states']
        references = [recurrent_scores(model, prompt, s, return_hidden=True) for s in suffixes]
        cached = [row[0] for row in references]
        hidden_max_error = (max((score['hidden_states']-reference[1]).abs().max().item()
                               for score, reference in zip(scores, references)) if args.hidden_replay else None)
        ids, targets, adv = pack(samples, tokenizer.encode_special('<|assistant_end|>'), model.get_device())
        model.train()
        calls = []
        original = model._run_trunk
        def trunk(*a):
            calls.append(torch.is_grad_enabled())
            return original(*a)
        model._run_trunk = trunk
        kwargs = {}
        if args.hidden_replay:
            from hidden_replay import pack_replay
            kwargs['replay_hidden'] = pack_replay(samples, ids, model.config.n_embd, COMPUTE_DTYPE)
        nll = scorer(ids, targets, prompt_lengths=[len(prompt)]*2, loss_reduction='none', **kwargs).view_as(targets)
        model._run_trunk = original
        assert calls == ([True] if args.hidden_replay else [False, False, True]), calls
        valid = targets.ne(-1)
        train = [(-nll[i].detach()[valid[i]]).tolist() for i in range(2)]
        with torch.no_grad():
            control = SoftThreePassLikelihood(model, tokenizer.get_bos_token_id())(
                ids, targets, prompt_lengths=[len(prompt)]*2, loss_reduction='none').view_as(targets)
            control_logp = [(-control[i][valid[i]]).tolist() for i in range(2)]
        loss = token_loss(nll, adv, valid, int(valid.sum()))
        loss.backward()
        active = model.latent_feedback.active_parameter_names()
        feedback_grad = {name: p.grad.float().norm().item() for name,p in model.latent_feedback.named_parameters() if name in active}
        assert all(value > 0 for value in feedback_grad.values())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
        before = {n:p.detach().clone() for n,p in model.latent_feedback.named_parameters() if n in active}
        # Disposable optimizer step: no production checkpoint is changed.
        torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3).step()
        changed = {n:bool(torch.any(p != before[n])) for n,p in model.latent_feedback.named_parameters() if n in active}
        assert all(changed.values())
        model.zero_grad(set_to_none=True)
        engine.sync_weights(model, 1)  # exact audit of every adapter tensor, including LF
        engine.generate_groups([prompt, prompt+prompt[1:]], [43, 44], 1, 8, 1,
                               return_hidden_states=args.hidden_replay)
        flat = lambda rows: [x for row in rows for x in row]
        rollout = [s['logprobs'] for s in scores]
        report = dict(checkpoint=str(directory/'model_004407.pt'),
                      checkpoint_sha256=sha256(directory/'model_004407.pt'),
                      metadata_sha256=sha256(directory/'meta_004407.json'),
                      source_sha256={p:sha256(Path(p)) for p in ('soft_likelihood.py','vllm_rollout.py','validate_soft.py','hidden_replay.py','replay_worker.py')},
                      gpu=torch.cuda.get_device_name(), tokens=int(valid.sum()),
                      pass_grad_enabled=calls, feedback_grad_norm=feedback_grad,
                      feedback_updated=changed, total_grad_norm=norm,
                      changed_weights_verified=True, synthetic_advantages=True,
                      rollout_hidden_vs_native_max_abs=hidden_max_error,
                      three_pass_control_vs_vllm=mismatch(flat(control_logp),flat(rollout)),
                      likelihood_estimator='hidden_state_replay' if args.hidden_replay else 'three_pass_detached',
                      score_vs_vllm=mismatch(flat(train),flat(rollout)),
                      recurrent_native_vs_vllm=mismatch(flat(cached),flat(rollout)),
                      first_three_scores_vs_native=mismatch(flat([s[:3] for s in train]),flat([s[:3] for s in cached])))
        (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        if args.hidden_replay:
            torch.save([sample.pop('rollout_hidden_states') for sample in samples], args.output/'rollout_hidden_states.pt')
        (args.output/'tokens.json').write_text(json.dumps(dict(samples=samples,ended=ended,train=train,cached=cached,rollout=rollout))+'\n')
        print(json.dumps(report), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
