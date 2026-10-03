"""Check cache alignment and parallel replay against independent model paths."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'online_rl_experiments'))
from soft_likelihood import SoftThreePassLikelihood
from replay_experiment import capture_rollout, make_cache, parallel_pass, tensor_metrics
from nanochat.gpt import GPT, GPTConfig


def tiny_model(mode):
    from nanochat import flash_attention
    flash_attention.USE_FA3 = False
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = GPT(GPTConfig(sequence_len=32, vocab_size=32, n_layer=2,
                         n_head=2, n_kv_head=2, n_embd=32, window_pattern='L',
                         latent_feedback=True, latent_feedback_mode=mode), pad_vocab_size_to=1)
    model.init_weights()
    # init_weights zeros output projections; exercise nontrivial attention.
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(0, 0.05)
            block.mlp.c_proj.weight.normal_(0, 0.05)
    return model.eval()


@pytest.mark.parametrize('mode', ['gate_product', 'concat_projection', 'linear_addition'])
@torch.inference_mode()
def test_oracle_and_causal_convergence_and_training_equivalence(mode):
    model = tiny_model(mode)
    ids = torch.tensor([[0, 2, 3, 4, 5, 6, 7, 8]])
    prompt_length = 3
    truth = make_cache(model, ids.shape[1])
    _, hidden = model(ids[:, :prompt_length], kv_cache=truth, return_hidden=True)
    states = [hidden]
    for pos in range(prompt_length, ids.shape[1]):
        _, hidden = model(ids[:, pos:pos+1], kv_cache=truth,
                          feedback_hidden=hidden[:, -1:], return_hidden=True)
        states.append(hidden)
    recurrent = torch.cat(states, 1)
    oracle, oracle_hidden = parallel_pass(model, ids, prompt_length, recurrent)
    torch.testing.assert_close(oracle_hidden, recurrent, atol=3e-6, rtol=3e-5)
    for field in ('k_cache', 'v_cache'):
        torch.testing.assert_close(getattr(oracle, field), getattr(truth, field), atol=3e-6, rtol=3e-5)
    previous = None
    for number in range(1, ids.shape[1]-prompt_length+2):
        cache, previous = parallel_pass(model, ids, prompt_length, previous)
        # Each additional pass recovers one more recurrent input position.
        end = min(prompt_length + number - 1, ids.shape[1])
        torch.testing.assert_close(previous[:, :end], recurrent[:, :end], atol=4e-6, rtol=4e-5)
        if number == 3:
            logits = model._project_and_loss(previous, None, 'mean')
            expected = SoftThreePassLikelihood(model, 0)(ids, prompt_lengths=[prompt_length])
            torch.testing.assert_close(logits, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(cache.k_cache, truth.k_cache, atol=4e-6, rtol=4e-5)
    torch.testing.assert_close(cache.v_cache, truth.v_cache, atol=4e-6, rtol=4e-5)


class Tokenizer:
    def get_bos_token_id(self):
        return 0

    def encode_special(self, token):
        assert token == '<|assistant_end|>'
        return 1


@torch.inference_mode()
def test_capture_excludes_last_sampled_token():
    model = tiny_model('gate_product')
    # Prevent early EOS; native Engine still chooses all tokens.
    with torch.no_grad():
        model.lm_head.weight[0:2].zero_()
    generated, cache, hidden, logits = capture_rollout(model, Tokenizer(), [0, 2, 3], 5, 42)
    assert cache.get_pos() == 3 + len(generated) - 1
    assert hidden.shape[1] == cache.get_pos()
    assert logits.shape[0] == len(generated)
    assert logits.argmax(-1).tolist() == generated
    ids = torch.tensor([[0, 2, 3] + generated[:-1]])
    replay, _ = parallel_pass(model, ids, 3, hidden)
    length = ids.shape[1]
    torch.testing.assert_close(replay.k_cache, cache.k_cache[:, :, :length], atol=3e-6, rtol=3e-5)


def test_metrics_known_scale():
    truth = torch.tensor([1., 2., 3.])
    metrics = tensor_metrics(2 * truth, truth)
    assert metrics['relative_l2'] == 1.0
    assert metrics['cosine'] == 1.0
