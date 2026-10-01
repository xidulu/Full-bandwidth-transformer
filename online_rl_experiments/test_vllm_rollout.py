from types import SimpleNamespace

import pytest
import torch

from main import validate_resume_config
from vllm_rollout import VLLMRollout, inference_weights, unpack_outputs, visible_device


def request(tokens, reason='stop'):
    return SimpleNamespace(finished=True, outputs=[SimpleNamespace(token_ids=tokens, finish_reason=reason)])


def test_terminal_tokens_and_length_limits_match_native():
    result = unpack_outputs([request([5, 9]), request([6, 7], 'length'), request([4, 0], 'length')],
                            3, [0, 9], 2)
    assert result == ([[5, 9], [6, 7], [4, 0]], [True, False, True])


@pytest.mark.parametrize('output', [request([1], 'stop'), request([9, 1]), request([], 'length'),
                                     request([1, 2, 3], 'length'), request([1], 'abort')])
def test_invalid_generations_fail_closed(output):
    with pytest.raises(RuntimeError):
        unpack_outputs([output], 1, [0, 9], 2)


def test_visible_device_remaps_torchrun_rank():
    assert visible_device(1, '3,5,6,7') == '5'
    assert visible_device(0, 'GPU-uuid') == 'GPU-uuid'


def test_standard_weights_crop_padding_and_preserve_tied_config():
    class Model:
        config = SimpleNamespace(vocab_size=3, weight_tying=True)
        def named_parameters(self):
            yield 'transformer.wte.weight', torch.ones(4, 2)
            yield 'lm_head.weight', torch.ones(4, 2)
            yield 'value_embeds.1.weight', torch.ones(4, 1)
            yield 'latent_feedback.state_proj.weight', torch.ones(2, 2)
            yield 'resid_lambdas', torch.ones(2)
    state = dict(inference_weights(Model()))
    assert set(state) == {'transformer.wte.weight', 'value_embeds.1.weight', 'resid_lambdas'}
    assert state['transformer.wte.weight'].shape == (3, 2)
    assert state['value_embeds.1.weight'].shape == (3, 1)
    soft_state = dict(inference_weights(Model(), 'soft'))
    assert 'latent_feedback.state_proj.weight' in soft_state
    assert set(soft_state) == set(state) | {'latent_feedback.state_proj.weight'}


def test_soft_concurrency_reserves_full_context_kv():
    from vllm_rollout import soft_sequence_capacity
    config = SimpleNamespace(n_layer=20,n_kv_head=10,n_embd=1280,n_head=10,sequence_len=2048)
    capacity = soft_sequence_capacity(config,8)
    assert 32 <= capacity < 40
    assert capacity*2048*20*2*1280*2 < 8*1024**3
    with pytest.raises(ValueError,match='KV budget'):
        soft_sequence_capacity(config,.01)


def test_stale_policy_cannot_generate():
    engine = object.__new__(VLLMRollout)
    engine.policy_version = 3
    with pytest.raises(RuntimeError, match='stale'):
        engine.generate_groups([[1]], [42], 8, 1024, policy_version=4)


def test_backend_change_requires_explicit_resume_branch():
    with pytest.raises(ValueError, match='engine changed'):
        validate_resume_config({}, {'rollout_engine': 'vllm'})
    validate_resume_config({}, {'rollout_engine': 'vllm', 'allow_rollout_engine_change': True})


def test_grouped_generation_preserves_prompt_and_seed_order():
    class Connection:
        def send(self, message):
            self.message = message
    engine = object.__new__(VLLMRollout)
    engine.policy_version = 7
    engine.connection = Connection()
    engine._receive = lambda: ([[1, 9], [2, 9], [3, 9], [4]], [True, True, True, False])
    groups = engine.generate_groups([[1, 2], [3, 4, 5]], [100, 200], 2, 10, 7)
    command, payload = engine.connection.message
    assert command == 'generate'
    assert payload['prompts'] == [[1, 2], [1, 2], [3, 4, 5], [3, 4, 5]]
    assert payload['seeds'] == [100, 101, 200, 201]
    assert groups == [([[1, 9], [2, 9]], [True, True]), ([[3, 9], [4]], [True, False])]


def test_scored_generation_preserves_token_alignment():
    class Connection:
        def send(self, message):
            self.message = message
    engine = object.__new__(VLLMRollout)
    engine.policy_version = 0
    engine.connection = Connection()
    scores = [{'logprobs': [-.3, -.4], 'top1': [1, 9]}]
    engine._receive = lambda: ([[1, 9]], [True], scores)
    assert engine.generate_groups([[2]], [42], 1, 2, 0, return_scores=True) == [([[1, 9]], [True], scores)]
    assert engine.connection.message[0] == 'generate_scored'


def test_mismatch_statistics_identity_and_known_ratio():
    from check_train_inference import mismatch
    same = mismatch([-.2, -1.], [-.2, -1.])
    assert same['k3_mean'] == 0
    assert same['token_probability_ratio']['mean'] == 1
    assert same['fraction_outside_08_12'] == 0
    difference = mismatch([-1., -1.], [-2., -2.])
    assert difference['log_ratio']['mean'] == 1
    assert difference['fraction_outside_08_12'] == 1
    assert difference['k3_mean'] == pytest.approx(2.718281828 - 2)


def test_hidden_replay_requires_opt_in_and_implicitly_returns_scores():
    engine=object.__new__(VLLMRollout)
    engine.policy_version=0
    with pytest.raises(ValueError,match='Enable hidden_replay'):
        engine.generate_groups([[2]],[42],1,2,0,return_hidden_states=True)
    class Connection:
        def send(self,message):self.message=message
    engine.connection=Connection();engine.hidden_replay=True
    scores=[dict(logprobs=[-.2],hidden_states=torch.ones(1,4))]
    engine._receive=lambda:([[9]],[True],scores)
    groups=engine.generate_groups([[2]],[42],1,2,0,return_hidden_states=True)
    assert groups[0][2] is not None
    assert engine.connection.message[0]=='generate_scored'
    assert engine.connection.message[1]['return_hidden_states'] is True
