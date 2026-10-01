from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from nanochat_vllm.runner import NanochatGPUModelRunner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _Buffer:
    def __init__(self, size):
        self.np = np.zeros(size, dtype=np.int32)
        self.gpu = torch.zeros(size, dtype=torch.int32)

    def copy_to_gpu(self, count):
        self.gpu[:count].copy_(torch.from_numpy(self.np[:count]))


def test_previous_token_ids_cover_prefill_decode_and_chunked_prefill():
    runner = NanochatGPUModelRunner.__new__(NanochatGPUModelRunner)
    runner.nanochat_prev_input_ids = _Buffer(8)
    runner.input_batch = SimpleNamespace(
        req_ids=["prefill", "decode", "chunk"],
        token_ids_cpu=np.array(
            [
                [10, 11, 12, 0, 0, 0],
                [20, 21, 22, 23, 0, 0],
                [30, 31, 32, 33, 34, 35],
            ],
            dtype=np.int32,
        ),
        num_computed_tokens_cpu=np.array([0, 3, 2], dtype=np.int32),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"prefill": 3, "decode": 1, "chunk": 2}
    )

    previous = runner._prepare_previous_token_ids(scheduler_output, 6)

    assert previous.tolist() == [10, 10, 11, 22, 31, 32]


def test_soft_runner_retains_last_hidden_per_request():
    runner = NanochatGPUModelRunner.__new__(NanochatGPUModelRunner)
    runner.nanochat_decode_mode = "soft"
    runner.nanochat_previous_hidden = {}
    runner.nanochat_forward_layout = [("request-a", 1), ("request-b", 4)]
    hidden_states = torch.arange(15, dtype=torch.float32).view(5, 3)

    with patch.object(GPUModelRunner, "_model_forward", return_value=hidden_states):
        returned = runner._model_forward()

    assert returned is hidden_states
    assert runner.nanochat_forward_layout is None
    assert torch.equal(runner.nanochat_previous_hidden["request-a"], hidden_states[1])
    assert torch.equal(runner.nanochat_previous_hidden["request-b"], hidden_states[4])
