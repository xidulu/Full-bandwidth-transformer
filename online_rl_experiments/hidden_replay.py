"""CPU storage for per-request recurrent states, indexed by absolute position."""
from contextlib import contextmanager

import torch


@contextmanager
def capture_request_ids(input_processor):
    """Capture vLLM's actual external/internal mapping without parsing UUIDs.

The rollout worker is synchronous and isolated; restore the instance attribute
even if input processing or generation fails.
    """
    mapping = {}
    original = input_processor.assign_request_id
    had_override = 'assign_request_id' in vars(input_processor)
    def assign(request):
        original(request)
        if request.external_req_id in mapping:
            raise RuntimeError('Duplicate external request ID in rollout batch')
        mapping[request.external_req_id] = request.request_id
    input_processor.assign_request_id = assign
    try:
        yield mapping
    finally:
        if had_override:
            input_processor.assign_request_id = original
        else:
            del input_processor.assign_request_id


class HiddenReplayBuffer:
    def __init__(self):
        self.rows = None

    def begin(self):
        if self.rows is not None:
            raise RuntimeError('Hidden replay capture already active')
        self.rows = {}

    def capture(self, layout, positions, hidden, prompt_lengths):
        if self.rows is None:
            return
        # A chunked prompt only contributes its final position. Decode contributes
        # the state used to predict each next token, including a terminal token.
        if not layout:
            return
        offsets = positions[[index for _, index in layout]].to(device='cpu').tolist()
        selected = [(req, index, position) for (req, index), position in zip(layout, offsets)
                    if position >= prompt_lengths[req] - 1]
        if not selected:
            return
        indices = [index for _, index, _ in selected]
        states = hidden[indices].detach().to(device='cpu').clone()
        for (req, _, position), state in zip(selected, states):
            history = self.rows.setdefault(req, [])
            expected = history[-1][0] + 1 if history else prompt_lengths[req] - 1
            if position != expected:
                raise RuntimeError(f'Non-contiguous replay for {req}: {position} != {expected}')
            history.append((position, state))

    def finish(self, requests):
        if self.rows is None:
            raise RuntimeError('Hidden replay capture is not active')
        try:
            result = []
            for req, prompt_length, count in requests:
                history = self.rows.get(req, [])
                if len(history) != count or not history or history[0][0] != prompt_length - 1:
                    raise RuntimeError(f'Hidden replay length/position mismatch for {req}')
                result.append(torch.stack([state for _, state in history]))
            if set(self.rows) != {req for req, _, _ in requests}:
                raise RuntimeError('Unexpected requests in hidden replay buffer')
            return result
        finally:
            self.rows = None


def pack_replay(samples, inputs, hidden_size, dtype):
    """Align predictor states [response length,H] to generated INPUT positions.

State j predicted response token j; it is the feedback input when processing
that token to predict j+1. The final predictor state is returned for completeness
but is not needed by the teacher-forced forward.
    """
    replay = torch.zeros((*inputs.shape, hidden_size), dtype=dtype, device=inputs.device)
    for i, sample in enumerate(samples):
        states = sample['rollout_hidden_states']
        count = len(sample['suffix'])
        if states.shape != (count, hidden_size):
            raise ValueError('Replay states must align one-to-one with response tokens')
        start = len(sample['prompt'])
        replay[i, start:start + count - 1] = states[:-1].detach().to(inputs.device, dtype=dtype)
    return replay
