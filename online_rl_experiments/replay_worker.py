"""Optional vLLM worker capturing soft hidden states without modifying vLLM."""
from nanochat_vllm.runner import NanochatGPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from hidden_replay import HiddenReplayBuffer


class ReplayGPUModelRunner(NanochatGPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_replay = HiddenReplayBuffer()

    def _model_forward(self, input_ids=None, positions=None, intermediate_tensors=None,
                       inputs_embeds=None, **model_kwargs):
        layout = self.nanochat_forward_layout
        hidden = super()._model_forward(input_ids=input_ids, positions=positions,
                                       intermediate_tensors=intermediate_tensors,
                                       inputs_embeds=inputs_embeds, **model_kwargs)
        if layout is not None and self.hidden_replay.rows is not None:
            lengths = {req: int(self.input_batch.num_prompt_tokens[i])
                       for i, req in enumerate(self.input_batch.req_ids)}
            self.hidden_replay.capture(layout, positions, hidden, lengths)
        return hidden


class ReplayWorker(GPUWorker):
    def init_device(self):
        if self.use_v2_model_runner:
            raise RuntimeError('Hidden replay requires the vLLM 0.14 V1 runner')
        import vllm.v1.worker.gpu_model_runner as module
        original = module.GPUModelRunner
        module.GPUModelRunner = ReplayGPUModelRunner
        try:
            super().init_device()
        finally:
            module.GPUModelRunner = original


def begin_replay(worker):
    if worker.model_runner.nanochat_decode_mode != 'soft':
        raise ValueError('Hidden replay requires soft decoding')
    worker.model_runner.hidden_replay.begin()


def finish_replay(worker, requests):
    return worker.model_runner.hidden_replay.finish(requests)
