"""Worker selection shim for the Nanochat-aware vLLM 0.14 GPU runner."""

from __future__ import annotations

from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class NanochatWorker(GPUWorker):
    """Install the standard/soft Nanochat runner without modifying vLLM."""

    def init_device(self) -> None:
        if self.use_v2_model_runner:
            raise RuntimeError(
                "Nanochat targets vLLM 0.14's default model runner; "
                "unset VLLM_USE_V2_MODEL_RUNNER"
            )

        # GPUWorker imports GPUModelRunner inside init_device. Replace that
        # module attribute only for the duration of construction so other vLLM
        # users in this process are unaffected.
        import vllm.v1.worker.gpu_model_runner as runner_module

        from .runner import NanochatGPUModelRunner

        original = runner_module.GPUModelRunner
        runner_module.GPUModelRunner = NanochatGPUModelRunner
        try:
            super().init_device()
        finally:
            runner_module.GPUModelRunner = original


__all__ = ["NanochatWorker"]
