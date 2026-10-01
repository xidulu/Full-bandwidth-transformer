"""Nanochat-aware vLLM runner for standard and soft decoding."""

from __future__ import annotations

from typing import Any

import torch

from vllm.distributed import get_pp_group
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class NanochatGPUModelRunner(GPUModelRunner):
    """vLLM 0.14 runner for Nanochat standard and soft decoding.

    vLLM's persistent input batch retains the complete token history for every
    active request. We gather the token immediately preceding every scheduled
    token to construct Nanochat's exact smeared standard inputs. In soft mode,
    the runner also retains the last normalized model hidden state per request
    and uses it to fuse generated-token inputs after ordinary prompt prefill.
    """

    def __init__(self, vllm_config, device: torch.device) -> None:
        parallel = vllm_config.parallel_config
        unsupported = []
        if parallel.tensor_parallel_size != 1:
            unsupported.append("tensor_parallel_size != 1")
        if parallel.pipeline_parallel_size != 1:
            unsupported.append("pipeline_parallel_size != 1")
        if parallel.data_parallel_size != 1:
            unsupported.append("data_parallel_size != 1")
        if vllm_config.speculative_config is not None:
            unsupported.append("speculative decoding")
        if vllm_config.model_config.enable_prompt_embeds:
            unsupported.append("prompt embeddings")
        if vllm_config.scheduler_config.async_scheduling:
            unsupported.append("async scheduling")
        if not vllm_config.model_config.enforce_eager:
            unsupported.append("CUDA graphs (pass enforce_eager=True)")
        decode_mode = vllm_config.model_config.hf_config.nanochat_decode_mode
        if decode_mode not in ("standard", "soft"):
            unsupported.append(f"decode_mode={decode_mode!r}")
        if decode_mode == "soft" and vllm_config.cache_config.enable_prefix_caching:
            unsupported.append(
                "prefix caching in soft mode (pass enable_prefix_caching=False)"
            )
        if decode_mode == "soft" and parallel.enable_dbo:
            unsupported.append("dual-batch overlap in soft mode")
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(f"Nanochat vLLM adapter does not support: {joined}")

        super().__init__(vllm_config, device)
        self.nanochat_prev_input_ids = self._make_buffer(
            self.max_num_tokens,
            dtype=torch.int32,
        )
        self.nanochat_decode_mode = decode_mode
        self.nanochat_previous_hidden: dict[str, torch.Tensor] = {}
        self.nanochat_forward_layout: list[tuple[str, int]] | None = None
        if decode_mode == "soft":
            self.nanochat_feedback_hidden = self._make_buffer(
                self.max_num_tokens,
                vllm_config.model_config.get_hidden_size(),
                dtype=self.dtype,
                numpy=False,
            )
            self.nanochat_feedback_mask = self._make_buffer(
                self.max_num_tokens,
                dtype=torch.bool,
            )
        else:
            self.nanochat_feedback_hidden = None
            self.nanochat_feedback_mask = None

    def _prepare_previous_token_ids(
        self,
        scheduler_output,
        num_scheduled_tokens: int,
    ) -> torch.Tensor:
        prev = self.nanochat_prev_input_ids.np
        cursor = 0
        history = self.input_batch.token_ids_cpu
        computed = self.input_batch.num_computed_tokens_cpu

        for req_index, req_id in enumerate(self.input_batch.req_ids):
            count = int(scheduler_output.num_scheduled_tokens[req_id])
            if count == 0:
                continue
            start_pos = int(computed[req_index])
            end = cursor + count
            if start_pos == 0:
                prev[cursor] = history[req_index, 0]
                if count > 1:
                    prev[cursor + 1 : end] = history[
                        req_index, 0 : count - 1
                    ]
            else:
                prev[cursor:end] = history[
                    req_index, start_pos - 1 : start_pos + count - 1
                ]
            cursor = end

        if cursor != num_scheduled_tokens:
            raise AssertionError(
                f"prepared {cursor} previous IDs for {num_scheduled_tokens} tokens"
            )
        self.nanochat_prev_input_ids.copy_to_gpu(num_scheduled_tokens)
        return self.nanochat_prev_input_ids.gpu[:num_scheduled_tokens]

    def _preprocess(
        self,
        scheduler_output,
        num_input_tokens: int,
        intermediate_tensors=None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        Any,
        dict[str, Any],
        Any,
    ]:
        result = super()._preprocess(
            scheduler_output,
            num_input_tokens,
            intermediate_tensors,
        )
        (
            input_ids,
            _inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        ) = result

        if not get_pp_group().is_first_rank:
            return result
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if num_input_tokens != num_scheduled_tokens:
            raise AssertionError(
                "Nanochat requires unpadded eager batches; "
                f"got {num_scheduled_tokens} scheduled and {num_input_tokens} input tokens"
            )

        current_ids = self.input_ids.gpu[:num_scheduled_tokens]
        previous_ids = self._prepare_previous_token_ids(
            scheduler_output,
            num_scheduled_tokens,
        )
        feedback_hidden = None
        feedback_mask = None
        forward_layout: list[tuple[str, int]] = []
        cursor = 0
        active_req_ids = set(self.input_batch.req_ids)
        for stale_req_id in self.nanochat_previous_hidden.keys() - active_req_ids:
            del self.nanochat_previous_hidden[stale_req_id]

        for req_index, req_id in enumerate(self.input_batch.req_ids):
            count = int(scheduler_output.num_scheduled_tokens[req_id])
            if count == 0:
                continue
            start_pos = int(self.input_batch.num_computed_tokens_cpu[req_index])
            if start_pos == 0:
                self.nanochat_previous_hidden.pop(req_id, None)
            end = cursor + count
            forward_layout.append((req_id, end - 1))

            if self.nanochat_decode_mode == "soft":
                prompt_len = int(self.input_batch.num_prompt_tokens[req_index])
                if start_pos < prompt_len < start_pos + count:
                    raise RuntimeError(
                        "a soft batch cannot cross directly from prompt prefill "
                        "into generated tokens"
                    )
                if start_pos >= prompt_len:
                    if count != 1:
                        raise RuntimeError(
                            "soft decoding requires one generated token per request "
                            "per engine step"
                        )
                    previous_hidden = self.nanochat_previous_hidden.get(req_id)
                    if previous_hidden is None:
                        raise RuntimeError(
                            f"missing prior hidden state for soft request {req_id!r}; "
                            "disable prefix caching and speculative decoding"
                        )
                    assert self.nanochat_feedback_hidden is not None
                    assert self.nanochat_feedback_mask is not None
                    self.nanochat_feedback_hidden.gpu[cursor:end].copy_(
                        previous_hidden.unsqueeze(0)
                    )
                    self.nanochat_feedback_mask.np[cursor:end] = True
            cursor = end

        if self.nanochat_decode_mode == "soft":
            assert self.nanochat_feedback_mask is not None
            has_feedback = bool(
                self.nanochat_feedback_mask.np[:num_scheduled_tokens].any()
            )
            if has_feedback:
                self.nanochat_feedback_mask.copy_to_gpu(num_scheduled_tokens)
                assert self.nanochat_feedback_hidden is not None
                feedback_hidden = self.nanochat_feedback_hidden.gpu[
                    :num_scheduled_tokens
                ]
                feedback_mask = self.nanochat_feedback_mask.gpu[:num_scheduled_tokens]
            self.nanochat_feedback_mask.np[:num_scheduled_tokens] = False

        prepared = self.model.prepare_decode_inputs(
            current_ids,
            previous_ids,
            positions[:num_scheduled_tokens],
            feedback_hidden=feedback_hidden,
            feedback_mask=feedback_mask,
        )
        self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(prepared)
        self.nanochat_forward_layout = forward_layout

        # Keep input_ids as well: Nanochat uses them for value embeddings in
        # every other transformer layer.
        return (
            current_ids,
            self.inputs_embeds.gpu[:num_scheduled_tokens],
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs,
    ) -> Any:
        layout = self.nanochat_forward_layout
        self.nanochat_forward_layout = None
        hidden_states = super()._model_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        if self.nanochat_decode_mode == "soft" and layout is not None:
            if not isinstance(hidden_states, torch.Tensor):
                raise TypeError("soft decoding expected tensor hidden states")
            for req_id, last_index in layout:
                self.nanochat_previous_hidden[req_id] = (
                    hidden_states[last_index].detach().clone()
                )
        return hidden_states


__all__ = ["NanochatGPUModelRunner"]
