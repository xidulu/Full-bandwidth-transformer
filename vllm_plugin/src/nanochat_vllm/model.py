"""Inference-only Nanochat model for vLLM 0.14.0 standard/soft decoding.

Soft decoding uses an ordinary prompt pass followed by recurrent latent feedback
for generated-token inputs. Fused prompt decoding is intentionally unsupported.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import AutoWeightsLoader


TIED_EMBEDDING_SCALE = 800.0
LATENT_FEEDBACK_MODES = (
    "gate_product",
    "concat_projection",
    "linear_addition",
)


def _norm(x: torch.Tensor) -> torch.Tensor:
    # Match nanochat.gpt.norm, including PyTorch's dtype-dependent default eps.
    return F.rms_norm(x, (x.size(-1),))


def _has_value_embedding(layer_idx: int, num_layers: int) -> bool:
    return layer_idx % 2 == (num_layers - 1) % 2


def _short_window(sequence_len: int) -> int:
    # Keep this expression identical to GPT._compute_window_sizes.
    return -(-sequence_len // 4 // 128) * 128


def _apply_nanochat_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """Apply Nanochat's checkpoint-compatible negative-angle half-split RoPE."""
    num_tokens, num_heads, head_dim = x.shape
    half_dim = head_dim // 2
    angles = torch.outer(positions.float(), inv_freq.float())
    cos = angles.cos().to(dtype=x.dtype).view(num_tokens, 1, half_dim)
    sin = angles.sin().to(dtype=x.dtype).view(num_tokens, 1, half_dim)
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class NanochatAttention(nn.Module):
    def __init__(
        self,
        *,
        config,
        layer_idx: int,
        cache_config,
        prefix: str,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = hidden_size // self.num_heads
        if hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        # The first adapter is deliberately TP=1. Replicated projections retain
        # the checkpoint's names, which also makes weight synchronization simple.
        self.c_q = ReplicatedLinear(
            hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            prefix=f"{prefix}.c_q",
        )
        self.c_k = ReplicatedLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            prefix=f"{prefix}.c_k",
        )
        self.c_v = ReplicatedLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            prefix=f"{prefix}.c_v",
        )
        self.c_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.c_proj",
        )

        self.ve_gate_channels = 12
        self.ve_gate = (
            ReplicatedLinear(
                self.ve_gate_channels,
                self.num_kv_heads,
                bias=False,
                prefix=f"{prefix}.ve_gate",
            )
            if _has_value_embedding(layer_idx, config.num_hidden_layers)
            else None
        )

        channel_range = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (float(config.rope_theta) ** (channel_range / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        window_pattern = config.nanochat_window_pattern.upper()
        is_sliding = window_pattern[layer_idx % len(window_pattern)] == "S"
        if layer_idx == config.num_hidden_layers - 1:
            is_sliding = False
        # Nanochat passes ``(left_tokens, 0)`` directly to FlashAttention.
        # vLLM interprets its integer as the total window size and converts it
        # to ``(window - 1, 0)``, so add one to preserve Nanochat's boundary.
        per_layer_sliding_window = (
            _short_window(config.max_position_embeddings) + 1
            if is_sliding
            else None
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            per_layer_sliding_window=per_layer_sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        value_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        q, _ = self.c_q(x)
        k, _ = self.c_k(x)
        v, _ = self.c_v(x)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if value_embedding is not None:
            if self.ve_gate is None:
                raise AssertionError("value embedding passed to a layer without a gate")
            gate, _ = self.ve_gate(x[..., : self.ve_gate_channels])
            gate = 3 * torch.sigmoid(gate)
            value_embedding = value_embedding.view(
                -1, self.num_kv_heads, self.head_dim
            ).to(v.dtype)
            v = v + gate.unsqueeze(-1) * value_embedding

        q = _apply_nanochat_rope(q, positions, self.inv_freq)
        k = _apply_nanochat_rope(k, positions, self.inv_freq)
        q = 1.2 * _norm(q)
        k = 1.2 * _norm(k)

        y = self.attn(q.flatten(1), k.flatten(1), v.flatten(1))
        y, _ = self.c_proj(y)
        return y


class NanochatMLP(nn.Module):
    def __init__(self, *, hidden_size: int, prefix: str) -> None:
        super().__init__()
        self.c_fc = ReplicatedLinear(
            hidden_size,
            4 * hidden_size,
            bias=False,
            prefix=f"{prefix}.c_fc",
        )
        self.c_proj = ReplicatedLinear(
            4 * hidden_size,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.c_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.c_fc(x)
        x = F.relu(x).square()
        x, _ = self.c_proj(x)
        return x


class NanochatLatentFeedback(nn.Module):
    """Checkpoint-compatible inference form of ``nanochat.gpt.LatentFeedback``."""

    def __init__(self, *, hidden_size: int, mode: str, prefix: str) -> None:
        super().__init__()
        if mode not in LATENT_FEEDBACK_MODES:
            raise ValueError(
                f"latent feedback mode must be one of {LATENT_FEEDBACK_MODES}, "
                f"got {mode!r}"
            )
        self.mode = mode
        self.state_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.state_proj",
        )
        self.token_gate = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.token_gate",
        )
        self.concat_proj = ReplicatedLinear(
            2 * hidden_size,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.concat_proj",
        )
        self.token_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.token_proj",
        )

    def forward(
        self,
        previous_hidden: torch.Tensor,
        token_input: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "gate_product":
            value, _ = self.state_proj(previous_hidden)
            gate, _ = self.token_gate(_norm(token_input))
            return _norm(value * torch.sigmoid(gate))
        if self.mode == "concat_projection":
            fused, _ = self.concat_proj(
                torch.cat((_norm(previous_hidden), _norm(token_input)), dim=-1)
            )
            return _norm(fused)
        if self.mode == "linear_addition":
            state, _ = self.state_proj(previous_hidden)
            token, _ = self.token_proj(_norm(token_input))
            return _norm(state + token)
        raise AssertionError(f"unhandled latent feedback mode: {self.mode}")


class NanochatBlock(nn.Module):
    def __init__(
        self,
        *,
        config,
        layer_idx: int,
        cache_config,
        prefix: str,
    ) -> None:
        super().__init__()
        self.attn = NanochatAttention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = NanochatMLP(
            hidden_size=config.hidden_size,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        value_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attn(positions, _norm(x), value_embedding)
        x = x + self.mlp(_norm(x))
        return x


class NanochatForCausalLM(nn.Module):
    """vLLM text-generation model for Nanochat standard and soft decoding."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if prefix:
            raise ValueError("Nanochat vLLM adapter does not support pipeline nesting")
        if vllm_config.quant_config is not None:
            raise ValueError("Nanochat vLLM adapter does not yet support quantization")

        config = vllm_config.model_config.hf_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.decode_mode = config.nanochat_decode_mode
        if self.decode_mode not in ("standard", "soft"):
            raise ValueError(
                "Nanochat vLLM supports decode_mode='standard' or 'soft', "
                f"got {self.decode_mode!r}"
            )
        if self.decode_mode == "soft" and not config.nanochat_latent_feedback:
            raise ValueError("soft decoding requires a latent-feedback checkpoint")

        transformer = nn.ModuleDict()
        transformer["wte"] = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix="transformer.wte",
        )
        transformer["h"] = nn.ModuleList(
            [
                NanochatBlock(
                    config=config,
                    layer_idx=i,
                    cache_config=vllm_config.cache_config,
                    prefix=f"transformer.h.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.transformer = transformer

        self.resid_lambdas = nn.Parameter(
            torch.empty(config.num_hidden_layers, dtype=torch.float32)
        )
        self.x0_lambdas = nn.Parameter(
            torch.empty(config.num_hidden_layers, dtype=torch.float32)
        )
        self.smear_gate = ReplicatedLinear(
            24,
            1,
            bias=False,
            prefix="smear_gate",
        )
        self.smear_lambda = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.backout_lambda = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.latent_feedback = (
            NanochatLatentFeedback(
                hidden_size=config.hidden_size,
                mode=config.nanochat_latent_feedback_mode,
                prefix="latent_feedback",
            )
            if self.decode_mode == "soft"
            else None
        )

        kv_dim = config.num_key_value_heads * (
            config.hidden_size // config.num_attention_heads
        )
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.num_hidden_layers)
                if _has_value_embedding(i, config.num_hidden_layers)
            }
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix="lm_head",
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.transformer["wte"])
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
        )

    @property
    def embed_tokens(self) -> VocabParallelEmbedding:
        return self.transformer["wte"]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _scaled_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.embed_input_ids(input_ids)
        if self.config.tie_word_embeddings:
            token_embeddings = token_embeddings * TIED_EMBEDDING_SCALE
        return token_embeddings

    def prepare_standard_inputs(
        self,
        input_ids: torch.Tensor,
        previous_token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Build exact standard-mode layer-0 inputs for flattened vLLM tokens."""
        token_embeddings = self._scaled_token_embeddings(input_ids)
        previous_embeddings = self._scaled_token_embeddings(previous_token_ids)
        x = _norm(token_embeddings)
        previous_x = _norm(previous_embeddings)
        gate, _ = self.smear_gate(x[..., :24])
        smeared = x + self.smear_lambda.to(x.dtype) * torch.sigmoid(gate) * previous_x
        return torch.where(positions.ne(0).unsqueeze(-1), smeared, x)

    def prepare_decode_inputs(
        self,
        input_ids: torch.Tensor,
        previous_token_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        feedback_hidden: torch.Tensor | None = None,
        feedback_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ordinary = self.prepare_standard_inputs(
            input_ids,
            previous_token_ids,
            positions,
        )
        if feedback_mask is None:
            return ordinary
        if self.latent_feedback is None or feedback_hidden is None:
            raise ValueError("soft inputs require latent feedback and prior hidden states")
        token_embeddings = self._scaled_token_embeddings(input_ids)
        ordinary[feedback_mask] = self.latent_feedback(
            feedback_hidden[feedback_mask],
            token_embeddings[feedback_mask],
        )
        return ordinary

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("Nanochat vLLM adapter currently requires pipeline_parallel_size=1")
        if input_ids is None:
            raise ValueError("Nanochat requires token IDs for its per-layer value embeddings")
        if inputs_embeds is None:
            # Used by vLLM's profile/dummy runs. Real requests use the custom
            # runner, which supplies exact previous-token IDs.
            inputs_embeds = self.prepare_standard_inputs(input_ids, input_ids, positions)

        x = inputs_embeds
        x0 = x
        backout_layer = self.config.num_hidden_layers // 2
        x_backout = None
        for i, block in enumerate(self.transformer["h"]):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            value_embedding = (
                self.value_embeds[str(i)](input_ids)
                if str(i) in self.value_embeds
                else None
            )
            x = block(positions, x, value_embedding)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return _norm(x)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        # Native Nanochat promotes the projection result before applying the
        # soft cap. Keeping this explicit avoids bf16 tanh changing close
        # greedy decisions.
        logits = logits.float()
        return 15.0 * torch.tanh(logits / 15.0)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        skip_prefixes = []
        if self.latent_feedback is None:
            skip_prefixes.append("latent_feedback.")
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)


__all__ = ["NanochatForCausalLM"]
