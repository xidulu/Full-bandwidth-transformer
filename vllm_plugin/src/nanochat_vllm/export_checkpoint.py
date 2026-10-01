"""Export a native Nanochat checkpoint as a vLLM-loadable model directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_paths(checkpoint: Path) -> tuple[Path, Path]:
    if checkpoint.suffix != ".pt" or not checkpoint.name.startswith("model_"):
        raise ValueError("checkpoint must look like .../model_000123.pt")
    suffix = checkpoint.stem.removeprefix("model_")
    meta = checkpoint.with_name(f"meta_{suffix}.json")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not meta.is_file():
        raise FileNotFoundError(meta)
    return checkpoint, meta


def _patched_model_config(meta: dict[str, Any]) -> dict[str, Any]:
    config = dict(meta["model_config"])
    config.setdefault("window_pattern", "L")
    config.setdefault("latent_feedback", False)
    config.setdefault("latent_feedback_mode", "gate_product")
    config.setdefault("weight_tying", False)
    return config


def _layer_types(num_layers: int, pattern: str) -> list[str]:
    pattern = pattern.upper()
    if not pattern or any(char not in "SL" for char in pattern):
        raise ValueError(f"invalid window_pattern: {pattern!r}")
    result = [
        "sliding_attention" if pattern[i % len(pattern)] == "S" else "full_attention"
        for i in range(num_layers)
    ]
    result[-1] = "full_attention"
    return result


def _short_window(sequence_len: int) -> int:
    return -(-sequence_len // 4 // 128) * 128


def _hf_config(config: dict[str, Any], decode_mode: str) -> dict[str, Any]:
    hidden = int(config["n_embd"])
    heads = int(config["n_head"])
    layers = int(config["n_layer"])
    sequence_len = int(config["sequence_len"])
    pattern = str(config["window_pattern"])
    return {
        "architectures": ["NanochatForCausalLM"],
        # LlamaConfig accepts and preserves Nanochat's extra fields while
        # supplying the standard attributes vLLM expects.
        "model_type": "llama",
        "vocab_size": int(config["vocab_size"]),
        "hidden_size": hidden,
        "intermediate_size": 4 * hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": int(config["n_kv_head"]),
        "head_dim": hidden // heads,
        "max_position_embeddings": sequence_len,
        "rope_theta": 100000.0,
        "hidden_act": "relu2",
        "rms_norm_eps": None,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": bool(config["weight_tying"]),
        "torch_dtype": "bfloat16",
        # vLLM's integer window includes the current token; Nanochat's raw
        # FlashAttention tuple counts tokens to the left of it.
        "sliding_window": _short_window(sequence_len) + 1,
        "layer_types": _layer_types(layers, pattern),
        "nanochat_window_pattern": pattern,
        "nanochat_latent_feedback": bool(config["latent_feedback"]),
        "nanochat_latent_feedback_mode": str(config["latent_feedback_mode"]),
        "nanochat_decode_mode": decode_mode,
        "transformers_version": "4.57.3",
    }


def _crop_vocab_padding(
    state: dict[str, torch.Tensor],
    vocab_size: int,
) -> dict[str, torch.Tensor]:
    vocab_keys = {"transformer.wte.weight", "lm_head.weight"}
    vocab_keys.update(
        key
        for key in state
        if key.startswith("value_embeds.") and key.endswith(".weight")
    )
    result: dict[str, torch.Tensor] = {}
    for raw_name, tensor in state.items():
        name = raw_name.removeprefix("_orig_mod.")
        if name in vocab_keys:
            if tensor.ndim != 2 or tensor.size(0) < vocab_size:
                raise ValueError(
                    f"{name} has incompatible shape {tuple(tensor.shape)} for "
                    f"vocab_size={vocab_size}"
                )
            tensor = tensor[:vocab_size]
        result[name] = tensor.detach().cpu().contiguous().clone()
    return result


def export_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    decode_mode: str = "standard",
    weight_format: str = "safetensors",
    force: bool = False,
) -> Path:
    checkpoint, meta_path = _checkpoint_paths(checkpoint.resolve())
    if decode_mode not in ("standard", "soft"):
        raise ValueError(f"unknown decode mode: {decode_mode}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_filename = (
        "model.safetensors"
        if weight_format == "safetensors"
        else "pytorch_model.bin"
    )
    managed_files = [
        output_dir / model_filename,
        output_dir / "config.json",
        output_dir / "nanochat_meta.json",
        output_dir / "nanochat_export.json",
    ]
    existing = [path for path in managed_files if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite {names}; pass --force")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    model_config = _patched_model_config(meta)
    if decode_mode == "soft" and not model_config["latent_feedback"]:
        raise ValueError("soft decoding requires a latent-feedback checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError("checkpoint is not a tensor state_dict")
    state = _crop_vocab_padding(state, int(model_config["vocab_size"]))
    if decode_mode == "soft":
        required_feedback = {
            "latent_feedback.state_proj.weight",
            "latent_feedback.token_gate.weight",
            "latent_feedback.concat_proj.weight",
            "latent_feedback.token_proj.weight",
        }
        missing_feedback = sorted(required_feedback - state.keys())
        if missing_feedback:
            raise ValueError(
                "soft checkpoint is missing latent-feedback weights: "
                + ", ".join(missing_feedback)
            )

    model_path = output_dir / model_filename
    if weight_format == "safetensors":
        from safetensors.torch import save_file

        save_file(state, model_path)
    elif weight_format == "pytorch":
        torch.save(state, model_path)
    else:
        raise ValueError(f"unknown weight format: {weight_format}")

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _hf_config(model_config, decode_mode),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    with (output_dir / "nanochat_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    export_meta = {
        "adapter": "nanochat-vllm",
        "adapter_version": "0.1.0",
        "decode_mode": decode_mode,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_meta": str(meta_path),
        "source_meta_sha256": _sha256(meta_path),
        "weight_file": model_filename,
        "weight_format": weight_format,
        "vllm_version": "0.14.0",
    }
    with (output_dir / "nanochat_export.json").open("w", encoding="utf-8") as handle:
        json.dump(export_meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decode-mode",
        choices=("standard", "soft"),
        default="standard",
    )
    parser.add_argument(
        "--weight-format",
        choices=("safetensors", "pytorch"),
        default="safetensors",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = export_checkpoint(
        args.checkpoint,
        args.output_dir,
        decode_mode=args.decode_mode,
        weight_format=args.weight_format,
        force=args.force,
    )
    print(output)


if __name__ == "__main__":
    main()
