import json

import torch

from nanochat_vllm.export_checkpoint import export_checkpoint


def test_export_crops_vocab_padding_and_preserves_metadata(tmp_path):
    checkpoint_dir = tmp_path / "source"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model_000123.pt"
    meta_path = checkpoint_dir / "meta_000123.json"
    vocab_size = 70
    padded_vocab_size = 128
    state = {
        "transformer.wte.weight": torch.randn(padded_vocab_size, 32),
        "transformer.h.0.attn.c_q.weight": torch.randn(32, 32),
        "value_embeds.0.weight": torch.randn(padded_vocab_size, 32),
        "lm_head.weight": torch.randn(padded_vocab_size, 32),
        "resid_lambdas": torch.ones(1),
        "x0_lambdas": torch.zeros(1),
        "smear_gate.weight": torch.randn(1, 24),
        "smear_lambda": torch.zeros(1),
        "backout_lambda": torch.tensor([0.2]),
    }
    torch.save(state, checkpoint)
    meta = {
        "model_config": {
            "sequence_len": 512,
            "vocab_size": vocab_size,
            "n_layer": 1,
            "n_head": 4,
            "n_kv_head": 2,
            "n_embd": 32,
            "window_pattern": "S",
            "latent_feedback": False,
            "latent_feedback_mode": "gate_product",
            "weight_tying": False,
        },
        "user_config": {"num_forward_passes": 1},
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    output = export_checkpoint(
        checkpoint,
        tmp_path / "exported",
        weight_format="pytorch",
    )

    exported = torch.load(output / "pytorch_model.bin", weights_only=True)
    assert exported["transformer.wte.weight"].shape == (vocab_size, 32)
    assert exported["value_embeds.0.weight"].shape == (vocab_size, 32)
    assert exported["lm_head.weight"].shape == (vocab_size, 32)

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["NanochatForCausalLM"]
    assert config["vocab_size"] == vocab_size
    assert config["nanochat_decode_mode"] == "standard"
    assert config["layer_types"] == ["full_attention"]
    assert config["sliding_window"] == 129
    assert json.loads(
        (output / "nanochat_meta.json").read_text(encoding="utf-8")
    ) == meta


def test_export_refuses_to_overwrite(tmp_path):
    checkpoint_dir = tmp_path / "source"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model_000001.pt"
    torch.save({"transformer.wte.weight": torch.randn(64, 8)}, checkpoint)
    (checkpoint_dir / "meta_000001.json").write_text(
        json.dumps(
            {
                "model_config": {
                    "sequence_len": 128,
                    "vocab_size": 64,
                    "n_layer": 1,
                    "n_head": 1,
                    "n_kv_head": 1,
                    "n_embd": 8,
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "exported"
    export_checkpoint(checkpoint, output, weight_format="pytorch")

    try:
        export_checkpoint(checkpoint, output, weight_format="pytorch")
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected overwrite protection")


def test_soft_export_requires_feedback_checkpoint_and_weights(tmp_path):
    checkpoint_dir = tmp_path / "source"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model_000002.pt"
    hidden_size = 8
    state = {"transformer.wte.weight": torch.randn(64, hidden_size)}
    torch.save(state, checkpoint)
    meta_path = checkpoint_dir / "meta_000002.json"
    model_config = {
        "sequence_len": 128,
        "vocab_size": 64,
        "n_layer": 1,
        "n_head": 1,
        "n_kv_head": 1,
        "n_embd": hidden_size,
        "latent_feedback": False,
    }
    meta_path.write_text(
        json.dumps({"model_config": model_config}),
        encoding="utf-8",
    )

    try:
        export_checkpoint(checkpoint, tmp_path / "no-feedback", decode_mode="soft")
    except ValueError as error:
        assert "latent-feedback checkpoint" in str(error)
    else:
        raise AssertionError("expected soft export to reject a standard checkpoint")

    model_config["latent_feedback"] = True
    meta_path.write_text(
        json.dumps({"model_config": model_config}),
        encoding="utf-8",
    )
    for name, in_features in (
        ("state_proj", hidden_size),
        ("token_gate", hidden_size),
        ("concat_proj", 2 * hidden_size),
        ("token_proj", hidden_size),
    ):
        state[f"latent_feedback.{name}.weight"] = torch.randn(
            hidden_size,
            in_features,
        )
    torch.save(state, checkpoint)

    output = export_checkpoint(
        checkpoint,
        tmp_path / "soft",
        decode_mode="soft",
        weight_format="pytorch",
    )
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "nanochat_export.json").read_text(encoding="utf-8")
    )
    assert config["nanochat_decode_mode"] == "soft"
    assert manifest["decode_mode"] == "soft"
