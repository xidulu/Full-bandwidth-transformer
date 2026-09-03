"""Focused tests for optional token-embedding/output-projection weight tying."""

import io
import unittest

import torch

import nanochat.flash_attention as flash_attention_module
from nanochat.checkpoint_manager import _patch_missing_config_keys
from nanochat.fp8 import Float8Linear, convert_to_float8_training
from nanochat.gpt import GPT, GPTConfig, TIED_EMBEDDING_SCALE, build_feedback_mask


def tiny_config(*, weight_tying, latent_feedback=False):
    return GPTConfig(
        sequence_len=8,
        vocab_size=19,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        latent_feedback=latent_feedback,
        weight_tying=weight_tying,
    )


def make_tiny_gpt(*, weight_tying, latent_feedback=False, seed=1234):
    # Match the production construction path. In particular, to_empty() may
    # independently replace parameters that were aliases on the meta device.
    with torch.device("meta"):
        model = GPT(
            tiny_config(
                weight_tying=weight_tying,
                latent_feedback=latent_feedback,
            ),
            pad_vocab_size_to=1,
        )
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    model.init_weights()
    return model


class TestWeightTying(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny test models run on CPU, even on hosts where FA3 is available.
        cls.original_use_fa3 = flash_attention_module.USE_FA3
        flash_attention_module.USE_FA3 = False

    @classmethod
    def tearDownClass(cls):
        flash_attention_module.USE_FA3 = cls.original_use_fa3

    def test_meta_materialization_init_restores_fp32_tie(self):
        tied = make_tiny_gpt(weight_tying=True)
        untied = make_tiny_gpt(weight_tying=False)

        shared = tied.transformer.wte.weight
        self.assertIs(shared, tied.lm_head.weight)
        self.assertEqual(
            shared.untyped_storage().data_ptr(),
            tied.lm_head.weight.untyped_storage().data_ptr(),
        )
        self.assertEqual(shared.dtype, torch.float32)
        self.assertTrue(torch.isfinite(shared).all())
        self.assertLess(abs(shared.mean().item()), 2e-4)
        self.assertGreater(shared.std().item(), 8e-4)
        self.assertLess(shared.std().item(), 1.2e-3)
        idx = torch.tensor([[1, 2, 3, 4]])
        token_embeddings = tied._embed_tokens(idx)
        prepared = tied._prepare_token_inputs(token_embeddings, kv_cache=None)
        prepared_rms = prepared.float().square().mean().sqrt().item()
        self.assertGreater(prepared_rms, 0.98)
        self.assertLess(prepared_rms, 1.1)

        self.assertFalse(untied.config.weight_tying)
        self.assertIsNot(untied.transformer.wte.weight, untied.lm_head.weight)
        self.assertNotEqual(
            untied.transformer.wte.weight.untyped_storage().data_ptr(),
            untied.lm_head.weight.untyped_storage().data_ptr(),
        )

        named_ids = [id(parameter) for _, parameter in tied.named_parameters()]
        self.assertEqual(len(named_ids), len(set(named_ids)))
        self.assertEqual(named_ids.count(id(shared)), 1)
        state = tied.state_dict()
        self.assertIn("transformer.wte.weight", state)
        self.assertIn("lm_head.weight", state)
        self.assertEqual(
            state["transformer.wte.weight"].untyped_storage().data_ptr(),
            state["lm_head.weight"].untyped_storage().data_ptr(),
        )

    def test_logits_match_equivalent_untied_model_and_gradients_sum(self):
        tied = make_tiny_gpt(weight_tying=True, seed=7)
        untied = make_tiny_gpt(weight_tying=False, seed=99)
        # A tied state dict contains both compatibility keys with equal values.
        # Loading it without assignment gives the untied reference two equal,
        # independent parameters while preserving each destination dtype. The
        # tied lookup activation is rescaled before RMSNorm, so incorporate the
        # same scale into the reference's standalone embedding matrix.
        untied.load_state_dict(tied.state_dict(), strict=True)
        with torch.no_grad():
            untied.transformer.wte.weight.copy_(
                tied.transformer.wte.weight.to(untied.transformer.wte.weight.dtype)
                * TIED_EMBEDDING_SCALE
            )

        idx = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]])
        targets = torch.tensor([[2, 3, 4, 5], [5, 6, 7, 8]])
        tied.eval()
        untied.eval()

        tied_logits = tied(idx)
        untied_logits = untied(idx)
        torch.testing.assert_close(tied_logits, untied_logits, rtol=0, atol=0)

        tied_loss = tied(idx, targets)
        untied_loss = untied(idx, targets)
        torch.testing.assert_close(tied_loss, untied_loss, rtol=0, atol=0)
        tied_loss.backward()
        untied_loss.backward()

        tied_grad = tied.transformer.wte.weight.grad
        expected_grad = (
            TIED_EMBEDDING_SCALE * untied.transformer.wte.weight.grad.float()
            + untied.lm_head.weight.grad.float()
        )
        self.assertIs(tied_grad, tied.lm_head.weight.grad)
        # The untied embedding stores its gradient in bf16, while the tied
        # matrix keeps an fp32 master gradient. Their decomposition therefore
        # differs by at most a few bf16 rounding quanta.
        torch.testing.assert_close(
            tied_grad.float(),
            expected_grad,
            rtol=3e-3,
            atol=6e-4,
        )

    def test_optimizer_groups_and_parameter_accounting_count_shared_weight_once(self):
        tied = make_tiny_gpt(weight_tying=True)
        untied = make_tiny_gpt(weight_tying=False)
        tied_counts = tied.num_scaling_params()
        untied_counts = untied.num_scaling_params()
        shared_numel = tied.transformer.wte.weight.numel()

        self.assertEqual(tied_counts["wte"], 0)
        self.assertEqual(tied_counts["lm_head"], shared_numel)
        self.assertEqual(
            untied_counts["total"] - tied_counts["total"],
            shared_numel,
        )
        self.assertEqual(tied.num_matmul_params(), untied.num_matmul_params())
        self.assertEqual(tied.estimate_flops(), untied.estimate_flops())
        self.assertEqual(
            tied_counts["total"],
            sum(parameter.numel() for parameter in tied.parameters()),
        )

        unembedding_lr = 0.007
        embedding_lr = 0.123
        optimizer = tied.setup_optimizer(
            unembedding_lr=unembedding_lr,
            embedding_lr=embedding_lr,
        )
        grouped = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertTrue(all(group["params"] for group in optimizer.param_groups))
        self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertEqual(
            {id(parameter) for parameter in grouped},
            {id(parameter) for parameter in tied.parameters()},
        )

        shared = tied.transformer.wte.weight
        shared_groups = [
            group
            for group in optimizer.param_groups
            if any(parameter is shared for parameter in group["params"])
        ]
        self.assertEqual(len(shared_groups), 1)
        shared_group = shared_groups[0]
        dmodel_lr_scale = (tied.config.n_embd / 768) ** -0.5
        self.assertEqual(shared_group["kind"], "adamw")
        self.assertAlmostEqual(
            shared_group["lr"],
            unembedding_lr * dmodel_lr_scale,
        )
        self.assertEqual(shared_group["betas"], (0.8, 0.96))
        self.assertEqual(shared_group["weight_decay"], 0.01)

        # Group topology must also be stable enough for optimizer checkpointing.
        reloaded_model = make_tiny_gpt(weight_tying=True, seed=8)
        reloaded_optimizer = reloaded_model.setup_optimizer(
            unembedding_lr=unembedding_lr,
            embedding_lr=embedding_lr,
        )
        reloaded_optimizer.load_state_dict(optimizer.state_dict())
        self.assertEqual(
            len(reloaded_optimizer.param_groups),
            len(optimizer.param_groups),
        )

    def test_fp8_module_conversion_preserves_shared_parameter(self):
        model = make_tiny_gpt(weight_tying=True)
        shared = model.transformer.wte.weight
        matmul_params = model.num_matmul_params()

        convert_to_float8_training(model)

        self.assertIsInstance(model.lm_head, Float8Linear)
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)
        self.assertIs(shared, model.lm_head.weight)
        self.assertEqual(model.num_matmul_params(), matmul_params)

    def test_strict_assign_roundtrip_reties_and_rejects_mismatched_matrices(self):
        source = make_tiny_gpt(weight_tying=True, seed=11)
        source.eval()
        idx = torch.tensor([[1, 2, 3, 4]])
        expected_logits = source(idx)

        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)

        restored = make_tiny_gpt(weight_tying=True, seed=12)
        restored.load_state_dict(checkpoint, strict=True, assign=True)
        self.assertIs(restored.transformer.wte.weight, restored.lm_head.weight)
        self.assertEqual(
            restored.transformer.wte.weight.untyped_storage().data_ptr(),
            restored.lm_head.weight.untyped_storage().data_ptr(),
        )
        restored.eval()
        torch.testing.assert_close(restored(idx), expected_logits, rtol=0, atol=0)

        mismatched = {
            key: value.detach().clone()
            for key, value in checkpoint.items()
        }
        mismatched["lm_head.weight"][0, 0].add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "Cannot load different"):
            restored.load_state_dict(mismatched, strict=True, assign=True)

        head_only = {"lm_head.weight": checkpoint["lm_head.weight"]}
        with self.assertRaisesRegex(RuntimeError, "must provide both"):
            restored.load_state_dict(head_only, strict=False, assign=True)

    def test_aot_eager_forward_backward_preserves_tie(self):
        model = make_tiny_gpt(weight_tying=True, seed=17)
        model.eval()
        idx = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]])
        targets = torch.tensor([[2, 3, 4, 5], [5, 6, 7, 8]])

        eager_loss = model(idx, targets)
        eager_loss.backward()
        eager_grad = model.transformer.wte.weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)

        compiled = torch.compile(model, backend="aot_eager", dynamic=False)
        compiled_loss = compiled(idx, targets)
        compiled_loss.backward()

        torch.testing.assert_close(compiled_loss, eager_loss, rtol=0, atol=0)
        torch.testing.assert_close(
            model.transformer.wte.weight.grad,
            eager_grad,
            rtol=0,
            atol=0,
        )
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)
        self.assertIs(
            compiled._orig_mod.transformer.wte.weight,
            compiled._orig_mod.lm_head.weight,
        )

    def test_weight_tying_composes_with_three_pass_latent_feedback(self):
        model = make_tiny_gpt(
            weight_tying=True,
            latent_feedback=True,
            seed=23,
        )
        model.eval()
        idx = torch.tensor([[1, 2, 3, 1, 4, 5], [1, 6, 7, 8, 9, 10]])
        targets = torch.tensor([[2, 3, 1, 4, 5, 6], [6, 7, 8, 9, 10, 11]])
        feedback_mask = build_feedback_mask(
            idx,
            bos_token_id=1,
            prefix_mixin=False,
        )
        feedback_masks = torch.stack((feedback_mask, feedback_mask))

        total, components = model(
            idx,
            targets,
            num_forward_passes=3,
            feedback_masks=feedback_masks,
            feedback_jitter=0.0,
            return_loss_components=True,
        )
        self.assertEqual(components.shape, (3,))
        self.assertTrue(torch.isfinite(components).all())
        torch.testing.assert_close(total, components[0] + components[1:].mean())
        total.backward()

        shared = model.transformer.wte.weight
        self.assertIs(shared, model.lm_head.weight)
        self.assertIsNotNone(shared.grad)
        self.assertTrue(torch.isfinite(shared.grad).all())
        self.assertGreater(shared.grad.abs().sum().item(), 0.0)
        active_parameter_ids = {
            id(parameter)
            for parameter in model.latent_feedback.active_parameters()
        }
        for parameter in model.latent_feedback.parameters():
            if id(parameter) in active_parameter_ids:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
            else:
                self.assertIsNone(parameter.grad)

    def test_legacy_config_defaults_to_untied_and_loads_strictly(self):
        self.assertFalse(GPTConfig().weight_tying)
        legacy_config = {
            "sequence_len": 8,
            "vocab_size": 19,
            "n_layer": 2,
            "n_head": 2,
            "n_kv_head": 2,
            "n_embd": 32,
        }
        _patch_missing_config_keys(legacy_config)
        self.assertEqual(legacy_config["window_pattern"], "L")
        self.assertEqual(legacy_config["latent_feedback_mode"], "gate_product")
        self.assertFalse(legacy_config["weight_tying"])

        source = make_tiny_gpt(weight_tying=False, seed=31)
        config = GPTConfig(**legacy_config)
        with torch.device("meta"):
            restored = GPT(config, pad_vocab_size_to=1)
        restored.to_empty(device="cpu")
        restored.init_weights()
        restored.load_state_dict(source.state_dict(), strict=True, assign=True)

        self.assertFalse(restored.config.weight_tying)
        self.assertIsNot(restored.transformer.wte.weight, restored.lm_head.weight)
        self.assertNotEqual(
            restored.transformer.wte.weight.untyped_storage().data_ptr(),
            restored.lm_head.weight.untyped_storage().data_ptr(),
        )
        for key, value in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
