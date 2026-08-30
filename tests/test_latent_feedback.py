"""Tests for latent-feedback fusion and the fixed multi-pass objective."""

import unittest

import torch
import torch.nn.functional as F

import nanochat.flash_attention as flash_attention_module
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig, LatentFeedback, build_feedback_mask, norm
from nanochat.loss_eval import evaluate_bpb, evaluate_bpb_per_pass


def make_tiny_gpt(latent_feedback=True, seed=1234):
    config = GPTConfig(
        sequence_len=8,
        vocab_size=19,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        latent_feedback=latent_feedback,
    )
    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    model.init_weights()
    return model


class TestLatentFeedback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The tiny test models live on CPU even when the test host has a FA3-capable GPU.
        cls.original_use_fa3 = flash_attention_module.USE_FA3
        flash_attention_module.USE_FA3 = False

    @classmethod
    def tearDownClass(cls):
        flash_attention_module.USE_FA3 = cls.original_use_fa3

    def test_fusion_matches_equation(self):
        module = LatentFeedback(4)
        previous_hidden = torch.tensor(
            [[[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]]]
        )
        token_input = torch.tensor(
            [[[0.5, 1.0, -1.5, 2.0], [2.0, -0.25, 0.75, -1.0]]]
        )
        with torch.no_grad():
            module.state_proj.weight.copy_(
                torch.tensor(
                    [
                        [0.5, -0.1, 0.2, 0.3],
                        [0.0, 0.4, -0.2, 0.1],
                        [-0.3, 0.2, 0.6, 0.0],
                        [0.1, 0.1, -0.4, 0.7],
                    ]
                )
            )
            module.token_gate.weight.copy_(
                torch.tensor(
                    [
                        [0.2, 0.1, 0.0, -0.3],
                        [-0.4, 0.5, 0.1, 0.0],
                        [0.3, -0.2, 0.4, 0.1],
                        [0.0, 0.2, -0.1, 0.6],
                    ]
                )
            )

        expected = norm(
            F.linear(previous_hidden, module.state_proj.weight)
            * torch.sigmoid(F.linear(norm(token_input), module.token_gate.weight))
        )
        actual = module(previous_hidden, token_input)

        self.assertIsNone(module.state_proj.bias)
        self.assertIsNone(module.token_gate.bias)
        self.assertEqual(actual.shape, previous_hidden.shape)
        self.assertEqual(actual.dtype, previous_hidden.dtype)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual.square().mean(dim=-1),
            torch.ones_like(actual[..., 0]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_init_weights_initializes_feedback_matrices(self):
        model = make_tiny_gpt()
        with torch.no_grad():
            model.latent_feedback.state_proj.weight.fill_(float("nan"))
            model.latent_feedback.token_gate.weight.fill_(float("nan"))

        torch.manual_seed(99)
        model.init_weights()

        self.assertTrue(torch.isfinite(model.latent_feedback.state_proj.weight).all())
        self.assertTrue(torch.isfinite(model.latent_feedback.token_gate.weight).all())

    def test_feedback_mask_resets_at_each_packed_bos(self):
        bos = 18
        idx = torch.tensor(
            [
                [bos, 1, 2, bos, 3, 4, 5, bos],
                [7, 8, bos, bos, 9, 10, bos, 11],
            ]
        )
        document_starts = idx.eq(bos)
        document_starts[:, 0] = True

        no_prefix_mask = build_feedback_mask(idx, bos, prefix_mixin=False)
        torch.testing.assert_close(no_prefix_mask, ~document_starts)

        first_generator = torch.Generator().manual_seed(123)
        second_generator = torch.Generator().manual_seed(123)
        mask = build_feedback_mask(idx, bos, prefix_mixin=True, generator=first_generator)
        repeated = build_feedback_mask(idx, bos, prefix_mixin=True, generator=second_generator)
        torch.testing.assert_close(mask, repeated)
        torch.testing.assert_close(
            mask,
            torch.tensor(
                [
                    [False, True, True, False, False, False, True, False],
                    [False, True, False, False, True, True, False, False],
                ]
            ),
        )
        self.assertFalse(mask[document_starts].any())

        # Each packed document is a contiguous plain prefix followed by a fused suffix.
        for row_idx in range(idx.size(0)):
            starts = document_starts[row_idx].nonzero(as_tuple=False).flatten().tolist()
            ends = starts[1:] + [idx.size(1)]
            for start, end in zip(starts, ends):
                segment = mask[row_idx, start:end]
                self.assertFalse(segment[0])
                self.assertFalse((segment[:-1] & ~segment[1:]).any())

    def test_one_pass_is_unchanged_when_feedback_is_enabled(self):
        standard_model = make_tiny_gpt(latent_feedback=False)
        feedback_model = make_tiny_gpt(latent_feedback=True)
        for key, value in standard_model.state_dict().items():
            torch.testing.assert_close(feedback_model.state_dict()[key], value, rtol=0, atol=0)
        load_result = feedback_model.load_state_dict(standard_model.state_dict(), strict=False)
        self.assertEqual(
            set(load_result.missing_keys),
            {
                "latent_feedback.state_proj.weight",
                "latent_feedback.token_gate.weight",
            },
        )
        self.assertEqual(load_result.unexpected_keys, [])
        self.assertFalse(any(key.startswith("latent_feedback.") for key in standard_model.state_dict()))

        idx = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]])
        targets = torch.tensor([[2, 3, 4, 5], [5, 6, 7, 8]])
        standard_model.eval()
        feedback_model.eval()
        with torch.no_grad():
            standard_logits = standard_model(idx)
            feedback_logits = feedback_model(idx, num_forward_passes=1)
            standard_loss = standard_model(idx, targets)
            feedback_loss = feedback_model(idx, targets, num_forward_passes=1)
            component_total, components = feedback_model(
                idx,
                targets,
                num_forward_passes=1,
                return_loss_components=True,
            )

        torch.testing.assert_close(feedback_logits, standard_logits, rtol=0, atol=0)
        torch.testing.assert_close(feedback_loss, standard_loss, rtol=0, atol=0)
        torch.testing.assert_close(component_total, feedback_loss)
        torch.testing.assert_close(components, feedback_loss.unsqueeze(0))

    def test_multi_pass_objective_uses_latest_hidden_and_equation_12(self):
        model = make_tiny_gpt(seed=7)
        model.eval()
        with torch.no_grad():
            model.smear_lambda.fill_(0.75)
            model.smear_gate.weight.fill_(0.1)
        idx = torch.tensor([[1, 2, 3, 1, 4, 5], [1, 6, 7, 8, 9, 10]])
        targets = torch.tensor([[2, 3, 1, 4, 5, 6], [6, 7, 8, 9, 10, 11]])
        feedback_mask = build_feedback_mask(idx, bos_token_id=1, prefix_mixin=False)
        feedback_masks = torch.stack((feedback_mask, feedback_mask))

        with torch.no_grad():
            token_embeddings = model._embed_tokens(idx)
            token_inputs = model._prepare_token_inputs(token_embeddings, kv_cache=None)
            loss_1, hidden_1 = model._forward_once(
                idx, token_inputs, targets, kv_cache=None, loss_reduction="mean"
            )

            fused_2 = model.latent_feedback(hidden_1[:, :-1], token_embeddings[:, 1:])
            smeared_fused_2 = model.latent_feedback(hidden_1[:, :-1], token_inputs[:, 1:])
            inputs_2 = torch.where(
                feedback_mask.unsqueeze(-1),
                torch.cat((token_inputs[:, :1], fused_2), dim=1),
                token_inputs,
            )
            loss_2, hidden_2 = model._forward_once(
                idx, inputs_2, targets, kv_cache=None, loss_reduction="mean"
            )

            fused_3 = model.latent_feedback(hidden_2[:, :-1], token_embeddings[:, 1:])
            inputs_3 = torch.where(
                feedback_mask.unsqueeze(-1),
                torch.cat((token_inputs[:, :1], fused_3), dim=1),
                token_inputs,
            )
            loss_3, _ = model._forward_once(
                idx, inputs_3, targets, kv_cache=None, loss_reduction="mean"
            )

            two_pass_loss, two_pass_components = model(
                idx,
                targets,
                num_forward_passes=2,
                feedback_masks=feedback_masks[:1],
                feedback_jitter=0.0,
                return_loss_components=True,
            )
            three_pass_loss, three_pass_components = model(
                idx,
                targets,
                num_forward_passes=3,
                feedback_masks=feedback_masks,
                feedback_jitter=0.0,
                return_loss_components=True,
            )

        torch.testing.assert_close(two_pass_loss, loss_1 + loss_2)
        torch.testing.assert_close(three_pass_loss, loss_1 + (loss_2 + loss_3) / 2)
        torch.testing.assert_close(two_pass_components, torch.stack((loss_1, loss_2)))
        torch.testing.assert_close(three_pass_components, torch.stack((loss_1, loss_2, loss_3)))
        self.assertFalse(torch.allclose(fused_2, smeared_fused_2))
        self.assertFalse(torch.isclose(three_pass_loss, loss_1 + loss_2 + loss_3))

    def test_pass_three_uses_shifted_pass_two_state(self):
        model = make_tiny_gpt(seed=11)
        idx = torch.tensor([[1, 2, 3, 4]])
        targets = torch.zeros_like(idx)
        token_embeddings = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1).expand(1, 4, 32)
        token_inputs = token_embeddings + 100.0
        feedback_masks = torch.tensor(
            [
                [[False, True, True, True]],
                [[False, True, True, True]],
            ]
        )
        seen_inputs = []
        hidden_states = []

        class EchoFeedback(torch.nn.Module):
            def forward(self, previous_hidden, current_embedding):
                return previous_hidden + 10.0 * current_embedding

        def embed_tokens(_idx):
            return token_embeddings

        def prepare_token_inputs(_token_embeddings, _kv_cache):
            return token_inputs

        def forward_once(_idx, inputs, _targets, _kv_cache, _loss_reduction):
            pass_number = len(seen_inputs) + 1
            seen_inputs.append(inputs.clone())
            hidden = inputs + 1000.0 * pass_number
            hidden_states.append(hidden)
            return inputs.new_tensor(float(pass_number)), hidden

        model.latent_feedback = EchoFeedback()
        model._embed_tokens = embed_tokens
        model._prepare_token_inputs = prepare_token_inputs
        model._forward_once = forward_once
        loss = model(
            idx,
            targets,
            num_forward_passes=3,
            feedback_masks=feedback_masks,
            feedback_jitter=0.0,
        )

        expected_second = torch.cat(
            (
                token_inputs[:, :1],
                hidden_states[0][:, :-1] + 10.0 * token_embeddings[:, 1:],
            ),
            dim=1,
        )
        expected_third = torch.cat(
            (
                token_inputs[:, :1],
                hidden_states[1][:, :-1] + 10.0 * token_embeddings[:, 1:],
            ),
            dim=1,
        )
        torch.testing.assert_close(seen_inputs[1], expected_second)
        torch.testing.assert_close(seen_inputs[2], expected_third)
        torch.testing.assert_close(loss, torch.tensor(1.0 + (2.0 + 3.0) / 2.0))

    def test_feedback_loss_backpropagates_through_first_pass(self):
        model = make_tiny_gpt(seed=21)
        idx = torch.tensor([[1, 2, 3, 4]])
        targets = torch.zeros_like(idx)
        feedback_mask = build_feedback_mask(idx, bos_token_id=1, prefix_mixin=False).unsqueeze(0)

        first_pass_states = []
        original_run_trunk = model._run_trunk

        def capture_run_trunk(run_idx, inputs, kv_cache):
            hidden = original_run_trunk(run_idx, inputs, kv_cache)
            hidden.retain_grad()
            first_pass_states.append(hidden)
            return hidden

        def feedback_only_loss(hidden, _targets, _loss_reduction):
            # Remove the direct first-pass loss so any nonzero first-state gradient
            # must have arrived through the feedback edge in the second pass.
            multiplier = 0.0 if len(first_pass_states) == 1 else 1.0
            return multiplier * hidden[..., 0].mean()

        model._run_trunk = capture_run_trunk
        model._project_and_loss = feedback_only_loss
        loss = model(
            idx,
            targets,
            num_forward_passes=2,
            feedback_masks=feedback_mask,
            feedback_jitter=0.0,
        )
        loss.backward()

        self.assertEqual(len(first_pass_states), 2)
        self.assertIsNotNone(first_pass_states[0].grad)
        self.assertGreater(first_pass_states[0].grad.abs().sum().item(), 0.0)
        for parameter in model.latent_feedback.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_jitter_is_training_only_and_bounded(self):
        model = make_tiny_gpt(seed=31)
        idx = torch.tensor([[1, 2, 3, 4]])
        targets = torch.zeros_like(idx)
        feedback_mask = build_feedback_mask(idx, bos_token_id=1, prefix_mixin=False).unsqueeze(0)
        token_embeddings = model._embed_tokens(idx)
        token_inputs = model._prepare_token_inputs(token_embeddings, kv_cache=None)
        _, first_hidden = model._forward_once(
            idx, token_inputs, targets, kv_cache=None, loss_reduction="mean"
        )
        expected_state = first_hidden[:, :-1]

        class CaptureFeedback(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.inputs = []

            def forward(self, previous_hidden, current_embedding):
                self.inputs.append(previous_hidden.detach().clone())
                return self.inner(previous_hidden, current_embedding)

        capture = CaptureFeedback(model.latent_feedback)
        model.latent_feedback = capture
        jitter = 0.05
        with torch.no_grad():
            model.train()
            torch.manual_seed(123)
            model(
                idx,
                targets,
                num_forward_passes=2,
                feedback_masks=feedback_mask,
                feedback_jitter=jitter,
            )
            train_delta = capture.inputs.pop() - expected_state
            self.assertGreater(train_delta.abs().max().item(), 0.0)
            self.assertLessEqual(train_delta.abs().max().item(), jitter)

            model.eval()
            model(
                idx,
                targets,
                num_forward_passes=2,
                feedback_masks=feedback_mask,
                feedback_jitter=jitter,
            )
            torch.testing.assert_close(capture.inputs.pop(), expected_state, rtol=0, atol=0)

    def test_compiled_two_pass_backward_and_optimizer_grouping(self):
        model = make_tiny_gpt(seed=41)
        compiled_model = torch.compile(model, backend="aot_eager", dynamic=False)
        optimizer = compiled_model.setup_optimizer()
        grouped_parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        model_parameter_ids = [id(parameter) for parameter in model.parameters()]
        self.assertEqual(len(grouped_parameter_ids), len(set(grouped_parameter_ids)))
        self.assertEqual(set(grouped_parameter_ids), set(model_parameter_ids))

        idx = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, 5]])
        feedback_mask = build_feedback_mask(idx, bos_token_id=1, prefix_mixin=False).unsqueeze(0)

        one_pass_loss, one_pass_components = compiled_model(
            idx,
            targets,
            num_forward_passes=1,
            return_loss_components=True,
        )
        one_pass_loss.backward()
        self.assertTrue(torch.isfinite(one_pass_loss))
        self.assertEqual(one_pass_components.shape, (1,))
        model.zero_grad(set_to_none=True)

        loss, pass_losses = compiled_model(
            idx,
            targets,
            num_forward_passes=2,
            feedback_masks=feedback_mask,
            feedback_jitter=0.0,
            return_loss_components=True,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(pass_losses.shape, (2,))
        for parameter in model.latent_feedback.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_cached_one_pass_matches_naive_forward(self):
        model = make_tiny_gpt(latent_feedback=True, seed=51)
        model.eval()
        idx = torch.tensor([[1, 2, 3, 4, 5]])
        head_dim = model.config.n_embd // model.config.n_head
        kv_cache = KVCache(
            batch_size=1,
            num_heads=model.config.n_kv_head,
            seq_len=model.config.sequence_len,
            head_dim=head_dim,
            num_layers=model.config.n_layer,
            device="cpu",
            dtype=torch.float32,
        )
        with torch.no_grad():
            naive_logits = model(idx)
            model(idx[:, :-1], kv_cache=kv_cache)
            cached_logits = model(idx[:, -1:], kv_cache=kv_cache)

        self.assertEqual(kv_cache.get_pos(), idx.size(1))
        torch.testing.assert_close(cached_logits[:, -1], naive_logits[:, -1])

    def test_per_pass_bpb_uses_component_losses_and_full_feedback(self):
        bos = 9
        x = torch.tensor([[bos, 1, 2, bos, 3]])
        y = torch.tensor([[2, 3, -1, 4, 0]])
        # Counted target bytes are 1 + 2 + 1 = 4. Token 4 is special (zero bytes),
        # and the -1 target is ignored regardless of its corresponding loss.
        token_bytes = torch.tensor([1, 1, 1, 2, 0, 1, 1, 1, 1, 0])

        class ComponentLossModel:
            def __init__(self):
                self.feedback_masks = None

            def get_device(self):
                return torch.device("cpu")

            def __call__(
                self,
                _x,
                _y,
                loss_reduction="mean",
                num_forward_passes=1,
                feedback_masks=None,
                feedback_jitter=0.02,
                return_loss_components=False,
            ):
                self.feedback_masks = None if feedback_masks is None else feedback_masks.clone()
                self.feedback_jitter = feedback_jitter
                base_loss = torch.arange(1, _y.numel() + 1, dtype=torch.float32)
                if num_forward_passes == 1:
                    return base_loss
                components = torch.stack(
                    tuple(base_loss + 10.0 * pass_idx for pass_idx in range(num_forward_passes))
                )
                total = components[0] + components[1:].mean(dim=0)
                self.assert_component_request = return_loss_components and loss_reduction == "none"
                return total, components

        model = ComponentLossModel()
        rng_state = torch.random.get_rng_state()
        per_pass_bpb = evaluate_bpb_per_pass(
            model,
            [(x, y)],
            steps=1,
            token_bytes=token_bytes,
            num_forward_passes=3,
            bos_token_id=bos,
        )
        torch.testing.assert_close(torch.random.get_rng_state(), rng_state)
        denominator = torch.log(torch.tensor(2.0)).item() * 4
        expected = [8 / denominator, 38 / denominator, 68 / denominator]

        self.assertEqual(len(per_pass_bpb), 3)
        for actual, expected_value in zip(per_pass_bpb, expected):
            self.assertAlmostEqual(actual, expected_value, places=6)
        self.assertTrue(model.assert_component_request)
        self.assertEqual(model.feedback_jitter, 0.0)
        torch.testing.assert_close(
            model.feedback_masks,
            torch.tensor(
                [
                    [[False, True, True, False, True]],
                    [[False, True, True, False, True]],
                ]
            ),
        )

        # The legacy evaluator remains a scalar, one-pass API.
        scalar_bpb = evaluate_bpb(model, [(x, y)], 1, token_bytes)
        self.assertIsInstance(scalar_bpb, float)
        self.assertAlmostEqual(scalar_bpb, expected[0], places=6)

    def test_actual_model_per_pass_bpb_is_finite_and_repeatable(self):
        model = make_tiny_gpt(seed=61)
        model.eval()
        compiled_model = torch.compile(model, backend="aot_eager", dynamic=False)
        idx = torch.tensor([[1, 2, 3, 1, 4, 5]])
        targets = torch.tensor([[2, 3, 1, 4, 5, 6]])
        token_bytes = torch.ones(model.config.vocab_size, dtype=torch.int64)
        token_bytes[1] = 0 # BOS is excluded from BPB

        def evaluate():
            return evaluate_bpb_per_pass(
                compiled_model,
                [(idx, targets)],
                steps=1,
                token_bytes=token_bytes,
                num_forward_passes=3,
                bos_token_id=1,
            )

        first = evaluate()
        second = evaluate()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in first))


if __name__ == "__main__":
    unittest.main()
