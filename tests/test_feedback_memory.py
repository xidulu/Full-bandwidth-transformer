"""Tests for multi-pass training with activation recomputation (nanochat/multi_pass.py)."""

import contextlib
import unittest

import torch

import nanochat.flash_attention as flash_attention_module
import nanochat.gpt as gpt_module
from nanochat.multi_pass import RECOMPUTE_MODES, RecomputeForward
from nanochat.gpt import GPT, GPTConfig, LATENT_FEEDBACK_MODES, build_feedback_mask

BOS = 1
JITTER = 0.02


def eager(fn):
    return fn


def aot_eager(fn):
    return torch.compile(fn, backend="aot_eager", dynamic=False)


@contextlib.contextmanager
def compute_dtype(dtype):
    """Run the model in this dtype. gpt.py reads its module variable COMPUTE_DTYPE at call time."""
    original = gpt_module.COMPUTE_DTYPE
    gpt_module.COMPUTE_DTYPE = dtype
    try:
        yield
    finally:
        gpt_module.COMPUTE_DTYPE = original


def make_tiny_gpt(latent_feedback_mode="gate_product", weight_tying=False, seed=1234, dtype=torch.float32):
    config = GPTConfig(
        sequence_len=8,
        vocab_size=19,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        latent_feedback=True,
        latent_feedback_mode=latent_feedback_mode,
        weight_tying=weight_tying,
    )
    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    model.init_weights()
    # init_weights sets some projections to 0. Add noise to each parameter, so that all
    # gradient paths have gradients that are not 0. This includes the feedback path between passes.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator).to(parameter.dtype))
    model.train()
    # The parameters, the buffers, and the activations use `dtype` (refer to compute_dtype) on all hosts.
    return model.to(dtype)


def make_batch(prefix_mixin, num_forward_passes, seed=0):
    # 2 of the rows have 2 packed documents each. The masks must thus also reset at a BOS token that is not in column 0.
    idx = torch.tensor([
        [BOS, 5, 6, 7, 8, 9, 10, 11],
        [BOS, 3, 4, BOS, 12, 13, 14, 15],
        [BOS, 16, 17, 18, 2, BOS, 6, 4],
    ])
    targets = torch.roll(idx, shifts=-1, dims=1)
    targets[:, -1] = -1  # ignored position, as in padded SFT rows
    feedback_masks = None
    if num_forward_passes > 1:
        generator = torch.Generator().manual_seed(seed)
        feedback_masks = torch.stack([
            build_feedback_mask(idx, BOS, prefix_mixin=prefix_mixin, generator=generator)
            for _ in range(num_forward_passes - 1)
        ])
    return idx, targets, feedback_masks


def loss_and_grads(forward, model, idx, targets, feedback_masks, num_forward_passes, jitter=JITTER, seed=7):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    loss, pass_losses = forward(
        idx,
        targets,
        num_forward_passes=num_forward_passes,
        feedback_masks=feedback_masks,
        feedback_jitter=jitter,
        return_loss_components=True,
    )
    loss.backward()
    rng_state = torch.get_rng_state()
    grads = {
        name: (None if parameter.grad is None else parameter.grad.detach().clone())
        for name, parameter in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    return loss.detach(), pass_losses.detach(), grads, rng_state


class TestRecomputeForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The small test models are on the CPU, also when the test host has a GPU that can use FA3.
        cls.original_use_fa3 = flash_attention_module.USE_FA3
        flash_attention_module.USE_FA3 = False
        # Use fp32 also on a CUDA host, where the default compute dtype is bf16.
        cls.exit_stack = contextlib.ExitStack()
        cls.exit_stack.enter_context(compute_dtype(torch.float32))

    @classmethod
    def tearDownClass(cls):
        cls.exit_stack.close()
        flash_attention_module.USE_FA3 = cls.original_use_fa3

    def assert_same_grads(self, reference, grads, rtol, scale_atol, norm_tol):
        # scale_atol is relative to the largest element of each gradient tensor. An element can be
        # near 0 because large terms cancel. Such an element has the rounding error of those terms.
        self.assertEqual(set(reference), set(grads))
        for name, expected in reference.items():
            actual = grads[name]
            if expected is None:
                self.assertIsNone(actual, f"{name} must have no gradient")
                continue
            self.assertIsNotNone(actual, f"{name} has no gradient")
            atol = scale_atol * expected.abs().max().item()
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=lambda m, name=name: f"{name}: {m}")
            relative_norm_error = ((actual - expected).norm() / expected.norm().clamp_min(1e-300)).item()
            self.assertLess(relative_norm_error, norm_tol, name)

    def test_matches_gpt_forward_for_all_modes(self):
        # A recomputed pass adds its parameter gradients to .grad in its own backward pass. The terms
        # of a gradient are thus added in a different sequence than in GPT.forward. In float32, the
        # rounding difference is as large as 4e-6 of a scalar gradient. Thus the comparison uses
        # float64. A real difference stays in float64, and the rounding difference does not (measured: 6e-15).
        with compute_dtype(torch.float64):
            self.check_matches_gpt_forward_for_all_modes()

    def check_matches_gpt_forward_for_all_modes(self):
        for latent_feedback_mode in LATENT_FEEDBACK_MODES:
            for weight_tying in (False, True):
                model = make_tiny_gpt(
                    latent_feedback_mode=latent_feedback_mode, weight_tying=weight_tying, dtype=torch.float64
                )
                for num_forward_passes in (1, 2, 3):
                    for prefix_mixin in (False, True):
                        idx, targets, masks = make_batch(prefix_mixin, num_forward_passes)
                        ref_loss, ref_components, ref_grads, ref_rng = loss_and_grads(
                            model, model, idx, targets, masks, num_forward_passes
                        )
                        # The feedback path must have a gradient. If not, the comparison shows nothing.
                        if num_forward_passes > 1:
                            active = model.latent_feedback.active_parameter_names()[0]
                            self.assertGreater(ref_grads[f"latent_feedback.{active}"].abs().sum().item(), 0.0)
                        for recompute in RECOMPUTE_MODES:
                            with self.subTest(
                                mode=latent_feedback_mode, weight_tying=weight_tying,
                                K=num_forward_passes, prefix_mixin=prefix_mixin, recompute=recompute,
                            ):
                                forward = RecomputeForward(model, recompute=recompute, compile_fn=eager)
                                loss, components, grads, rng = loss_and_grads(
                                    forward, model, idx, targets, masks, num_forward_passes
                                )
                                torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=0.0)
                                torch.testing.assert_close(components, ref_components, rtol=1e-6, atol=0.0)
                                self.assertEqual(components.shape, (num_forward_passes,))
                                self.assert_same_grads(ref_grads, grads, rtol=1e-9, scale_atol=1e-12, norm_tol=1e-10)
                                # The 2 paths use the same quantity of random numbers: 1 jitter sample for each feedback pass.
                                self.assertTrue(torch.equal(rng, ref_rng))

    def test_passes_run_2k_minus_1_times_with_pass_recompute(self):
        for num_forward_passes in (1, 2, 3):
            for recompute, expected in (("none", num_forward_passes), ("pass", 2 * num_forward_passes - 1)):
                with self.subTest(K=num_forward_passes, recompute=recompute):
                    model = make_tiny_gpt()
                    trunk_calls, head_calls = [], []
                    original_run_trunk = model._run_trunk
                    original_project_and_loss = model._project_and_loss

                    def counting_run_trunk(idx, inputs, kv_cache):
                        trunk_calls.append(1)
                        return original_run_trunk(idx, inputs, kv_cache)

                    def counting_project_and_loss(hidden, targets, loss_reduction):
                        head_calls.append(1)
                        return original_project_and_loss(hidden, targets, loss_reduction)

                    model._run_trunk = counting_run_trunk
                    model._project_and_loss = counting_project_and_loss
                    idx, targets, masks = make_batch(False, num_forward_passes)
                    forward = RecomputeForward(model, recompute=recompute, compile_fn=eager)
                    loss = forward(idx, targets, num_forward_passes=num_forward_passes, feedback_masks=masks)
                    # The forward pass runs each pass 1 time, with checkpoints and with no checkpoints.
                    self.assertEqual(len(trunk_calls), num_forward_passes)
                    self.assertEqual(len(head_calls), num_forward_passes)
                    loss.backward()
                    # A checkpoint contains a full pass (trunk and head). All passes have one, but not the last pass.
                    self.assertEqual(len(trunk_calls), expected)
                    self.assertEqual(len(head_calls), expected)

    def test_gpt_forward_and_recompute_share_one_loop(self):
        # GPT.forward and RecomputeForward use the same function, multi_pass_loss. This test confirms
        # that gpt.py has the same function object, and that a direct call gives the loss of GPT.forward.
        import nanochat.gpt as gpt_module
        import nanochat.multi_pass as multi_pass_module

        model = make_tiny_gpt()
        idx, targets, masks = make_batch(False, 2)
        self.assertIs(gpt_module.multi_pass_loss, multi_pass_module.multi_pass_loss)
        reference = model(idx, targets, num_forward_passes=2, feedback_masks=masks, feedback_jitter=0.0)
        direct = multi_pass_module.multi_pass_loss(model, idx, targets, 2, feedback_masks=masks, feedback_jitter=0.0)
        torch.testing.assert_close(direct, reference, rtol=0.0, atol=0.0)

    def test_no_grad_and_eval_need_no_recomputation(self):
        model = make_tiny_gpt()
        idx, targets, masks = make_batch(False, 3)
        forward = RecomputeForward(model, recompute="pass", compile_fn=eager)
        model.eval()
        rng_before = torch.get_rng_state()
        with torch.no_grad():
            loss, components = forward(idx, targets, num_forward_passes=3, feedback_masks=masks, return_loss_components=True)
            reference, reference_components = model(
                idx, targets, num_forward_passes=3, feedback_masks=masks, return_loss_components=True
            )
        # Only training uses jitter, so evaluation uses no random numbers.
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertFalse(loss.requires_grad)
        torch.testing.assert_close(loss, reference, rtol=1e-6, atol=0.0)
        torch.testing.assert_close(components, reference_components, rtol=1e-6, atol=0.0)

    def test_compiled_pieces_match_gpt_forward(self):
        for weight_tying in (False, True):
            for num_forward_passes in (1, 3):
                with self.subTest(weight_tying=weight_tying, K=num_forward_passes):
                    model = make_tiny_gpt(weight_tying=weight_tying)
                    idx, targets, masks = make_batch(False, num_forward_passes)
                    ref_loss, ref_components, ref_grads, _ = loss_and_grads(
                        model, model, idx, targets, masks, num_forward_passes
                    )
                    forward = RecomputeForward(model, recompute="pass", compile_fn=aot_eager)
                    # Do 2 steps. Step 2 uses the compiled parts again.
                    for _ in range(2):
                        loss, components, grads, _ = loss_and_grads(
                            forward, model, idx, targets, masks, num_forward_passes
                        )
                        torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=0.0)
                        torch.testing.assert_close(components, ref_components, rtol=1e-5, atol=0.0)
                        self.assert_same_grads(ref_grads, grads, rtol=1e-4, scale_atol=1e-5, norm_tol=1e-4)

    def test_dormant_fusion_matrices_stay_without_gradient(self):
        # The optimizer skips a feedback group only when the gradient of the group is None. Thus the
        # recomputation must not give gradients of 0 to the fusion matrices that are not in use.
        for latent_feedback_mode in LATENT_FEEDBACK_MODES:
            with self.subTest(mode=latent_feedback_mode):
                model = make_tiny_gpt(latent_feedback_mode=latent_feedback_mode)
                idx, targets, masks = make_batch(False, 3)
                forward = RecomputeForward(model, recompute="pass", compile_fn=eager)
                loss = forward(idx, targets, num_forward_passes=3, feedback_masks=masks)
                loss.backward()
                active = {f"latent_feedback.{name}" for name in model.latent_feedback.active_parameter_names()}
                for name, parameter in model.named_parameters():
                    if name.startswith("latent_feedback.") and name not in active:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_recompute_requires_pass_inputs_with_grad(self):
        # The reentrant checkpoint gives no gradients to a pass when the pass inputs have
        # requires_grad=False, and PyTorch gives only a warning. RecomputeForward must give an error.
        model = make_tiny_gpt(weight_tying=False)
        for parameter in (model.transformer.wte.weight, model.smear_gate.weight, model.smear_lambda):
            parameter.requires_grad_(False)
        idx, targets, masks = make_batch(False, 2)
        with self.assertRaisesRegex(AssertionError, "requires_grad=False"):
            RecomputeForward(model, recompute="pass", compile_fn=eager)(
                idx, targets, num_forward_passes=2, feedback_masks=masks
            )
        # With no checkpoints, the same model trains correctly.
        loss = RecomputeForward(model, recompute="none", compile_fn=eager)(
            idx, targets, num_forward_passes=2, feedback_masks=masks
        )
        loss.backward()
        self.assertIsNone(model.transformer.wte.weight.grad)
        self.assertIsNotNone(model.transformer.h[0].mlp.c_fc.weight.grad)

    def test_rejects_invalid_arguments(self):
        model = make_tiny_gpt()
        with self.assertRaises(ValueError):
            RecomputeForward(model, recompute="layer", compile_fn=eager)
        forward = RecomputeForward(model, recompute="pass", compile_fn=eager)
        idx, targets, masks = make_batch(False, 3)
        with self.assertRaises(AssertionError):
            forward(idx, targets, num_forward_passes=3, feedback_masks=None)
        with self.assertRaises(AssertionError):
            forward(idx, targets, num_forward_passes=2, feedback_masks=masks)  # masks are for K=3
        with self.assertRaises(AssertionError):
            forward(idx, None, num_forward_passes=1)


if __name__ == "__main__":
    unittest.main()
