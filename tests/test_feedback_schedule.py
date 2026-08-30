"""Focused tests for the one-pass to latent-feedback training schedule."""

import unittest

import torch

import nanochat.flash_attention as flash_attention_module
from nanochat.gpt import GPT, GPTConfig, build_feedback_mask
from nanochat.optim import MuonAdamW
from nanochat.train_utils import get_active_forward_passes


def make_tiny_gpt(seed=1234):
    config = GPTConfig(
        sequence_len=8,
        vocab_size=19,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        latent_feedback=True,
    )
    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    model.init_weights()
    return model


class TestFeedbackSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny models run on CPU even on hosts where Flash Attention is available.
        cls.original_use_fa3 = flash_attention_module.USE_FA3
        flash_attention_module.USE_FA3 = False

    @classmethod
    def tearDownClass(cls):
        flash_attention_module.USE_FA3 = cls.original_use_fa3

    def test_half_schedule_switches_at_exact_even_boundary(self):
        active = [
            get_active_forward_passes(
                step,
                num_iterations=10,
                max_forward_passes=3,
                feedback_start_fraction=0.5,
            )
            for step in range(10)
        ]
        self.assertEqual(active, [1] * 5 + [3] * 5)

        # A resumed run uses the absolute checkpoint step, so it makes the same
        # decision on either side of the transition as an uninterrupted run.
        self.assertEqual(
            get_active_forward_passes(4, 10, 3, 0.5),
            1,
        )
        self.assertEqual(
            get_active_forward_passes(5, 10, 3, 0.5),
            3,
        )

    def test_half_schedule_rounds_odd_boundary_up(self):
        # ceil(0.5 * 5) == 3: steps 0, 1, and 2 are one-pass, and the
        # first multi-pass optimizer step is step 3.
        active = [
            get_active_forward_passes(step, 5, 3, 0.5)
            for step in range(5)
        ]
        self.assertEqual(active, [1, 1, 1, 3, 3])

    def test_schedule_extremes_preserve_fixed_k_behavior(self):
        self.assertEqual(
            [get_active_forward_passes(step, 4, 3, 0.0) for step in range(4)],
            [3, 3, 3, 3],
        )
        self.assertEqual(
            [get_active_forward_passes(step, 4, 3, 1.0) for step in range(4)],
            [1, 1, 1, 1],
        )
        self.assertEqual(
            [get_active_forward_passes(step, 4, 1, 0.5) for step in range(4)],
            [1, 1, 1, 1],
        )

    def test_compiled_k1_to_k3_activates_dedicated_feedback_group(self):
        model = make_tiny_gpt(seed=77)
        compiled_model = torch.compile(model, backend="aot_eager", dynamic=False)
        optimizer = compiled_model.setup_optimizer(separate_feedback_params=True)

        feedback_params = list(model.latent_feedback.parameters())
        feedback_ids = {id(parameter) for parameter in feedback_params}
        feedback_groups = [
            group for group in optimizer.param_groups if group.get("is_feedback", False)
        ]
        self.assertEqual(len(feedback_groups), 1)
        self.assertTrue(feedback_groups[0].get("allow_no_grad", False))
        self.assertEqual(
            {id(parameter) for parameter in feedback_groups[0]["params"]},
            feedback_ids,
        )
        for group in optimizer.param_groups:
            if not group.get("is_feedback", False):
                self.assertTrue(
                    feedback_ids.isdisjoint(id(parameter) for parameter in group["params"])
                )

        idx = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]])
        targets = torch.tensor([[2, 3, 4, 5], [5, 6, 7, 8]])
        feedback_before = [parameter.detach().clone() for parameter in feedback_params]

        # Exercise the compiled one-pass graph used before the schedule boundary.
        loss, components = compiled_model(
            idx,
            targets,
            num_forward_passes=1,
            return_loss_components=True,
        )
        loss.backward()
        self.assertEqual(components.shape, (1,))
        self.assertTrue(all(parameter.grad is None for parameter in feedback_params))

        # Stub only the numerical/collective phases. MuonAdamW.step's group
        # preflight remains real, allowing us to verify that it omits the
        # dormant feedback group without compiling optimizer kernels on CPU.
        reduced_groups = []

        def fake_reduce(group, _world_size):
            reduced_groups.append(group)
            return {"group": group}

        optimizer._reduce_adamw = fake_reduce
        optimizer._reduce_muon = fake_reduce
        optimizer._compute_adamw = lambda *_args: None
        optimizer._compute_muon = lambda *_args: None
        optimizer._finish_gathers = lambda _gathers: None
        optimizer.step()

        self.assertFalse(any(group.get("is_feedback", False) for group in reduced_groups))
        for parameter, before in zip(feedback_params, feedback_before):
            torch.testing.assert_close(parameter, before, rtol=0, atol=0)
            self.assertNotIn(parameter, optimizer.state)

        model.zero_grad(set_to_none=True)
        reduced_groups.clear()
        feedback_mask = build_feedback_mask(
            idx, bos_token_id=1, prefix_mixin=False
        )
        feedback_masks = torch.stack((feedback_mask, feedback_mask))

        # Switching the Python K argument creates/uses the K=3 compiled graph;
        # it must now produce gradients and activate the feedback optimizer group.
        loss, components = compiled_model(
            idx,
            targets,
            num_forward_passes=3,
            feedback_masks=feedback_masks,
            feedback_jitter=0.0,
            return_loss_components=True,
        )
        loss.backward()
        self.assertEqual(components.shape, (3,))
        for parameter in feedback_params:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

        optimizer.step()
        self.assertTrue(any(group.get("is_feedback", False) for group in reduced_groups))

    def test_only_explicitly_dormant_optimizer_groups_may_lack_gradients(self):
        def muon_group(params, *, allow_no_grad=False):
            return {
                "kind": "muon",
                "params": params,
                "lr": 0.02,
                "momentum": 0.95,
                "ns_steps": 5,
                "beta2": 0.9,
                "weight_decay": 0.0,
                "allow_no_grad": allow_no_grad,
            }

        ordinary = [
            torch.nn.Parameter(torch.zeros(2, 2)),
            torch.nn.Parameter(torch.zeros(2, 2)),
        ]
        optimizer = MuonAdamW([muon_group(ordinary)])
        with self.assertRaisesRegex(RuntimeError, "has no gradients"):
            optimizer.step()

        ordinary[0].grad = torch.ones_like(ordinary[0])
        optimizer.param_groups[0]["allow_no_grad"] = True
        with self.assertRaisesRegex(RuntimeError, "1/2 parameters with grad=None"):
            optimizer.step()

        dormant = [
            torch.nn.Parameter(torch.zeros(2, 2)),
            torch.nn.Parameter(torch.zeros(2, 2)),
        ]
        optimizer = MuonAdamW([muon_group(dormant, allow_no_grad=True)])
        optimizer._reduce_muon = lambda *_args: self.fail(
            "a dormant group must not launch reduction"
        )
        optimizer.step()
        self.assertEqual(len(optimizer.state), 0)


if __name__ == "__main__":
    unittest.main()
