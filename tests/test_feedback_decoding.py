"""Focused tests for the paper's SOFT and FUSED decoding recurrences."""

import unittest
from dataclasses import dataclass

import torch

import nanochat.flash_attention as flash_attention_module
from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import Engine, KVCache
from nanochat.gpt import GPT, GPTConfig
from tests.test_engine import ByteTokenizer


@dataclass
class _TraceConfig:
    n_kv_head: int = 2
    n_head: int = 2
    n_embd: int = 4
    n_layer: int = 1
    sequence_len: int = 32


class _TracingModel:
    """Small protocol mock whose call log exposes recurrence and cache alignment."""

    def __init__(self, *, latent_feedback=True, vocab_size=262):
        self.config = _TraceConfig()
        self.latent_feedback = object() if latent_feedback else None
        self.vocab_size = vocab_size
        self.calls = []

    def get_device(self):
        return torch.device("cpu")

    def forward(
        self,
        ids,
        targets=None,
        kv_cache=None,
        feedback_hidden=None,
        feedback_mask=None,
        return_hidden=False,
        **_kwargs,
    ):
        del targets
        batch_size, num_tokens = ids.shape
        cache_pos_before = None if kv_cache is None else kv_cache.get_pos()

        # Make every state identify both its token and absolute cache position. This
        # lets the tests distinguish the correctly shifted state from an off-by-one.
        positions = torch.arange(
            cache_pos_before or 0,
            (cache_pos_before or 0) + num_tokens,
            dtype=torch.float32,
        ).view(1, num_tokens, 1)
        channels = torch.arange(self.config.n_embd, dtype=torch.float32).view(1, 1, -1)
        hidden = ids.float().unsqueeze(-1) * 10 + positions + channels / 10
        hidden = hidden.expand(batch_size, -1, -1).clone()

        # Pick a finite, non-terminal token deterministically. A different argmax on
        # every model call makes it clear which pass supplied the sampled logits.
        logits = torch.zeros(batch_size, num_tokens, self.vocab_size)
        argmax_token = 11 + len(self.calls)
        logits[..., argmax_token] = 5.0

        call = {
            "ids": ids.detach().clone(),
            "cache": kv_cache,
            "cache_pos_before": cache_pos_before,
            "feedback_hidden": None if feedback_hidden is None else feedback_hidden.detach().clone(),
            "feedback_mask": None if feedback_mask is None else feedback_mask.detach().clone(),
            "return_hidden": return_hidden,
            "hidden": hidden.detach().clone(),
            "logits": logits.detach().clone(),
        }
        if kv_cache is not None:
            kv_cache.advance(num_tokens)
            call["cache_pos_after"] = kv_cache.get_pos()
        self.calls.append(call)

        return (logits, hidden) if return_hidden else logits


def _make_tiny_feedback_gpt(seed=1234):
    config = GPTConfig(
        sequence_len=16,
        vocab_size=262,
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
    model.eval()
    return model


def _make_cache(model, *, batch_size, seq_len):
    config = model.config
    return KVCache(
        batch_size=batch_size,
        num_heads=config.n_kv_head,
        seq_len=seq_len,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device=model.get_device(),
        dtype=COMPUTE_DTYPE,
    )


class TestFeedbackDecoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The real-model equivalence test deliberately runs on CPU.
        cls.original_use_fa3 = flash_attention_module.USE_FA3
        flash_attention_module.USE_FA3 = False

    @classmethod
    def tearDownClass(cls):
        flash_attention_module.USE_FA3 = cls.original_use_fa3

    def test_standard_mode_is_the_default_and_never_requests_feedback(self):
        prompt = [261, 21, 22]
        default_model = _TracingModel()
        explicit_model = _TracingModel()

        default = Engine(default_model, ByteTokenizer()).generate_batch(
            prompt,
            max_tokens=3,
            temperature=0.0,
            use_calculator=False,
        )
        explicit = Engine(explicit_model, ByteTokenizer()).generate_batch(
            prompt,
            max_tokens=3,
            temperature=0.0,
            decode_mode="standard",
            use_calculator=False,
        )

        self.assertEqual(default, explicit)
        for call in (*default_model.calls, *explicit_model.calls):
            self.assertIsNone(call["feedback_hidden"])
            self.assertIsNone(call["feedback_mask"])
            self.assertFalse(call["return_hidden"])

    def test_soft_samples_from_shared_prefill_then_fuses_the_sampled_token(self):
        prompt = [261, 31, 32]
        model = _TracingModel()
        stream = Engine(model, ByteTokenizer()).generate(
            prompt,
            max_tokens=2,
            temperature=0.0,
            decode_mode="soft",
            use_calculator=False,
        )

        first_column, _ = next(stream)
        self.assertEqual(first_column, [11])
        self.assertEqual(len(model.calls), 1)
        prefill = model.calls[0]
        self.assertIsNone(prefill["feedback_hidden"])
        self.assertTrue(prefill["return_hidden"])
        self.assertEqual(prefill["cache_pos_before"], 0)
        self.assertEqual(prefill["cache_pos_after"], len(prompt))

        second_column, _ = next(stream)
        self.assertEqual(second_column, [12])
        self.assertEqual(len(model.calls), 2)
        decode = model.calls[1]
        torch.testing.assert_close(decode["ids"], torch.tensor([[first_column[0]]]))
        torch.testing.assert_close(
            decode["feedback_hidden"],
            prefill["hidden"][:, -1:, :],
        )
        self.assertIsNone(decode["feedback_mask"])
        self.assertEqual(decode["cache_pos_before"], len(prompt))
        self.assertEqual(decode["cache_pos_after"], len(prompt) + 1)
        stream.close()

    def test_soft_expands_the_last_prompt_state_for_every_sample(self):
        prompt = [261, 41, 42]
        num_samples = 4
        model = _TracingModel()
        stream = Engine(model, ByteTokenizer()).generate(
            prompt,
            num_samples=num_samples,
            max_tokens=2,
            temperature=0.0,
            decode_mode="soft",
            use_calculator=False,
        )

        first_column, _ = next(stream)
        next(stream)
        prefill, decode = model.calls[:2]
        self.assertEqual(first_column, [11] * num_samples)
        self.assertEqual(tuple(decode["feedback_hidden"].shape), (num_samples, 1, model.config.n_embd))
        expected = prefill["hidden"][:, -1:, :].expand(num_samples, -1, -1)
        torch.testing.assert_close(decode["feedback_hidden"], expected)
        self.assertEqual(decode["cache"].batch_size, num_samples)
        stream.close()

    def test_fused_prefill_rebuilds_a_fresh_cache_and_shifts_pass1_states(self):
        prompt = [261, 51, 52, 53]
        model = _TracingModel()
        stream = Engine(model, ByteTokenizer()).generate(
            prompt,
            max_tokens=1,
            temperature=0.0,
            decode_mode="fused",
            use_calculator=False,
        )

        first_column, _ = next(stream)
        self.assertEqual(len(model.calls), 2)
        pass1, pass2 = model.calls
        self.assertEqual(first_column, [12])  # pass-2 logits, not pass-1 logits
        self.assertIsNot(pass1["cache"], pass2["cache"])
        self.assertEqual(pass1["cache_pos_before"], 0)
        self.assertEqual(pass2["cache_pos_before"], 0)
        self.assertEqual(pass2["cache_pos_after"], len(prompt))
        self.assertNotEqual(pass2["cache_pos_after"], 2 * len(prompt))
        torch.testing.assert_close(
            pass2["feedback_hidden"],
            torch.roll(pass1["hidden"], shifts=1, dims=1),
        )
        torch.testing.assert_close(
            pass2["feedback_mask"],
            torch.tensor([[False, True, True, True]]),
        )
        stream.close()

    def test_fused_first_token_matches_a_manual_second_prefill_pass(self):
        model = _make_tiny_feedback_gpt(seed=2026)
        tokenizer = ByteTokenizer()
        prompt = [tokenizer.get_bos_token_id(), 61, 62, 63]
        ids = torch.tensor([prompt])

        pass1_cache = _make_cache(model, batch_size=1, seq_len=len(prompt))
        _, pass1_hidden = model.forward(ids, kv_cache=pass1_cache, return_hidden=True)
        pass2_cache = _make_cache(model, batch_size=1, seq_len=len(prompt))
        shifted_hidden = torch.roll(pass1_hidden, shifts=1, dims=1)
        feedback_mask = ids.ne(tokenizer.get_bos_token_id())
        feedback_mask[:, 0] = False
        expected_logits, expected_hidden = model.forward(
            ids,
            kv_cache=pass2_cache,
            feedback_hidden=shifted_hidden,
            feedback_mask=feedback_mask,
            return_hidden=True,
        )
        expected_token = expected_logits[:, -1, :].argmax(dim=-1).item()

        stream = Engine(model, tokenizer).generate(
            prompt,
            max_tokens=1,
            temperature=0.0,
            decode_mode="fused",
            use_calculator=False,
        )
        actual_column, _ = next(stream)
        stream.close()

        self.assertEqual(actual_column, [expected_token])
        self.assertEqual(pass1_cache.get_pos(), len(prompt))
        self.assertEqual(pass2_cache.get_pos(), len(prompt))
        self.assertTrue(torch.isfinite(expected_logits).all())
        self.assertTrue(torch.isfinite(expected_hidden).all())

    def test_one_token_prompt_works_for_both_feedback_modes(self):
        prompt = [261]
        for mode in ("soft", "fused"):
            with self.subTest(mode=mode):
                model = _TracingModel()
                stream = Engine(model, ByteTokenizer()).generate(
                    prompt,
                    max_tokens=1,
                    temperature=0.0,
                    decode_mode=mode,
                    use_calculator=False,
                )
                token_column, _ = next(stream)
                stream.close()
                self.assertEqual(len(token_column), 1)
                self.assertTrue(torch.isfinite(model.calls[-1]["logits"]).all())
                if mode == "fused":
                    self.assertEqual(len(model.calls), 2)
                    torch.testing.assert_close(
                        model.calls[1]["feedback_mask"],
                        torch.tensor([[False]]),
                    )
                    self.assertEqual(model.calls[1]["cache_pos_after"], 1)

    def test_feedback_modes_reject_models_without_latent_feedback(self):
        prompt = [261, 71]
        for mode in ("soft", "fused"):
            with self.subTest(mode=mode):
                engine = Engine(_TracingModel(latent_feedback=False), ByteTokenizer())
                with self.assertRaisesRegex(ValueError, "requires a model trained with latent feedback"):
                    engine.generate_batch(
                        prompt,
                        max_tokens=1,
                        temperature=0.0,
                        decode_mode=mode,
                        use_calculator=False,
                    )

    def test_unknown_decode_mode_is_rejected(self):
        engine = Engine(_TracingModel(), ByteTokenizer())
        with self.assertRaisesRegex(ValueError, "decode_mode must be one of"):
            engine.generate_batch(
                [261],
                max_tokens=1,
                decode_mode="unknown",
                use_calculator=False,
            )


if __name__ == "__main__":
    unittest.main()
