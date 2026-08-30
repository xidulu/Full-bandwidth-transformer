"""Focused checks for deterministic GSM8K evaluator sharding."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fbt_experiments.evaluate_checkpoint import evaluate_gsm8k, load_gsm8k_rows, parse_args


def _write_rows(path, count):
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            row = {"row": index, "context": "\n\nQ:".join(["Q:d"] * 8 + [f"target {index}"]), "answer": index}
            handle.write(json.dumps(row) + "\n")


def test_load_gsm8k_rows_selects_exact_slice_and_checks_bounds(tmp_path):
    dataset = tmp_path / "gsm8k.jsonl"
    _write_rows(dataset, 5)

    assert [row["row"] for row in load_gsm8k_rows(dataset, 2, start=2)] == [2, 3]
    assert [row["row"] for row in load_gsm8k_rows(dataset, 2)] == [0, 1]

    for count, start, message in ((1, -1, "non-negative"), (0, 0, "positive"), (2, 4, "[4:6]")):
        try:
            load_gsm8k_rows(dataset, count, start=start)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("Expected an invalid GSM8K range to fail")


def test_gsm8k_start_cli_defaults_to_zero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_checkpoint.py", "--checkpoint", "dummy.pt"])
    assert parse_args().gsm8k_start == 0

    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_checkpoint.py", "--checkpoint", "dummy.pt", "--gsm8k-start", "37"],
    )
    assert parse_args().gsm8k_start == 37


def test_evaluation_records_global_example_indices(tmp_path):
    dataset_dir = tmp_path / "eval_bundle" / "eval_data" / "symbolic_problem_solving"
    dataset_dir.mkdir(parents=True)
    _write_rows(dataset_dir / "gsm8k_prepended_8shot.jsonl", 5)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    class FakeTokenizer:
        def get_bos_token_id(self):
            return 0

        def encode(self, _text, prepend=None):
            return [prepend, 1] if prepend is not None else [1]

        def encode_special(self, _text):
            return 99

        def decode(self, _ids):
            return ""

    class FakeModel:
        config = SimpleNamespace(sequence_len=32)

        def get_device(self):
            return torch.device("cpu")

    class FakeEngine:
        def __init__(self, _model, _tokenizer):
            pass

        def generate(self, *_args, **_kwargs):
            def stream():
                yield [99], [1]

            return stream()

    args = SimpleNamespace(
        num_gsm8k=2,
        gsm8k_start=2,
        gsm8k_shots=0,
        max_new_tokens=2,
        seed=42,
    )
    with patch("nanochat.engine.Engine", FakeEngine):
        evaluate_gsm8k(FakeModel(), FakeTokenizer(), args, output_dir, tmp_path)

    records = [json.loads(line) for line in (output_dir / "gsm8k_generations.jsonl").read_text().splitlines()]
    assert [record["example_index"] for record in records] == [2, 3]

