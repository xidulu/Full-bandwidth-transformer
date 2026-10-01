import json
from types import SimpleNamespace

from vllm_plugin.evaluate_gsm8k import merge_modes


def _row(index, mode, *, correct, token_ids):
    return {
        "example_index": index,
        "prompt": f"prompt {index}",
        "prompt_token_ids": [1, index + 2],
        "prompt_tokens": 2,
        "reference_answer": str(index),
        "mode": mode,
        "completion": "answer",
        "raw_completion_through_stop": "answer",
        "completion_token_ids": token_ids,
        "completion_tokens": len(token_ids),
        "predicted_answer": str(index) if correct else "wrong",
        "parse_method": "test",
        "answer_parsed": True,
        "correct": correct,
        "stop_reason": "length",
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_merge_modes_computes_paired_counts(tmp_path):
    standard_path = tmp_path / "standard.jsonl"
    soft_path = tmp_path / "soft.jsonl"
    _write_jsonl(
        standard_path,
        [
            _row(0, "standard", correct=True, token_ids=[7, 8]),
            _row(1, "standard", correct=False, token_ids=[9]),
        ],
    )
    _write_jsonl(
        soft_path,
        [
            _row(0, "soft", correct=False, token_ids=[7, 10]),
            _row(1, "soft", correct=True, token_ids=[9]),
        ],
    )
    output_dir = tmp_path / "merged"

    merge_modes(
        SimpleNamespace(
            standard=standard_path,
            soft=soft_path,
            output_dir=output_dir,
            native_reference=None,
            expected_start=0,
            expected_count=2,
        )
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["standard"]["correct"] == 1
    assert metrics["soft"]["correct"] == 1
    assert metrics["pairwise"]["standard_soft_same_first_token"] == 1.0
    assert metrics["pairwise"]["standard_soft_identical"] == 0.5
    assert metrics["paired_accuracy"] == {
        "both_correct": 0,
        "standard_only_correct": 1,
        "soft_only_correct": 1,
        "neither_correct": 0,
        "accuracy_delta_soft_minus_standard": 0.0,
        "exact_mcnemar_p": 1.0,
    }
