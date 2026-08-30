#!/usr/bin/env python3
"""Validate and merge disjoint GSM8K evaluation shards.

The input directories must be completed outputs from evaluate_checkpoint.py.
The merger trusts raw generation rows rather than shard-level aggregate metrics:
it validates every row, sorts by the global ``example_index``, and recomputes
the full metrics and Markdown report. The destination must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

try:
    # Package import (tests and `python -m fbt_experiments.merge_gsm8k_shards`).
    from .evaluate_checkpoint import (
        MODES,
        build_gsm8k_prompt,
        dump_json,
        dump_jsonl,
        exact_mcnemar_p,
        extract_gsm8k_answer,
        normalize_number,
        render_summary,
        wilson_interval,
    )
except ImportError:
    # Direct `python fbt_experiments/merge_gsm8k_shards.py` execution.
    from evaluate_checkpoint import (  # type: ignore[no-redef]
        MODES,
        build_gsm8k_prompt,
        dump_json,
        dump_jsonl,
        exact_mcnemar_p,
        extract_gsm8k_answer,
        normalize_number,
        render_summary,
        wilson_interval,
    )


REQUIRED_FILES = (
    "run_config.json",
    "checkpoint_meta.json",
    "metrics.json",
    "gsm8k_generations.jsonl",
)
PROTOCOL_KEYS = (
    "checkpoint",
    "checkpoint_size_bytes",
    "nanochat_base_dir",
    "step",
    "seed",
    "gsm8k_shots",
    "max_new_tokens",
    "modes",
)
MODE_REQUIRED_KEYS = (
    "completion",
    "raw_completion_through_stop",
    "completion_token_ids",
    "completion_tokens",
    "sampled_tokens",
    "predicted_answer",
    "parse_method",
    "answer_parsed",
    "correct",
    "seconds",
    "tokens_per_second",
    "stop_reason",
)
PAIRWISE_KEYS = (
    "standard_soft_same_first_token",
    "standard_soft_identical",
    "standard_fused_identical",
    "soft_fused_identical",
)
GSM8K_RELATIVE_PATH = Path(
    "eval_bundle/eval_data/symbolic_problem_solving/gsm8k_prepended_8shot.jsonl"
)
DECODE_PROTOCOL = {
    "num_samples": 1,
    "temperature": 0.0,
    "top_k": None,
    "use_calculator": False,
    "modes": list(MODES),
}


class MergeError(ValueError):
    """Raised when shard inputs are incomplete or incompatible."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "shard_dirs",
        nargs="+",
        type=Path,
        help="Two or more completed shard result directories",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-start",
        type=int,
        default=0,
        help="Required first global example index (default: 0)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Required total row count; strongly recommended for a full benchmark merge",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MergeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise MergeError(f"Expected a JSON object at {path}:{line_number}")
                rows.append(row)
    except OSError as exc:
        raise MergeError(f"Could not read {path}: {exc}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MergeError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _validate_mode_row(mode_row: Any, label: str, reference: str | None) -> None:
    if not isinstance(mode_row, dict):
        raise MergeError(f"{label} must be an object")
    missing = [key for key in MODE_REQUIRED_KEYS if key not in mode_row]
    if missing:
        raise MergeError(f"{label} is missing fields: {', '.join(missing)}")

    token_ids = mode_row["completion_token_ids"]
    if not isinstance(token_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in token_ids
    ):
        raise MergeError(f"{label}.completion_token_ids must be a list of non-negative integers")
    completion_tokens = _require_nonnegative_int(
        mode_row["completion_tokens"], f"{label}.completion_tokens"
    )
    if completion_tokens != len(token_ids):
        raise MergeError(
            f"{label}.completion_tokens={completion_tokens} does not match "
            f"the {len(token_ids)} stored token IDs"
        )
    sampled_tokens = _require_nonnegative_int(mode_row["sampled_tokens"], f"{label}.sampled_tokens")
    if sampled_tokens > completion_tokens:
        raise MergeError(
            f"{label}.sampled_tokens={sampled_tokens} exceeds completion_tokens={completion_tokens}"
        )

    completion = mode_row["completion"]
    raw_completion = mode_row["raw_completion_through_stop"]
    if not isinstance(completion, str) or not isinstance(raw_completion, str):
        raise MergeError(f"{label}.completion and raw_completion_through_stop must be strings")
    if raw_completion.split("\n\nQ:", 1)[0] != completion:
        raise MergeError(f"{label}.completion is inconsistent with raw_completion_through_stop")

    seconds = mode_row["seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise MergeError(f"{label}.seconds must be numeric")
    if not math.isfinite(seconds) or seconds < 0:
        raise MergeError(f"{label}.seconds must be finite and non-negative, got {seconds!r}")

    predicted = mode_row["predicted_answer"]
    if predicted is not None and not isinstance(predicted, str):
        raise MergeError(f"{label}.predicted_answer must be a string or null")
    for key in ("answer_parsed", "correct"):
        if not isinstance(mode_row[key], bool):
            raise MergeError(f"{label}.{key} must be boolean")
    if mode_row["answer_parsed"] != (predicted is not None):
        raise MergeError(f"{label}.answer_parsed is inconsistent with predicted_answer")
    if mode_row["correct"] != (predicted == reference):
        raise MergeError(f"{label}.correct is inconsistent with predicted/reference answers")
    extracted_answer, extracted_method = extract_gsm8k_answer(completion)
    if (predicted, mode_row["parse_method"]) != (extracted_answer, extracted_method):
        raise MergeError(
            f"{label}: stored predicted_answer/parse_method do not match the evaluator's parser"
        )

    tokens_per_second = mode_row["tokens_per_second"]
    expected_rate = completion_tokens / seconds if seconds > 0 else None
    if expected_rate is None:
        if tokens_per_second is not None:
            raise MergeError(f"{label}.tokens_per_second must be null when seconds is zero")
    elif (
        isinstance(tokens_per_second, bool)
        or not isinstance(tokens_per_second, (int, float))
        or not math.isfinite(tokens_per_second)
        or not math.isclose(tokens_per_second, expected_rate, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise MergeError(
            f"{label}.tokens_per_second={tokens_per_second!r}, expected {expected_rate!r}"
        )
    if not isinstance(mode_row["stop_reason"], str):
        raise MergeError(f"{label}.stop_reason must be a string")


def _expected_pairwise(row: dict[str, Any]) -> dict[str, bool]:
    tokens = {mode: row["modes"][mode]["completion_token_ids"] for mode in MODES}
    return {
        "standard_soft_same_first_token": bool(
            tokens["standard"]
            and tokens["soft"]
            and tokens["standard"][0] == tokens["soft"][0]
        ),
        "standard_soft_identical": tokens["standard"] == tokens["soft"],
        "standard_fused_identical": tokens["standard"] == tokens["fused"],
        "soft_fused_identical": tokens["soft"] == tokens["fused"],
    }


def _validate_record(row: dict[str, Any], expected_index: int, shots: int, source: Path) -> None:
    label = f"{source}/gsm8k_generations.jsonl example {expected_index}"
    if type(row.get("example_index")) is not int or row["example_index"] != expected_index:
        raise MergeError(
            f"{label}: expected global example_index {expected_index}, "
            f"got {row.get('example_index')!r}"
        )
    if row.get("gsm8k_shots") != shots:
        raise MergeError(
            f"{label}: gsm8k_shots={row.get('gsm8k_shots')!r}, expected {shots}"
        )
    if not isinstance(row.get("prompt"), str):
        raise MergeError(f"{label}: prompt must be a string")
    _require_nonnegative_int(row.get("prompt_tokens"), f"{label}: prompt_tokens")
    reference = row.get("reference_answer")
    if not isinstance(reference, str):
        raise MergeError(f"{label}: reference_answer must be a string")
    modes = row.get("modes")
    if not isinstance(modes, dict) or set(modes) != set(MODES):
        raise MergeError(f"{label}: modes must be exactly {list(MODES)}")
    for mode in MODES:
        _validate_mode_row(modes[mode], f"{label}.modes.{mode}", reference)

    pairwise = row.get("pairwise")
    if not isinstance(pairwise, dict) or set(pairwise) != set(PAIRWISE_KEYS):
        raise MergeError(f"{label}: pairwise fields must be exactly {list(PAIRWISE_KEYS)}")
    expected_pairwise = _expected_pairwise(row)
    if pairwise != expected_pairwise:
        raise MergeError(
            f"{label}: stored pairwise flags do not match token IDs; "
            f"expected {expected_pairwise}, got {pairwise}"
        )


def summarize_gsm8k_records(
    records: list[dict[str, Any]], num_shots: int
) -> dict[str, Any]:
    """Recompute the evaluator's GSM8K aggregate schema from raw records."""
    if not records:
        raise MergeError("Cannot summarize zero GSM8K records")
    summary: dict[str, Any] = {}
    for mode in MODES:
        seconds = sum(record["modes"][mode]["seconds"] for record in records)
        tokens = sum(record["modes"][mode]["completion_tokens"] for record in records)
        correct = sum(record["modes"][mode]["correct"] for record in records)
        parsed = sum(record["modes"][mode]["answer_parsed"] for record in records)
        summary[mode] = {
            "examples": len(records),
            "correct": correct,
            "accuracy": correct / len(records),
            "answers_parsed": parsed,
            "answer_parse_rate": parsed / len(records),
            "completion_tokens": tokens,
            "seconds": seconds,
            "tokens_per_second": tokens / seconds if seconds > 0 else None,
            "accuracy_wilson_95": wilson_interval(correct, len(records)),
        }

    summary["pairwise"] = {
        key: sum(record["pairwise"][key] for record in records) / len(records)
        for key in PAIRWISE_KEYS
    }
    summary["paired_accuracy"] = {}
    for left, right in (("standard", "soft"), ("standard", "fused"), ("soft", "fused")):
        both_correct = sum(
            record["modes"][left]["correct"] and record["modes"][right]["correct"]
            for record in records
        )
        left_only = sum(
            record["modes"][left]["correct"] and not record["modes"][right]["correct"]
            for record in records
        )
        right_only = sum(
            record["modes"][right]["correct"] and not record["modes"][left]["correct"]
            for record in records
        )
        summary["paired_accuracy"][f"{left}_vs_{right}"] = {
            "both_correct": both_correct,
            f"{left}_only_correct": left_only,
            f"{right}_only_correct": right_only,
            "neither_correct": len(records) - both_correct - left_only - right_only,
            f"accuracy_delta_{right}_minus_{left}": (right_only - left_only) / len(records),
            "exact_mcnemar_p": exact_mcnemar_p(left_only, right_only),
        }
    summary["num_shots"] = num_shots
    return summary


def _json_values_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality that does not conflate booleans with integers."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _assert_same(label: str, expected: Any, actual: Any, source: Path) -> None:
    if not _json_values_equal(expected, actual):
        raise MergeError(
            f"Shard {source} has incompatible {label}: expected {expected!r}, got {actual!r}"
        )


def _assert_metric_matches(expected: Any, actual: Any, path: str) -> None:
    """Compare stored and recomputed metrics, allowing only float summation noise."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise MergeError(f"Stored shard metric {path} has a different schema")
        for key in expected:
            _assert_metric_matches(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise MergeError(f"Stored shard metric {path} has a different list shape")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_metric_matches(expected_item, actual_item, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(
            expected, actual, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise MergeError(f"Stored shard metric {path} is {actual!r}; recomputed {expected!r}")
        return
    if type(expected) is not type(actual) or expected != actual:
        raise MergeError(f"Stored shard metric {path} is {actual!r}; recomputed {expected!r}")


def _load_shard(source: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise MergeError(f"Shard directory does not exist: {source}")
    missing = [filename for filename in REQUIRED_FILES if not (source / filename).is_file()]
    if missing:
        raise MergeError(f"Shard {source} is missing required files: {', '.join(missing)}")

    config = _read_json(source / "run_config.json")
    meta = _read_json(source / "checkpoint_meta.json")
    stored_metrics = _read_json(source / "metrics.json")
    rows = _read_jsonl(source / "gsm8k_generations.jsonl")

    for key in PROTOCOL_KEYS:
        if key not in config:
            raise MergeError(f"Shard {source} run_config.json is missing {key!r}")
    if not isinstance(config["checkpoint"], str) or not config["checkpoint"]:
        raise MergeError(f"Shard {source} checkpoint must be a non-empty string")
    _require_nonnegative_int(config["checkpoint_size_bytes"], f"{source}: checkpoint_size_bytes")
    _require_nonnegative_int(config["step"], f"{source}: step")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise MergeError(f"{source}: seed must be an integer")
    max_new_tokens = _require_nonnegative_int(
        config["max_new_tokens"], f"{source}: max_new_tokens"
    )
    if max_new_tokens == 0:
        raise MergeError(f"{source}: max_new_tokens must be positive")
    start = _require_nonnegative_int(config.get("gsm8k_start", 0), f"{source}: gsm8k_start")
    count = _require_nonnegative_int(config.get("num_gsm8k"), f"{source}: num_gsm8k")
    shots = _require_nonnegative_int(config["gsm8k_shots"], f"{source}: gsm8k_shots")
    if not 0 <= shots <= 8:
        raise MergeError(f"{source}: gsm8k_shots must be in [0, 8]")
    if config.get("skip_gsm8k") is not False:
        raise MergeError(f"Shard {source} did not run GSM8K (skip_gsm8k must be false)")
    if config["modes"] != list(MODES):
        raise MergeError(f"Shard {source} modes must be exactly {list(MODES)}")
    if len(rows) != count:
        raise MergeError(
            f"Shard {source} contains {len(rows)} complete rows but run_config requests {count}; "
            "the shard may be incomplete"
        )
    if count == 0:
        raise MergeError(f"Shard {source} contains no GSM8K rows")
    for offset, row in enumerate(rows):
        _validate_record(row, start + offset, shots, source)

    if meta.get("step") != config["step"]:
        raise MergeError(
            f"Shard {source} checkpoint metadata step {meta.get('step')!r} "
            f"does not match run_config step {config['step']!r}"
        )
    if stored_metrics.get("checkpoint") != config["checkpoint"]:
        raise MergeError(f"Shard {source} metrics checkpoint does not match run_config")
    if stored_metrics.get("step") != config["step"]:
        raise MergeError(f"Shard {source} metrics step does not match run_config")
    if stored_metrics.get("modes") != list(MODES):
        raise MergeError(f"Shard {source} metrics modes do not match the evaluator modes")
    if "gsm8k" not in stored_metrics:
        raise MergeError(f"Shard {source} metrics.json has no GSM8K aggregate")
    recomputed = summarize_gsm8k_records(rows, shots)
    _assert_metric_matches(recomputed, stored_metrics["gsm8k"], f"{source}/metrics.json.gsm8k")

    return {
        "path": source,
        "config": config,
        "meta": meta,
        "rows": rows,
        "start": start,
        "count": count,
    }


def _validate_source_dataset(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Tie merged prompts and references back to the canonical source file."""
    base_dir_value = config.get("nanochat_base_dir")
    if not isinstance(base_dir_value, str) or not base_dir_value:
        raise MergeError("run_config.nanochat_base_dir must be a non-empty string")
    dataset_path = Path(base_dir_value).expanduser().resolve() / GSM8K_RELATIVE_PATH
    if not dataset_path.is_file():
        raise MergeError(f"Canonical GSM8K source dataset is unavailable: {dataset_path}")
    source_rows = _read_jsonl(dataset_path)
    shots = config["gsm8k_shots"]
    for record in records:
        index = record["example_index"]
        if index >= len(source_rows):
            raise MergeError(
                f"Merged example_index {index} exceeds the {len(source_rows)}-row source dataset"
            )
        source_row = source_rows[index]
        if not isinstance(source_row.get("context"), str) or "answer" not in source_row:
            raise MergeError(f"Malformed canonical GSM8K source row {index} in {dataset_path}")
        expected_prompt = build_gsm8k_prompt(source_row["context"], shots) + "\n\nA:"
        expected_reference = normalize_number(str(source_row["answer"]))
        if record["prompt"] != expected_prompt:
            raise MergeError(
                f"Merged example {index} prompt does not match canonical source {dataset_path}"
            )
        if record["reference_answer"] != expected_reference:
            raise MergeError(
                f"Merged example {index} reference {record['reference_answer']!r} does not match "
                f"canonical source value {expected_reference!r}"
            )
    return {
        "path": str(dataset_path),
        "sha256": _sha256(dataset_path),
        "size_bytes": dataset_path.stat().st_size,
        "rows": len(source_rows),
    }


def _source_manifest(shard: dict[str, Any]) -> dict[str, Any]:
    source = shard["path"]
    return {
        "directory": str(source),
        "gsm8k_start": shard["start"],
        "num_gsm8k": shard["count"],
        "first_example_index": shard["rows"][0]["example_index"],
        "last_example_index": shard["rows"][-1]["example_index"],
        "files": {
            filename: {
                "path": str(source / filename),
                "sha256": _sha256(source / filename),
            }
            for filename in REQUIRED_FILES
        },
    }


def _merged_run_config(
    first_config: dict[str, Any],
    shards: list[dict[str, Any]],
    start: int,
    count: int,
    command: list[str],
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "command": command,
        "merged_from_shards": True,
        "source_shards": [str(shard["path"]) for shard in shards],
    }
    for key in (
        "checkpoint",
        "checkpoint_size_bytes",
        "matching_meta",
        "nanochat_base_dir",
        "step",
        "seed",
        "gsm8k_shots",
        "max_new_tokens",
        "modes",
    ):
        if key in first_config:
            config[key] = first_config[key]
    config.update(
        {
            "gsm8k_start": start,
            "num_gsm8k": count,
            "skip_continuation": True,
            "skip_gsm8k": False,
            "source_execution_environments": [
                {
                    key: shard["config"].get(key)
                    for key in (
                        "device",
                        "torch_version",
                        "compute_dtype",
                        "compute_dtype_reason",
                        "cuda_device",
                    )
                }
                for shard in shards
            ],
        }
    )
    return config


def merge_shards(
    shard_dirs: Iterable[Path],
    output_dir: Path,
    *,
    expected_start: int = 0,
    expected_count: int | None = None,
    command: list[str] | None = None,
) -> Path:
    """Merge shards and return the resolved destination path."""
    expected_start = _require_nonnegative_int(expected_start, "expected_start")
    if expected_count is not None:
        expected_count = _require_nonnegative_int(expected_count, "expected_count")
        if expected_count == 0:
            raise MergeError("expected_count must be positive")

    sources = [Path(path).expanduser().resolve() for path in shard_dirs]
    if len(sources) < 2:
        raise MergeError("At least two shard directories are required")
    if len(set(sources)) != len(sources):
        raise MergeError("Each shard directory must be supplied exactly once")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise MergeError(f"Refusing to overwrite existing output path: {output_dir}")
    if output_dir in sources:
        raise MergeError("The output directory cannot also be an input shard")

    shards = [_load_shard(source) for source in sources]
    shards.sort(key=lambda shard: shard["start"])
    first = shards[0]
    for shard in shards[1:]:
        for key in PROTOCOL_KEYS:
            _assert_same(
                f"run_config.{key}", first["config"][key], shard["config"][key], shard["path"]
            )
        _assert_same("checkpoint_meta.json", first["meta"], shard["meta"], shard["path"])

    rows = sorted(
        (row for shard in shards for row in shard["rows"]),
        key=lambda row: row["example_index"],
    )
    actual_indices = [row["example_index"] for row in rows]
    expected_indices = list(range(expected_start, expected_start + len(rows)))
    if actual_indices != expected_indices:
        missing = sorted(set(expected_indices) - set(actual_indices))
        duplicates = sorted(index for index, count in Counter(actual_indices).items() if count > 1)
        unexpected = sorted(set(actual_indices) - set(expected_indices))
        raise MergeError(
            "Shard indices are not one unique contiguous range beginning at "
            f"{expected_start}; missing={missing[:10]}, duplicates={duplicates[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    if expected_count is not None and len(rows) != expected_count:
        raise MergeError(f"Expected {expected_count} merged rows, found {len(rows)}")

    shots = first["config"]["gsm8k_shots"]
    source_dataset = _validate_source_dataset(first["config"], rows)
    gsm8k_metrics = summarize_gsm8k_records(rows, shots)
    metrics = {
        "checkpoint": first["config"]["checkpoint"],
        "step": first["config"]["step"],
        "modes": list(MODES),
        "gsm8k": gsm8k_metrics,
    }
    invoked_command = command or [sys.executable, *sys.argv]
    run_config = _merged_run_config(
        first["config"], shards, expected_start, len(rows), invoked_command
    )
    run_config["source_dataset"] = source_dataset
    run_config["decoding"] = DECODE_PROTOCOL
    summary_args = SimpleNamespace(
        checkpoint=Path(first["config"]["checkpoint"]),
        gsm8k_shots=shots,
    )
    summary = render_summary(metrics, first["meta"], summary_args)
    summary += (
        "\n## Merge provenance\n\n"
        f"This report was recomputed from {len(shards)} validated, disjoint shards "
        f"covering global examples {expected_start}–{expected_start + len(rows) - 1}. "
        "The per-mode `seconds` totals are summed GPU-seconds, not parallel wall time. "
        "The three reported McNemar p-values are unadjusted for multiple comparisons. "
        "See `merge_manifest.json` for source paths and SHA-256 hashes.\n"
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.merge-", dir=output_dir.parent))
    try:
        dump_jsonl(temporary / "gsm8k_generations.jsonl", rows)
        dump_json(temporary / "metrics.json", metrics)
        dump_json(temporary / "run_config.json", run_config)
        dump_json(temporary / "checkpoint_meta.json", first["meta"])
        (temporary / "summary.md").write_text(summary, encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": invoked_command,
            "checkpoint": first["config"]["checkpoint"],
            "step": first["config"]["step"],
            "protocol": {key: first["config"][key] for key in PROTOCOL_KEYS},
            "decoding": DECODE_PROTOCOL,
            "source_dataset": source_dataset,
            "gsm8k_start": expected_start,
            "num_gsm8k": len(rows),
            "first_example_index": rows[0]["example_index"],
            "last_example_index": rows[-1]["example_index"],
            "sources": [_source_manifest(shard) for shard in shards],
            "outputs": {
                filename: {"sha256": _sha256(temporary / filename)}
                for filename in (
                    "run_config.json",
                    "checkpoint_meta.json",
                    "gsm8k_generations.jsonl",
                    "metrics.json",
                    "summary.md",
                )
            },
        }
        dump_json(temporary / "merge_manifest.json", manifest)
        # Recheck immediately before rename. On POSIX, renaming a directory onto
        # a non-empty directory also fails, but this explicitly protects empty
        # destinations and symlinks created during the validation window.
        if output_dir.exists() or output_dir.is_symlink():
            raise MergeError(
                f"Refusing to overwrite output path created during merge: {output_dir}"
            )
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir


def main() -> None:
    args = parse_args()
    try:
        output = merge_shards(
            args.shard_dirs,
            args.output_dir,
            expected_start=args.expected_start,
            expected_count=args.expected_count,
        )
    except MergeError as exc:
        raise SystemExit(f"merge_gsm8k_shards: error: {exc}") from exc
    print(f"wrote merged evaluation assets to {output}")


if __name__ == "__main__":
    main()
