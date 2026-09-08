#!/usr/bin/env python3
"""Regrade saved MATH-500 generations with Hugging Face Math-Verify.

This script does not rerun model inference. It reads ``math500_generations.jsonl``
from one or more completed MATH-500 result directories and writes:

- ``math500_generations_math_verify.jsonl``
- ``metrics_math_verify.json``
- ``summary_math_verify.md``
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from math_verify import LatexExtractionConfig, parse, verify

try:
    from .evaluate_checkpoint import dump_json, dump_jsonl, exact_mcnemar_p, wilson_interval
except ImportError:
    from evaluate_checkpoint import dump_json, dump_jsonl, exact_mcnemar_p, wilson_interval  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--parse-timeout", type=int, default=8)
    parser.add_argument("--verify-timeout", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Math-Verify regrade artifacts",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stringify_parsed(parsed: Any) -> list[str]:
    if not isinstance(parsed, list):
        parsed = [parsed]
    return [str(item) for item in parsed]


def parse_gold(reference_answer: str, timeout: int) -> list[Any]:
    # MATH-500 gold answers are clean LaTeX snippets, not necessarily wrapped in
    # a math environment. Math-Verify expects a LaTeX environment for reliable
    # extraction of tuples, sets, symbolic expressions, and text answers.
    return parse(
        f"${reference_answer}$",
        extraction_config=[LatexExtractionConfig()],
        parsing_timeout=timeout,
    )


def summarize(rows: list[dict[str, Any]], modes: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in modes:
        correct = sum(row["modes"][mode]["math_verify_correct"] for row in rows)
        parsed = sum(row["modes"][mode]["math_verify_answer_parsed"] for row in rows)
        seconds = sum(row["modes"][mode]["math_verify_seconds"] for row in rows)
        tokens = sum(row["modes"][mode].get("completion_tokens", 0) for row in rows)
        summary[mode] = {
            "examples": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
            "standard_error": math.sqrt((correct / len(rows)) * (1.0 - correct / len(rows)) / len(rows)),
            "answers_parsed": parsed,
            "answer_parse_rate": parsed / len(rows),
            "completion_tokens": tokens,
            "original_generation_seconds": sum(row["modes"][mode].get("seconds", 0.0) for row in rows),
            "math_verify_seconds": seconds,
            "accuracy_wilson_95": wilson_interval(correct, len(rows)),
        }

    summary["paired_accuracy"] = {}
    for left, right in (("standard", "soft"), ("standard", "fused"), ("soft", "fused")):
        if left not in modes or right not in modes:
            continue
        both_correct = sum(
            row["modes"][left]["math_verify_correct"] and row["modes"][right]["math_verify_correct"]
            for row in rows
        )
        left_only = sum(
            row["modes"][left]["math_verify_correct"] and not row["modes"][right]["math_verify_correct"]
            for row in rows
        )
        right_only = sum(
            row["modes"][right]["math_verify_correct"] and not row["modes"][left]["math_verify_correct"]
            for row in rows
        )
        summary["paired_accuracy"][f"{left}_vs_{right}"] = {
            "both_correct": both_correct,
            f"{left}_only_correct": left_only,
            f"{right}_only_correct": right_only,
            "neither_correct": len(rows) - both_correct - left_only - right_only,
            f"accuracy_delta_{right}_minus_{left}": (right_only - left_only) / len(rows),
            "exact_mcnemar_p": exact_mcnemar_p(left_only, right_only),
        }
    return summary


def render_summary(result_dir: Path, metrics: dict[str, Any]) -> str:
    lines = [
        "# MATH-500 Math-Verify regrade",
        "",
        f"Source result dir: `{result_dir}`",
        f"Checkpoint: `{metrics['checkpoint']}`",
        "",
        "| mode | correct | accuracy | SE | 95% Wilson CI | parsed | verifier seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in metrics["modes"]:
        row = metrics["math500_math_verify"][mode]
        ci_low, ci_high = row["accuracy_wilson_95"]
        lines.append(
            f"| {mode} | {row['correct']}/{row['examples']} | {row['accuracy']:.1%} | "
            f"{row['standard_error']:.2%} | {ci_low:.1%}–{ci_high:.1%} | "
            f"{row['answer_parse_rate']:.1%} | {row['math_verify_seconds']:.1f} |"
        )
    paired = metrics["math500_math_verify"].get("paired_accuracy", {})
    if paired:
        lines.extend(["", "Paired exact McNemar p-values:"])
        for name, row in paired.items():
            delta_key = next(key for key in row if key.startswith("accuracy_delta_"))
            lines.append(f"- {name}: delta={row[delta_key]:+.3%}, p={row['exact_mcnemar_p']:.4g}")
    lines.extend(
        [
            "",
            "Verifier: `math-verify` using LaTeX gold extraction and default prediction extraction.",
            "This reuses saved model generations; no inference was rerun.",
            "",
        ]
    )
    return "\n".join(lines)


def regrade_dir(result_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    result_dir = result_dir.expanduser().resolve()
    run_config = read_json(result_dir / "run_config.json")
    old_metrics = read_json(result_dir / "metrics.json")
    rows = read_jsonl(result_dir / "math500_generations.jsonl")
    modes = tuple(run_config["modes"])

    outputs = [
        result_dir / "math500_generations_math_verify.jsonl",
        result_dir / "metrics_math_verify.json",
        result_dir / "summary_math_verify.md",
    ]
    if not args.force:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                f"{result_dir} already has Math-Verify artifacts: {existing}. Use --force to overwrite."
            )

    gold_cache: dict[str, list[Any]] = {}
    regraded_rows = []
    for row_index, row in enumerate(rows, start=1):
        reference = row["reference_answer"]
        if reference not in gold_cache:
            gold_cache[reference] = parse_gold(reference, args.parse_timeout)
        gold = gold_cache[reference]
        row = dict(row)
        row["math_verify_gold_parsed"] = stringify_parsed(gold)
        row["math_verify_gold_parse_ok"] = bool(gold)
        row["modes"] = {mode: dict(mode_row) for mode, mode_row in row["modes"].items()}
        for mode in modes:
            mode_row = row["modes"][mode]
            completion = mode_row["completion"]
            started = time.perf_counter()
            try:
                prediction = parse(completion, parsing_timeout=args.parse_timeout)
                correct = bool(
                    gold
                    and prediction
                    and verify(
                        gold,
                        prediction,
                        timeout_seconds=args.verify_timeout,
                        raise_on_error=False,
                    )
                )
                error = None
            except Exception as exc:  # Math-Verify should usually fail closed.
                prediction = []
                correct = False
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            mode_row["math_verify_prediction_parsed"] = stringify_parsed(prediction)
            mode_row["math_verify_answer_parsed"] = bool(prediction)
            mode_row["math_verify_correct"] = correct
            mode_row["math_verify_seconds"] = elapsed
            mode_row["math_verify_error"] = error
        regraded_rows.append(row)
        if row_index % 25 == 0 or row_index == len(rows):
            print(f"{result_dir.name}: regraded {row_index}/{len(rows)}", flush=True)

    metrics = {
        "checkpoint": old_metrics["checkpoint"],
        "step": old_metrics["step"],
        "modes": list(modes),
        "source_metrics_path": str(result_dir / "metrics.json"),
        "math_verify": {
            "package": "math-verify",
            "parse_timeout": args.parse_timeout,
            "verify_timeout": args.verify_timeout,
            "gold_extraction": "LatexExtractionConfig on $reference_answer$",
            "prediction_extraction": "default parse(completion)",
        },
        "math500_math_verify": summarize(regraded_rows, modes),
    }
    dump_jsonl(outputs[0], regraded_rows)
    dump_json(outputs[1], metrics)
    with outputs[2].open("w", encoding="utf-8") as handle:
        handle.write(render_summary(result_dir, metrics))
    return metrics


def main() -> None:
    args = parse_args()
    for result_dir in args.result_dirs:
        metrics = regrade_dir(result_dir, args)
        compact = {
            mode: {
                "correct": metrics["math500_math_verify"][mode]["correct"],
                "accuracy": metrics["math500_math_verify"][mode]["accuracy"],
            }
            for mode in metrics["modes"]
        }
        print(json.dumps({"result_dir": str(result_dir), "results": compact}, indent=2))


if __name__ == "__main__":
    main()
