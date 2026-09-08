#!/usr/bin/env python3
"""Merge disjoint MATH-500 shard outputs from evaluate_math500.py."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from .evaluate_checkpoint import dump_json, dump_jsonl
    from .evaluate_math500 import render_summary, summarize_records
except ImportError:
    from evaluate_checkpoint import dump_json, dump_jsonl  # type: ignore[no-redef]
    from evaluate_math500 import render_summary, summarize_records  # type: ignore[no-redef]


class MergeError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-start", type=int, default=0)
    parser.add_argument("--expected-count", type=int, default=500)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise MergeError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise MergeError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def load_shard(path: Path) -> dict[str, Any]:
    for name in ("run_config.json", "checkpoint_meta.json", "metrics.json", "math500_generations.jsonl"):
        if not (path / name).is_file():
            raise MergeError(f"{path} missing {name}")
    config = read_json(path / "run_config.json")
    metrics = read_json(path / "metrics.json")
    rows = read_jsonl(path / "math500_generations.jsonl")
    modes = tuple(config["modes"])
    if tuple(metrics["modes"]) != modes:
        raise MergeError(f"{path} modes mismatch between run_config and metrics")
    if len(rows) != config["num_math500"]:
        raise MergeError(f"{path} contains {len(rows)} rows, expected {config['num_math500']}")
    expected_indices = list(range(config["math500_start"], config["math500_start"] + config["num_math500"]))
    actual_indices = [row.get("example_index") for row in rows]
    if actual_indices != expected_indices:
        raise MergeError(f"{path} row indices are not contiguous as configured")
    for row in rows:
        row_modes = tuple(row["modes"].keys())
        if row_modes != modes:
            raise MergeError(f"{path} row {row.get('example_index')} modes {row_modes} != {modes}")
    return {"path": path, "config": config, "metrics": metrics, "rows": rows, "modes": modes}


def assert_same(label: str, left: Any, right: Any, path: Path) -> None:
    if left != right:
        raise MergeError(f"{label} mismatch for {path}: {right!r} != {left!r}")


def merge_shards(args: argparse.Namespace) -> None:
    shards = [load_shard(path) for path in args.shard_dirs]
    first = shards[0]
    for shard in shards[1:]:
        for key in (
            "checkpoint",
            "checkpoint_size_bytes",
            "matching_meta",
            "nanochat_base_dir",
            "step",
            "seed",
            "math500_prompt_format",
            "math500_shots",
            "max_new_tokens",
            "modes",
        ):
            assert_same(f"run_config.{key}", first["config"][key], shard["config"][key], shard["path"])

    rows = [row for shard in shards for row in shard["rows"]]
    rows.sort(key=lambda row: row["example_index"])
    expected_indices = list(range(args.expected_start, args.expected_start + args.expected_count))
    actual_indices = [row["example_index"] for row in rows]
    if actual_indices != expected_indices:
        raise MergeError("Merged rows do not match expected contiguous MATH500 range")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise MergeError(f"Output directory already exists: {output_dir}")

    run_config = dict(first["config"])
    run_config["math500_start"] = args.expected_start
    run_config["num_math500"] = args.expected_count
    run_config["source_shards"] = [str(shard["path"]) for shard in shards]
    metrics = {
        "checkpoint": first["metrics"]["checkpoint"],
        "step": first["metrics"]["step"],
        "modes": list(first["modes"]),
        "math500": summarize_records(rows, first["modes"]),
    }
    summary_args = SimpleNamespace(
        checkpoint=run_config["checkpoint"],
        math500_start=args.expected_start,
        num_math500=args.expected_count,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as tmp_name:
        tmp = Path(tmp_name)
        shutil.copyfile(first["path"] / "checkpoint_meta.json", tmp / "checkpoint_meta.json")
        dump_json(tmp / "run_config.json", run_config)
        dump_json(tmp / "metrics.json", metrics)
        dump_jsonl(tmp / "math500_generations.jsonl", rows)
        with (tmp / "summary.md").open("w", encoding="utf-8") as handle:
            handle.write(render_summary(metrics, summary_args))
        tmp.rename(output_dir)
    print(f"merged {len(rows)} rows into {output_dir}")


def main() -> None:
    merge_shards(parse_args())


if __name__ == "__main__":
    main()
