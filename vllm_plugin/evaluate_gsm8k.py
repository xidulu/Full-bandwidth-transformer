#!/usr/bin/env python3
"""Run and merge vLLM standard/soft GSM8K evaluations for Nanochat."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fbt_experiments.evaluate_checkpoint import (  # noqa: E402
    build_gsm8k_chat_prompt_ids,
    exact_mcnemar_p,
    extract_gsm8k_answer,
    load_gsm8k_rows,
    normalize_number,
    wilson_interval,
)


MODES = ("standard", "soft")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_meta_path(checkpoint: Path) -> Path:
    if not checkpoint.name.startswith("model_") or checkpoint.suffix != ".pt":
        raise ValueError("checkpoint must be named model_<step>.pt")
    return checkpoint.with_name(
        f"meta_{checkpoint.stem.removeprefix('model_')}.json"
    )


def _validate_linear_addition(checkpoint: Path, model_dir: Path, mode: str) -> None:
    meta = _read_json(_checkpoint_meta_path(checkpoint))
    model_config = meta["model_config"]
    if not model_config.get("latent_feedback"):
        raise ValueError("checkpoint does not enable latent feedback")
    if model_config.get("latent_feedback_mode") != "linear_addition":
        raise ValueError(
            "expected latent_feedback_mode='linear_addition', got "
            f"{model_config.get('latent_feedback_mode')!r}"
        )
    exported = _read_json(model_dir / "config.json")
    if exported.get("nanochat_decode_mode") != mode:
        raise ValueError(
            f"exported model mode is {exported.get('nanochat_decode_mode')!r}, "
            f"expected {mode!r}"
        )
    if exported.get("nanochat_latent_feedback_mode") != "linear_addition":
        raise ValueError("exported model is not linear_addition")


def _truncate_next_question(tokenizer, token_ids: list[int]) -> tuple[list[int], str]:
    for end in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(token_ids[:end])
        if "\n\nQ:" in decoded:
            return token_ids[:end], "next_question"
    return token_ids, "vllm_stop_or_max_tokens"


def run_mode(args: argparse.Namespace) -> None:
    from nanochat.tokenizer import get_tokenizer
    from vllm import LLM, SamplingParams

    checkpoint = args.checkpoint.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f"{args.mode}_generations.jsonl"
    metrics_path = output_dir / f"{args.mode}_metrics.json"
    config_path = output_dir / f"{args.mode}_run_config.json"
    existing = [path for path in (rows_path, metrics_path, config_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite completed artifacts: "
            + ", ".join(path.name for path in existing)
        )

    _validate_linear_addition(checkpoint, model_dir, args.mode)
    base_dir = Path(args.nanochat_base_dir).expanduser().resolve()
    dataset_path = (
        base_dir
        / "eval_bundle/eval_data/symbolic_problem_solving/gsm8k_prepended_8shot.jsonl"
    )
    examples = load_gsm8k_rows(dataset_path, args.count, args.start)
    tokenizer = get_tokenizer()
    prompts = []
    prompt_texts = []
    for example in examples:
        prompt_ids, prompt_text = build_gsm8k_chat_prompt_ids(
            tokenizer, example["context"]
        )
        if len(prompt_ids) + args.max_new_tokens > 2048:
            raise ValueError("prompt plus generation exceeds the checkpoint context")
        prompts.append({"prompt_token_ids": prompt_ids})
        prompt_texts.append(prompt_text)

    llm = LLM(
        model=str(model_dir),
        skip_tokenizer_init=True,
        worker_cls="nanochat_vllm.worker.NanochatWorker",
        enforce_eager=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop_token_ids=[
            tokenizer.encode_special("<|assistant_end|>"),
            tokenizer.get_bos_token_id(),
        ],
    )
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    elapsed = time.perf_counter() - started
    if len(outputs) != len(examples):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(examples)} prompts")

    rows = []
    for offset, (example, prompt, prompt_text, request_output) in enumerate(
        zip(examples, prompts, prompt_texts, outputs, strict=True)
    ):
        generated = request_output.outputs[0]
        raw_ids = list(generated.token_ids)
        token_ids, local_stop_reason = _truncate_next_question(tokenizer, raw_ids)
        raw_completion = tokenizer.decode(token_ids)
        completion = raw_completion.split("\n\nQ:", 1)[0]
        predicted, parse_method = extract_gsm8k_answer(completion)
        reference = normalize_number(str(example["answer"]))
        rows.append(
            {
                "example_index": args.start + offset,
                "prompt": prompt_text,
                "prompt_token_ids": prompt["prompt_token_ids"],
                "prompt_tokens": len(prompt["prompt_token_ids"]),
                "reference_answer": reference,
                "mode": args.mode,
                "completion": completion,
                "raw_completion_through_stop": raw_completion,
                "completion_token_ids": token_ids,
                "completion_tokens": len(token_ids),
                "predicted_answer": predicted,
                "parse_method": parse_method,
                "answer_parsed": predicted is not None,
                "correct": predicted == reference,
                "stop_reason": (
                    local_stop_reason
                    if local_stop_reason == "next_question"
                    else generated.finish_reason
                ),
            }
        )

    correct = sum(row["correct"] for row in rows)
    parsed = sum(row["answer_parsed"] for row in rows)
    completion_tokens = sum(row["completion_tokens"] for row in rows)
    metrics = {
        "mode": args.mode,
        "examples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "accuracy_wilson_95": wilson_interval(correct, len(rows)),
        "answers_parsed": parsed,
        "answer_parse_rate": parsed / len(rows),
        "completion_tokens": completion_tokens,
        "wall_seconds": elapsed,
        "aggregate_tokens_per_second": completion_tokens / elapsed,
    }
    config = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "matching_meta": str(_checkpoint_meta_path(checkpoint)),
        "matching_meta_sha256": _sha256(_checkpoint_meta_path(checkpoint)),
        "model_dir": str(model_dir),
        "mode": args.mode,
        "start": args.start,
        "count": args.count,
        "gsm8k_shots": 0,
        "gsm8k_prompt_format": "chat",
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "top_k": None,
        "seed": args.seed,
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    _write_jsonl(rows_path, rows)
    _write_json(metrics_path, metrics)
    _write_json(config_path, config)
    print(json.dumps(metrics, indent=2), flush=True)


def _validate_mode_rows(
    rows: list[dict[str, Any]], mode: str
) -> dict[int, dict[str, Any]]:
    by_index = {}
    for row in rows:
        index = row.get("example_index")
        if not isinstance(index, int) or index in by_index:
            raise ValueError(f"invalid or duplicate {mode} example index: {index!r}")
        if row.get("mode") != mode:
            raise ValueError(f"row {index} has mode={row.get('mode')!r}, expected {mode}")
        by_index[index] = row
    return by_index


def _native_comparison(
    native_path: Path | None,
    combined: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if native_path is None:
        return None
    native_rows = {
        row["example_index"]: row for row in _read_jsonl(native_path.expanduser().resolve())
    }
    result = {}
    for mode in MODES:
        token_matches = 0
        correctness_matches = 0
        for row in combined:
            native_mode = native_rows[row["example_index"]]["modes"][mode]
            current_mode = row["modes"][mode]
            token_matches += (
                native_mode["completion_token_ids"]
                == current_mode["completion_token_ids"]
            )
            correctness_matches += native_mode["correct"] == current_mode["correct"]
        result[mode] = {
            "exact_completion_matches": token_matches,
            "exact_completion_match_rate": token_matches / len(combined),
            "correctness_agreements": correctness_matches,
            "correctness_agreement_rate": correctness_matches / len(combined),
        }
    return result


def merge_modes(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = (
        output_dir / "gsm8k_generations.jsonl",
        output_dir / "metrics.json",
        output_dir / "run_config.json",
        output_dir / "summary.md",
    )
    existing = [path for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite final artifacts: "
            + ", ".join(path.name for path in existing)
        )

    standard = _validate_mode_rows(_read_jsonl(args.standard), "standard")
    soft = _validate_mode_rows(_read_jsonl(args.soft), "soft")
    if set(standard) != set(soft):
        raise ValueError("standard and soft example indices differ")
    indices = sorted(standard)
    if indices != list(range(args.expected_start, args.expected_start + args.expected_count)):
        raise ValueError("merged rows do not cover the expected contiguous GSM8K range")

    combined = []
    for index in indices:
        left, right = standard[index], soft[index]
        for key in ("prompt_token_ids", "reference_answer"):
            if left[key] != right[key]:
                raise ValueError(f"mode mismatch for example {index}: {key}")
        left_tokens = left["completion_token_ids"]
        right_tokens = right["completion_token_ids"]
        combined.append(
            {
                "example_index": index,
                "prompt": left["prompt"],
                "prompt_token_ids": left["prompt_token_ids"],
                "prompt_tokens": left["prompt_tokens"],
                "reference_answer": left["reference_answer"],
                "modes": {"standard": left, "soft": right},
                "pairwise": {
                    "standard_soft_same_first_token": bool(
                        left_tokens
                        and right_tokens
                        and left_tokens[0] == right_tokens[0]
                    ),
                    "standard_soft_identical": left_tokens == right_tokens,
                },
            }
        )

    mode_metrics = {}
    for mode in MODES:
        correct = sum(row["modes"][mode]["correct"] for row in combined)
        parsed = sum(row["modes"][mode]["answer_parsed"] for row in combined)
        mode_metrics[mode] = {
            "examples": len(combined),
            "correct": correct,
            "accuracy": correct / len(combined),
            "accuracy_wilson_95": wilson_interval(correct, len(combined)),
            "answers_parsed": parsed,
            "answer_parse_rate": parsed / len(combined),
            "completion_tokens": sum(
                row["modes"][mode]["completion_tokens"] for row in combined
            ),
        }
    both = sum(
        row["modes"]["standard"]["correct"]
        and row["modes"]["soft"]["correct"]
        for row in combined
    )
    standard_only = sum(
        row["modes"]["standard"]["correct"]
        and not row["modes"]["soft"]["correct"]
        for row in combined
    )
    soft_only = sum(
        row["modes"]["soft"]["correct"]
        and not row["modes"]["standard"]["correct"]
        for row in combined
    )
    paired = {
        "both_correct": both,
        "standard_only_correct": standard_only,
        "soft_only_correct": soft_only,
        "neither_correct": len(combined) - both - standard_only - soft_only,
        "accuracy_delta_soft_minus_standard": (
            soft_only - standard_only
        ) / len(combined),
        "exact_mcnemar_p": exact_mcnemar_p(standard_only, soft_only),
    }
    metrics = {
        "checkpoint_feedback_mode": "linear_addition",
        "gsm8k_shots": 0,
        "gsm8k_prompt_format": "chat",
        "standard": mode_metrics["standard"],
        "soft": mode_metrics["soft"],
        "pairwise": {
            "standard_soft_same_first_token": sum(
                row["pairwise"]["standard_soft_same_first_token"] for row in combined
            )
            / len(combined),
            "standard_soft_identical": sum(
                row["pairwise"]["standard_soft_identical"] for row in combined
            )
            / len(combined),
        },
        "paired_accuracy": paired,
    }
    native = _native_comparison(args.native_reference, combined)
    if native is not None:
        metrics["native_engine_comparison"] = native

    lines = [
        "# Linear-addition FBT GSM8K with vLLM",
        "",
        f"Examples: {len(combined)}; zero-shot Nanochat chat prompt; greedy; max 192 tokens.",
        "",
        "| mode | correct | accuracy | parsed |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        row = mode_metrics[mode]
        lines.append(
            f"| {mode} | {row['correct']}/{row['examples']} | "
            f"{row['accuracy']:.3%} | {row['answer_parse_rate']:.3%} |"
        )
    lines.extend(
        [
            "",
            f"Paired standard-only/soft-only: {standard_only}/{soft_only}; "
            f"exact McNemar p={paired['exact_mcnemar_p']:.6g}.",
            "",
            f"First-token match: {metrics['pairwise']['standard_soft_same_first_token']:.3%}; "
            f"identical completions: {metrics['pairwise']['standard_soft_identical']:.3%}.",
            "",
        ]
    )
    _write_jsonl(final_paths[0], combined)
    _write_json(final_paths[1], metrics)
    _write_json(
        final_paths[2],
        {
            "standard_rows": str(args.standard.resolve()),
            "soft_rows": str(args.soft.resolve()),
            "native_reference": (
                str(args.native_reference.resolve()) if args.native_reference else None
            ),
            "expected_start": args.expected_start,
            "expected_count": args.expected_count,
        },
    )
    final_paths[3].write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--mode", choices=MODES, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--nanochat-base-dir", type=Path, required=True)
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--count", type=int, default=1319)
    run.add_argument("--max-new-tokens", type=int, default=192)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-num-seqs", type=int, default=256)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.8)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--standard", type=Path, required=True)
    merge.add_argument("--soft", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--native-reference", type=Path)
    merge.add_argument("--expected-start", type=int, default=0)
    merge.add_argument("--expected-count", type=int, default=1319)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run_mode(args)
    else:
        merge_modes(args)


if __name__ == "__main__":
    main()
