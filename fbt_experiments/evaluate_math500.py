#!/usr/bin/env python3
"""Evaluate one checkpoint on zero-shot MATH-500 with chat-template prompts."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch

try:
    from .evaluate_checkpoint import (
        MODES,
        checkpoint_paths,
        dump_json,
        exact_mcnemar_p,
        parse_decode_modes,
        synchronize,
        wilson_interval,
    )
except ImportError:
    from evaluate_checkpoint import (  # type: ignore[no-redef]
        MODES,
        checkpoint_paths,
        dump_json,
        exact_mcnemar_p,
        parse_decode_modes,
        synchronize,
        wilson_interval,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BOXED_RE = re.compile(r"\\boxed\s*{")
NUMBER_RE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:/[1-9]\d*)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--math500-start", type=int, default=0)
    parser.add_argument("--num-math500", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--decode-modes",
        default="standard,soft,fused",
        help="Comma-separated modes from: standard, soft, fused",
    )
    return parser.parse_args()


def load_math500_rows(count: int, start: int = 0) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError(f"math500_start must be non-negative, got {start}")
    if count <= 0:
        raise ValueError(f"num_math500 must be positive, got {count}")
    from tasks.common import load_hub_dataset

    ds = load_hub_dataset("HuggingFaceH4/MATH-500", "default", split="test")
    end = start + count
    if end > len(ds):
        raise ValueError(
            f"Requested MATH500 rows [{start}:{end}], but dataset contains {len(ds)} rows"
        )
    return [ds[index] for index in range(start, end)]


def build_math500_chat_prompt_ids(tokenizer, problem: str) -> tuple[list[int], str]:
    content = problem.strip()
    if not content:
        raise ValueError("MATH500 problem is empty")
    content += "\n\nPut your final answer in \\boxed{}."
    conversation = {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": ""},
        ]
    }
    prompt_ids = tokenizer.render_for_completion(conversation)
    return prompt_ids, tokenizer.decode(prompt_ids)


def _balanced_brace_content(text: str, open_brace_index: int) -> str | None:
    depth = 0
    chars: list[str] = []
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            if depth > 0:
                chars.append(char)
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            if depth < 0:
                return None
            chars.append(char)
        else:
            if depth > 0:
                chars.append(char)
    return None


def extract_boxed_answer(text: str) -> str | None:
    matches = list(BOXED_RE.finditer(text))
    for match in reversed(matches):
        answer = _balanced_brace_content(text, match.end() - 1)
        if answer:
            return answer
    return None


def extract_math500_answer(completion: str) -> tuple[str | None, str | None]:
    boxed = extract_boxed_answer(completion)
    if boxed:
        return boxed, "boxed"

    patterns = [
        r"(?:final answer|answer)\s*(?:is|:)\s*([^\n]+)",
        r"(?:therefore|thus),?\s*(?:the answer is)?\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, completion, flags=re.IGNORECASE)
        if matches:
            value = matches[-1].strip().strip("$").strip().rstrip(".").strip()
            if value:
                return value, "answer_phrase"

    fallback = NUMBER_RE.findall(completion)
    if fallback:
        return fallback[-1], "last_number_fallback"
    return None, None


def _strip_latex_wrappers(value: str) -> str:
    value = value.strip()
    if value.startswith("$") and value.endswith("$"):
        value = value[1:-1]
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = re.sub(r"\\(?:mathrm|text)\s*{([^{}]*)}", r"\1", value)
    value = value.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    value = value.replace("−", "-")
    return value.strip()


def normalize_math_answer(value: str | None) -> str | None:
    if value is None:
        return None
    value = _strip_latex_wrappers(value)
    value = value.strip().rstrip(".").strip()
    value = value.replace(" ", "")
    value = value.replace("{", "").replace("}", "")
    value = value.replace("\\", "")
    value = value.lower()
    return value or None


def _decimal_from_string(value: str) -> Decimal | None:
    value = value.replace(",", "").strip()
    try:
        return Decimal(value)
    except InvalidOperation:
        if "/" in value:
            left, right = value.split("/", 1)
            try:
                return Decimal(left) / Decimal(right)
            except (InvalidOperation, ZeroDivisionError):
                return None
    return None


def _decimal_from_latex(value: str) -> Decimal | None:
    value = _strip_latex_wrappers(value)
    frac_match = re.fullmatch(r"\\frac\s*{([^{}]+)}\s*{([^{}]+)}", value)
    if frac_match:
        numerator = _decimal_from_string(frac_match.group(1))
        denominator = _decimal_from_string(frac_match.group(2))
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator
    return _decimal_from_string(value)


def _numeric_equivalent(predicted: str | None, reference: str | None) -> bool:
    if predicted is None or reference is None:
        return False
    pred_num = _decimal_from_latex(predicted)
    ref_num = _decimal_from_latex(reference)
    if pred_num is None or ref_num is None:
        return False
    return abs(pred_num - ref_num) <= Decimal("1e-9")


def answers_equivalent(predicted: str | None, reference: str | None) -> bool:
    if _numeric_equivalent(predicted, reference):
        return True
    pred_norm = normalize_math_answer(predicted)
    ref_norm = normalize_math_answer(reference)
    if pred_norm is None or ref_norm is None:
        return False
    return pred_norm == ref_norm


def summarize_records(records: list[dict[str, Any]], modes: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in modes:
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

    summary["paired_accuracy"] = {}
    for left, right in (("standard", "soft"), ("standard", "fused"), ("soft", "fused")):
        if left not in modes or right not in modes:
            continue
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
    return summary


def render_summary(metrics: dict[str, Any], args: argparse.Namespace) -> str:
    lines = [
        "# MATH-500 zero-shot chat evaluation",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        "",
        f"Examples: `{args.num_math500}` starting at `{args.math500_start}`.",
        f"Prompt: zero-shot chat template; final-answer instruction requests `\\boxed{{}}`.",
        "",
        "| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    math500 = metrics["math500"]
    for mode in metrics["modes"]:
        row = math500[mode]
        ci_low, ci_high = row["accuracy_wilson_95"]
        rate = row["tokens_per_second"]
        rate_text = f"{rate:.2f}" if rate is not None else "—"
        lines.append(
            f"| {mode} | {row['correct']}/{row['examples']} | {row['accuracy']:.1%} | "
            f"{ci_low:.1%}–{ci_high:.1%} | {row['answer_parse_rate']:.1%} | "
            f"{row['completion_tokens']} | {rate_text} |"
        )
    if math500["paired_accuracy"]:
        lines.extend(["", "Paired exact McNemar p-values:"])
        for name, row in math500["paired_accuracy"].items():
            lines.append(
                f"- {name}: delta={row[next(k for k in row if k.startswith('accuracy_delta_'))]:+.3%}, "
                f"p={row['exact_mcnemar_p']:.4g}"
            )
    lines.extend(
        [
            "",
            "Grading note: this evaluator uses boxed-answer extraction plus normalized exact/numeric matching.",
            "The raw generations are saved so the run can be regraded with a stronger verifier later.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def evaluate_math500(model, tokenizer, args: argparse.Namespace) -> dict[str, Any]:
    from nanochat.engine import Engine

    examples = load_math500_rows(args.num_math500, args.math500_start)
    engine = Engine(model, tokenizer)
    bos_id = tokenizer.get_bos_token_id()
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    records: list[dict[str, Any]] = []
    generations_path = args.output_dir / "math500_generations.jsonl"

    with generations_path.open("w", encoding="utf-8") as generations_handle:
        for shard_offset, example in enumerate(examples):
            example_index = args.math500_start + shard_offset
            prompt_ids, prompt = build_math500_chat_prompt_ids(tokenizer, example["problem"])
            if len(prompt_ids) + args.max_new_tokens > model.config.sequence_len:
                raise ValueError(
                    f"MATH500 example {example_index} would exceed context: "
                    f"{len(prompt_ids)} + {args.max_new_tokens} > {model.config.sequence_len}"
                )
            reference = str(example["answer"])
            reference_normalized = normalize_math_answer(reference)
            record = {
                "example_index": example_index,
                "unique_id": example.get("unique_id"),
                "subject": example.get("subject"),
                "level": example.get("level"),
                "problem": example["problem"],
                "prompt": prompt,
                "prompt_tokens": len(prompt_ids),
                "reference_answer": reference,
                "reference_answer_normalized": reference_normalized,
                "modes": {},
            }

            for mode in args.decode_modes:
                synchronize(model.get_device())
                started = time.perf_counter()
                stream = engine.generate(
                    prompt_ids,
                    num_samples=1,
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_k=None,
                    seed=args.seed,
                    decode_mode=mode,
                    use_calculator=False,
                )
                suffix_ids = []
                sampled_tokens = 0
                stop_reason = "max_new_tokens"
                try:
                    for token_column, token_masks in stream:
                        token_id = token_column[0]
                        if token_id in (bos_id, assistant_end):
                            stop_reason = "terminal_token"
                            break
                        suffix_ids.append(token_id)
                        sampled_tokens += token_masks[0]
                finally:
                    stream.close()
                synchronize(model.get_device())
                elapsed = time.perf_counter() - started
                completion = tokenizer.decode(suffix_ids)
                predicted, parse_method = extract_math500_answer(completion)
                predicted_normalized = normalize_math_answer(predicted)
                correct = answers_equivalent(predicted, reference)
                record["modes"][mode] = {
                    "completion": completion,
                    "completion_token_ids": suffix_ids,
                    "completion_tokens": len(suffix_ids),
                    "sampled_tokens": int(sampled_tokens),
                    "predicted_answer": predicted,
                    "predicted_answer_normalized": predicted_normalized,
                    "parse_method": parse_method,
                    "answer_parsed": predicted is not None,
                    "correct": correct,
                    "seconds": elapsed,
                    "tokens_per_second": len(suffix_ids) / elapsed if elapsed > 0 else None,
                    "stop_reason": stop_reason,
                }
                print(
                    f"math500 example={shard_offset + 1}/{len(examples)} "
                    f"global_index={example_index} mode={mode} "
                    f"correct={correct} pred={predicted!r} ref={reference!r} "
                    f"tokens={len(suffix_ids)} seconds={elapsed:.3f}",
                    flush=True,
                )
            records.append(record)
            generations_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            generations_handle.flush()

    return summarize_records(records, args.decode_modes)


def run(args: argparse.Namespace) -> None:
    args.decode_modes = parse_decode_modes(args.decode_modes)
    checkpoint, meta_path, step, base_dir = checkpoint_paths(args.checkpoint)
    args.checkpoint = checkpoint
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import os

    os.environ["NANOCHAT_BASE_DIR"] = str(base_dir)
    shutil.copyfile(meta_path, args.output_dir / "checkpoint_meta.json")

    from nanochat.checkpoint_manager import build_model
    from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    config = {
        "command": [sys.executable, *sys.argv],
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "matching_meta": str(meta_path),
        "nanochat_base_dir": str(base_dir),
        "step": step,
        "device": str(device),
        "torch_version": torch.__version__,
        "compute_dtype": str(COMPUTE_DTYPE),
        "compute_dtype_reason": COMPUTE_DTYPE_REASON,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "math500_start": args.math500_start,
        "num_math500": args.num_math500,
        "math500_prompt_format": "chat",
        "math500_shots": 0,
        "max_new_tokens": args.max_new_tokens,
        "modes": list(args.decode_modes),
    }
    dump_json(args.output_dir / "run_config.json", config)
    print(json.dumps(config, indent=2), flush=True)

    model, tokenizer, loaded_meta = build_model(str(checkpoint.parent), step, device, phase="eval")
    has_latent_feedback = loaded_meta["model_config"].get("latent_feedback") is True
    requested_feedback_decode = any(mode in {"soft", "fused"} for mode in args.decode_modes)
    if requested_feedback_decode and not has_latent_feedback:
        raise RuntimeError("SOFT/FUSED decoding requires latent-feedback checkpoint metadata")
    print(f"loaded model parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)

    metrics = {
        "checkpoint": str(checkpoint),
        "step": step,
        "modes": list(args.decode_modes),
        "math500": evaluate_math500(model, tokenizer, args),
    }
    dump_json(args.output_dir / "metrics.json", metrics)
    with (args.output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(render_summary(metrics, args))
    print(f"wrote evaluation assets to {args.output_dir}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
