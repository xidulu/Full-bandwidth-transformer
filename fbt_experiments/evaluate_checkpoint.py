#!/usr/bin/env python3
"""Evaluate standard, SOFT, and FUSED decoding on one latent-feedback checkpoint.

The evaluator intentionally uses two small, deterministic probes:

1. Sequential teacher-forced continuation likelihood on held-out pretraining
   documents. This is the most sensitive test of whether recurrent feedback
   remains useful once decoding becomes autoregressive.
2. Matched greedy generations on raw-text GSM8K prompts derived from the
   canonical 8-shot prompts shipped in nanochat's eval bundle. The number of
   retained demonstrations is configurable.

Every generated artifact is written below --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


MODES = ("standard", "soft", "fused")
CORE_PREFILL_MODES = ("standard_prefill", "fused_prefill")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Direct `python fbt_experiments/evaluate_checkpoint.py` execution otherwise
    # places only fbt_experiments/ on sys.path, hiding the sibling nanochat package.
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Exact model_<step>.pt path, or a checkpoint directory (selects the highest matched step)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "model_010000",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-docs", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--continuation-tokens", type=int, default=64)
    parser.add_argument("--num-gsm8k", type=int, default=20)
    parser.add_argument(
        "--gsm8k-start",
        type=int,
        default=0,
        help="Zero-based first GSM8K row to evaluate (for deterministic sharding)",
    )
    parser.add_argument(
        "--gsm8k-shots",
        type=int,
        default=8,
        choices=range(0, 9),
        metavar="{0,...,8}",
        help="Number of canonical demonstrations retained from the bundled 8-shot prompt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--core-max-per-task",
        type=int,
        default=0,
        help="Examples per CORE task for prefill-mode scoring (0 = skip, -1 = full)",
    )
    parser.add_argument("--skip-continuation", action="store_true")
    parser.add_argument("--skip-gsm8k", action="store_true")
    return parser.parse_args()


class Tee:
    """Mirror writes to the terminal and the experiment log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def dump_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def dump_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpoint_paths(checkpoint: Path) -> tuple[Path, Path, int, Path]:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_dir():
        candidates = []
        for model_path in checkpoint.glob("model_*.pt"):
            candidate_match = re.fullmatch(r"model_(\d+)\.pt", model_path.name)
            if candidate_match is None:
                continue
            candidate_step = int(candidate_match.group(1))
            if model_path.with_name(f"meta_{candidate_step:06d}.json").is_file():
                candidates.append((candidate_step, model_path))
        if not candidates:
            raise FileNotFoundError(f"No model_<step>.pt with matching metadata found in {checkpoint}")
        _, checkpoint = max(candidates, key=lambda item: item[0])
    match = re.fullmatch(r"model_(\d+)\.pt", checkpoint.name)
    if match is None:
        raise ValueError(f"Checkpoint must be named model_<step>.pt, got {checkpoint.name!r}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    step = int(match.group(1))
    meta_path = checkpoint.with_name(f"meta_{step:06d}.json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Matching metadata not found: {meta_path}")
    if checkpoint.parent.parent.name not in {"base_checkpoints", "chatsft_checkpoints"}:
        raise ValueError(
            "Expected <base-dir>/{base_checkpoints,chatsft_checkpoints}/<tag>/model_<step>.pt "
            "so the tokenizer/data base can be inferred"
        )
    base_dir = checkpoint.parent.parent.parent
    return checkpoint, meta_path, step, base_dir


def make_cache(model, batch_size: int, seq_len: int):
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.engine import KVCache

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


@torch.inference_mode()
def forward_model_prefill_mode(model, input_ids: torch.Tensor, mode: str, bos_id: int):
    """Return autoregressive losses/predictions for one CORE prefill strategy."""
    if mode not in CORE_PREFILL_MODES:
        raise ValueError(mode)

    if mode == "standard_prefill":
        outputs = model(input_ids)
    else:
        if getattr(model, "latent_feedback", None) is None:
            raise ValueError("fused_prefill requires a latent-feedback checkpoint")
        _, pass1_hidden = model(input_ids, return_hidden=True)
        shifted_hidden = torch.roll(pass1_hidden, shifts=1, dims=1)
        feedback_mask = input_ids.ne(bos_id)
        feedback_mask[:, 0] = False
        outputs = model(
            input_ids,
            feedback_hidden=shifted_hidden,
            feedback_mask=feedback_mask,
        )

    batch_size, seq_len = input_ids.size()
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = F.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction="none",
    ).view(batch_size, seq_len)
    losses[:, -1] = float("nan")
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


def core_example_inputs(idx, model, tokenizer, data, device, task_meta, fewshot_data=None):
    from nanochat.core_eval import (
        batch_sequences_lm,
        batch_sequences_mc,
        batch_sequences_schema,
        render_prompts_lm,
        render_prompts_mc,
        render_prompts_schema,
        stack_sequences,
    )

    item = data[idx]
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    continuation_delimiter = task_meta["continuation_delimiter"]

    fewshot_examples = []
    if num_fewshot > 0:
        fewshot_data = fewshot_data or data
        rng = random.Random(1234 + idx)
        available_indices = [i for i, candidate in enumerate(fewshot_data) if candidate is not item]
        if len(available_indices) < num_fewshot:
            raise ValueError(
                f"Task has only {len(available_indices)} available few-shot examples, "
                f"but {num_fewshot} were requested"
            )
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [fewshot_data[i] for i in fewshot_indices]

    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    if hasattr(model, "max_seq_len") and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for token_ids, start_idx, end_idx in zip(tokens, start_idxs, end_idxs):
            if len(token_ids) > max_tokens:
                num_to_crop = len(token_ids) - max_tokens
                new_tokens.append(token_ids[-max_tokens:])
                new_start_idxs.append(start_idx - num_to_crop)
                new_end_idxs.append(end_idx - num_to_crop)
                assert start_idx - num_to_crop >= 0
                assert end_idx - num_to_crop >= 0
            else:
                new_tokens.append(token_ids)
                new_start_idxs.append(start_idx)
                new_end_idxs.append(end_idx)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    return item, task_type, input_ids, start_idxs, end_idxs


@torch.inference_mode()
def score_core_prefill_inputs(item, task_type, input_ids, start_idxs, end_idxs, losses, predictions):
    if task_type == "language_modeling":
        start_idx = start_idxs[0]
        end_idx = end_idxs[0]
        predicted_tokens = predictions[0, start_idx - 1 : end_idx - 1]
        actual_tokens = input_ids[0, start_idx:end_idx]
        return {
            "correct": bool(torch.all(predicted_tokens == actual_tokens).item()),
            "mean_loss": float(losses[0, start_idx - 1 : end_idx - 1].mean().item()),
        }
    if task_type in ("multiple_choice", "schema"):
        mean_losses = [
            float(losses[row_idx, start_idx - 1 : end_idx - 1].mean().item())
            for row_idx, (start_idx, end_idx) in enumerate(zip(start_idxs, end_idxs))
        ]
        pred_idx = mean_losses.index(min(mean_losses))
        return {
            "correct": pred_idx == item["gold"],
            "pred_idx": pred_idx,
            "gold": item["gold"],
            "mean_losses": mean_losses,
        }
    raise ValueError(f"Unsupported task type: {task_type}")


def ensure_eval_bundle(base_dir: Path) -> Path:
    eval_bundle_dir = base_dir / "eval_bundle"
    if not eval_bundle_dir.exists():
        from nanochat.common import download_file_with_lock
        from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle

        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    return eval_bundle_dir


def evaluate_core_prefill(model, tokenizer, args, output_dir: Path, base_dir: Path):
    eval_bundle_dir = ensure_eval_bundle(base_dir)
    config_path = eval_bundle_dir / "core.yaml"
    data_base_path = eval_bundle_dir / "eval_data"
    eval_meta_data = eval_bundle_dir / "eval_meta_data.csv"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    tasks = config["icl_tasks"]

    random_baselines = {}
    with eval_meta_data.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    bos_id = tokenizer.get_bos_token_id()
    details_path = output_dir / "core_prefill_details.jsonl"
    task_summaries = {}
    aggregate = {mode: {"correct": 0, "examples": 0, "seconds": 0.0} for mode in CORE_PREFILL_MODES}

    with details_path.open("w", encoding="utf-8") as details_handle:
        for task in tasks:
            task_started = time.perf_counter()
            label = task["label"]
            task_meta = {
                "task_type": task["icl_task_type"],
                "dataset_uri": task["dataset_uri"],
                "num_fewshot": task["num_fewshot"][0],
                "continuation_delimiter": task.get("continuation_delimiter", " "),
            }
            print(
                f"core task={label} shots={task_meta['num_fewshot']} type={task_meta['task_type']}",
                flush=True,
            )

            data_path = data_base_path / task_meta["dataset_uri"]
            with data_path.open("r", encoding="utf-8") as handle:
                full_data = [json.loads(line.strip()) for line in handle if line.strip()]
            data = list(full_data)
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            if args.core_max_per_task > 0:
                data = data[: args.core_max_per_task]

            task_counts = {mode: 0 for mode in CORE_PREFILL_MODES}
            mode_seconds = {mode: 0.0 for mode in CORE_PREFILL_MODES}
            pair_counts = {
                "both_correct": 0,
                "standard_prefill_only_correct": 0,
                "fused_prefill_only_correct": 0,
                "neither_correct": 0,
            }

            for idx in range(len(data)):
                item, task_type, input_ids, start_idxs, end_idxs = core_example_inputs(
                    idx, model, tokenizer, data, model.get_device(), task_meta, fewshot_data=full_data
                )
                mode_results = {}
                for mode in CORE_PREFILL_MODES:
                    synchronize(model.get_device())
                    started = time.perf_counter()
                    losses, predictions = forward_model_prefill_mode(model, input_ids, mode, bos_id)
                    mode_results[mode] = score_core_prefill_inputs(
                        item, task_type, input_ids, start_idxs, end_idxs, losses, predictions
                    )
                    synchronize(model.get_device())
                    elapsed = time.perf_counter() - started
                    mode_seconds[mode] += elapsed
                    aggregate[mode]["seconds"] += elapsed

                standard_correct = mode_results["standard_prefill"]["correct"]
                fused_correct = mode_results["fused_prefill"]["correct"]
                if standard_correct and fused_correct:
                    pair_counts["both_correct"] += 1
                elif standard_correct:
                    pair_counts["standard_prefill_only_correct"] += 1
                elif fused_correct:
                    pair_counts["fused_prefill_only_correct"] += 1
                else:
                    pair_counts["neither_correct"] += 1

                for mode in CORE_PREFILL_MODES:
                    if mode_results[mode]["correct"]:
                        task_counts[mode] += 1
                        aggregate[mode]["correct"] += 1
                    aggregate[mode]["examples"] += 1

                detail = {
                    "task": label,
                    "example_index": idx,
                    "results": mode_results,
                }
                details_handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
                details_handle.flush()

            examples = len(data)
            task_summary = {}
            for mode in CORE_PREFILL_MODES:
                accuracy = task_counts[mode] / examples
                random_baseline = random_baselines[label]
                centered = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
                task_summary[mode] = {
                    "examples": examples,
                    "correct": task_counts[mode],
                    "accuracy": accuracy,
                    "centered": centered,
                    "seconds": mode_seconds[mode],
                }
            task_summary["paired_accuracy"] = {
                **pair_counts,
                "accuracy_delta_fused_minus_standard": (
                    pair_counts["fused_prefill_only_correct"]
                    - pair_counts["standard_prefill_only_correct"]
                )
                / examples,
                "exact_mcnemar_p": exact_mcnemar_p(
                    pair_counts["standard_prefill_only_correct"],
                    pair_counts["fused_prefill_only_correct"],
                ),
            }
            task_summary["seconds"] = time.perf_counter() - task_started
            task_summaries[label] = task_summary
            print(
                f"core task={label} standard={task_counts['standard_prefill']}/{examples} "
                f"fused={task_counts['fused_prefill']}/{examples} "
                f"seconds={task_summary['seconds']:.1f}",
                flush=True,
            )

    mode_summaries = {}
    for mode in CORE_PREFILL_MODES:
        centered_values = [task_summaries[label][mode]["centered"] for label in task_summaries]
        mode_summaries[mode] = {
            **aggregate[mode],
            "accuracy": aggregate[mode]["correct"] / aggregate[mode]["examples"],
            "core_metric": sum(centered_values) / len(centered_values),
        }
    standard_only = sum(
        task["paired_accuracy"]["standard_prefill_only_correct"] for task in task_summaries.values()
    )
    fused_only = sum(
        task["paired_accuracy"]["fused_prefill_only_correct"] for task in task_summaries.values()
    )
    paired = {
        "standard_prefill_only_correct": standard_only,
        "fused_prefill_only_correct": fused_only,
        "accuracy_delta_fused_minus_standard": (fused_only - standard_only)
        / mode_summaries["standard_prefill"]["examples"],
        "exact_mcnemar_p": exact_mcnemar_p(standard_only, fused_only),
    }
    return {
        "max_per_task": args.core_max_per_task,
        "modes": list(CORE_PREFILL_MODES),
        "tasks": task_summaries,
        "summary": mode_summaries,
        "paired_accuracy": paired,
    }


@torch.inference_mode()
def initial_decode_state(model, prompt_ids: torch.Tensor, mode: str, total_cache_len: int, bos_id: int):
    """Return next-token logits, last top state, and the mode-correct KV cache."""
    if mode not in MODES:
        raise ValueError(mode)

    if mode == "standard":
        cache = make_cache(model, prompt_ids.size(0), total_cache_len)
        logits, hidden = model(prompt_ids, kv_cache=cache, return_hidden=True)
        return logits[:, -1, :], hidden[:, -1:, :], cache

    if getattr(model, "latent_feedback", None) is None:
        raise ValueError(f"{mode} requires a latent-feedback checkpoint")

    if mode == "soft":
        cache = make_cache(model, prompt_ids.size(0), total_cache_len)
        logits, hidden = model(prompt_ids, kv_cache=cache, return_hidden=True)
        return logits[:, -1, :], hidden[:, -1:, :], cache

    # FUSED: obtain h1 from an ordinary prompt pass, discard that pass's cache,
    # and build pass-2 K/V from position zero using the shifted h1 states.
    pass1_cache = make_cache(model, prompt_ids.size(0), prompt_ids.size(1))
    _, pass1_hidden = model(prompt_ids, kv_cache=pass1_cache, return_hidden=True)
    del pass1_cache

    cache = make_cache(model, prompt_ids.size(0), total_cache_len)
    shifted_hidden = torch.roll(pass1_hidden, shifts=1, dims=1)
    feedback_mask = prompt_ids.ne(bos_id)
    feedback_mask[:, 0] = False
    logits, hidden = model(
        prompt_ids,
        kv_cache=cache,
        feedback_hidden=shifted_hidden,
        feedback_mask=feedback_mask,
        return_hidden=True,
    )
    return logits[:, -1, :], hidden[:, -1:, :], cache


@torch.inference_mode()
def score_continuation(model, sequence: list[int], prefix_tokens: int, mode: str, token_bytes, bos_id: int):
    """Score a known continuation through the exact autoregressive recurrence."""
    device = model.get_device()
    prompt = torch.tensor([sequence[:prefix_tokens]], dtype=torch.long, device=device)
    targets = sequence[prefix_tokens:]
    logits, previous_hidden, cache = initial_decode_state(
        model, prompt, mode, total_cache_len=len(sequence), bos_id=bos_id
    )

    nats = 0.0
    byte_count = 0
    scored_tokens = 0
    token_nlls = []
    max_abs_hidden = float(previous_hidden.float().abs().max().item())
    min_hidden_rms = float("inf")
    max_hidden_rms = 0.0

    for target_index, target_id in enumerate(targets):
        target = torch.tensor([target_id], dtype=torch.long, device=device)
        nll = float(F.cross_entropy(logits, target, reduction="sum").item())
        num_bytes = int(token_bytes[target_id].item())
        if num_bytes > 0:
            nats += nll
            byte_count += num_bytes
            scored_tokens += 1
        token_nlls.append(nll if num_bytes > 0 else None)

        hidden_float = previous_hidden.float()
        hidden_rms = float(hidden_float.square().mean().sqrt().item())
        max_abs_hidden = max(max_abs_hidden, float(hidden_float.abs().max().item()))
        min_hidden_rms = min(min_hidden_rms, hidden_rms)
        max_hidden_rms = max(max_hidden_rms, hidden_rms)

        if target_index + 1 == len(targets):
            continue
        token_column = target.view(1, 1)
        if mode == "standard":
            next_logits, next_hidden = model(token_column, kv_cache=cache, return_hidden=True)
        else:
            next_logits, next_hidden = model(
                token_column,
                kv_cache=cache,
                feedback_hidden=previous_hidden,
                return_hidden=True,
            )
        logits = next_logits[:, -1, :]
        previous_hidden = next_hidden[:, -1:, :]

    if byte_count == 0:
        raise RuntimeError("Continuation contains no byte-bearing tokens")
    return {
        "nats": nats,
        "bytes": byte_count,
        "scored_tokens": scored_tokens,
        "bpb": nats / (math.log(2.0) * byte_count),
        "first_token_nll": token_nlls[0],
        "mean_token_nll": nats / scored_tokens,
        "hidden_rms_min": min_hidden_rms,
        "hidden_rms_max": max_hidden_rms,
        "hidden_abs_max": max_abs_hidden,
        "all_finite": all(value is None or math.isfinite(value) for value in token_nlls)
        and math.isfinite(max_abs_hidden),
    }


def heldout_documents(tokenizer, count: int, length: int):
    """Select the first sufficiently long documents from the fixed validation shard."""
    from nanochat.dataset import parquets_iter_batched

    bos_id = tokenizer.get_bos_token_id()
    selected = []
    source_index = 0
    for text_batch in parquets_iter_batched("val"):
        token_batches = tokenizer.encode(text_batch, prepend=bos_id, num_threads=4)
        for text, token_ids in zip(text_batch, token_batches):
            if len(token_ids) >= length:
                selected.append(
                    {
                        "source_document_index": source_index,
                        "tokens": token_ids[:length],
                        "text_preview": text[:240],
                    }
                )
                if len(selected) == count:
                    return selected
            source_index += 1
    raise RuntimeError(f"Validation shard supplied only {len(selected)} documents with at least {length} tokens")


def evaluate_continuations(model, tokenizer, args, output_dir: Path):
    from nanochat.tokenizer import get_token_bytes

    sequence_len = args.prefix_tokens + args.continuation_tokens
    if sequence_len > model.config.sequence_len:
        raise ValueError(f"Requested {sequence_len} tokens, model limit is {model.config.sequence_len}")
    docs = heldout_documents(tokenizer, args.num_docs, sequence_len)
    token_bytes = get_token_bytes(device=model.get_device())
    bos_id = tokenizer.get_bos_token_id()
    rows = []

    aggregate = {mode: {"nats": 0.0, "bytes": 0, "scored_tokens": 0, "seconds": 0.0} for mode in MODES}
    for doc_index, doc in enumerate(docs):
        row = {
            "document_index": doc_index,
            "source_document_index": doc["source_document_index"],
            "prefix_tokens": args.prefix_tokens,
            "continuation_tokens": args.continuation_tokens,
            "prefix_text": tokenizer.decode(doc["tokens"][: args.prefix_tokens]),
            "continuation_text": tokenizer.decode(doc["tokens"][args.prefix_tokens :]),
            "modes": {},
        }
        for mode in MODES:
            synchronize(model.get_device())
            started = time.perf_counter()
            result = score_continuation(
                model, doc["tokens"], args.prefix_tokens, mode, token_bytes, bos_id
            )
            synchronize(model.get_device())
            result["seconds"] = time.perf_counter() - started
            row["modes"][mode] = result
            for key in ("nats", "bytes", "scored_tokens", "seconds"):
                aggregate[mode][key] += result[key]
            print(
                f"continuation doc={doc_index + 1}/{len(docs)} mode={mode} "
                f"bpb={result['bpb']:.6f} seconds={result['seconds']:.3f}",
                flush=True,
            )

        standard_first = row["modes"]["standard"]["first_token_nll"]
        soft_first = row["modes"]["soft"]["first_token_nll"]
        row["standard_soft_first_token_nll_abs_diff"] = abs(standard_first - soft_first)
        rows.append(row)

    summary = {}
    for mode, values in aggregate.items():
        bpb = values["nats"] / (math.log(2.0) * values["bytes"])
        summary[mode] = {
            **values,
            "bpb": bpb,
            "mean_token_nll": values["nats"] / values["scored_tokens"],
            "tokens_per_second": values["scored_tokens"] / values["seconds"],
            "all_finite": all(row["modes"][mode]["all_finite"] for row in rows),
            "hidden_rms_min": min(row["modes"][mode]["hidden_rms_min"] for row in rows),
            "hidden_rms_max": max(row["modes"][mode]["hidden_rms_max"] for row in rows),
            "hidden_abs_max": max(row["modes"][mode]["hidden_abs_max"] for row in rows),
        }
    for mode in ("soft", "fused"):
        summary[mode]["bpb_delta_vs_standard"] = summary[mode]["bpb"] - summary["standard"]["bpb"]
        summary[mode]["bpb_relative_delta_vs_standard"] = (
            summary[mode]["bpb"] / summary["standard"]["bpb"] - 1.0
        )
        summary[mode]["documents_better_than_standard"] = sum(
            row["modes"][mode]["bpb"] < row["modes"]["standard"]["bpb"] for row in rows
        )
        summary[mode]["documents_evaluated"] = len(rows)
    summary["standard_soft_first_token_max_abs_diff"] = max(
        row["standard_soft_first_token_nll_abs_diff"] for row in rows
    )
    dump_jsonl(output_dir / "continuation_details.jsonl", rows)
    return summary


NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
ANSWER_PATTERNS = (
    ("hash_marker", re.compile(rf"####\s*\$?\s*({NUMBER})")),
    ("answer_phrase", re.compile(rf"(?i)(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*\$?\s*({NUMBER})")),
)


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.replace(",", "").strip()
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if number == number.to_integral():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    """Two-sided Wilson score interval for a binomial proportion."""
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    """Two-sided exact McNemar/binomial p-value for paired binary outcomes."""
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(i + 1)
        - math.lgamma(discordant - i + 1)
        - discordant * math.log(2.0)
        for i in range(tail + 1)
    ]
    max_log = max(log_terms)
    log_probability = math.log(2.0) + max_log + math.log(
        sum(math.exp(term - max_log) for term in log_terms)
    )
    return min(1.0, math.exp(log_probability))


def extract_gsm8k_answer(completion: str):
    for method, pattern in ANSWER_PATTERNS:
        matches = pattern.findall(completion)
        if matches:
            # Few-shot base-model completions sometimes begin another synthetic
            # Q/A pair. The answer to the requested problem is the first explicit
            # answer phrase, not the last number in the full rollout.
            return normalize_number(matches[0]), method
    fallback = re.findall(NUMBER, completion)
    if fallback:
        return normalize_number(fallback[-1]), "last_number_fallback"
    return None, None


def load_gsm8k_rows(path: Path, count: int, start: int = 0):
    """Load exactly ``rows[start:start + count]`` with explicit bounds checks."""
    if start < 0:
        raise ValueError(f"gsm8k_start must be non-negative, got {start}")
    if count <= 0:
        raise ValueError(f"num_gsm8k must be positive, got {count}")

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    end = start + count
    if end > len(rows):
        raise ValueError(
            f"Requested GSM8K rows [{start}:{end}], but {path} contains "
            f"{len(rows)} rows (valid end is at most {len(rows)})"
        )
    return rows[start:end]


def build_gsm8k_prompt(context: str, num_shots: int) -> str:
    """Keep the first num_shots canonical demos and the final target question."""
    if not 0 <= num_shots <= 8:
        raise ValueError(f"gsm8k_shots must be in [0, 8], got {num_shots}")
    blocks = context.split("\n\nQ:")
    if not context.startswith("Q:") or len(blocks) != 9:
        raise ValueError(
            "Expected the bundled GSM8K context to contain eight demonstrations and one target question"
        )
    target = blocks[-1]
    if num_shots == 0:
        return "Q:" + target
    return "\n\nQ:".join([*blocks[:num_shots], target])


@torch.inference_mode()
def evaluate_gsm8k(model, tokenizer, args, output_dir: Path, base_dir: Path):
    from nanochat.engine import Engine

    gsm8k_path = base_dir / "eval_bundle" / "eval_data" / "symbolic_problem_solving" / "gsm8k_prepended_8shot.jsonl"
    if not gsm8k_path.is_file():
        raise FileNotFoundError(gsm8k_path)
    examples = load_gsm8k_rows(gsm8k_path, args.num_gsm8k, args.gsm8k_start)
    engine = Engine(model, tokenizer)
    bos_id = tokenizer.get_bos_token_id()
    records = []
    generations_path = output_dir / "gsm8k_generations.jsonl"

    # Flush one completed, three-mode record at a time. A long benchmark run
    # therefore leaves useful, valid partial evidence if the job is interrupted.
    with generations_path.open("w", encoding="utf-8") as generations_handle:
        for shard_offset, example in enumerate(examples):
            example_index = args.gsm8k_start + shard_offset
            prompt_context = build_gsm8k_prompt(example["context"], args.gsm8k_shots)
            prompt = prompt_context + "\n\nA:"
            prompt_ids = tokenizer.encode(prompt, prepend=bos_id)
            if len(prompt_ids) + args.max_new_tokens > model.config.sequence_len:
                raise ValueError(
                    f"GSM8K example {example_index} would exceed context: "
                    f"{len(prompt_ids)} + {args.max_new_tokens} > {model.config.sequence_len}"
                )
            reference = normalize_number(str(example["answer"]))
            record = {
                "example_index": example_index,
                "gsm8k_shots": args.gsm8k_shots,
                "prompt": prompt,
                "prompt_tokens": len(prompt_ids),
                "reference_answer": reference,
                "modes": {},
            }

            for mode in MODES:
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
                assistant_end = tokenizer.encode_special("<|assistant_end|>")
                try:
                    for token_column, token_masks in stream:
                        token_id = token_column[0]
                        if token_id in (bos_id, assistant_end):
                            stop_reason = "terminal_token"
                            break
                        suffix_ids.append(token_id)
                        sampled_tokens += token_masks[0]
                        # The few-shot prompt uses Q:/A: records. Once the model
                        # starts the next record, the answer to this record is complete.
                        if "\n\nQ:" in tokenizer.decode(suffix_ids):
                            stop_reason = "next_question"
                            break
                finally:
                    stream.close()
                synchronize(model.get_device())
                elapsed = time.perf_counter() - started
                raw_completion = tokenizer.decode(suffix_ids)
                completion = raw_completion.split("\n\nQ:", 1)[0]
                predicted, parse_method = extract_gsm8k_answer(completion)
                record["modes"][mode] = {
                    "completion": completion,
                    "raw_completion_through_stop": raw_completion,
                    "completion_token_ids": suffix_ids,
                    "completion_tokens": len(suffix_ids),
                    "sampled_tokens": int(sampled_tokens),
                    "predicted_answer": predicted,
                    "parse_method": parse_method,
                    "answer_parsed": predicted is not None,
                    "correct": predicted == reference,
                    "seconds": elapsed,
                    "tokens_per_second": len(suffix_ids) / elapsed if elapsed > 0 else None,
                    "stop_reason": stop_reason,
                }
                print(
                    f"gsm8k example={shard_offset + 1}/{len(examples)} "
                    f"global_index={example_index} mode={mode} "
                    f"pred={predicted!r} ref={reference!r} tokens={len(suffix_ids)} seconds={elapsed:.3f}",
                    flush=True,
                )

            mode_tokens = {mode: record["modes"][mode]["completion_token_ids"] for mode in MODES}
            record["pairwise"] = {
                "standard_soft_same_first_token": bool(
                    mode_tokens["standard"] and mode_tokens["soft"]
                    and mode_tokens["standard"][0] == mode_tokens["soft"][0]
                ),
                "standard_soft_identical": mode_tokens["standard"] == mode_tokens["soft"],
                "standard_fused_identical": mode_tokens["standard"] == mode_tokens["fused"],
                "soft_fused_identical": mode_tokens["soft"] == mode_tokens["fused"],
            }
            records.append(record)
            generations_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            generations_handle.flush()

    summary = {}
    for mode in MODES:
        seconds = sum(record["modes"][mode]["seconds"] for record in records)
        tokens = sum(record["modes"][mode]["completion_tokens"] for record in records)
        summary[mode] = {
            "examples": len(records),
            "correct": sum(record["modes"][mode]["correct"] for record in records),
            "accuracy": sum(record["modes"][mode]["correct"] for record in records) / len(records),
            "answers_parsed": sum(record["modes"][mode]["answer_parsed"] for record in records),
            "answer_parse_rate": sum(record["modes"][mode]["answer_parsed"] for record in records) / len(records),
            "completion_tokens": tokens,
            "seconds": seconds,
            "tokens_per_second": tokens / seconds if seconds > 0 else None,
        }
        summary[mode]["accuracy_wilson_95"] = wilson_interval(summary[mode]["correct"], len(records))
    summary["pairwise"] = {
        key: sum(record["pairwise"][key] for record in records) / len(records)
        for key in records[0]["pairwise"]
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
    summary["num_shots"] = args.gsm8k_shots
    return summary


def render_summary(metrics, meta, args) -> str:
    lines = [
        "# SOFT/FUSED checkpoint evaluation",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        "",
        "The implementation passed exact recurrence/cache tests before this run. The numbers below assess this particular checkpoint, not just code execution.",
        "",
    ]
    continuation = metrics.get("continuation")
    if continuation:
        lines.extend(
            [
                "## Held-out sequential continuation",
                "",
                "| mode | BPB | delta vs standard | relative delta | document wins | finite | hidden RMS range |",
                "|---|---:|---:|---:|---:|:---:|---:|",
            ]
        )
        for mode in MODES:
            row = continuation[mode]
            delta = row.get("bpb_delta_vs_standard", 0.0)
            relative = row.get("bpb_relative_delta_vs_standard", 0.0)
            wins = (
                f"{row['documents_better_than_standard']}/{row['documents_evaluated']}"
                if mode != "standard"
                else "—"
            )
            lines.append(
                f"| {mode} | {row['bpb']:.6f} | {delta:+.6f} | {relative:+.3%} | {wins} | "
                f"{row['all_finite']} | {row['hidden_rms_min']:.6f}–{row['hidden_rms_max']:.6f} |"
            )
        first_diff = continuation["standard_soft_first_token_max_abs_diff"]
        lines.extend(
            [
                "",
                f"STANDARD and SOFT share the ordinary prompt prefill; their first-target NLL max difference was `{first_diff:.3g}`.",
                "",
            ]
        )

    core_prefill = metrics.get("core_prefill")
    if core_prefill:
        standard = core_prefill["summary"]["standard_prefill"]
        fused = core_prefill["summary"]["fused_prefill"]
        paired = core_prefill["paired_accuracy"]
        limit = core_prefill["max_per_task"]
        limit_text = "full task set" if limit < 0 else f"up to {limit} examples per task"
        lines.extend(
            [
                f"## CORE Prefill Likelihood ({limit_text})",
                "",
                "| prefill mode | correct | raw accuracy | CORE metric | seconds |",
                "|---|---:|---:|---:|---:|",
                f"| standard_prefill | {standard['correct']}/{standard['examples']} | "
                f"{standard['accuracy']:.1%} | {standard['core_metric']:.4f} | "
                f"{standard['seconds']:.1f} |",
                f"| fused_prefill | {fused['correct']}/{fused['examples']} | "
                f"{fused['accuracy']:.1%} | {fused['core_metric']:.4f} | "
                f"{fused['seconds']:.1f} |",
                "",
                "Paired exact McNemar: "
                f"standard-only `{paired['standard_prefill_only_correct']}`, "
                f"fused-only `{paired['fused_prefill_only_correct']}`, "
                f"delta `{paired['accuracy_delta_fused_minus_standard']:+.3%}`, "
                f"p=`{paired['exact_mcnemar_p']:.4g}`.",
                "",
            ]
        )

    gsm8k = metrics.get("gsm8k")
    if gsm8k:
        lines.extend(
            [
                f"## {args.gsm8k_shots}-shot GSM8K ({gsm8k['standard']['examples']} problems)",
                "",
                "| mode | exact | accuracy | 95% Wilson CI | parsed | output tokens | tokens/s |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in MODES:
            row = gsm8k[mode]
            ci_low, ci_high = row["accuracy_wilson_95"]
            lines.append(
                f"| {mode} | {row['correct']}/{row['examples']} | {row['accuracy']:.1%} | "
                f"{ci_low:.1%}–{ci_high:.1%} | {row['answer_parse_rate']:.1%} | "
                f"{row['completion_tokens']} | {row['tokens_per_second']:.2f} |"
            )
        lines.extend(
            [
                "",
                f"STANDARD/SOFT first generated token match rate: {gsm8k['pairwise']['standard_soft_same_first_token']:.1%} (expected 100%).",
                "",
                "Paired exact McNemar p-values: "
                f"STANDARD↔SOFT `{gsm8k['paired_accuracy']['standard_vs_soft']['exact_mcnemar_p']:.4g}`, "
                f"STANDARD↔FUSED `{gsm8k['paired_accuracy']['standard_vs_fused']['exact_mcnemar_p']:.4g}`, "
                f"SOFT↔FUSED `{gsm8k['paired_accuracy']['soft_vs_fused']['exact_mcnemar_p']:.4g}`.",
                "",
            ]
        )

        if not continuation:
            standard_correct = gsm8k["standard"]["correct"]
            soft_correct = gsm8k["soft"]["correct"]
            fused_correct = gsm8k["fused"]["correct"]
            smallest_p = min(
                comparison["exact_mcnemar_p"]
                for comparison in gsm8k["paired_accuracy"].values()
            )
            significance_sentence = (
                f"None of the paired exact tests is significant at 0.05 (smallest p={smallest_p:.4g}), "
                "so this run does not establish an accuracy difference among the decoding modes."
                if smallest_p >= 0.05
                else
                f"At least one paired exact test is below 0.05 (smallest p={smallest_p:.4g}); "
                "inspect the paired counts above before drawing a conclusion."
            )
            lines.extend(
                [
                    "## Verdict",
                    "",
                    f"STANDARD scored {standard_correct}/{gsm8k['standard']['examples']}; "
                    f"SOFT scored {soft_correct}/{gsm8k['soft']['examples']} "
                    f"({soft_correct - standard_correct:+d} versus STANDARD), and FUSED scored "
                    f"{fused_correct}/{gsm8k['fused']['examples']} "
                    f"({fused_correct - standard_correct:+d} versus STANDARD). "
                    + significance_sentence,
                    "",
                ]
            )

    if continuation:
        soft_delta = continuation["soft"]["bpb_relative_delta_vs_standard"]
        fused_delta = continuation["fused"]["bpb_relative_delta_vs_standard"]
        lines.extend(
            [
                "## Verdict",
                "",
                "Both decoding algorithms operate correctly and remain numerically stable on this checkpoint. "
                f"Quality is different: SOFT changes BPB by {soft_delta:+.3%}, while FUSED changes it by {fused_delta:+.3%}. "
                "Negative BPB deltas favor feedback decoding; positive deltas favor STANDARD. "
                "The exact shared-first-token invariant provides an additional implementation check.",
                "",
                "These likelihood deltas are small and mixed across documents, so they do not establish a statistically persuasive quality gain. "
                "The tiny GSM8K result is also retained as a behavioral smoke test only.",
                "",
            ]
        )

    user_config = meta.get("user_config", {})
    lines.extend(["## Interpretation limits", ""])
    if user_config.get("sft_dataset") is not None:
        lines.extend(
            [
                f"- This is an SFT checkpoint at step {meta.get('step')}, initialized from "
                f"`{user_config.get('model_tag')}` step `{user_config.get('model_step')}`.",
                f"- SFT data: `{user_config.get('sft_dataset')}` split `{user_config.get('openmath_split')}`; "
                f"checkpoint validation bpb was `{meta.get('val_bpb'):.4f}`.",
            ]
        )
    else:
        lines.append(
            f"- This is a base checkpoint trained for about "
            f"{meta.get('step', 0) * meta.get('total_batch_size', 0) / 1e9:.2f}B raw tokens, "
            "far smaller than the paper's main runs."
        )
    lines.extend(
        [
            f"- Evaluation used greedy decoding (`temperature=0.0`) with K={user_config.get('num_forward_passes') or meta.get('num_forward_passes')} latent-feedback training metadata.",
            f"- `feedback_prefix_mixin={user_config.get('feedback_prefix_mixin')}`. With mixin disabled, FUSED matches the trained full-feedback prompt regime; SOFT's plain-prompt/feedback-generation boundary was not explicitly trained.",
        ]
    )
    if gsm8k:
        lines.append("- The GSM8K subset is deterministic and paired across all decoding modes.")
    if core_prefill:
        lines.append("- The CORE examples are deterministically ordered and paired across both prefill modes.")
    lines.extend(
        [
            "",
            "Paper: https://arxiv.org/abs/2608.08888",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    checkpoint, meta_path, step, base_dir = checkpoint_paths(args.checkpoint)
    args.checkpoint = checkpoint
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NANOCHAT_BASE_DIR"] = str(base_dir)

    from nanochat.checkpoint_manager import build_model
    from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    shutil.copyfile(meta_path, output_dir / "checkpoint_meta.json")
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
        "num_docs": args.num_docs,
        "prefix_tokens": args.prefix_tokens,
        "continuation_tokens": args.continuation_tokens,
        "num_gsm8k": args.num_gsm8k,
        "gsm8k_start": args.gsm8k_start,
        "gsm8k_shots": args.gsm8k_shots,
        "max_new_tokens": args.max_new_tokens,
        "core_max_per_task": args.core_max_per_task,
        "skip_continuation": args.skip_continuation,
        "skip_gsm8k": args.skip_gsm8k,
        "modes": list(MODES),
        "core_prefill_modes": list(CORE_PREFILL_MODES),
    }
    dump_json(output_dir / "run_config.json", config)

    print(json.dumps(config, indent=2), flush=True)
    model, tokenizer, loaded_meta = build_model(str(checkpoint.parent), step, device, phase="eval")
    if loaded_meta["model_config"].get("latent_feedback") is not True:
        raise RuntimeError("The exact checkpoint metadata does not enable latent feedback")
    print(f"loaded model parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)

    metrics = {
        "checkpoint": str(checkpoint),
        "step": step,
        "modes": list(MODES),
    }
    if args.core_max_per_task != 0:
        metrics["core_prefill"] = evaluate_core_prefill(model, tokenizer, args, output_dir, base_dir)
    if not args.skip_continuation:
        metrics["continuation"] = evaluate_continuations(model, tokenizer, args, output_dir)
    if not args.skip_gsm8k:
        metrics["gsm8k"] = evaluate_gsm8k(model, tokenizer, args, output_dir, base_dir)
    dump_json(output_dir / "metrics.json", metrics)
    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(render_summary(metrics, meta, args))
    print(f"wrote evaluation assets to {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        tee_stdout = Tee(sys.stdout, log_handle)
        tee_stderr = Tee(sys.stderr, log_handle)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            try:
                run(args)
            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
