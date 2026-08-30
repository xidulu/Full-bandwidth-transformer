"""
Evaluate nanochat checkpoints on Python code-generation benchmarks.

This is intentionally separate from scripts/chat_eval.py because the Stack-Edu
SFT checkpoint was trained as raw code continuation, not chat instruction
following. Prompts here are direct code/text continuations and outputs are
executed against benchmark tests.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_dist_info, print0
from nanochat.engine import Engine
from nanochat.execution import execute_code
from tasks.common import load_hub_dataset


def comment_block(text: str) -> str:
    return "\n".join("# " + line for line in text.strip().splitlines())


def strip_code_fences(text: str) -> str:
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip() + "\n"
    return text


def truncate_completion(text: str, task: str | None = None) -> str:
    # Generated benchmark continuations often drift into markdown or another
    # problem. Keep this conservative: valid Python before these markers remains.
    text = strip_code_fences(text)
    stop_markers = [
        "\n```",
        "\n# Task:",
        "\n# Problem:",
        "\nif __name__ == \"__main__\":",
        "\nif __name__ == '__main__':",
    ]
    end = len(text)
    for marker in stop_markers:
        idx = text.find(marker)
        if idx >= 0:
            end = min(end, idx)
    text = text[:end]

    # Raw-code models tend to continue with another function variant after
    # solving the current prompt. Standard code-eval harnesses stop generation
    # at such boundaries; Engine does not support stop strings, so do it here.
    lines = text.splitlines()
    if task == "humaneval":
        for i, line in enumerate(lines):
            if i > 0 and re.match(r"^(def|class)\s+", line):
                lines = lines[:i]
                break
    elif task and task.startswith("mbpp"):
        seen_definition = False
        for i, line in enumerate(lines):
            if re.match(r"^(def|class)\s+", line):
                if seen_definition:
                    lines = lines[:i]
                    break
                seen_definition = True
            elif seen_definition and line.strip() and not line.startswith((" ", "\t", "@")):
                lines = lines[:i]
                break
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class Problem:
    task: str
    task_id: str
    prompt: str
    eval_prefix: str
    tests: str
    metadata: dict[str, Any]


class HumanEvalDirect:
    name = "humaneval"

    def __init__(self):
        self.ds = load_hub_dataset(
            "openai/openai_humaneval",
            subset="openai_humaneval",
            split="test",
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = self.ds[index]
        return Problem(
            task=self.name,
            task_id=str(row["task_id"]),
            prompt=row["prompt"],
            eval_prefix=row["prompt"],
            tests=row["test"] + "\n" + f"check({row['entry_point']})\n",
            metadata={"entry_point": row["entry_point"]},
        )


class MBPPDirect:
    name = "mbpp"

    def __init__(self, config="full"):
        if config not in {"full", "sanitized"}:
            raise ValueError("--mbpp-config must be full or sanitized")
        self.config = config
        self.ds = load_hub_dataset(
            "google-research-datasets/mbpp",
            subset=config,
            split="test",
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = self.ds[index]
        if self.config == "sanitized":
            task_text = row["prompt"]
            setup = "\n".join(row.get("test_imports") or [])
        else:
            task_text = row["text"]
            setup = row.get("test_setup_code") or ""
        test_list = row["test_list"]
        commented_tests = "\n".join(comment_block(test) for test in test_list[:3])
        prompt = (
            f"{comment_block(task_text)}\n"
            f"# Your code should pass these tests:\n"
            f"{commented_tests}\n\n"
        )
        eval_prefix = (setup.rstrip() + "\n\n" if setup.strip() else "") + prompt
        return Problem(
            task=f"mbpp_{self.config}",
            task_id=str(row["task_id"]),
            prompt=prompt,
            eval_prefix=eval_prefix,
            tests="\n".join(test_list) + "\n",
            metadata={"config": self.config},
        )


def build_task(name, mbpp_config):
    if name == "humaneval":
        return HumanEvalDirect()
    if name == "mbpp":
        return MBPPDirect(config=mbpp_config)
    raise ValueError(f"unknown task {name!r}")


def evaluate_problem(problem: Problem, completion: str, timeout: float) -> tuple[bool, str | None]:
    code = truncate_completion(completion, problem.task)
    program = problem.eval_prefix + code + "\n" + problem.tests
    result = execute_code(program, timeout=timeout)
    return result.success, result.error


def main():
    parser = argparse.ArgumentParser(description="Evaluate code-generation benchmarks")
    parser.add_argument("--source", type=str, default="sft", choices=["base", "sft", "rl"])
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--tasks", type=str, default="humaneval,mbpp")
    parser.add_argument("--mbpp-config", type=str, default="full", choices=["full", "sanitized"])
    parser.add_argument("--decode-modes", type=str, default="standard,soft,fused")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-problems", type=int, default=-1)
    parser.add_argument("--execution-timeout", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")))
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"])
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must be in [0, --num-shards)")

    os.makedirs(args.output_dir, exist_ok=True)

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)
    bos = tokenizer.get_bos_token_id()

    task_names = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
    decode_modes = [mode.strip() for mode in args.decode_modes.split(",") if mode.strip()]
    local_stats = {(task_name, mode): [0, 0] for task_name in task_names for mode in decode_modes}
    worker_rank = args.shard_index * ddp_world_size + ddp_rank
    worker_world_size = args.num_shards * ddp_world_size

    for task_name in task_names:
        task = build_task(task_name, args.mbpp_config)
        num_problems = len(task) if args.max_problems < 0 else min(len(task), args.max_problems)
        for mode in decode_modes:
            shard_path = os.path.join(
                args.output_dir,
                f"{task_name}_{mode}_shard{args.shard_index:03d}_rank{ddp_rank}.jsonl",
            )
            passed_local, total_local = 0, 0
            with open(shard_path, "w", encoding="utf-8") as f:
                for idx in range(worker_rank, num_problems, worker_world_size):
                    problem = task[idx]
                    prompt_ids = tokenizer.encode(problem.prompt, prepend=bos)
                    results, _ = engine.generate_batch(
                        prompt_ids,
                        num_samples=1,
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        decode_mode=mode,
                        use_calculator=False,
                    )
                    completion = tokenizer.decode(results[0][len(prompt_ids):])
                    passed, error = evaluate_problem(problem, completion, args.execution_timeout)
                    passed_local += int(passed)
                    total_local += 1
                    record = {
                        "task": problem.task,
                        "task_id": problem.task_id,
                        "decode_mode": mode,
                        "passed": passed,
                        "error": error,
                        "prompt": problem.prompt,
                        "completion": completion,
                        "truncated_completion": truncate_completion(completion, problem.task),
                        "metadata": problem.metadata,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    print(
                        f"\r\033[Krank {ddp_rank} {task_name}/{mode}: "
                        f"{passed_local}/{total_local} ({100 * passed_local / max(total_local, 1):.2f}%)",
                        end="",
                        flush=True,
                    )
            print()
            local_stats[(task_name, mode)] = [passed_local, total_local]

    metrics = {}
    for key, (passed_local, total_local) in local_stats.items():
        counts = torch.tensor([passed_local, total_local], dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        task_name, mode = key
        passed = int(counts[0].item())
        total = int(counts[1].item())
        metrics[f"{task_name}/{mode}"] = {
            "passed": passed,
            "total": total,
            "accuracy": passed / total if total else 0.0,
        }

    if ddp_rank == 0:
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "meta": meta, "metrics": metrics}, f, indent=2)
        summary_path = os.path.join(args.output_dir, "summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"# Code eval: {args.model_tag} step {args.step}\n\n")
            f.write("| task/mode | pass@1 |\n")
            f.write("|---|---:|\n")
            for name, metric in metrics.items():
                f.write(
                    f"| {name} | {metric['passed']}/{metric['total']} "
                    f"({100 * metric['accuracy']:.2f}%) |\n"
                )
        print0(json.dumps(metrics, indent=2))
        print0(f"Wrote {metrics_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
