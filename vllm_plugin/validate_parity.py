"""Compare native and vLLM greedy standard decoding in separate processes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Running this file directly puts only vllm_plugin/ on sys.path. Add the
# repository root so the native Nanochat package is importable without an
# editable install of the parent project.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _checkpoint_step(path: Path) -> int:
    return int(path.stem.removeprefix("model_"))


def _prompt_ids(tokenizer, prompt: str) -> list[int]:
    return [
        tokenizer.get_bos_token_id(),
        tokenizer.encode_special("<|user_start|>"),
        *tokenizer.encode(prompt),
        tokenizer.encode_special("<|user_end|>"),
        tokenizer.encode_special("<|assistant_start|>"),
    ]


def run_native(args) -> None:
    import torch

    from nanochat.checkpoint_manager import build_model
    from nanochat.engine import Engine

    checkpoint = args.checkpoint.resolve()
    model, tokenizer, _meta = build_model(
        str(checkpoint.parent),
        _checkpoint_step(checkpoint),
        torch.device("cuda"),
        phase="eval",
    )
    prompt_ids = _prompt_ids(tokenizer, args.prompt)
    engine = Engine(model, tokenizer)
    sequences, _masks = engine.generate_batch(
        prompt_ids,
        num_samples=1,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_k=None,
        decode_mode=args.decode_mode,
        use_calculator=False,
    )
    completion_ids = sequences[0][len(prompt_ids) :]
    payload = {
        "prompt": args.prompt,
        "decode_mode": args.decode_mode,
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "completion": tokenizer.decode(completion_ids),
    }
    args.result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_vllm(args) -> None:
    from nanochat.tokenizer import get_tokenizer
    from vllm import LLM, SamplingParams

    expected = json.loads(args.result.read_text(encoding="utf-8"))
    if expected["decode_mode"] != args.decode_mode:
        raise ValueError(
            f"native result used {expected['decode_mode']!r}, "
            f"requested {args.decode_mode!r}"
        )
    exported_config = json.loads(
        (args.export_dir / "config.json").read_text(encoding="utf-8")
    )
    if exported_config["nanochat_decode_mode"] != args.decode_mode:
        raise ValueError(
            "exported model uses "
            f"{exported_config['nanochat_decode_mode']!r}, "
            f"requested {args.decode_mode!r}"
        )
    tokenizer = get_tokenizer()
    llm = LLM(
        model=str(args.export_dir.resolve()),
        skip_tokenizer_init=True,
        worker_cls="nanochat_vllm.worker.NanochatWorker",
        enforce_eager=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.6,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop_token_ids=[
            tokenizer.encode_special("<|assistant_end|>"),
            tokenizer.get_bos_token_id(),
        ],
    )
    outputs = llm.generate(
        [{"prompt_token_ids": expected["prompt_ids"]}],
        params,
    )
    actual = list(outputs[0].outputs[0].token_ids)
    wanted = expected["completion_ids"]
    if actual != wanted:
        first_difference = next(
            (
                index
                for index, (left, right) in enumerate(zip(wanted, actual))
                if left != right
            ),
            min(len(wanted), len(actual)),
        )
        raise AssertionError(
            "greedy token mismatch at completion index "
            f"{first_difference}: native={wanted[first_difference:first_difference + 8]}, "
            f"vllm={actual[first_difference:first_difference + 8]}"
        )
    print(
        json.dumps(
            {
                "match": True,
                "decode_mode": args.decode_mode,
                "num_prompt_tokens": len(expected["prompt_ids"]),
                "num_completion_tokens": len(actual),
                "completion": tokenizer.decode(actual),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("native", "vllm"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prompt", default="What is 17 times 23?")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--decode-mode",
        choices=("standard", "soft"),
        default="standard",
    )
    args = parser.parse_args()
    if args.phase == "native":
        if args.checkpoint is None:
            parser.error("native phase requires --checkpoint")
        run_native(args)
    else:
        if args.export_dir is None:
            parser.error("vllm phase requires --export-dir")
        run_vllm(args)


if __name__ == "__main__":
    main()
