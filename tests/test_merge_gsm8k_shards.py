"""CPU-only checks for validated GSM8K shard merging."""

import json
import tempfile
import unittest
from pathlib import Path

from fbt_experiments.merge_gsm8k_shards import (
    MODES,
    MergeError,
    build_gsm8k_prompt,
    merge_shards,
    summarize_gsm8k_records,
)


def _dump_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _record(index, correct_modes=()):
    reference = str(100 + index)
    source_context = _source_context(index)
    modes = {}
    for mode_index, mode in enumerate(MODES):
        predicted = reference if mode in correct_modes else str(900 + mode_index)
        token_ids = [7, index + 20] if mode != "fused" else [8, index + 20]
        modes[mode] = {
            "completion": f"The answer is {predicted}.",
            "raw_completion_through_stop": f"The answer is {predicted}.",
            "completion_token_ids": token_ids,
            "completion_tokens": len(token_ids),
            "sampled_tokens": len(token_ids),
            "predicted_answer": predicted,
            "parse_method": "answer_phrase",
            "answer_parsed": True,
            "correct": mode in correct_modes,
            "seconds": 1.0 + mode_index,
            "tokens_per_second": len(token_ids) / (1.0 + mode_index),
            "stop_reason": "terminal_token",
        }
    return {
        "example_index": index,
        "gsm8k_shots": 5,
        "prompt": build_gsm8k_prompt(source_context, 5) + "\n\nA:",
        "prompt_tokens": 20,
        "reference_answer": reference,
        "modes": modes,
        "pairwise": {
            "standard_soft_same_first_token": True,
            "standard_soft_identical": True,
            "standard_fused_identical": False,
            "soft_fused_identical": False,
        },
    }


def _source_context(index):
    demonstrations = [f"Q: demo {demo}\n\nA: The answer is {demo}." for demo in range(8)]
    target = f"Q: synthetic problem {index}"
    return "\n\n".join([*demonstrations, target])


class TestMergeGsm8kShards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache_dir = self.root / "cache"
        dataset_path = (
            self.cache_dir
            / "eval_bundle/eval_data/symbolic_problem_solving/gsm8k_prepended_8shot.jsonl"
        )
        dataset_path.parent.mkdir(parents=True)
        with dataset_path.open("w", encoding="utf-8") as handle:
            for index in range(8):
                handle.write(
                    json.dumps({"context": _source_context(index), "answer": str(100 + index)})
                    + "\n"
                )
        self.base_config = {
            "checkpoint": "/checkpoints/model_000050.pt",
            "checkpoint_size_bytes": 12345,
            "matching_meta": "/checkpoints/meta_000050.json",
            "nanochat_base_dir": str(self.cache_dir),
            "step": 50,
            "device": "cuda",
            "torch_version": "test",
            "compute_dtype": "torch.bfloat16",
            "compute_dtype_reason": "test",
            "cuda_device": "synthetic",
            "seed": 42,
            "gsm8k_shots": 5,
            "max_new_tokens": 192,
            "skip_continuation": True,
            "skip_gsm8k": False,
            "modes": list(MODES),
        }
        self.meta = {
            "step": 50,
            "total_batch_size": 16,
            "user_config": {
                "num_forward_passes": 3,
                "feedback_start_fraction": 0.5,
                "feedback_prefix_mixin": False,
            },
        }

    def make_shard(self, name, start, records, **config_updates):
        path = self.root / name
        path.mkdir()
        config = {
            **self.base_config,
            "gsm8k_start": start,
            "num_gsm8k": len(records),
            **config_updates,
        }
        _dump_json(path / "run_config.json", config)
        _dump_json(path / "checkpoint_meta.json", self.meta)
        _dump_json(
            path / "metrics.json",
            {
                "checkpoint": config["checkpoint"],
                "step": config["step"],
                "modes": list(MODES),
                "gsm8k": summarize_gsm8k_records(records, config["gsm8k_shots"]),
            },
        )
        with (path / "gsm8k_generations.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return path

    def test_merge_sorts_rows_and_recomputes_metrics_and_provenance(self):
        first = self.make_shard(
            "shard_a",
            0,
            [_record(0, ("standard", "soft")), _record(1, ())],
        )
        second = self.make_shard(
            "shard_b",
            2,
            [_record(2, ("standard",)), _record(3, ("fused",))],
        )
        output = self.root / "merged"

        # Deliberately supply the shards out of order.
        merge_shards(
            [second, first],
            output,
            expected_count=4,
            command=["synthetic-merge"],
        )

        rows = [
            json.loads(line)
            for line in (output / "gsm8k_generations.jsonl").read_text().splitlines()
        ]
        self.assertEqual([row["example_index"] for row in rows], [0, 1, 2, 3])
        metrics = json.loads((output / "metrics.json").read_text())["gsm8k"]
        self.assertEqual(metrics["standard"]["correct"], 2)
        self.assertEqual(metrics["soft"]["correct"], 1)
        self.assertEqual(metrics["fused"]["correct"], 1)
        self.assertEqual(metrics["standard"]["examples"], 4)
        self.assertIn("accuracy_wilson_95", metrics["standard"])
        self.assertIn("exact_mcnemar_p", metrics["paired_accuracy"]["standard_vs_soft"])

        config = json.loads((output / "run_config.json").read_text())
        self.assertTrue(config["merged_from_shards"])
        self.assertEqual(config["gsm8k_start"], 0)
        self.assertEqual(config["num_gsm8k"], 4)
        manifest = json.loads((output / "merge_manifest.json").read_text())
        self.assertEqual(len(manifest["sources"]), 2)
        self.assertEqual(manifest["last_example_index"], 3)
        self.assertIn("sha256", manifest["outputs"]["metrics.json"])
        self.assertIn("4 problems", (output / "summary.md").read_text())

    def test_rejects_gap_and_protocol_mismatch(self):
        first = self.make_shard("shard_a", 0, [_record(0), _record(1)])
        gap = self.make_shard("shard_gap", 3, [_record(3), _record(4)])
        with self.assertRaisesRegex(MergeError, "not one unique contiguous range"):
            merge_shards([first, gap], self.root / "gap-output")

        incompatible = self.make_shard(
            "shard_incompatible",
            2,
            [_record(2), _record(3)],
            max_new_tokens=128,
        )
        with self.assertRaisesRegex(MergeError, "incompatible run_config.max_new_tokens"):
            merge_shards([first, incompatible], self.root / "protocol-output")

    def test_refuses_to_overwrite_destination(self):
        first = self.make_shard("shard_a", 0, [_record(0)])
        second = self.make_shard("shard_b", 1, [_record(1)])
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("do not replace", encoding="utf-8")

        with self.assertRaisesRegex(MergeError, "Refusing to overwrite"):
            merge_shards([first, second], output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "do not replace")


if __name__ == "__main__":
    unittest.main()
