"""Merge JSONL shards from fbt_experiments.evaluate_code."""

import argparse
import glob
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Merge code eval shards")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stats = defaultdict(lambda: [0, 0])
    for path in sorted(glob.glob(os.path.join(args.output_dir, "*_shard*_rank*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = f"{rec['task']}/{rec['decode_mode']}"
                stats[key][0] += int(bool(rec["passed"]))
                stats[key][1] += 1

    metrics = {
        key: {
            "passed": passed,
            "total": total,
            "accuracy": passed / total if total else 0.0,
        }
        for key, (passed, total) in sorted(stats.items())
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics}, f, indent=2)

    summary_path = os.path.join(args.output_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Code eval summary\n\n")
        f.write("| task/mode | pass@1 |\n")
        f.write("|---|---:|\n")
        for name, metric in metrics.items():
            f.write(
                f"| {name} | {metric['passed']}/{metric['total']} "
                f"({100 * metric['accuracy']:.2f}%) |\n"
            )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
