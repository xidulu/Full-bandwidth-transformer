"""
Stack-Edu text-only coding data.
https://huggingface.co/datasets/HuggingFaceTB/stack-edu

The Hugging Face dataset contains Software Heritage blob ids, not file
contents. Run scripts/materialize_stackedu.py first to build the local parquet
cache consumed by this task.
"""

import os

import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from tasks.common import HubDataset, Task


class StackEduText(Task):
    """Materialized Stack-Edu source files as plain causal-LM text examples."""

    def __init__(self, path=None, ds=None, shuffle_seed=42, **kwargs):
        super().__init__(**kwargs)
        if path is None:
            path = os.path.join(
                get_base_dir(),
                "task_data",
                "HuggingFaceTB--stack-edu",
                "materialized",
                "Python",
                "stackedu_python_budget.parquet",
            )
        self.path = path
        self.shuffle_seed = shuffle_seed
        if ds is not None:
            self.ds = ds
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Materialized Stack-Edu parquet not found: {path}. "
                    "Run scripts/materialize_stackedu.py first."
                )
            table = pq.read_table(path)
            if "text" not in table.column_names:
                raise ValueError(f"Stack-Edu parquet must contain a 'text' column: {path}")
            self.ds = HubDataset(table).shuffle(seed=shuffle_seed)

    def slice(self, **kwargs):
        return StackEduText(path=self.path, ds=self.ds, shuffle_seed=self.shuffle_seed, **kwargs)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        text = row["text"]
        assert isinstance(text, str) and text, "Stack-Edu text must be non-empty"
        return {
            "text": text,
            "language": row.get("language"),
            "repo_name": row.get("repo_name"),
            "path": row.get("path"),
            "blob_id": row.get("blob_id"),
            "length_bytes": row.get("length_bytes"),
            "score": row.get("score"),
        }
