"""
OpenMathInstruct-2 supervised math instruction data.
https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
"""

from tasks.common import Task, load_hub_dataset


class OpenMathInstruct2(Task):
    """NVIDIA OpenMathInstruct-2 problem/solution pairs."""

    valid_splits = {"train", "train_1M", "train_2M", "train_5M"}

    def __init__(self, split="train_1M", ds=None, **kwargs):
        super().__init__(**kwargs)
        assert split in self.valid_splits, f"OpenMathInstruct-2 split must be one of {sorted(self.valid_splits)}"
        self.split = split
        self.ds = ds if ds is not None else load_hub_dataset("nvidia/OpenMathInstruct-2", split=split).shuffle(seed=42)

    def slice(self, **kwargs):
        return OpenMathInstruct2(split=self.split, ds=self.ds, **kwargs)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        problem = row["problem"]
        solution = row["generated_solution"]
        assert isinstance(problem, str) and problem, "OpenMathInstruct-2 problem must be non-empty text"
        assert isinstance(solution, str) and solution, "OpenMathInstruct-2 solution must be non-empty text"
        return {
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ],
            "expected_answer": row.get("expected_answer"),
            "problem_source": row.get("problem_source"),
        }
