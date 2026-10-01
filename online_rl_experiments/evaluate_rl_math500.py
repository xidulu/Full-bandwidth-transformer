"""Use the repository MATH-500 evaluator with an external-cache RL checkpoint."""
import os
from pathlib import Path
import re

from fbt_experiments import evaluate_math500 as evaluator


def checkpoint_paths(checkpoint):
    checkpoint = checkpoint.expanduser().resolve()
    match = re.fullmatch(r'model_(\d+)\.pt', checkpoint.name)
    if not match or not checkpoint.is_file():
        raise ValueError(f'Expected an existing model_STEP.pt: {checkpoint}')
    step = int(match.group(1))
    metadata = checkpoint.with_name(f'meta_{step:06d}.json')
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    base = Path(os.environ['NANOCHAT_BASE_DIR']).resolve()
    if not (base/'tokenizer').is_dir():
        raise FileNotFoundError(base/'tokenizer')
    return checkpoint, metadata, step, base


if __name__ == '__main__':
    # Only path resolution differs: generation, prompts and metrics are unchanged.
    evaluator.checkpoint_paths = checkpoint_paths
    evaluator.main()
