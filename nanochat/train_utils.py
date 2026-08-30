"""Small, import-safe helpers shared by training entrypoints and their tests."""

import math


def get_feedback_start_step(num_iterations: int, feedback_start_fraction: float) -> int:
    """Return the first optimization step that uses latent feedback."""
    if not isinstance(num_iterations, int) or num_iterations <= 0:
        raise ValueError("num_iterations must be a positive integer")
    if not math.isfinite(feedback_start_fraction) or not 0.0 <= feedback_start_fraction <= 1.0:
        raise ValueError("feedback_start_fraction must be in [0, 1]")
    return math.ceil(feedback_start_fraction * num_iterations)


def get_active_forward_passes(
    step: int,
    num_iterations: int,
    max_forward_passes: int,
    feedback_start_fraction: float,
) -> int:
    """Return K for a training step: K=1 before the boundary, then max K."""
    if not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if not isinstance(max_forward_passes, int) or max_forward_passes < 1:
        raise ValueError("max_forward_passes must be a positive integer")
    start_step = get_feedback_start_step(num_iterations, feedback_start_fraction)
    return 1 if step < start_step else max_forward_passes
