"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist

from nanochat.gpt import build_feedback_mask


@torch.no_grad()
def evaluate_bpb_per_pass(
    model,
    batches,
    steps,
    token_bytes,
    *,
    num_forward_passes,
    bos_token_id=None,
):
    """
    Return one bits-per-byte value for each recurrent forward pass.

    Instead of the naive 'mean loss', BPB is a tokenization vocab size-independent
    metric, meaning you are still comparing apples:apples if you change the vocab size.
    The way this works is that instead of just calculating the average loss as usual,
    you calculate the sum loss, and independently also the sum bytes (of all the target
    tokens), and divide. This normalizes the loss by the number of bytes that the target
    tokens represent.

    Feedback validation is deterministic and uses the full fused suffix on every later
    pass. Prefix mixin and jitter are training-only; packed BOS positions remain plain.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    token_bytes is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    assert isinstance(num_forward_passes, int) and num_forward_passes >= 1
    if num_forward_passes > 1:
        assert bos_token_id is not None, "Multi-pass BPB evaluation requires bos_token_id"

    total_nats = torch.zeros(
        num_forward_passes,
        dtype=torch.float32,
        device=model.get_device(),
    )
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        if num_forward_passes == 1:
            loss_by_pass = model(x, y, loss_reduction='none').reshape(1, -1)
        else:
            # Each later pass performs a full fused prefill. Reuse the deterministic
            # packed-document reset mask; only the carried hidden state changes by pass.
            feedback_mask = build_feedback_mask(x, bos_token_id, prefix_mixin=False)
            feedback_masks = feedback_mask.unsqueeze(0).expand(
                num_forward_passes - 1, -1, -1
            )
            _, loss_by_pass = model(
                x,
                y,
                loss_reduction='none',
                num_forward_passes=num_forward_passes,
                feedback_masks=feedback_masks,
                feedback_jitter=0.0,
                return_loss_components=True,
            )
            loss_by_pass = loss_by_pass.reshape(num_forward_passes, -1)

        y = y.reshape(-1)
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # Any target token < 0 is ignored; do not index token_bytes with negatives.
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            num_bytes = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype),
            )
        else:
            num_bytes = token_bytes[y]

        counted_tokens = num_bytes > 0
        total_nats += (loss_by_pass * counted_tokens.unsqueeze(0)).sum(dim=1)
        total_bytes += num_bytes.sum()

    # Sum reduce across all ranks.
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    total_bytes_int = total_bytes.item()
    if total_bytes_int == 0:
        return [float('inf')] * num_forward_passes
    return (total_nats / (math.log(2) * total_bytes_int)).tolist()


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Return standard one-pass bits per byte (BPB).

    This preserves the original scalar API. Use evaluate_bpb_per_pass for recurrent
    latent-feedback validation.

    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    return evaluate_bpb_per_pass(
        model,
        batches,
        steps,
        token_bytes,
        num_forward_passes=1,
    )[0]
