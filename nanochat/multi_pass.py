"""
The K-pass latent-feedback training objective with optional recomputation.
GPT.forward calls multi_pass_loss for K > 1.
"""

from collections import namedtuple

import torch
from torch.utils.checkpoint import checkpoint

RECOMPUTE_MODES = ("none", "pass")

MultiPassParts = namedtuple("MultiPassParts", ["token_inputs", "run_pass", "fuse"])


def make_parts(model, loss_reduction='mean'):
    """Make the parts of the K-pass loop for a model. The closures find the model methods at call time."""

    def token_inputs(idx):
        token_embeddings = model._embed_tokens(idx)
        return token_embeddings, model._prepare_token_inputs(token_embeddings, None)

    def run_pass(idx, inputs, targets):
        # Returns (loss, top-layer hidden states) of one full pass: the Transformer trunk and the output head.
        return model._forward_once(idx, inputs, targets, None, loss_reduction)

    def fuse(hidden, token_embeddings, token_inputs, feedback_mask, feedback_jitter):
        previous_hidden = hidden[:, :-1]
        if model.training and feedback_jitter > 0.0:
            noise = torch.empty_like(previous_hidden).uniform_(-feedback_jitter, feedback_jitter)
            previous_hidden = previous_hidden + noise
        fused_suffix = model.latent_feedback(previous_hidden, token_embeddings[:, 1:])
        fused_inputs = torch.cat((token_inputs[:, :1], fused_suffix), dim=1)
        return torch.where(feedback_mask.unsqueeze(-1), fused_inputs, token_inputs)

    return MultiPassParts(token_inputs, run_pass, fuse)


def multi_pass_loss(
    model,
    idx,
    targets,
    num_forward_passes,
    feedback_masks=None,
    feedback_jitter=0.02,
    loss_reduction='mean',
    return_loss_components=False,
    parts=None,
    checkpoint_passes=False,
):
    """
    L = L_standard + mean(L_feedback)

    Pass 1 is a usual forward pass. Each subsequent pass moves the top-layer states of
    the previous pass 1 position to the right. The pass fuses these states with the
    token embeddings where feedback_masks is True. Then the pass runs the full model again.

    parts: the parts from make_parts, compiled or not compiled. The default is the methods of the model.
    checkpoint_passes: if True, the backward pass computes each pass again, but not the last pass.
    """
    assert isinstance(num_forward_passes, int) and num_forward_passes >= 1
    assert feedback_jitter >= 0.0
    assert targets is not None, "Multi-pass feedback is currently implemented only for the training objective"
    if num_forward_passes > 1:
        assert model.latent_feedback is not None, "Multi-pass training requires GPTConfig(latent_feedback=True)"
        B, T = idx.shape
        assert T > 1, "Multi-pass feedback requires at least two token positions"
        assert feedback_masks is not None, "Multi-pass feedback requires explicit prefix/reset masks"
        expected_shape = (num_forward_passes - 1, B, T)
        assert feedback_masks.shape == expected_shape, f"Expected feedback_masks shape {expected_shape}, got {feedback_masks.shape}"
        assert feedback_masks.dtype == torch.bool
    if parts is None:
        parts = make_parts(model, loss_reduction)
    # Recomputation is not necessary when autograd records no graph.
    checkpoint_passes = checkpoint_passes and torch.is_grad_enabled()

    token_embeddings, token_inputs = parts.token_inputs(idx)
    inputs = token_inputs
    pass_losses = []
    for pass_idx in range(num_forward_passes):
        last_pass = pass_idx == num_forward_passes - 1
        if checkpoint_passes and not last_pass:
            # PyTorch recommends use_reentrant=False. But with False, all recomputed passes are
            # in one large graph. All passes use the same parameters. Thus autograd holds the
            # accumulated parameter gradients of the later passes while it computes the gradients
            # of the earlier passes. Until autograd computes the gradients of the first pass, one
            # more copy of the parameter gradients is in memory. With use_reentrant=True, each
            # recomputed pass does its own backward pass and accumulates its gradients immediately.
            # The cost is 2 limits. Only loss.backward() gives the gradients (torch.autograd.grad
            # does not). The input of the pass must have requires_grad=True.
            assert inputs.requires_grad, (
                "Pass recomputation is not possible: the pass input has requires_grad=False, and the "
                "recomputed pass then gets no gradients. Use recompute='none' if the embeddings are frozen."
            )
            pass_loss, hidden = checkpoint(parts.run_pass, idx, inputs, targets, use_reentrant=True)
        else:
            # The backward pass starts at the last pass, so a recomputation of the last pass saves no memory.
            pass_loss, hidden = parts.run_pass(idx, inputs, targets)
        pass_losses.append(pass_loss)
        if not last_pass:
            # The jitter and the fusion are outside the checkpoint. Thus the code samples the noise
            # 1 time for each pass, and a recomputed pass gets the same inputs as the initial pass.
            inputs = parts.fuse(hidden, token_embeddings, token_inputs, feedback_masks[pass_idx], feedback_jitter)

    output, feedback_losses = pass_losses[0], pass_losses[1:]
    if not feedback_losses:
        return (output, output.unsqueeze(0)) if return_loss_components else output
    feedback_loss = torch.stack(feedback_losses, dim=0).mean(dim=0)
    total_loss = output + feedback_loss
    if return_loss_components:
        return total_loss, torch.stack((output, *feedback_losses), dim=0)
    return total_loss


def _default_compile(fn):
    return torch.compile(fn, dynamic=False)


class RecomputeForward:
    """
    A replacement for the training call of GPT.forward, with the same arguments.

    Checkpoints operate only when the pass loop runs outside compiled code. Thus this
    class compiles the parts separately and does not compile the full model.

    recompute="pass": the backward pass computes each pass again, but not the last pass.
    recompute="none": the same computation with separate parts and no checkpoints.
    compile_fn: the function that wraps each part. Use `lambda fn: fn` for no compilation.
    """

    def __init__(self, model, recompute="pass", compile_fn=_default_compile):
        if recompute not in RECOMPUTE_MODES:
            raise ValueError(f"recompute must be one of {RECOMPUTE_MODES}, got {recompute!r}")
        self.model = model
        self.recompute = recompute
        self.parts = MultiPassParts(*(compile_fn(fn) for fn in make_parts(model)))

    def __call__(
        self,
        idx,
        targets,
        num_forward_passes=1,
        feedback_masks=None,
        feedback_jitter=0.02,
        return_loss_components=False,
    ):
        return multi_pass_loss(
            self.model,
            idx,
            targets,
            num_forward_passes,
            feedback_masks=feedback_masks,
            feedback_jitter=feedback_jitter,
            return_loss_components=return_loss_components,
            parts=self.parts,
            checkpoint_passes=self.recompute == "pass",
        )
