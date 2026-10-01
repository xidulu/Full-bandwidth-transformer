"""Three-pass teacher-forced SOFT score; only the final pass has autograd.

Prompt positions stay ordinary in every pass. Generated input position t uses
the previous pass's hidden state at t-1. This is a finite Jacobi approximation
to recurrent SOFT decoding, not its exact sequence likelihood.
"""
import torch
from torch import nn


def configure_feedback_gradients(model, decode_mode):
    """Train the active fusion path only; dormant paths must not enter DDP."""
    if decode_mode not in ('standard', 'soft'):
        raise ValueError(f'Unsupported decode mode: {decode_mode}')
    feedback = model.latent_feedback
    if decode_mode == 'soft' and feedback is None:
        raise ValueError('Soft post-training requires a latent-feedback checkpoint')
    if feedback is not None:
        active = set(feedback.active_parameter_names()) if decode_mode == 'soft' else set()
        for name, parameter in feedback.named_parameters():
            parameter.requires_grad_(name in active)


class SoftThreePassLikelihood(nn.Module):
    def __init__(self, model, bos_token_id):
        super().__init__()
        if model.latent_feedback is None:
            raise ValueError('Three-pass scoring requires latent feedback')
        self.model = model
        self.bos_token_id = bos_token_id

    def feedback_mask(self, idx, prompt_lengths, targets=None):
        lengths = torch.as_tensor(prompt_lengths, device=idx.device, dtype=torch.long)
        if lengths.shape != (idx.shape[0],):
            raise ValueError('Expected one prompt length per sequence')
        if bool(((lengths < 1) | (lengths > idx.shape[1])).any()):
            raise ValueError('Prompt length outside the input sequence')
        positions = torch.arange(idx.shape[1], device=idx.device)[None, :]
        mask = (positions >= lengths[:, None]) & idx.ne(self.bos_token_id)
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError('Targets and inputs must have identical shapes')
            mask = mask & targets.ne(-1)
        return mask

    def _fuse(self, previous_hidden, embeddings, ordinary, mask):
        # A causal shift is required: using h[t] would implement a different
        # recurrence from Engine.generate(decode_mode='soft').
        fused = self.model.latent_feedback(previous_hidden[:, :-1], embeddings[:, 1:])
        inputs = torch.cat((ordinary[:, :1], fused), dim=1)
        return torch.where(mask.unsqueeze(-1), inputs, ordinary)

    def forward(self, idx, targets=None, *, prompt_lengths, loss_reduction='mean'):
        if idx.ndim != 2 or idx.shape[1] < 2:
            raise ValueError('Three-pass scoring requires [batch, time>=2] inputs')
        mask = self.feedback_mask(idx, prompt_lengths, targets)
        model = self.model
        with torch.no_grad():
            embeddings = model._embed_tokens(idx)
            ordinary = model._prepare_token_inputs(embeddings, None)
            hidden1 = model._run_trunk(idx, ordinary, None)
            inputs2 = self._fuse(hidden1, embeddings, ordinary, mask)
            hidden2 = model._run_trunk(idx, inputs2, None)
        # Re-embed under autograd: pass 3 must train its embedding/fusion inputs
        # as well as its Transformer and output head. Only h2 is detached.
        del hidden1, inputs2, embeddings, ordinary
        embeddings = model._embed_tokens(idx)
        ordinary = model._prepare_token_inputs(embeddings, None)
        inputs3 = self._fuse(hidden2.detach(), embeddings, ordinary, mask)
        hidden3 = model._run_trunk(idx, inputs3, None)
        # No first-/second-pass logits or auxiliary losses are constructed.
        return model._project_and_loss(hidden3, targets, loss_reduction)


class SoftReplayLikelihood(SoftThreePassLikelihood):
    """One differentiable trunk pass with detached recurrent rollout feedback.

Reproduces recurrent forward inputs at the sampled weights (up to backend
roundoff), but intentionally omits gradients through the rollout recurrence.
    """
    def forward(self, idx, targets=None, *, prompt_lengths, replay_hidden,
                loss_reduction='mean'):
        mask = self.feedback_mask(idx, prompt_lengths, targets)
        model = self.model
        if replay_hidden.shape != (*idx.shape, model.config.n_embd):
            raise ValueError('Replay hidden shape must match [batch,time,hidden]')
        embeddings = model._embed_tokens(idx)
        ordinary = model._prepare_token_inputs(embeddings, None)
        fused = model.latent_feedback(replay_hidden.detach().to(embeddings.dtype), embeddings)
        inputs = torch.where(mask.unsqueeze(-1), fused, ordinary)
        hidden = model._run_trunk(idx, inputs, None)
        return model._project_and_loss(hidden, targets, loss_reduction)
