"""The predictor's error per principal axis of the latent: where does the one-step loss come from, and which variance is free?

The training loss is the squared error between the predicted and the encoded latent of the next frame. For a latent
whose axes are the principal axes of the pooled frames (`variance.pca`), the error splits exactly over the axes. Per axis:

    variance        what SIGReg sees of it
    error           the predictor's squared error on that axis (teacher-forced, the training's history)
    copy            the squared error of copying the last latent on that axis (how fast the axis moves)
    error / variance   what the loss still has to pay for the axis; near 0 is a free direction
    error / copy       below 1 the predictor knows something copying does not (the action, the dynamics)

`free_share`: the variance-weighted share of axes whose error is under `FREE` of their variance. High variance with a
near-zero error is what a slow, cheap feature looks like to the loss.
"""

from __future__ import annotations

import numpy as np
import torch

from . import variance as va

FREE = 0.05  # an axis is free when the predictor's error is under 5% of its variance (a heuristic)


def action_embedding(jepa, action: np.ndarray, num_actions: int) -> torch.Tensor:
    """The predictor's action input (B, T, A): the one-hot of the executed action (a NaN, the last frame of an episode,
    is all zero), through the model's action encoder and embedding as in training. `action` is (B, T, 1)."""
    a = torch.from_numpy(action[..., 0])
    valid = ~torch.isnan(a)
    onehot = torch.nn.functional.one_hot(a.nan_to_num(0).long(), num_actions).float() * valid.unsqueeze(-1)
    with torch.no_grad():
        return jepa.action_embed(jepa.action_encoder(onehot)).float().cpu()


@torch.no_grad()
def one_step(jepa, latent: np.ndarray, action: np.ndarray, num_actions: int, *, history: int, device="cpu", chunk=512):
    """Predicted and encoded next latents and the last latent, over every window: (N, D) each, for the transitions
    t -> t + 1 with `history` frames of context, and the validity of the action. The inputs are (E, W, L, D) and
    (E, W, L, 1)."""
    E, W, L, D = latent.shape
    emb = torch.from_numpy(latent).reshape(E * W, L, D).float()
    act = action_embedding(jepa, action.reshape(E * W, L, 1), num_actions)
    n = L - history  # contexts [t - history + 1, t] for t = history - 1 .. L - 2
    ctx = emb.unfold(1, history, 1)[:, :n].permute(0, 1, 3, 2)  # (B, n, history, D)
    ctx_act = act.unfold(1, history, 1)[:, :n].permute(0, 1, 3, 2)
    target = emb[:, history:]
    valid = ~torch.from_numpy(np.isnan(action.reshape(E * W, L))[:, history - 1 : L - 1])
    ctx, ctx_act = ctx.flatten(0, 1), ctx_act.flatten(0, 1)
    out = []
    jepa.predictor.eval()
    for a in range(0, len(ctx), chunk):
        out.append(jepa.predict(ctx[a : a + chunk].to(device), ctx_act[a : a + chunk].to(device))[:, -1].float().cpu())
    keep = valid.flatten()
    return torch.cat(out)[keep].numpy(), target.flatten(0, 1)[keep].numpy(), ctx[:, -1][keep].numpy()


def per_axis_error(latent: np.ndarray, pred: np.ndarray, target: np.ndarray, last: np.ndarray) -> dict:
    """Variance, error and copy error on each principal axis of the pooled latent (largest first)."""
    mean, axes, values = va.pca(latent)
    project = lambda x: (np.asarray(x, np.float64) - mean) @ axes  # noqa: E731
    z_pred, z_target, z_last = project(pred), project(target), project(last)
    error = ((z_pred - z_target) ** 2).mean(0)
    copy = ((z_last - z_target) ** 2).mean(0)
    free = error / np.maximum(values, 1e-12) < FREE
    return {
        "variance": values,
        "error": error,
        "copy": copy,
        "free": free,
        "free_share": float(values[free].sum() / values.sum()),
        "error_share_of_variance": float(error.sum() / values.sum()),
        "pred_over_copy": float(error.sum() / copy.sum()),
    }
