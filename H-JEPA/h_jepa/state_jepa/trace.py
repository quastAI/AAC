"""What the latent holds at one step of the training, on a fixed sample of held-out episodes.

The sample is `slowness`'s nested one (episodes x windows x consecutive frames of the validation file), so the numbers
read like the slowness reports of B1. Per check:

    r2              linear read (ridge, held-out episodes) of every factor of `factors.scalar_targets`: the inventory items,
                    the vitals, daylight, the mobs of each type in view, the tiles of each block type in view
    near            variance-weighted R^2 of the four tiles next to the player (block type, one-hot) from the latent
    rank            effective rank of the latent; `dead_axes`: principal axes with under a tenth of the mean variance
    clip_share      the share of the latent's variance inside a clip of 4 frames (what changes from frame to frame)
    pred_over_copy  the predictor's one-step error over the error of copying the last latent, on the sample's windows
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from probing import targets as tg
from slowness import factors as fa
from slowness import sample as sm
from slowness import variance as va

COLUMNS = ("state", "inventory", "vitals", "daylight", "action")
DEAD = 0.1  # a principal axis with under this fraction of the mean variance counts as dead (heuristic)


class Sample:
    """The fixed held-out sample of a run: the states, the factors to read and the episode split of the ridge fits."""

    def __init__(self, path, windows: int = sm.WINDOWS, length: int = sm.LENGTH, seed: int = 0, max_episodes=None):
        import h5py

        with h5py.File(path, "r") as f:
            lengths, offsets = f["ep_len"][:], f["ep_offset"][:]
        self.sample = sm.nested_sample(lengths, offsets, windows=windows, length=length, seed=seed, max_episodes=max_episodes)
        cols = sm.read_columns(path, self.sample, COLUMNS)
        self.state = torch.from_numpy(cols["state"].astype(np.float32))  # (E, W, L, 1345)
        self.action = torch.from_numpy(np.nan_to_num(cols["action"][..., 0], nan=-1).astype(np.int64))  # (E, W, L)
        self.targets = fa.scalar_targets(cols)
        blocks, _ = tg.tile_maps(self.state.reshape(-1, tg.STATE_DIM))
        near = F.one_hot(blocks[:, list(fa.NEIGHBOURS)], tg.NUM_BLOCKS).flatten(1)  # the four neighbours, as `factors.near`
        self.near = near.numpy().astype(np.float32).reshape(*self.state.shape[:3], -1)  # (E, W, L, 68)
        self.split = fa.episode_split(self.state.shape[0], seed=seed)


@torch.no_grad()
def check(model, s: Sample, chunk: int = 8192) -> dict:
    """The readouts of the model's latent on the sample (see the module docstring)."""
    was = model.training
    model.eval()
    try:
        E, W, L, _ = s.state.shape
        flat = s.state.reshape(-1, s.state.shape[-1])
        z = torch.cat([model.encode(flat[a : a + chunk]) for a in range(0, len(flat), chunk)]).reshape(E, W, L, -1)
        error, copy = prediction_errors(model, z, s.action)
    finally:
        model.train(was)
    x = z.numpy().astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):  # a factor constant in a part of the split: no R^2 (None)
        r2 = fa.decodability(x, s.targets, s.split)
        tr, te = s.split
        xs_tr, xs_te = fa.standardise(x[tr].reshape(-1, x.shape[-1]), x[te].reshape(-1, x.shape[-1]))
        y_tr, y_te = s.near[tr].reshape(-1, s.near.shape[-1]), s.near[te].reshape(-1, s.near.shape[-1])
        near = fa.r2_weighted(y_te, fa.ridge_cv(xs_tr, y_tr, xs_te))
    values = va.pca(x)[2]
    return {
        "r2": {k: (None if np.isnan(v) else float(v)) for k, v in r2.items()},
        "near": float(near),
        "rank": va.effective_rank(values),
        "dead_axes": int((values < DEAD * values.mean()).sum()),
        "clip_share": va.shares(va.decompose(x, clip=4))["clip"],
        "pred_error": error,
        "pred_over_copy": error / max(copy, 1e-12),
    }


def prediction_errors(model, z: torch.Tensor, action: torch.Tensor) -> tuple[float, float]:
    """Mean squared one-step error of the predictor and of copying the last latent, over the frames of the sample's
    windows that have `history` frames before them and an action at the frame before."""
    E, W, L, D = z.shape
    h = model.cfg.history
    zz, aa = z.reshape(E * W, L, D), action.reshape(E * W, L)
    pred = model.predict(zz, aa)  # (B, L - 1, D): frame t from frames before it
    target, last = zz[:, 1:], zz[:, :-1]
    keep = (torch.arange(1, L) >= h)[None, :] & (aa[:, :-1] >= 0)
    error = (pred - target).pow(2).mean(-1)[keep].mean()
    copy = (last - target).pow(2).mean(-1)[keep].mean()
    return float(error), float(copy)
