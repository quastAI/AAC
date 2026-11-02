"""What the latent of a model holds, on a fixed sample of held-out episodes (the readouts of one check).

The sample is `slowness`'s nested one (episodes x windows x consecutive frames of the validation file), so the numbers read like the
slowness reports. A model is a saved `HJEPA` object: its level-1 encoder reads `obs_key` (the pixels of B1, or the simulator state of
A3 and A4). Per check:

    r2              linear read (ridge, held-out episodes) of every factor of `factors.scalar_targets`: the inventory items, the
                    vitals, daylight, the mobs of each type in view, the tiles of each block type in view
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
from slowness import predict as pe
from slowness import sample as sm
from slowness import variance as va

DEAD = 0.1  # a principal axis with under this fraction of the mean variance counts as dead (heuristic)


class Sample:
    """The fixed held-out sample of a run: the observations the model reads, the actions and the factors to read."""

    def __init__(self, path, obs_key: str = "pixels", windows: int = sm.WINDOWS, length: int = sm.LENGTH, seed: int = 0,
                 max_episodes=None):  # fmt: skip
        import h5py

        with h5py.File(path, "r") as f:
            lengths, offsets = f["ep_len"][:], f["ep_offset"][:]
        self.obs_key = obs_key
        self.layout = sm.nested_sample(lengths, offsets, windows=windows, length=length, seed=seed, max_episodes=max_episodes)
        columns = sm.FACTORS + (("pixels",) if obs_key == "pixels" else ())
        cols = sm.read_columns(path, self.layout, columns)
        self.obs = cols["pixels"] if obs_key == "pixels" else cols["state"].astype(np.float32)
        self.action = cols["action"]  # (E, W, L, 1), NaN at the end of an episode
        self.targets = fa.scalar_targets(cols)
        state = torch.from_numpy(cols["state"].reshape(-1, tg.STATE_DIM)).float()
        blocks, _ = tg.tile_maps(state)
        near = F.one_hot(blocks[:, list(fa.NEIGHBOURS)], tg.NUM_BLOCKS).flatten(1)  # the four neighbours, as `factors.near`
        self.near = near.numpy().astype(np.float32).reshape(*cols["state"].shape[:3], -1)
        self.shape = cols["state"].shape[:3]
        self.split = fa.episode_split(self.shape[0], seed=seed)


@torch.no_grad()
def encode(model, s: Sample, device: str = "cpu", chunk: int = 1024) -> np.ndarray:
    """The latent (E, W, L, D) of every frame of the sample."""
    from main_hjepa import _normalize_uint8_images_on_device
    from probing.encoder import encode_pixels

    was = model.training
    model.eval()
    model.to(device)
    try:
        flat = s.obs.reshape(-1, *s.obs.shape[3:])
        if s.obs_key == "pixels":
            z, _ = encode_pixels(model, torch.from_numpy(np.ascontiguousarray(flat.transpose(0, 3, 1, 2))),
                                 _normalize_uint8_images_on_device, device, chunk=chunk)  # fmt: skip
        else:
            enc = model.get_level(1).encoder
            z = torch.cat([enc(torch.from_numpy(flat[a : a + chunk]).to(device)).float().cpu() for a in range(0, len(flat), chunk)])
    finally:
        model.train(was)
    return z.numpy().reshape(*s.shape, -1)


@torch.no_grad()
def check(model, s: Sample, history: int, num_actions: int, device: str = "cpu") -> dict:
    """The readouts of the model's latent on the sample (see the module docstring)."""
    x = encode(model, s, device).astype(np.float64)
    jepa = model.get_level(1)
    jepa.eval()
    pred, target, last = pe.one_step(jepa, x.astype(np.float32), s.action, num_actions, history=history, device=device)
    error, copy = float(((pred - target) ** 2).mean()), float(((last - target) ** 2).mean())
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
