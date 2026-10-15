"""Whole code sequences of the training data, for the upper-level CEM.

The planner of a level above 1 searches discrete codes. Searching all combinations uniformly lets
it pick codes that never occur together in the data, which the model has no reason to predict
well. The bank holds, per level, real sequences of that level's codes: encode clips of the
training episodes through the whole hierarchy, take the code indices of every upper step, and cut
windows of the planning horizon. `CategoricalCEMSolver.set_proposal` mixes them into its samples
(a stopgap until learned actors propose the codes).
"""

from __future__ import annotations

import math

import h5py
import numpy as np
import torch
from loguru import logger as logging
from torchvision import tv_tensors


def steps_per_step(model, level: int) -> int:
    """Level-1 steps one step of `level` spans (the product of the strides up to it)."""
    return max(1, math.prod(int(model.get_level(k).temporal_stride) for k in range(2, level + 1)))


def clip_frames(model, horizons: dict[int, int]) -> int:
    """Level-1 frames of a clip that holds `horizons[level]` complete steps of every level."""
    return max(h * steps_per_step(model, level) + steps_per_step(model, level) + 1 for level, h in horizons.items())


@torch.no_grad()
def build_code_bank(
    model,
    h5_path,
    transform,
    *,
    num_actions: int,
    horizons: dict[int, int],
    n_clips: int = 512,
    exclude_sources=(2,),
    seed: int = 0,
    batch: int = 16,
) -> dict[int, torch.Tensor]:
    """{level: (M, horizon, groups) code indices} for the levels in `horizons` (levels 2 and up).

    `transform` turns an uint8 CHW image into a model input (the evaluation's pixel transform).
    Clips come from random places of the episodes whose source is not in `exclude_sources`
    (uniform random episodes are noise as macro-actions). Returns {} when no episode is long enough.
    """
    frames = clip_frames(model, horizons)
    with h5py.File(h5_path, "r") as f:
        lengths, offsets = f["ep_len"][:].astype(np.int64), f["ep_offset"][:].astype(np.int64)
        source = f["source"][offsets] if "source" in f else np.zeros(len(lengths), np.int8)
        windows = np.where(np.isin(source, list(exclude_sources)), 0, np.maximum(lengths - frames + 1, 0))
        if windows.sum() == 0:
            logging.warning(f"code bank: no episode of {h5_path} has {frames} frames of an allowed source")
            return {}
        rng = np.random.default_rng(seed)
        episode = rng.choice(len(lengths), size=n_clips, p=windows / windows.sum())
        start = (rng.random(n_clips) * windows[episode]).astype(np.int64)
        order = np.argsort(offsets[episode] + start)  # read the file in order
        pixels = np.stack([f["pixels"][offsets[e] + s : offsets[e] + s + frames] for e, s in zip(episode[order], start[order])])
        actions = np.stack([f["action"][offsets[e] + s : offsets[e] + s + frames, 0] for e, s in zip(episode[order], start[order])])

    device = next(model.parameters()).device
    levels = sorted(horizons)
    jepas = [model.get_level(k) for k in range(1, model.num_levels + 1)]
    saved = [j.target_length for j in jepas]
    was_training = model.training
    for j in jepas:
        j.target_length = None  # keep every step of every level
    model.eval()
    bank = {level: [] for level in levels}
    try:
        for s in range(0, len(pixels), batch):
            x = torch.stack(
                [torch.stack([transform(tv_tensors.Image(np.transpose(fr, (2, 0, 1)))) for fr in clip]) for clip in pixels[s : s + batch]]
            )
            a = torch.from_numpy(actions[s : s + batch])
            onehot = torch.zeros(*a.shape, num_actions)
            valid = torch.isfinite(a)  # the last action of an episode is NaN: no action
            onehot[valid] = torch.nn.functional.one_hot(a[valid].long(), num_actions).float()
            out = model.encode_hierarchical({"pixels_level1": x.to(device), "action_level1": onehot.to(device)})
            for level in levels:
                codes = out[f"action_index_{level}"][:, :-1]  # the last chunk is cut by the clip
                windows_ = codes.unfold(1, horizons[level], 1).permute(0, 1, 3, 2)  # (B, W, H, groups)
                bank[level].append(windows_.flatten(0, 1).cpu())
    finally:
        for j, t in zip(jepas, saved):
            j.target_length = t
        model.train(was_training)
    return {level: torch.cat(parts) for level, parts in bank.items()}
