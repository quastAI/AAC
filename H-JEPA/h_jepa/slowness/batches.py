"""SIGReg of the same latents under different batch compositions (no training).

A training batch is B = 128 clips of B different episodes ("iid"). SIGReg sees only the marginal of the batch, so what a
batch is made of decides what variance it can ask for. Here the F1 latents are put into other batches and SIGReg is
computed with the module of the training (`loss.SIGReg`, the same knots and 1024 projections, a fixed seed per batch):

    iid        B frames of B different episodes (what the training does)
    episodes   E episodes x K = B / E frames spread over each episode (slow factors still vary inside it)
    window     E episodes x K frames of one window of an episode (slow factors nearly constant inside it)

`centered`: the group mean is removed and the group is rescaled by sqrt(K / (K - 1)) (centering over K samples shrinks
the variance by (K - 1) / K); the groups are the E x K groups of the composition, and for `iid` the same number of
random groups: the null, since nothing ties a group of an iid batch together. A rise of the centered value above the null
is what episode identity (window identity) pays for the isotropy.
"""

from __future__ import annotations

import numpy as np
import torch

from loss import SIGReg

BATCH = 128


def draw(x: np.ndarray, kind: str, episodes: int, rng: np.random.Generator, batch: int = BATCH):
    """One batch of x (E_all, W, L, D): the frames (batch, D) and the group of each frame (batch,)."""
    E_all, W, L, _ = x.shape
    if kind == "iid":
        ep = rng.choice(E_all, batch, replace=False)
        frame = x[ep, rng.integers(0, W, batch), rng.integers(0, L, batch)]
        return frame, np.arange(batch) % episodes  # random groups of the same size: the centering null
    K = batch // episodes
    chosen = rng.choice(E_all, episodes, replace=False)
    frames = []
    for ep in chosen:
        if kind == "episodes":  # one frame from each of K equal slices of the episode's W * L frames
            edges = np.linspace(0, W * L, K + 1).astype(int)
            pos = np.array([rng.integers(edges[k], max(edges[k + 1], edges[k] + 1)) for k in range(K)])
            frames.append(x[ep].reshape(W * L, -1)[pos])
        elif kind == "window":
            frames.append(x[ep, rng.integers(0, W)][rng.choice(L, K, replace=False)])
        else:
            raise ValueError(kind)
    return np.concatenate(frames), np.repeat(np.arange(episodes), K)


def center(frames: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Remove each group's mean and undo the shrinkage of centering over K samples."""
    out = frames.astype(np.float64).copy()
    for g in np.unique(group):
        rows = group == g
        k = rows.sum()
        out[rows] = (out[rows] - out[rows].mean(0)) * np.sqrt(k / (k - 1))
    return out


@torch.no_grad()
def sigreg(frames: np.ndarray, seed: int, module: SIGReg | None = None) -> float:
    module = module or SIGReg()
    torch.manual_seed(seed)
    return float(module(torch.from_numpy(np.ascontiguousarray(frames, np.float32))[None]))


def compare(x: np.ndarray, episodes=(4, 8, 16), n: int = 100, seed: int = 0, batch: int = BATCH) -> list[dict]:
    """Per composition and E: SIGReg raw and centered (mean and standard deviation over n batches)."""
    module = SIGReg()
    rows = []
    for kind in ("iid", "episodes", "window"):
        for e in episodes:
            rng = np.random.default_rng(seed)
            raw, cen = [], []
            for i in range(n):
                frames, group = draw(x, kind, e, rng, batch)
                raw.append(sigreg(frames, seed + i, module))
                cen.append(sigreg(center(frames, group), seed + i, module))
            rows.append(
                {
                    "kind": kind,
                    "episodes": e,
                    "frames_per_episode": batch // e,
                    "raw": (float(np.mean(raw)), float(np.std(raw))),
                    "centered": (float(np.mean(cen)), float(np.std(cen))),
                }
            )
    return rows
