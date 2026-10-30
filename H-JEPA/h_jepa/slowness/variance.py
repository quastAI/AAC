"""The variance of a feature by time scale, for a balanced nested sample x of shape (E, W, L, D).

With population variances the total splits exactly (law of total variance, the design is balanced):

    total = episode      the episode means around the grand mean
          + window       the window means around their episode's mean (minutes apart)
          + clip_means   the means of the training clips (`clip` consecutive frames) around their window's mean
          + clip         the frames around their clip's mean (what a training clip itself shows)

The first two are slow: an episode's identity and progress; `clip` is what changes within the 4 frames a training clip
holds. Shares are variance-weighted: sums over the dimensions, so a few large dimensions weigh more than many small ones,
as they do in SIGReg's batch variance.
"""

from __future__ import annotations

import numpy as np

CLIP = 4
LEVELS = ("episode", "window", "clip_means", "clip")


def _blocks(x: np.ndarray, clip: int) -> np.ndarray:
    E, W, L, D = x.shape
    if L % clip:
        raise ValueError(f"the window length {L} is not a multiple of the clip length {clip}")
    return x.reshape(E, W, L // clip, clip, D)


def decompose(x: np.ndarray, clip: int = CLIP) -> dict[str, np.ndarray]:
    """Per dimension: the variance at each time scale (they add up to the total)."""
    b = _blocks(np.asarray(x, np.float64), clip)
    clip_mean = b.mean(3)
    window_mean = clip_mean.mean(2)
    episode_mean = window_mean.mean(1)
    grand = episode_mean.mean(0)
    out = {
        "episode": ((episode_mean - grand) ** 2).mean(0),
        "window": ((window_mean - episode_mean[:, None]) ** 2).mean((0, 1)),
        "clip_means": ((clip_mean - window_mean[:, :, None]) ** 2).mean((0, 1, 2)),
        "clip": ((b - clip_mean[:, :, :, None]) ** 2).mean((0, 1, 2, 3)),
    }
    out["total"] = ((b - grand) ** 2).mean((0, 1, 2, 3))
    return out


def shares(parts: dict[str, np.ndarray]) -> dict[str, float]:
    """The variance-weighted share of each time scale of the total (sums over dimensions)."""
    total = float(parts["total"].sum())
    return {k: float(parts[k].sum()) / total for k in LEVELS}


def null_shares(x: np.ndarray, clip: int = CLIP, n: int = 10, seed: int = 0) -> dict[str, tuple[float, float]]:
    """The shares when whole windows are dealt to random episodes: nothing links a window to its episode, so the episode
    share is what the finite sample alone gives (about (1 / W) times the window share) and the rest is unchanged.
    Returns mean and standard deviation over `n` deals."""
    rng = np.random.default_rng(seed)
    E, W = x.shape[:2]
    runs = []
    for _ in range(n):
        order = rng.permutation(E * W)
        runs.append(shares(decompose(x.reshape(E * W, *x.shape[2:])[order].reshape(x.shape), clip)))
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in LEVELS}


def pca(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, principal axes (D, D; columns, largest first) and their variances of the pooled frames."""
    flat = np.asarray(x, np.float64).reshape(-1, x.shape[-1])
    mean = flat.mean(0)
    values, vectors = np.linalg.eigh(np.cov(flat - mean, rowvar=False, bias=True))
    order = np.argsort(values)[::-1]
    return mean, vectors[:, order], np.clip(values[order], 0, None)


def lag_correlation(x: np.ndarray, lags=(1, 8, 32)) -> dict[int, np.ndarray]:
    """Per dimension: the correlation of frames `lag` apart inside a window (pairs pooled over all windows)."""
    x = np.asarray(x, np.float64)
    out = {}
    for lag in lags:
        a, b = x[:, :, :-lag], x[:, :, lag:]
        a = a - a.mean((0, 1, 2))
        b = b - b.mean((0, 1, 2))
        out[lag] = (a * b).mean((0, 1, 2)) / np.sqrt((a**2).mean((0, 1, 2)) * (b**2).mean((0, 1, 2)) + 1e-12)
    return out


def effective_rank(values: np.ndarray) -> float:
    """Exponential of the entropy of the normalised covariance spectrum (as health.effective_rank)."""
    values = np.clip(np.asarray(values, np.float64), 0, None)
    p = values / max(values.sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def within_rank(x: np.ndarray, level: str = "episode") -> float:
    """Effective rank of the frames after each episode's (or window's) mean is removed: the dimensions that vary inside
    one episode (or window), without the ones that only tell episodes (windows) apart."""
    x = np.asarray(x, np.float64)
    if level == "episode":
        centered = x - x.mean((1, 2), keepdims=True)
    elif level == "window":
        centered = x - x.mean(2, keepdims=True)
    else:
        raise ValueError(level)
    values = np.linalg.eigvalsh(np.cov(centered.reshape(-1, x.shape[-1]), rowvar=False, bias=True))
    return effective_rank(values)


def per_axis(x: np.ndarray, clip: int = CLIP, lags=(1, 8, 32)) -> dict:
    """The slowness spectrum: for each principal axis of the pooled frames its variance, the share of that variance at
    each time scale and its lag correlations. High variance with slow shares and correlations near 1 is the
    slow-feature signature."""
    mean, axes, values = pca(x)
    z = (np.asarray(x, np.float64) - mean) @ axes
    parts = decompose(z, clip)
    corr = lag_correlation(z, lags)
    total = np.maximum(parts["total"], 1e-12)
    return {
        "variance": values,
        "share": {k: parts[k] / total for k in LEVELS},
        "lag_correlation": corr,
    }
