"""Time scales of state trajectories: spectral centroids, bands, the entity frequency gap and event times.

numpy only, pure functions on arrays (the tests have known answers: a sine of period p has the centroid period p,
two entities with periods 4 and 32 have a gap of 8).

Spectral side (H-JEPA, appendix F): every dimension of an entity is standardised per episode (the centroid does not
depend on the amplitude), cut into segments of `window` frames (hop window / 2, Hann taper, segment mean removed), and
the power per frequency bin is averaged over the dimensions and summed over the segments of all episodes. The entity's
centroid is sum(f P) / sum(P) without the DC bin, its period is 1 / centroid (in frames), the gap is the ratio of the
largest to the smallest entity period.

Staircase signals (inventory, achievements) and anything slower than the window pile their power into the lowest
bins: the estimate is then limited by the window, not by the data. `limited` flags it, the period is a lower bound
there, and a gap whose slowest entity is limited is a lower bound too.

Event side: an event is a frame where something changed; the inter-event times of a group of events are the
candidates for a fixed stride (median) and a band (p10 to p90) of a level.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6
LOW_BINS = 2  # the lowest non-DC bins that decide whether an estimate is limited by the window
LIMITED_SHARE = 0.5  # more than this share of the power in the lowest bins: limited by the window
CLUSTER_RATIO = 2.0  # neighbouring entity periods this far apart are two time scales (heuristic from the paper)
QUANTILES = (10, 25, 50, 75, 90)


# --------------------------------------------------------------------------
# spectra


def standardise(x) -> np.ndarray | None:
    """Zero mean and unit variance per dimension over the episode; constant dimensions are dropped (None if all are)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    std = x.std(0)
    keep = std > EPS
    if not keep.any():
        return None
    return (x[:, keep] - x[:, keep].mean(0)) / std[keep]


def episode_power(x, window: int, hop: int | None = None, max_cells: float = 2e7) -> tuple[np.ndarray, int]:
    """Welch power of one episode (per bin, dimension-averaged, summed over its segments) and its segment count.

    Episodes shorter than `window`, and episodes without any varying dimension, have no segment (zeros, 0).
    """
    bins = window // 2 + 1
    power = np.zeros(bins)
    n = len(x)
    if n < window:
        return power, 0
    z = standardise(x)
    if z is None:
        return power, 0
    starts = np.arange(0, n - window + 1, hop or window // 2)
    taper = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window) / window)).astype(np.float32)
    chunk = max(1, int(max_cells // (window * z.shape[1])))
    for i in range(0, len(starts), chunk):
        segments = z[starts[i : i + chunk, None] + np.arange(window)]  # (S, window, dims)
        segments = (segments - segments.mean(1, keepdims=True)) * taper[None, :, None]
        power += (np.abs(np.fft.rfft(segments, axis=1)) ** 2).mean(2).sum(0)
    return power, len(starts)


def centroid_frequency(psd: np.ndarray, window: int) -> float:
    """Spectral centroid in cycles per frame, without the DC bin."""
    f = np.arange(len(psd)) / window
    total = psd[1:].sum()
    return float((f[1:] * psd[1:]).sum() / total) if total > 0 else float("nan")


def power_period(psd: np.ndarray, window: int, q: float) -> float:
    """Period (frames) of the frequency below which a share q of the power lies (cumulated from the slow end)."""
    cum = np.cumsum(psd[1:]) / psd[1:].sum()
    return window / (int(np.searchsorted(cum, q)) + 1)


def spectrum_summary(psd: np.ndarray, window: int) -> dict | None:
    """Centroid period, power band and the window-limit flag of one pooled spectrum."""
    if psd[1:].sum() <= 0:
        return None
    centroid = centroid_frequency(psd, window)
    low_share = float(psd[1 : 1 + LOW_BINS].sum() / psd[1:].sum())
    return {
        "period": 1.0 / centroid,
        "centroid": centroid,
        "band": {  # the period range that holds 10% to 90% of the power
            "fast_edge": power_period(psd, window, 0.9),
            "median": power_period(psd, window, 0.5),
            "slow_edge": power_period(psd, window, 0.1),
        },
        "low_share": low_share,
        "limited": low_share > LIMITED_SHARE,
    }


def frequency_gap(summaries: dict[str, dict]) -> dict | None:
    """Largest over smallest entity period. A lower bound when the slowest entity is limited by the window."""
    periods = {n: s["period"] for n, s in summaries.items() if s is not None}
    if len(periods) < 2:
        return None
    slowest, fastest = max(periods, key=periods.get), min(periods, key=periods.get)
    return {
        "ratio": periods[slowest] / periods[fastest],
        "slowest": slowest,
        "fastest": fastest,
        "lower_bound": bool(summaries[slowest]["limited"]),
    }


def bootstrap_gap(acc: dict[str, np.ndarray], window: int, n_boot: int = 200, seed: int = 0) -> dict | None:
    """Percentiles of the gap over resamples of the episodes. `acc[name]` is (episodes, bins), `episode_power` per row.

    Pooling is a sum over segments, and the centroid does not depend on the normalisation, so a resample is one
    matrix product of episode counts and spectra.
    """
    names = list(acc)
    if not names:
        return None
    episodes = acc[names[0]].shape[0]
    rng = np.random.default_rng(seed)
    counts = np.stack(
        [np.bincount(rng.integers(0, episodes, episodes), minlength=episodes) for _ in range(n_boot)]
    ).astype(float)
    f = np.arange(acc[names[0]].shape[1])[1:] / window
    periods = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for n in names:
            psd = counts @ acc[n]
            periods.append(psd[:, 1:].sum(1) / (psd[:, 1:] @ f))
        periods = np.stack(periods, 1)
        ratio = np.nanmax(periods, 1) / np.nanmin(periods, 1)
    ratio = ratio[np.isfinite(ratio)]
    if not len(ratio):
        return None
    return {
        "p5": float(np.percentile(ratio, 5)),
        "p50": float(np.percentile(ratio, 50)),
        "p95": float(np.percentile(ratio, 95)),
    }


# --------------------------------------------------------------------------
# events


def change_mask(x) -> np.ndarray:
    """(n - 1,) bool: the step t -> t + 1 changed any dimension."""
    x = np.asarray(x)
    x = x.reshape(len(x), -1)
    return (x[1:] != x[:-1]).any(1)


def rise_mask(x) -> np.ndarray:
    x = np.asarray(x).astype(np.int32)
    x = x.reshape(len(x), -1)
    return (x[1:] > x[:-1]).any(1)


def drop_mask(x) -> np.ndarray:
    x = np.asarray(x).astype(np.int32)
    x = x.reshape(len(x), -1)
    return (x[1:] < x[:-1]).any(1)


def union_masks(groups: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Per episode, the OR of the masks of several event types (`groups[type][episode]`)."""
    return [np.logical_or.reduce(per_episode) for per_episode in zip(*groups)]


def inter_event_stats(masks: list[np.ndarray]) -> dict:
    """Rate and inter-event times of the events in per-episode masks (the event happens at frame t + 1 for mask[t]).

    Gaps are measured between consecutive events inside an episode. Each episode also has one gap that the episode end
    cuts off (after its last event); `censored_share` is their share of all gaps, and quantiles of the complete gaps
    are biased short when it is large. `mode` is the most frequent gap (for clocks).
    """
    frames = sum(len(m) for m in masks)
    times = [np.flatnonzero(m) for m in masks]
    events = int(sum(len(t) for t in times))
    gaps = np.concatenate([np.diff(t) for t in times if len(t) > 1] or [np.zeros(0, np.int64)])
    out = {"events": events, "rate": events / frames if frames else 0.0, "gaps": int(len(gaps))}
    out["censored_share"] = len(masks) / (len(gaps) + len(masks)) if masks else None
    if len(gaps):
        out |= {f"p{q}": float(np.percentile(gaps, q)) for q in QUANTILES}
        out["mean"] = float(gaps.mean())
        out["cv"] = float(gaps.std() / gaps.mean())
        counts = np.bincount(gaps)
        out["mode"], out["mode_share"] = int(counts.argmax()), float(counts.max() / len(gaps))
    return out


# --------------------------------------------------------------------------
# ladder


def cluster_periods(periods: dict[str, float], ratio: float = CLUSTER_RATIO) -> list[list[str]]:
    """Entities sorted by period, split wherever two neighbours are at least `ratio` apart (fast cluster first)."""
    order = sorted(periods, key=periods.get)
    clusters = [[order[0]]] if order else []
    for prev, cur in zip(order, order[1:]):
        if periods[cur] / periods[prev] >= ratio:
            clusters.append([])
        clusters[-1].append(cur)
    return clusters


def boundaries(clusters: list[list[str]], periods: dict[str, float], summaries: dict[str, dict]) -> list[dict]:
    """Between two neighbouring clusters: the interval of periods between them, and whether their power bands overlap.

    The interval runs from the slowest period of the faster cluster to the fastest of the slower one; a cumulative
    step of a level boundary lies inside it. Bands that overlap are expected for spectra that fall like 1 / f^2, so the
    overlap is information and not a failure.
    """
    out = []
    for fast, slow in zip(clusters, clusters[1:]):
        low, high = max(periods[n] for n in fast), min(periods[n] for n in slow)
        fast_slow_edge = max(summaries[n]["band"]["slow_edge"] for n in fast)
        slow_fast_edge = min(summaries[n]["band"]["fast_edge"] for n in slow)
        out.append({"fast": fast, "slow": slow, "from": low, "to": high, "ratio": high / low,
                    "bands_overlap": bool(fast_slow_edge > slow_fast_edge)})  # fmt: skip
    return out
