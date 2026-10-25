"""The entity frequency gap of the paper (H-JEPA, appendix F) on 7 entities of the game (the probe targets of the first
probe script; the probe suite, H-JEPA/h_jepa/probing, replaced it on 2026-10-25).

Every dimension of an entity is standardised per episode (the centroid does not depend on the amplitude), cut into
segments of `window` frames (hop window / 2, Hann taper, segment mean removed), and the power per frequency bin is averaged
over the dimensions and summed over the segments of all episodes. The entity's centroid is sum(f P) / sum(P) without the DC
bin, its period is 1 / centroid (in frames), the gap is the largest over the smallest entity period (tests: a sine of period
p has the centroid period p, two entities with periods 4 and 32 have a gap of 8).

Staircase signals (inventory, achievements) and anything slower than the window pile their power into the lowest bins: the
estimate is then limited by the window, not by the data. `limited` flags it, the period is a lower bound there, and a gap
whose slowest entity is limited is a lower bound too. The slowest entity here is the daylight, a clock that no action
changes: `without_daylight` gives the gap without it.
"""

from __future__ import annotations

import numpy as np

from .data import Split

EPS = 1e-6
LOW_BINS = 2  # the lowest non-DC bins that decide whether an estimate is limited by the window
LIMITED_SHARE = 0.5  # more than this share of the power in the lowest bins: limited by the window
ENTITIES = {  # the targets of the first probe script with their probe group
    "position": "fast", "facing": "fast", "mobs_in_view": "fast", "vitals": "medium",
    "inventory": "slow", "achievements": "slow", "daylight": "slow",
}  # fmt: skip
WINDOWS = (64, 128, 256, 512)  # Welch windows (frames): the sensitivity
PRIMARY_WINDOW = 256
N_BOOT = 200  # episode bootstraps


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


def _entity_arrays(s: Split, name: str):
    arrays = s.episodes(name)
    if name == "facing":  # a direction 1-4: one-hot, so that no order is assumed
        return (np.eye(4, dtype=np.float32)[np.clip(x[:, 0].astype(int), 1, 4) - 1] for x in arrays)
    return (x.reshape(len(x), -1) for x in arrays)


def entity_power(s: Split, name: str, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per episode: the Welch power of an entity per bin (rows) and its number of segments."""
    acc = np.zeros((s.n_episodes, window // 2 + 1))
    segments = np.zeros(s.n_episodes, np.int64)
    for row, x in enumerate(_entity_arrays(s, name)):
        acc[row], segments[row] = episode_power(x, window)
    return acc, segments


def _gap_pair(summaries: dict) -> dict:
    return {
        "all": frequency_gap(summaries),
        "without_daylight": frequency_gap({n: v for n, v in summaries.items() if n != "daylight"}),
    }


def entity_gap(s: Split, windows=WINDOWS, primary=PRIMARY_WINDOW, n_boot=N_BOOT) -> dict | None:
    """Period and window limit of every entity at the primary window, the gap with an episode bootstrap, the gap at the
    other windows. Episodes shorter than a window have no segment at it: `episodes_used` says how many remain."""
    names = [n for n in ENTITIES if s.has(n)]
    if len(names) < 2:
        return None
    power = {n: entity_power(s, n, primary) for n in names}
    segments = np.max([power[n][1] for n in names], axis=0)
    here = {n: spectrum_summary(power[n][0].sum(0), primary) for n in names}
    entities = {n: {**v, "probe_group": ENTITIES[n], "segments": int(power[n][1].sum())} for n, v in here.items() if v}
    present = {n: power[n][0] for n, v in here.items() if v}
    boot = {
        "all": bootstrap_gap(present, primary, n_boot),
        "without_daylight": bootstrap_gap({n: v for n, v in present.items() if n != "daylight"}, primary, n_boot),
    }
    gap = {k: (v and {**v, "ci": boot[k]}) for k, v in _gap_pair(here).items()}
    sensitivity = {}
    for w in sorted(set(windows) | {primary}):
        summaries = here if w == primary else {n: spectrum_summary(entity_power(s, n, w)[0].sum(0), w) for n in names}
        sensitivity[w] = {
            "gap": _gap_pair(summaries),
            "entities": {n: {"period": v["period"], "limited": v["limited"]} for n, v in summaries.items() if v},
        }
    return {
        "episodes": s.n_episodes, "episodes_used": int((segments > 0).sum()), "segments": int(segments.sum()),
        "primary_window": primary, "entities": entities, "gap": gap, "sensitivity": sensitivity,
    }  # fmt: skip
