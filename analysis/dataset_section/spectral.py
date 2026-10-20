"""The entity frequency gap of the paper (H-JEPA, appendix F) on the 7 probe entities of `probe_eval.py`.

The spectral estimators are in `timescales.py` (a sine of period p has the centroid period p). Per entity the period is
1 / spectral centroid of the pooled Welch spectrum (every dimension standardised per episode), the gap is the largest over
the smallest entity period. An entity whose power sits in the two lowest bins is `limited` by the window (staircases, the
day): its period, and a gap that rests on it, are lower bounds. The slowest entity here is the daylight, a clock that no
action changes: `without_daylight` gives the gap without it.
"""

from __future__ import annotations

import numpy as np

import timescales as tsc

from .data import Split

ENTITIES = {  # the probe targets of probe_eval.py with their probe group
    "position": "fast", "facing": "fast", "mobs_in_view": "fast", "vitals": "medium",
    "inventory": "slow", "achievements": "slow", "daylight": "slow",
}  # fmt: skip
WINDOWS = (64, 128, 256, 512)  # Welch windows (frames): the sensitivity
PRIMARY_WINDOW = 256
N_BOOT = 200  # episode bootstraps


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
        acc[row], segments[row] = tsc.episode_power(x, window)
    return acc, segments


def _gap_pair(summaries: dict) -> dict:
    return {
        "all": tsc.frequency_gap(summaries),
        "without_daylight": tsc.frequency_gap({n: v for n, v in summaries.items() if n != "daylight"}),
    }


def entity_gap(s: Split, windows=WINDOWS, primary=PRIMARY_WINDOW, n_boot=N_BOOT) -> dict | None:
    """Period and window limit of every entity at the primary window, the gap with an episode bootstrap, the gap at the
    other windows. Episodes shorter than a window have no segment at it: `episodes_used` says how many remain."""
    names = [n for n in ENTITIES if s.has(n)]
    if len(names) < 2:
        return None
    power = {n: entity_power(s, n, primary) for n in names}
    segments = np.max([power[n][1] for n in names], axis=0)
    here = {n: tsc.spectrum_summary(power[n][0].sum(0), primary) for n in names}
    entities = {n: {**v, "probe_group": ENTITIES[n], "segments": int(power[n][1].sum())} for n, v in here.items() if v}
    present = {n: power[n][0] for n, v in here.items() if v}
    boot = {
        "all": tsc.bootstrap_gap(present, primary, n_boot),
        "without_daylight": tsc.bootstrap_gap({n: v for n, v in present.items() if n != "daylight"}, primary, n_boot),
    }
    gap = {k: (v and {**v, "ci": boot[k]}) for k, v in _gap_pair(here).items()}
    sensitivity = {}
    for w in sorted(set(windows) | {primary}):
        summaries = here if w == primary else {n: tsc.spectrum_summary(entity_power(s, n, w)[0].sum(0), w) for n in names}
        sensitivity[w] = {
            "gap": _gap_pair(summaries),
            "entities": {n: {"period": v["period"], "limited": v["limited"]} for n, v in summaries.items() if v},
        }
    return {
        "episodes": s.n_episodes, "episodes_used": int((segments > 0).sum()), "segments": int(segments.sum()),
        "primary_window": primary, "entities": entities, "gap": gap, "sensitivity": sensitivity,
    }  # fmt: skip
