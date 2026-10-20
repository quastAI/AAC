"""Episode length, time to the diamond, achievements per episode and the first-unlock ladder, clip coverage."""

from __future__ import annotations

import numpy as np

from .data import ACHIEVEMENTS, DIAMOND_INDEX, Split

LENGTH_QUANTILES = (10, 25, 50, 75, 90)
STRIDES = (1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400)  # frame spacing of a clip
CONTEXTS = (1, 5)  # frames of context (`history`)
ROLLOUTS = (1, 5)  # frames to predict


def quantile_row(x, quantiles=LENGTH_QUANTILES) -> dict:
    x = np.asarray(x, dtype=float)
    out = {"n": int(len(x)), "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}
    out |= {f"p{q}": float(np.percentile(x, q)) for q in quantiles}
    return out


def length_numbers(s: Split) -> dict:
    """Episode lengths (frames, episode-weighted) with their quantiles; every length is kept for the histogram."""
    lengths = np.sort(s.lengths)
    return {
        "episodes": int(len(lengths)), "frames": int(lengths.sum()), "quantiles": quantile_row(lengths),
        "lengths": lengths.tolist(),
    }  # fmt: skip


def time_to_diamond(s: Split) -> dict:
    """Steps from the reset to the first frame with the diamond, of the episodes that reach it."""
    flags = s.episodes("achievements")
    steps = [int(f[:, DIAMOND_INDEX].argmax()) for f in flags if f[:, DIAMOND_INDEX].any()]
    if not steps:
        return {"episodes": 0, "of": len(flags)}
    return {"episodes": len(steps), "of": len(flags), "quantiles": quantile_row(steps), "steps": sorted(steps)}


def achievement_numbers(s: Split) -> dict:
    """Achievements per episode (histogram over 0 to 22) and per achievement the share of episodes that unlock it and
    the quantiles of the frame of its first unlock (among the episodes that do)."""
    counts = s.final_achievements.sum(1)
    first = []
    for f in s.episodes("achievements"):
        got = f.any(0)
        first.append(np.where(got, f.argmax(0), -1))
    first = np.stack(first)
    ladder = {}
    for i, name in enumerate(ACHIEVEMENTS):
        got = first[:, i] >= 0
        entry = {"share": float(got.mean()), "episodes": int(got.sum())}
        if got.any():
            entry |= {f"p{q}": float(np.percentile(first[got, i], q)) for q in (10, 50, 90)}
            entry["mean"] = float(first[got, i].mean())
        ladder[name] = entry
    histogram = np.bincount(counts, minlength=len(ACHIEVEMENTS) + 1).tolist()
    return {"per_episode": {"histogram": histogram, "mean": float(counts.mean())}, "ladder": ladder}


def clip_counts(s: Split, strides=STRIDES, context=1, rollout=1) -> list[dict]:
    """Clips (all overlapping ones) of `context + rollout` frames spaced `stride` frames apart, per stride.

    A clip needs (context + rollout - 1) x stride + 1 frames of an episode. Episodes that ended in death are extended by
    stride - 1 copies of the dead frame, as in the training dataset (`terminal_pad: auto`).
    """
    rows = []
    for stride in strides:
        span = (context + rollout - 1) * stride + 1
        padded = s.lengths + np.where(s.dead, stride - 1, 0)
        per_episode = np.maximum(padded - span + 1, 0)
        rows.append({
            "stride": int(stride), "span": int(span), "clips": int(per_episode.sum()),
            "episodes_long_enough": int((padded >= span).sum()), "independent_clips": int((padded // span).sum()),
        })  # fmt: skip
    return rows


def clip_coverage(s: Split, strides=STRIDES) -> dict:
    """Clip counts per stride for every (context, rollout)."""
    out = {"episodes": s.n_episodes, "frames": s.total, "strides": list(strides), "settings": {}}
    for context in CONTEXTS:
        for rollout in ROLLOUTS:
            rows = clip_counts(s, strides, context, rollout)
            out["settings"][f"context{context}_rollout{rollout}"] = {
                "context": context, "rollout": rollout,
                **{k: [r[k] for r in rows] for k in ("span", "clips", "episodes_long_enough", "independent_clips")},
            }  # fmt: skip
    return out
