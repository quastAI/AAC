"""Action coverage, and the information that chunks of actions carry about their effect."""

from __future__ import annotations

import math

import numpy as np

from .data import ACTIONS, Split

CHUNKS = (1, 2, 3, 4, 6, 8, 12, 16)  # chunk lengths (steps)
MIN_CHUNK_COUNT = 20  # chunks seen fewer times are left out of the mutual information (plug-in bias)
MAX_WINDOWS = 500_000  # windows per chunk length (a random subset above this)


def action_numbers(s: Split) -> dict:
    """Steps per action (the frames with an action: all but the last of an episode) over the 17 actions."""
    action = s.cols["action"][:, 0]
    steps = np.bincount(action[~np.isnan(action)].astype(int), minlength=len(ACTIONS))
    return {"steps": {n: int(c) for n, c in zip(ACTIONS, steps)}, "share": {n: float(c / steps.sum()) for n, c in zip(ACTIONS, steps)}}


def _windows(s: Split, k: int) -> np.ndarray:
    """Start frames of non-overlapping chunks of k steps inside an episode."""
    starts = [o + np.arange(0, n - k, k) for o, n in zip(s.offsets, s.lengths)]  # the last frame has no action
    return np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)


def _entropy_bits(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _entropy_mm_bits(labels: np.ndarray) -> float:
    """Entropy of integer labels in bits with the Miller-Madow correction for the plug-in bias."""
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    return _entropy_bits(counts) + (len(counts) - 1) / (2 * counts.sum() * math.log(2))


def mutual_information(chunk: np.ndarray, effect: np.ndarray, min_count: int = MIN_CHUNK_COUNT) -> dict | None:
    """I(chunk; effect) in bits, Miller-Madow corrected, over the windows whose chunk was seen at least `min_count` times.

    `coverage` is the share of windows that qualify; the other chunks are too rare for a conditional entropy (the plug-in
    estimate would read every rare chunk as a deterministic effect). Chunks and effects are integer labels.
    """
    keep = np.bincount(chunk)[chunk] >= min_count
    if not keep.any():
        return None
    c, e = chunk[keep], effect[keep]
    order = np.argsort(c, kind="stable")
    c, e = c[order], e[order]
    edges = np.flatnonzero(np.diff(c)) + 1
    conditional = sum(len(g) / len(e) * _entropy_mm_bits(np.unique(g, return_inverse=True)[1].reshape(-1)) for g in np.split(e, edges))
    h_effect = _entropy_mm_bits(np.unique(e, return_inverse=True)[1].reshape(-1))
    return {"mi_bits": float(h_effect - conditional), "h_effect_bits": float(h_effect), "coverage": float(keep.mean()), "chunk_classes": len(edges) + 1}


def chunk_numbers(s: Split, chunks=CHUNKS, max_windows=MAX_WINDOWS) -> dict:
    """Per chunk length k: distinct chunks, chunk entropy, distinct effects and the information of the chunk about them.

    Effect `interaction`: change of inventory, achievements and the sign of the change of the vitals over the window;
    `with_motion` adds the player's displacement (clipped to k).
    """
    action = np.nan_to_num(s.cols["action"][:, 0], nan=0).astype(np.int8)
    inventory, achievements, vitals, position = (s.cols[c] for c in ("inventory", "achievements", "vitals", "position"))
    rng = np.random.default_rng(0)
    out = {}
    for k in chunks:
        starts = _windows(s, k)
        if len(starts) > max_windows:
            starts = rng.choice(starts, max_windows, replace=False)
        if not len(starts):
            continue
        grams = action[starts[:, None] + np.arange(k)]
        _, chunk_label, counts = np.unique(grams, axis=0, return_inverse=True, return_counts=True)
        chunk_label = chunk_label.reshape(-1)
        end = starts + k
        inv = inventory[end].astype(np.int16) - inventory[starts]
        ach = achievements[end].astype(np.int8) - achievements[starts]
        vit = np.sign(vitals[end].astype(np.int16) - vitals[starts]).astype(np.int8)
        pos = np.clip(position[end].astype(np.int16) - position[starts], -k, k)
        interaction = np.concatenate([inv, ach, vit], axis=1)
        _, int_label, _ = np.unique(interaction, axis=0, return_inverse=True, return_counts=True)
        _, all_label, c_all = np.unique(np.concatenate([interaction, pos], axis=1), axis=0, return_inverse=True, return_counts=True)
        out[str(k)] = {
            "windows": int(len(starts)), "distinct_chunks": int(len(counts)), "chunk_entropy_bits": _entropy_bits(counts),
            "distinct_effects": int(len(c_all)), "effect_entropy_bits": _entropy_bits(c_all),
            "information_with_motion": mutual_information(chunk_label, all_label.reshape(-1)),
            "information_interaction": mutual_information(chunk_label, int_label.reshape(-1)),
        }  # fmt: skip
    return out
