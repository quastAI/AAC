"""Balanced clip sampling, for the ablation A1 only (not a model candidate).

The labels that decide the weights (the block the player faces, the events that change the inventory) come from the
simulator's `state`, `inventory` and `vitals` columns: data a real agent would not have. The ablation asks what the
latent keeps when the rare faced blocks and rare events (iron, coal, diamond, crafts) are seen far more often than
the expert meets them; it says nothing about a model we could train on real data.

A clip holds the faced block of each of its frames and the successful events of its transitions (the event table of the
probe suite, `probing.targets`). Its weight is the largest of the weights of its items: an item that is rarer than
`share` of the clips is lifted toward that share, at most `max_weight` times, and only if it comes from at least
`min_scenes` separate scenes (runs of consecutive clips). The cap and the scene minimum are what keep the few
transitions that exist (a diamond is collected about once per episode, 1,800 times in all) from being drawn so often
that the model memorises them. Items above `share` keep weight 1; the number of draws per epoch is not changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from probing import targets as tg

EVENTS = list(dict.fromkeys([*tg.PLACE_MAKE, *(v[0] for v in tg.COLLECT.values())]))
ITEMS = [f"faced:{b}" for b in tg.BLOCK_NAMES] + [f"event:{e}" for e in EVENTS]
STATE_CHUNK = 50_000
CHUNK = 200_000  # clips per step of the weight computation


def faced_blocks(f: h5py.File, total: int) -> np.ndarray:
    """The block type of the tile the player faces, per frame, from `state` and `facing` (read in chunks)."""
    out = np.empty(total, np.int16)
    for a in range(0, total, STATE_CHUNK):
        b = min(a + STATE_CHUNK, total)
        tile = tg.faced_index(torch.from_numpy(f["facing"][a:b].reshape(-1).astype(np.int64))).numpy()
        x = f["state"][a:b, : tg.VIEW_TILES * tg.VIEW_CHANNELS].reshape(b - a, tg.VIEW_TILES, tg.VIEW_CHANNELS)
        out[a:b] = x[np.arange(b - a), tile, : tg.NUM_BLOCKS].argmax(1)
    return out


def frame_events(action, inventory, vitals, faced, lengths, offsets) -> np.ndarray:
    """The event of the transition t -> t + 1 per frame (index into EVENTS, -1 for none), with the success rule of the
    probe suite: the item an action changes rose or fell, or `do` on a block that gives something gave it."""
    n = len(faced)
    event = np.full(n, -1, np.int16)
    last = np.zeros(n, bool)
    last[offsets + lengths - 1] = True
    t = np.flatnonzero(~np.isnan(action) & ~last)
    act = action[t].astype(int)
    cols = {"inventory": inventory, "vitals": vitals}

    def rose(rows, group, index, sign):
        return sign * (cols[group][rows + 1, index].astype(np.int64) - cols[group][rows, index].astype(np.int64)) > 0

    for name, (group, index, sign) in tg.PLACE_MAKE.items():
        rows = t[act == tg.ACTION_NAMES.index(name)]
        event[rows[rose(rows, group, index, sign)]] = EVENTS.index(name)
    do = t[act == tg.DO]
    for block, (name, group, index, sign) in tg.COLLECT.items():
        rows = do[faced[do] == tg.BLOCK_NAMES.index(block)]
        event[rows[rose(rows, group, index, sign)]] = EVENTS.index(name)
    return event


def clip_table(faced, event, clip_indices, offsets, lengths, span):
    """(clips, items) bool: which items each clip holds, and the episode and start of every clip. Padded frames after a
    death repeat the last real frame (no event)."""
    ep = np.fromiter((c[0] for c in clip_indices), np.int64, len(clip_indices))
    start = np.fromiter((c[1] for c in clip_indices), np.int64, len(clip_indices))
    rows = np.arange(len(ep))
    blocks = len(tg.BLOCK_NAMES)
    has = np.zeros((len(ep), blocks + len(EVENTS)), bool)
    for k in range(span):
        frame = offsets[ep] + np.minimum(start + k, lengths[ep] - 1)
        has[rows, faced[frame]] = True
        if k < span - 1:
            e = event[frame]
            keep = e >= 0
            has[rows[keep], blocks + e[keep]] = True
    return has, ep, start


def item_stats(has, ep, start) -> dict:
    """Per item: clips, share of the clips, episodes and scenes (runs of consecutive clips of one episode)."""
    n = len(has)
    follows = np.zeros(n, bool)
    follows[1:] = (ep[1:] == ep[:-1]) & (start[1:] == start[:-1] + 1)
    before = np.zeros_like(has)
    before[1:] = has[:-1] & follows[1:, None]
    return {
        "clips": has.sum(0),
        "share": has.sum(0) / max(n, 1),
        "scenes": (has & ~before).sum(0),
        "episodes": np.array([len(np.unique(ep[has[:, i]])) for i in range(has.shape[1])]),
    }


def item_ratios(stats: dict, share: float, max_weight: float, min_scenes: int) -> np.ndarray:
    """Per item: how often its clips are drawn (1 = as the data has them), 0 for items that are not lifted."""
    eligible = (stats["scenes"] >= min_scenes) & (stats["share"] > 0) & (stats["share"] < share)
    return np.where(eligible, np.minimum(share / np.maximum(stats["share"], 1e-12), max_weight), 0.0)


def clip_weights(has, ratios) -> np.ndarray:
    """The weight of each clip: the largest ratio among its items, at least 1."""
    out = np.ones(len(has))
    for a in range(0, len(has), CHUNK):
        out[a : a + CHUNK] = np.maximum(1.0, (has[a : a + CHUNK] * ratios).max(1))
    return out


def summarize(has, weights, stats, ratios, params) -> dict:
    """The record of what the balancing did: per item its clips, scenes and share before and after, and the totals."""
    total = weights.sum()
    items = {}
    for i, name in enumerate(ITEMS):
        if not stats["clips"][i]:
            continue
        items[name] = {
            "clips": int(stats["clips"][i]),
            "scenes": int(stats["scenes"][i]),
            "episodes": int(stats["episodes"][i]),
            "share": float(stats["share"][i]),
            "share_balanced": float(weights[has[:, i]].sum() / total),
            "lifted": bool(ratios[i] > 0),
            "ratio": float(ratios[i]),
        }
    return {
        **params,
        "clips": int(len(weights)),
        "effective_clips": float(total**2 / (weights**2).sum()),  # equally drawn clips with the same variance
        "mean_weight": float(weights.mean()),
        "max_weight_used": float(weights.max()),
        "items": items,
    }


def compute(dataset, share: float, max_weight: float, min_scenes: int):
    """Weights per clip (in the order of `dataset.clip_indices`) and their summary."""
    with h5py.File(dataset.h5_path, "r") as f:
        total = int(dataset.lengths.sum())
        faced = faced_blocks(f, total)
        event = frame_events(
            f["action"][:total, 0].astype(np.float64),
            f["inventory"][:total],
            f["vitals"][:total],
            faced,
            dataset.lengths,
            dataset.offsets,
        )
    span = int(dataset.last_level["span"])
    has, ep, start = clip_table(faced, event, dataset.clip_indices, dataset.offsets, dataset.lengths, span)
    stats = item_stats(has, ep, start)
    ratios = item_ratios(stats, share, max_weight, min_scenes)
    weights = clip_weights(has, ratios)
    params = {"share": share, "max_weight": max_weight, "min_scenes": min_scenes, "span": span}
    return weights, summarize(has, weights, stats, ratios, params)


def clip_weights_for(dataset, cfg, cache_dir=None):
    """`compute`, cached next to the data (the pass over `state` takes about a minute); the key names the file, the
    clips and the settings."""
    share, cap, minimum = float(cfg.share), float(cfg.max_weight), int(cfg.min_scenes)
    path = Path(dataset.h5_path)
    span = dataset.last_level["span"]
    key = f"{path.name}:{path.stat().st_size}:{len(dataset.clip_indices)}:{span}:{share}:{cap}:{minimum}:v1"
    cache = Path(cache_dir or path.parent) / f"balance_{hashlib.sha1(key.encode()).hexdigest()[:12]}.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if str(saved["key"]) == key:
            return saved["weights"], json.loads(str(saved["summary"]))
    weights, summary = compute(dataset, share, cap, minimum)
    np.savez(cache, key=key, weights=weights, summary=json.dumps(summary))
    return weights, summary


def build_sampler(dataset, cfg, cache_dir=None, generator=None):
    """A weighted sampler that draws as many clips per epoch as the dataset has, with replacement, and its summary."""
    weights, summary = clip_weights_for(dataset, cfg, cache_dir)
    assert len(weights) == len(dataset.clip_indices), "the cached weights do not match the dataset's clips"
    lifted = {k: v for k, v in summary["items"].items() if v["lifted"]}
    logging.info(
        "balanced sampling: %d items lifted, mean weight %.2f, effective clips %.0f of %d: %s",
        len(lifted),
        summary["mean_weight"],
        summary["effective_clips"],
        summary["clips"],
        {k: f"{v['share']:.2%} -> {v['share_balanced']:.2%}" for k, v in lifted.items()},
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True, generator=generator
    )
    return sampler, summary
