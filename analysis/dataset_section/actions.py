"""Action coverage, and the information that chunks of actions carry about their effect."""

from __future__ import annotations

import math

import numpy as np

from .data import ACTIONS, CREATURES, FACING_STEP, PLAYER_TILE, VIEW_COLS, VIEW_ROWS, Split

CHUNKS = (1, 2, 3, 4, 6, 8, 12, 16)  # chunk lengths (steps)
MIN_CHUNK_COUNT = 20  # chunks seen fewer times are left out of the mutual information (plug-in bias)
MAX_WINDOWS = 500_000  # windows per chunk length (a random subset above this)


# The actions are the ones the game executed (`Split.executed_action`): while the player is asleep the game replaces every
# action by a noop, so those steps are noops (the policy's choice is kept in `action_policy`). Success as the state shows it:
# moves: the player's position
# changed (a blocked move only turns the player). Place and make: the placed or crafted item changed in the inventory (SUCCESS:
# column, index, +1 if it must rise, -1 if it must fall). `do`: an item was gained, food or drink rose (eating, drinking), an
# achievement was unlocked, or a zombie, cow or skeleton vanished from the tile the player faces (a hit that does not kill counts
# as failed). `sleep`: the player fell asleep (it fails with full energy). `noop` is not split: it cannot succeed. Without the
# `state` column (the sleeping flag) the actions are used as stored, and `do` and `sleep` are not split.
SUCCESS = {
    "place_stone": ("inventory", 1, -1), "place_table": ("inventory", 0, -1), "place_furnace": ("inventory", 1, -1),
    "place_plant": ("inventory", 5, -1),
    "make_wood_pickaxe": ("inventory", 6, 1), "make_stone_pickaxe": ("inventory", 7, 1), "make_iron_pickaxe": ("inventory", 8, 1),
    "make_wood_sword": ("inventory", 9, 1), "make_stone_sword": ("inventory", 10, 1), "make_iron_sword": ("inventory", 11, 1),
}  # fmt: skip
MOVES = ("left", "right", "up", "down")
ITEMS = 6  # wood, stone, coal, iron, diamond, sapling: what `do` can gain


def _removed_from_facing_tile(s: Split, at: np.ndarray) -> np.ndarray:
    """Per step: a zombie, cow or skeleton was on the tile the player faces before the step and is gone after it."""
    mob = s.view().mob.reshape(-1, VIEW_ROWS, VIEW_COLS)
    facing = s.cols["facing"][at, 0].astype(int)
    out = np.zeros(len(at), bool)
    for direction, (dr, dc) in FACING_STEP.items():
        here = facing == direction
        r, c = PLAYER_TILE[0] + dr, PLAYER_TILE[1] + dc
        before, after = mob[at[here], r, c] & CREATURES, mob[at[here] + 1, r, c] & CREATURES
        out[here] = (before & ~after) != 0
    return out


def _succeeded(s: Split, name: str, at: np.ndarray) -> np.ndarray | None:
    """Per step of the action `name` (frames `at`): did it succeed, if it was not ignored? None for actions that are not split."""
    if name in MOVES:
        return (s.cols["position"][at + 1] != s.cols["position"][at]).any(1)
    if name in SUCCESS:
        column, index, sign = SUCCESS[name]
        return sign * (s.cols[column][at + 1, index].astype(np.int64) - s.cols[column][at, index]) > 0
    if name == "do" and s.state_ok:
        inventory, vitals, achievements = (s.cols[c] for c in ("inventory", "vitals", "achievements"))
        gained = (inventory[at + 1, :ITEMS] > inventory[at, :ITEMS]).any(1)
        ate = (vitals[at + 1, 1:3] > vitals[at, 1:3]).any(1)  # food, drink: they only fall on their own
        unlocked = (achievements[at + 1] > achievements[at]).any(1)
        return gained | ate | unlocked | _removed_from_facing_tile(s, at)
    if name == "sleep" and s.state_ok:
        return s.view().sleeping[at + 1]
    return None


def action_numbers(s: Split) -> dict:
    """Steps per executed action (the frames with an action: all but the last of an episode) over the 17 actions, and, for
    the actions whose success the state shows, how many succeeded and how many failed. `asleep_steps`: the steps taken while
    asleep, all of them noops."""
    action = s.executed_action()
    steps_at = np.flatnonzero(~np.isnan(action))
    steps_at = steps_at[steps_at + 1 < s.total]  # every action has a next frame of its episode: the last frame has none
    taken = action[steps_at].astype(int)
    steps = np.bincount(taken, minlength=len(ACTIONS))
    outcome = {}
    for i, name in enumerate(ACTIONS):
        ok = _succeeded(s, name, steps_at[taken == i])
        if ok is not None:
            outcome[name] = {"succeeded": int(ok.sum()), "failed": int((~ok).sum())}
    return {
        "steps": {n: int(c) for n, c in zip(ACTIONS, steps)}, "share": {n: float(c / steps.sum()) for n, c in zip(ACTIONS, steps)},
        "outcome": outcome, "asleep_steps": s.asleep_steps(),
    }  # fmt: skip


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
    action = np.nan_to_num(s.executed_action(), nan=0).astype(np.int8)
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
