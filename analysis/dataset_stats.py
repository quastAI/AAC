"""Statistics of a Craftax-Classic episode file that the design decisions depend on (D0).

Reads the collector's HDF5 files (numpy and h5py only, no pixels) and reports facts about the data, with no
assumed strides or depth. Spectra, events, the ladder, tasks and the achievement ladder use the expert episodes
(source 0, the planning tasks come from them) when the file has them; the gap is also given per source and snapshot.

* composition: episodes, lengths, sources (and, for early-checkpoint episodes, the snapshots), provenance, deaths;
* is a hierarchy useful at all: the entity frequency gap of the paper (spectral centroids of the 7 probe entities, with
  an episode bootstrap, a window sensitivity, bands, the view and terrain from the `state` column as extras), the
  distance horizon of each entity, and the inter-event times that a stride or a band is read from;
* the timescale ladder: entities clustered by period, the intervals between neighbouring clusters, a fixed-stride and
  a band candidate per cluster (median and p10 to p90 of the inter-event times, CV as irregularity), the clips left;
* clip counts per top-level step T (the product of all strides), and the T needed to reach the task offsets and the
  expert's time to the diamond, per planning horizon;
* action chunks per length k: distinct chunks and effects, their mutual information, action coverage per source;
* tasks (per offset, how many expert episodes have a pair with progress), the achievement ladder per source and
  snapshot, and survival (critical frames, the run into death).

    python dataset_stats.py --data expert_train.h5 --label train \
        --data expert_val.h5 --label val --out report/

The sections that need the derived columns (inventory, vitals, ...) are skipped for files written by an older collector.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

import timescales as tsc

ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie",
    "collect_stone", "place_stone", "eat_plant", "defeat_skeleton",
    "make_stone_pickaxe", "make_stone_sword", "wake_up", "place_furnace",
    "collect_coal", "collect_iron", "collect_diamond", "make_iron_pickaxe",
    "make_iron_sword",
]  # fmt: skip
DIAMOND_INDEX = 19  # collect_diamond in the Craftax-Classic achievement order
ACTIONS = [
    "noop", "left", "right", "up", "down", "do", "sleep", "place_stone", "place_table",
    "place_furnace", "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword", "make_iron_sword",
]  # fmt: skip
# The 7 probe targets of H-JEPA/h_jepa/probe_eval.py with their probe group: the entities of the frequency gap.
ENTITIES = {
    "position": "fast", "facing": "fast", "mobs_in_view": "fast", "vitals": "medium",
    "inventory": "slow", "achievements": "slow", "daylight": "slow",
}  # fmt: skip
EXTRA_ENTITIES = ("terrain", "view")  # from the `state` column: block counts and block one-hot of the 7x9 view
VITALS = ("health", "food", "drink", "energy")  # column order of `vitals`
ENTITY_EVENTS = {  # the event types of an entity (`event_masks`); the daylight has none
    "position": ["position"], "facing": ["facing"], "mobs_in_view": ["mobs_in_view"],
    "vitals": [f"{v}_{k}" for v in VITALS for k in ("drop", "rise")],
    "inventory": ["inventory"], "achievements": ["unlock"], "daylight": [],
}  # fmt: skip
STATE_DIM, VIEW_DIM, VIEW_TILES, VIEW_CHANNELS, BLOCK_TYPES = 1345, 1323, 63, 21, 17  # layout of `state` (renderer)
LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
CHUNKS = (1, 2, 3, 4, 6, 8, 12, 16)
MAX_FRAMES = 2_000_000
SOURCES = {0: "expert", 1: "noised_expert", 2: "random", 3: "early"}  # the collector's `source` column
# The top level of a model steps T environment steps, T = the product of all its strides. With window 1 and
# `history + rollout` = 4 states at the top, a clip needs 3 T + 1 level-1 frames and a plan of H top-level steps
# reaches H x T steps: clip counts and reach depend on T only, not on the depth or on how T splits into strides.
TOP_STEPS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
HORIZONS = (2, 4, 8)  # top-level planning horizons (a planner setting)
OFFSETS = (25, 50, 100, 200)  # task offsets of protocol A (experiments.yaml)
UPPER_EXCLUDED_SOURCES = (2,)  # `upper_loss_exclude_sources` of the training config: they train level 1 only
WINDOWS = (64, 128, 256, 512)  # Welch windows (frames) of the frequency gap: the sensitivity
PRIMARY_WINDOW = 256  # covers the clip span up to T = 64 (193 frames) and is shorter than an expert episode
N_BOOT = 200  # episode bootstraps of the gap
VIEW_FRAMES = 300_000  # frames for the spectrum of the 1071-dimensional view (every k-th episode)
GAP_THRESHOLD = 2.0  # entity frequency gap below this: little selective abstraction (six environments of the paper)
CV_REGULAR, CV_IRREGULAR = 0.5, 1.0  # inter-event CV below / above: regular events (a band has no room) / irregular
CLOCKS = {"food_drop": 26, "drink_drop": 21, "energy_drop": 31}  # inter-event times of the env's needs (awake)
DAYLIGHT_CHECK_WINDOW = 256  # the day is 300 steps: the daylight is limited by this window
CRITICAL_NEED, CRITICAL_HEALTH = 2, 3  # a frame is critical when food, drink or energy is at most 2, or health at most 3
MIN_GROUP_EPISODES = 5  # a source or snapshot with fewer episodes gets no gap of its own
MIN_CHUNK_COUNT = 20  # chunks seen fewer times are left out of the mutual information (plug-in bias)
CODEBOOK = (2, 64)  # groups x codes of the VQ (placeholder, configs)
MIN_ACTION_STEPS = 500  # an action seen fewer times than this is reported as thin (placeholder, tune on the data)


# --------------------------------------------------------------------------
# loading


class Data:
    """The small columns of one file, with episode boundaries.

    Lengths, sources, seeds and the final achievements are read for every episode. The frame
    columns are read for at most `max_frames` frames: above that, every k-th episode (spread over
    the whole file, so every source is in it) is kept, and `sampled` says so.
    """

    COLUMNS = ["action", "achievements", "inventory", "vitals", "facing", "daylight", "position", "mobs_in_view", "source", "seed"]

    def __init__(self, path, max_frames: int = MAX_FRAMES):
        self.path = str(path)
        with h5py.File(path, "r") as f:
            self.all_lengths = f["ep_len"][:].astype(np.int64)
            all_offsets = f["ep_offset"][:].astype(np.int64)
            self.total_all = int(self.all_lengths.sum())
            self.all_sources = f["source"][all_offsets] if "source" in f else None
            self.all_seeds = f["seed"][all_offsets] if "seed" in f else None
            last = all_offsets + self.all_lengths - 1
            self.all_final_achievements = f["achievements"][last] if "achievements" in f else None
            self.all_final_dead = f["dead"][last].astype(bool) if "dead" in f else None
            self.all_final_vitals = f["vitals"][last] if "vitals" in f else None
            self.all_policy_updates = f["policy_updates"][all_offsets] if "policy_updates" in f else None
            self.runs = json.loads(f.attrs["runs"]) if "runs" in f.attrs else []  # the collector's provenance

            step = max(1, -(-self.total_all // max(1, max_frames)))  # ceil
            keep = np.arange(0, len(self.all_lengths), step)
            self.lengths, self.offsets_in_file = self.all_lengths[keep], all_offsets[keep]
            if step == 1:
                self.cols = {c: f[c][: self.total_all] for c in self.COLUMNS if c in f}
            else:
                self.cols = {
                    c: np.concatenate([f[c][o : o + n] for o, n in zip(self.offsets_in_file, self.lengths)])
                    for c in self.COLUMNS
                    if c in f
                }
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64)
        self.n_episodes = len(self.lengths)
        self.total = int(self.lengths.sum())
        self.sampled = None if step == 1 else (self.n_episodes, len(self.all_lengths))
        self.episode = np.repeat(np.arange(self.n_episodes), self.lengths)
        self.seeds = self.all_seeds
        # per analysed episode: source, snapshot (training updates) and whether it ended in death (None if not recorded)
        self.keep = keep
        self.ep_source = None if self.all_sources is None else self.all_sources[keep]
        self.ep_updates = None if self.all_policy_updates is None else self.all_policy_updates[keep]
        self.ep_dead = None if self.all_final_dead is None else self.all_final_dead[keep]

    def has(self, *names) -> bool:
        return all(n in self.cols for n in names)

    def episodes(self, name, idx=None):
        """The rows of a column per analysed episode (all of them, or the episodes `idx`)."""
        arr = self.cols[name]
        idx = range(self.n_episodes) if idx is None else idx
        return [arr[self.offsets[i] : self.offsets[i] + self.lengths[i]] for i in idx]

    def has_state(self) -> bool:
        with h5py.File(self.path, "r") as f:
            return "state" in f and f["state"].shape[1] == STATE_DIM

    def state_features(self, idx, kind: str):
        """Per episode, features of the 7x9 view from the `state` column, read one episode at a time.

        `terrain`: how many tiles of each block type are in view (17), `view`: the block one-hot of every tile (1071).
        """
        with h5py.File(self.path, "r") as f:
            for i in idx:
                o, n = self.offsets_in_file[i], self.lengths[i]
                blocks = f["state"][o : o + n, :VIEW_DIM].reshape(n, VIEW_TILES, VIEW_CHANNELS)[:, :, :BLOCK_TYPES]
                yield blocks.sum(1) if kind == "terrain" else blocks.reshape(n, -1)


# --------------------------------------------------------------------------
# sections


def composition(d: Data) -> dict:
    """Counts over every episode of the file, also when the frame columns were sampled."""
    lengths = d.all_lengths
    out = {
        "episodes": len(lengths),
        "frames": d.total_all,
        "length": {q: float(np.percentile(lengths, p)) for q, p in (("min", 0), ("p25", 25), ("median", 50), ("p75", 75), ("max", 100))},
    }
    if d.sampled:
        out["sampled"] = {"episodes_analysed": d.sampled[0], "episodes": d.sampled[1]}
    if d.all_sources is not None:
        out["by_source"] = {n: int((d.all_sources == i).sum()) for i, n in SOURCES.items() if (d.all_sources == i).any()}
        out["frames_by_source"] = {n: int(lengths[d.all_sources == i].sum()) for i, n in SOURCES.items() if (d.all_sources == i).any()}
        out["frame_share_by_source"] = {n: v / d.total_all for n, v in out["frames_by_source"].items()}
        if d.all_policy_updates is not None and (d.all_sources == 3).any():
            early = d.all_sources == 3
            out["early_checkpoints"] = {
                str(int(u)): {"episodes": int((early & (d.all_policy_updates == u)).sum()),
                              "frames": int(lengths[early & (d.all_policy_updates == u)].sum())}
                for u in np.unique(d.all_policy_updates[early])
            }
    if d.all_final_dead is not None:
        out["deaths"] = death_statistics(d)
    if d.all_final_achievements is not None:
        final = d.all_final_achievements.astype(bool)
        out["diamond_episodes"] = int(final[:, ACHIEVEMENTS.index("collect_diamond")].sum())
        out["achievements_per_episode_mean"] = float(final.sum(1).mean())
    return out


def death_statistics(d: Data) -> dict:
    """How the episodes end, per source: deaths, their causes and when they happen.

    Cause from the last frame: lava (dead with health left), exhaustion (health used up while food,
    drink or energy was 0: it takes about 330 steps from a fresh start) or damage (health used up
    with all three above 0). The ends that are not deaths are the diamond cut and the horizon.
    """
    dead, lengths = d.all_final_dead, d.all_lengths
    sources = d.all_sources if d.all_sources is not None else np.zeros(len(lengths), np.int8)
    out = {}
    for code, name in SOURCES.items():
        here = sources == code
        if not here.any():
            continue
        died = here & dead
        entry = {"episodes": int(here.sum()), "died": int(died.sum()), "died_share": float(died.sum() / here.sum())}
        if died.any():
            entry["steps_to_death"] = {q: float(np.percentile(lengths[died] - 1, p)) for q, p in (("p25", 25), ("median", 50), ("p75", 75))}
            if d.all_final_vitals is not None:
                vitals = d.all_final_vitals[died].astype(int)
                lava = vitals[:, 0] > 0
                exhausted = (vitals[:, 0] <= 0) & (vitals[:, 1:].min(1) <= 0)
                entry["causes"] = {"lava": int(lava.sum()), "exhaustion": int(exhausted.sum()),
                                   "damage": int((~lava & ~exhausted).sum())}
        out[name] = entry
    return out


def clip_counts(d: Data, totals=TOP_STEPS, history=3, rollout=1, upper_excluded=UPPER_EXCLUDED_SOURCES) -> list[dict]:
    """Clips a model whose top level steps T environment steps can draw, for every T in `totals`.

    The same for any depth and any split of T into strides (window 1): a clip needs (history + rollout - 1) T + 1
    level-1 frames. Episodes that ended in death are extended by T - 1 copies of the dead frame (`terminal_pad:
    auto`, as in the training dataset). `clips_upper` leaves out the sources that train level 1 only.
    """
    steps = history + rollout
    lengths = d.all_lengths
    dead = d.all_final_dead if d.all_final_dead is not None else np.zeros(len(lengths), bool)
    sources = d.all_sources if d.all_sources is not None else np.zeros(len(lengths), np.int8)
    present = {name: code for code, name in SOURCES.items() if (sources == code).any()}
    rows = []
    for top in totals:
        span = (steps - 1) * top + 1
        padded = lengths + np.where(dead, top - 1, 0)
        per_episode = np.maximum(padded - span + 1, 0)
        row = {
            "T": int(top), "span": int(span),
            "episodes_long_enough": int((padded >= span).sum()),
            "clips": int(per_episode.sum()),
            "clips_upper": int(per_episode[~np.isin(sources, upper_excluded)].sum()) if top > 1 else None,
            "independent_clips": int((padded // span).sum()),
        }  # fmt: skip
        if len(present) > 1:
            row["by_source"] = {name: int(per_episode[sources == code].sum()) for name, code in present.items()}
        rows.append(row)
    return rows


def event_durations(d: Data, idx=None) -> dict:
    flags = d.episodes("achievements", idx)
    unlock = {}  # achievement -> first frame per episode that has it
    gaps = []
    for f in flags:
        first = np.where(f.any(0), f.argmax(0), -1)
        for i, frame in enumerate(first):
            if frame >= 0:
                unlock.setdefault(i, []).append(int(frame))
        frames = np.sort(first[first >= 0])
        gaps += np.diff(frames).tolist()
    table = {
        ACHIEVEMENTS[i]: {
            "episodes": len(v), "median": float(np.median(v)),
            "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
        }
        for i, v in sorted(unlock.items(), key=lambda kv: np.median(kv[1]))
    }  # fmt: skip
    gaps = np.array([g for g in gaps if g > 0])
    out = {"unlock_frames": table}
    if len(gaps):
        out["gap_between_unlocks"] = {f"p{p}": float(np.percentile(gaps, p)) for p in (10, 25, 50, 75, 90)}
    return out


def required_top_step(d: Data, horizons=HORIZONS, offsets=OFFSETS, quantiles=(50, 75, 90)) -> dict | None:
    """The top-level step T that a plan of `horizon` top-level steps needs to reach a distance: T >= distance / horizon.

    Distances: the task offsets, and the expert's time to the first diamond (quantiles over the expert episodes,
    source 0 when the file has sources). T is the product of all strides, so this does not assume a depth or strides;
    how T is split into strides is a separate decision. A design rule (the plan covers the distance), not a measurement.
    """
    flags = d.episodes("achievements")
    keep = np.ones(len(flags), bool) if "source" not in d.cols else d.cols["source"][d.offsets] == 0
    times = [int(np.argmax(f[:, DIAMOND_INDEX])) for f, k in zip(flags, keep) if k and f[:, DIAMOND_INDEX].any()]
    distances = {f"offset {o}": float(o) for o in offsets}
    if times:
        distances |= {f"diamond p{q}": float(np.percentile(times, q)) for q in quantiles}
    rows = {name: {h: int(math.ceil(dist / h)) for h in horizons} for name, dist in distances.items()}
    return {
        "expert_episodes_with_diamond": len(times),
        "steps_to_diamond": {f"p{q}": float(np.percentile(times, q)) for q in quantiles} if times else None,
        "distances": distances, "horizons": list(horizons), "required_top_step": rows,
    }  # fmt: skip


def _entity_matrix(d: Data, name: str, frames) -> np.ndarray:
    x = d.cols[name].reshape(d.total, -1).astype(np.float32)[frames]
    return (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)


def distance_horizon(d: Data, idx=None, lags=LAGS, pairs=200_000, seed=0, min_pairs=1000) -> dict:
    """How far a distance in an entity still carries the time gap: distance of states `lag` steps apart against unrelated states (1 = unrelated).

    A cost on a flat latent is such a distance. A ratio that is already 1 at a short lag means the cost has no signal
    beyond it (the saturation lag, the first lag where the ratio reaches 0.9; None if it never does within `lags`).
    `changed` is the share of pairs in which anything in the entity differs. Lags with fewer than `min_pairs` pairs
    inside episodes are left out (too noisy). Over the episodes `idx` (default: all).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(d.n_episodes) if idx is None else np.asarray(idx)
    frames = np.concatenate([np.arange(d.offsets[i], d.offsets[i] + d.lengths[i]) for i in idx])
    episode, total = d.episode[frames], len(frames)
    out = {}
    for name, group in ENTITIES.items():
        if not d.has(name):
            continue
        x = _entity_matrix(d, name, frames)
        i = rng.integers(0, total, pairs)
        j = rng.integers(0, total, pairs)
        different = episode[i] != episode[j]
        unrelated = float(np.abs(x[i[different]] - x[j[different]]).mean()) if different.any() else 1.0
        unrelated = unrelated or 1.0
        ratios, changed = {}, {}
        for lag in lags:
            if lag >= total:
                continue
            valid = episode[lag:] == episode[:-lag]
            if valid.sum() < min_pairs:
                continue
            delta = np.abs(x[lag:] - x[:-lag])[valid]
            ratios[lag] = float(delta.mean() / unrelated)
            changed[lag] = float((delta.max(1) > 1e-6).mean())
        saturation = next((lag for lag, r in ratios.items() if r >= 0.9), None)
        out[name] = {"probe_group": group, "ratio": ratios, "changed": changed, "saturation_lag": saturation}
    return out


def _windows(d: Data, k: int):
    """Start frames of non-overlapping chunks of k steps inside an episode."""
    starts = []
    for o, n in zip(d.offsets, d.lengths):
        starts.append(o + np.arange(0, n - k, k))  # the last frame has no action
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

    `coverage` is the share of windows that qualify; the other chunks are too rare for a conditional entropy
    (the plug-in estimate would read every rare chunk as a deterministic effect). Chunks and effects are integer labels.
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


def chunk_diversity(d: Data, chunks=CHUNKS, max_windows=500_000) -> dict:
    """Distinct action chunks and distinct effects of the chunks, per chunk length k."""
    action = np.nan_to_num(d.cols["action"][:, 0], nan=0).astype(np.int8)
    out = {}
    rng = np.random.default_rng(0)
    for k in chunks:
        starts = _windows(d, k)
        if len(starts) > max_windows:
            starts = rng.choice(starts, max_windows, replace=False)
        if not len(starts):
            continue
        grams = action[starts[:, None] + np.arange(k)]
        _, chunk_label, counts = np.unique(grams, axis=0, return_inverse=True, return_counts=True)
        chunk_label = chunk_label.reshape(-1)
        entry = {
            "windows": int(len(starts)),
            "distinct_chunks": int(len(counts)),
            "chunk_entropy_bits": _entropy_bits(counts),
        }
        if d.has("inventory", "achievements", "vitals", "position"):
            end = starts + k
            inv = d.cols["inventory"][end].astype(np.int16) - d.cols["inventory"][starts]
            ach = d.cols["achievements"][end].astype(np.int8) - d.cols["achievements"][starts]
            vit = np.sign(d.cols["vitals"][end].astype(np.int16) - d.cols["vitals"][starts]).astype(np.int8)
            pos = np.clip(d.cols["position"][end].astype(np.int16) - d.cols["position"][starts], -k, k)
            interaction = np.concatenate([inv, ach, vit], axis=1)
            _, int_label, c_int = np.unique(interaction, axis=0, return_inverse=True, return_counts=True)
            _, all_label, c_all = np.unique(np.concatenate([interaction, pos], axis=1), axis=0, return_inverse=True, return_counts=True)
            entry |= {
                "distinct_interaction_effects": int(len(c_int)),
                "interaction_entropy_bits": _entropy_bits(c_int),
                "distinct_effects_with_motion": int(len(c_all)),
                "effect_entropy_bits": _entropy_bits(c_all),
                "no_effect_share": float((~interaction.any(1) & ~pos.any(1)).mean()),
                "information": {
                    "interaction": mutual_information(chunk_label, int_label.reshape(-1)),
                    "with_motion": mutual_information(chunk_label, all_label.reshape(-1)),
                },
            }
        out[k] = entry
    return out


def _action_steps(d: Data):
    """Indices of the steps with an action inside an episode, the action index, and whether the step changed the state."""
    action = d.cols["action"][:, 0]
    valid = ~np.isnan(action)
    nxt = np.arange(d.total) + 1
    valid &= nxt < d.total
    valid[valid] &= d.episode[nxt[valid]] == d.episode[valid]
    idx = np.nonzero(valid)[0]
    a = action[idx].astype(int)
    changed = np.zeros(len(idx), dtype=bool)
    for name in ("inventory", "achievements", "vitals", "position", "facing"):
        if d.has(name):
            col = d.cols[name].reshape(d.total, -1)
            changed |= (col[idx + 1] != col[idx]).any(1)
    return idx, a, changed


def action_effects(d: Data) -> list[dict]:
    """Share of each action in the data, and how often it changes the state."""
    _, a, changed = _action_steps(d)
    rows = []
    for i, name in enumerate(ACTIONS):
        m = a == i
        rows.append({"action": name, "share": float(m.mean()), "changes_state": float(changed[m].mean()) if m.any() else None})
    return rows


def action_coverage(d: Data) -> dict | None:
    """How often every action is taken, per source, and the actions that are thin overall.

    Without uniform random episodes the action space is only covered by what the policies do; an action
    that is (almost) never taken, or never changes anything, is a hole the planner can fall into.
    """
    if not d.has("source"):
        return None
    idx, a, changed = _action_steps(d)
    source = d.cols["source"][idx].astype(int)
    steps = np.array([(a == i).sum() for i in range(len(ACTIONS))])
    out = {"by_source": {}, "thin": [n for n, c in zip(ACTIONS, steps) if c < MIN_ACTION_STEPS], "min_steps": MIN_ACTION_STEPS}
    for code, name in SOURCES.items():
        here = source == code
        if not here.any():
            continue
        out["by_source"][name] = {
            "steps": int(here.sum()),
            "actions": {n: {"steps": int((here & (a == i)).sum()), "changes_state": int((here & (a == i) & changed).sum())}
                        for i, n in enumerate(ACTIONS)},
        }
    return out


# --------------------------------------------------------------------------
# which episodes


def primary_episodes(d: Data) -> tuple[np.ndarray, str]:
    """The episodes the structure of the data is read from: the expert's (source 0, the tasks come from them), else all."""
    if d.ep_source is not None and (d.ep_source == 0).any():
        return np.nonzero(d.ep_source == 0)[0], "expert"
    return np.arange(d.n_episodes), "all episodes"


def episode_groups(d: Data, min_episodes: int = 1) -> dict[str, np.ndarray]:
    """Analysed-episode indices per source, and per snapshot (`early@<updates>`) for the early rollouts."""
    if d.ep_source is None:
        return {"all": np.arange(d.n_episodes)}
    out = {}
    for code, name in SOURCES.items():
        rows = np.nonzero(d.ep_source == code)[0]
        if len(rows) >= min_episodes:
            out[name] = rows
        if code == 3 and d.ep_updates is not None and len(rows):
            updates = np.unique(d.ep_updates[rows])
            for u in updates if len(updates) > 1 else []:
                sub = rows[d.ep_updates[rows] == u]
                if len(sub) >= min_episodes:
                    out[f"early@{int(u)}"] = sub
    return out


# --------------------------------------------------------------------------
# the entity frequency gap


def _entity_arrays(d: Data, name: str, idx):
    if name in EXTRA_ENTITIES:
        return d.state_features(idx, name)
    arrays = d.episodes(name, idx)
    if name == "facing":  # a direction 1-4: one-hot, so that no order is assumed
        return (np.eye(4, dtype=np.float32)[np.clip(x[:, 0].astype(int), 1, 4) - 1] for x in arrays)
    return (x.reshape(len(x), -1) for x in arrays)


def entity_power(d: Data, name: str, window: int, idx) -> tuple[np.ndarray, np.ndarray]:
    """Per episode of `idx`: the Welch power of an entity per bin (rows) and its number of segments."""
    acc = np.zeros((len(idx), window // 2 + 1))
    segments = np.zeros(len(idx), np.int64)
    for row, x in enumerate(_entity_arrays(d, name, idx)):
        acc[row], segments[row] = tsc.episode_power(x, window)
    return acc, segments


def _gap_pair(summaries: dict) -> dict:
    """The gap of the entities, and without the daylight (a clock that no action changes)."""
    return {
        "all": tsc.frequency_gap(summaries),
        "without_daylight": tsc.frequency_gap({n: s for n, s in summaries.items() if n != "daylight"}),
    }


def _thin(d: Data, rows: np.ndarray, budget: int) -> np.ndarray:
    step = max(1, -(-int(d.lengths[rows].sum()) // budget))
    return rows[::step]


def spectral_analysis(d: Data, windows=WINDOWS, primary=PRIMARY_WINDOW, n_boot=N_BOOT, view_frames=VIEW_FRAMES, mix=None, use_view=True) -> dict | None:
    """The entity frequency gap of the paper on the 7 probe entities, with what it needs to be read.

    At the primary window: the period and power band of every entity, the gap with an episode bootstrap, the gap per
    source and snapshot, for the mix `mix` (source name -> share of frames) if given, and at the other windows. The
    terrain counts and the block one-hot of the view are reported as extras (they come from the `state` column).
    Episodes shorter than a window have no segment at it: `episodes_used` says how many remain.
    """
    names = [n for n in ENTITIES if d.has(n)]
    if len(names) < 2:
        return None
    primary_rows, label = primary_episodes(d)
    everything = np.arange(d.n_episodes)
    power = {n: entity_power(d, n, primary, everything) for n in names}
    segments = np.max([power[n][1] for n in names], axis=0)

    def summaries(rows):
        return {n: tsc.spectrum_summary(power[n][0][rows].sum(0), primary) for n in names}

    here = summaries(primary_rows)
    entities = {n: {**s, "probe_group": ENTITIES[n], "segments": int(power[n][1][primary_rows].sum())} for n, s in here.items() if s}
    present = {n: power[n][0][primary_rows] for n, s in here.items() if s}
    boot = {
        "all": tsc.bootstrap_gap(present, primary, n_boot),
        "without_daylight": tsc.bootstrap_gap({n: v for n, v in present.items() if n != "daylight"}, primary, n_boot),
    }
    gap = {k: (v and {**v, "ci": boot[k]}) for k, v in _gap_pair(here).items()}

    sensitivity = {}
    for w in sorted(set(windows) | {primary}):
        if w == primary:
            s = here
        else:
            s = {n: tsc.spectrum_summary(entity_power(d, n, w, primary_rows)[0].sum(0), w) for n in names}
        sensitivity[w] = {"gap": _gap_pair(s), "entities": {n: {"period": v["period"], "limited": v["limited"]} for n, v in s.items() if v}}

    groups = {}
    for name, rows in episode_groups(d, MIN_GROUP_EPISODES).items():
        s = summaries(rows)
        groups[name] = {"episodes": len(rows), "segments": int(segments[rows].sum()), "gap": _gap_pair(s)}

    mixed = None
    if mix:
        by_source = {n: r for n, r in episode_groups(d).items() if n in mix and segments[r].sum() > 0}
        if by_source:
            total = sum(mix[n] for n in by_source)
            psd = {n: sum(mix[g] / total * power[n][0][r].sum(0) / power[n][1][r].sum() for g, r in by_source.items() if power[n][1][r].sum() > 0) for n in names}
            mixed = {"shares": {g: mix[g] / total for g in by_source}, "gap": _gap_pair({n: tsc.spectrum_summary(np.asarray(p), primary) if np.any(p) else None for n, p in psd.items()})}

    extra = {}
    if use_view and d.has_state():
        for name in EXTRA_ENTITIES:
            rows = _thin(d, primary_rows, view_frames) if name == "view" else primary_rows
            extra[name] = tsc.spectrum_summary(entity_power(d, name, primary, rows)[0].sum(0), primary)
            if extra[name]:
                extra[name]["episodes"] = len(rows)

    by_group = {}
    for n, e in entities.items():
        by_group.setdefault(e["probe_group"], []).append(e["period"])
    order = {g: float(np.mean(v)) for g, v in by_group.items()}
    ranked = [order[g] for g in ("fast", "medium", "slow") if g in order]
    psd = {n: (power[n][0][primary_rows].sum(0) / max(power[n][0][primary_rows].sum(), 1e-12)).tolist() for n in entities}
    daylight = sensitivity.get(DAYLIGHT_CHECK_WINDOW, {}).get("entities", {}).get("daylight")
    return {
        "subset": label, "episodes": len(primary_rows), "primary_window": primary,
        "episodes_used": int((segments[primary_rows] > 0).sum()), "segments": int(segments[primary_rows].sum()),
        "entities": entities, "gap": gap, "sensitivity": sensitivity, "groups": groups, "mix": mixed, "extra": extra,
        "probe_group_periods": order, "probe_groups_ordered": ranked == sorted(ranked), "psd": psd,
        "daylight_check": None if daylight is None else {"window": DAYLIGHT_CHECK_WINDOW, "limited": daylight["limited"]},
    }  # fmt: skip


# --------------------------------------------------------------------------
# events and the ladder


def event_masks(d: Data, idx) -> dict[str, list[np.ndarray]]:
    """Per event type and episode of `idx`, the (n - 1,) mask of the steps where it happens (see `ENTITY_EVENTS`)."""
    out = {}
    for name in ("position", "facing", "mobs_in_view", "inventory"):
        if d.has(name):
            out[name] = [tsc.change_mask(x) for x in d.episodes(name, idx)]
    if d.has("vitals"):
        vitals = d.episodes("vitals", idx)
        for i, vital in enumerate(VITALS):
            out[f"{vital}_drop"] = [tsc.drop_mask(x[:, i]) for x in vitals]
            out[f"{vital}_rise"] = [tsc.rise_mask(x[:, i]) for x in vitals]
    if d.has("achievements"):
        out["unlock"] = [tsc.rise_mask(x) for x in d.episodes("achievements", idx)]
    return out


def event_table(masks: dict) -> dict:
    """Inter-event statistics per event type, and per entity (the events of its types together)."""
    types = {name: tsc.inter_event_stats(m) for name, m in masks.items()}
    entities = {}
    for entity, names in ENTITY_EVENTS.items():
        if names and all(n in masks for n in names):
            entities[entity] = tsc.inter_event_stats(tsc.union_masks([masks[n] for n in names]))
    return {"types": types, "entities": entities}


def clock_check(events: dict, min_gaps: int = 50) -> list[dict]:
    """The environment's own clocks: the most frequent time between two drops of food, drink and energy."""
    out = []
    for name, expected in CLOCKS.items():
        stats = events["types"].get(name)
        if stats and stats["gaps"] >= min_gaps:
            out.append({"event": name, "expected": expected, "mode": stats["mode"], "mode_share": stats["mode_share"], "ok": stats["mode"] == expected})
    return out


def regularity(cv: float | None) -> str:
    return "-" if cv is None else "regular" if cv < CV_REGULAR else "irregular" if cv > CV_IRREGULAR else "mixed"


def timescale_ladder(d: Data, spectral: dict, masks: dict) -> dict | None:
    """The time scales of the entities in order: clusters, the intervals between them, a stride and a band per cluster.

    The scale of an entity is the median time between its events (position, facing and mobs change nearly every step,
    vitals every few steps, inventory and achievements rarely). The spectral period cannot order the slow end: at one
    window the staircases, a walk and the slow clocks are all limited by it and look alike. Entities are clustered
    wherever two neighbours are at least 2x apart (heuristic). Per cluster, from the events of its entities together:
    the median (the fixed stride at which a level sees about one event per step), p10 to p90 (a band), the CV, the
    clips left at a top step equal to the p90, and the spectral periods of its members as a cross-check. The cumulative
    step of a level boundary lies between the medians of two clusters. A reading of the data, not the best strides:
    later experiments decide. Entities without events (the daylight, a clock) are listed apart.
    """
    entities = spectral["entities"]
    window = spectral["primary_window"]
    events = {
        n: tsc.inter_event_stats(tsc.union_masks([masks[t] for t in ENTITY_EVENTS[n]]))
        for n in entities if ENTITY_EVENTS[n] and all(t in masks for t in ENTITY_EVENTS[n])
    }  # fmt: skip
    medians = {n: s["p50"] for n, s in events.items() if "p50" in s}
    if len(medians) < 2:
        return None
    clusters = tsc.cluster_scales(medians)
    rows = []
    for members in clusters:
        together = tsc.inter_event_stats(tsc.union_masks([masks[t] for n in members for t in ENTITY_EVENTS[n]]))
        pooled = tsc.spectrum_summary(np.mean([np.asarray(spectral["psd"][n]) for n in members], axis=0), window)
        row = {
            "entities": members, "events": together, "regularity": regularity(together.get("cv")),
            "spectral_periods": {n: {"period": entities[n]["period"], "limited": entities[n]["limited"]} for n in members},
            "band": pooled["band"] if pooled else None,
        }  # fmt: skip
        if "p90" in together and together["p90"] >= 1:
            row["clips_at_p90"] = clip_counts(d, totals=(int(math.ceil(together["p90"])),))[0]
        rows.append(row)
    periods = {n: e["period"] for n, e in entities.items() if n != "daylight"}
    gap = spectral["gap"]["all"]
    return {
        "clusters": rows,
        "boundaries": tsc.boundaries(clusters, medians),
        "levels_upper_bound": len(clusters),
        "without_events": {n: {"period": e["period"], "limited": e["limited"]} for n, e in entities.items() if n not in medians},
        "verdict": {
            "gap": gap["ratio"] if gap else None,
            "gap_at_least_threshold": bool(gap and gap["ratio"] >= GAP_THRESHOLD),
            "ci_above_threshold": bool(gap and gap["ci"] and gap["ci"]["p5"] >= GAP_THRESHOLD),
            # the slowest spectral cluster (the daylight is a clock, left out) holds what the goal test depends on
            "slow_end_has_progress": bool({"inventory", "achievements"} & set(tsc.cluster_scales(periods)[-1])),
        },
    }  # fmt: skip


# --------------------------------------------------------------------------
# tasks, achievements per source, survival


def task_feasibility(d: Data, idx, offsets=OFFSETS, num_tasks: int | None = None) -> dict | None:
    """Per offset: the expert episodes that have a start with progress at it (a task needs one), and what the tasks require.

    The pairs are those of `eval_tasks.py`: goal = start + offset, the goal frame has an achievement the start lacks.
    `possible_tasks` is the number of episodes with a progressing pair at any offset (one task per episode).
    """
    flags = d.episodes("achievements", idx)
    per_offset, any_progress = {}, np.zeros(len(flags), bool)
    for offset in offsets:
        has, starts, with_progress, sizes, required = np.zeros(len(flags), bool), 0, 0, [], np.zeros(len(ACHIEVEMENTS))
        for e, f in enumerate(flags):
            if len(f) <= offset:
                continue
            new = f[offset:].astype(bool) & ~f[:-offset].astype(bool)
            progress = new.any(1)
            starts += len(progress)
            with_progress += int(progress.sum())
            has[e] = progress.any()
            sizes.append(new.sum(1)[progress])
            required += new[progress].sum(0)
        any_progress |= has
        sizes = np.concatenate(sizes) if sizes else np.zeros(0)
        per_offset[offset] = {
            "episodes_with_progress": int(has.sum()),
            "share_of_starts": with_progress / starts if starts else None,
            "required_median": float(np.median(sizes)) if len(sizes) else None,
            "required_p90": float(np.percentile(sizes, 90)) if len(sizes) else None,
            "most_required": [(ACHIEVEMENTS[i], float(required[i] / with_progress)) for i in np.argsort(-required)[:4] if required[i] > 0],
        }
    return {
        "episodes": len(flags), "offsets": per_offset, "possible_tasks": int(any_progress.sum()), "num_tasks": num_tasks,
        "short": bool(num_tasks and any_progress.sum() < num_tasks),
    }  # fmt: skip


def achievement_ladder(d: Data) -> dict:
    """Per source and snapshot: the share of episodes that unlock each achievement, and the median frame of the first unlock."""
    out = {}
    for name, rows in episode_groups(d).items():
        flags = d.episodes("achievements", rows)
        first = np.stack([np.where(f.any(0), f.argmax(0), -1) for f in flags])
        out[name] = {
            "episodes": len(rows),
            "achievements": {
                ACHIEVEMENTS[i]: {"share": float((first[:, i] >= 0).mean()), "median_frame": float(np.median(first[first[:, i] >= 0, i])) if (first[:, i] >= 0).any() else None}
                for i in range(len(ACHIEVEMENTS))
            },
        }
    return out


def survival(d: Data, need=CRITICAL_NEED, health=CRITICAL_HEALTH) -> dict | None:
    """How dangerous the data is: critical frames per source, whether a critical stretch ends in recovery or in death.

    A frame is critical when food, drink or energy is at most `need` (a few ticks from zero: they fall every 21 to 31
    steps) or health is at most `health`. A death is preceded by a terminal critical stretch when the episode's last
    frames are all critical; its length is the time the agent had between the last safe frame and the death.
    """
    if not d.has("vitals") or d.ep_dead is None:
        return None
    out = {}
    for name, rows in episode_groups(d).items():
        frames = critical = runs = recovered = deaths = deaths_critical = 0
        terminal = []
        for i, v in zip(rows, d.episodes("vitals", rows)):
            v = v.astype(int)
            c = (v[:, 1:].min(1) <= need) | (v[:, 0] <= health)
            frames, critical = frames + len(c), critical + int(c.sum())
            edges = np.diff(np.concatenate([[0], c.astype(np.int8), [0]]))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            runs += len(starts)
            recovered += int((ends < len(c)).sum())
            if d.ep_dead[i]:
                deaths += 1
                if c[-1]:
                    deaths_critical += 1
                    terminal.append(len(c) - starts[-1])
        out[name] = {
            "episodes": len(rows), "critical_share": critical / frames, "critical_runs": runs,
            "recovered_share": recovered / runs if runs else None, "deaths": deaths,
            "deaths_after_critical": deaths_critical / deaths if deaths else None,
            "terminal_run": {f"p{q}": float(np.percentile(terminal, q)) for q in (25, 50, 75, 90)} if terminal else None,
        }  # fmt: skip
    return out


# --------------------------------------------------------------------------
# report


def seed_overlap(datas: dict[str, Data]) -> list[str]:
    notes = []
    labels = list(datas)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            if datas[a].seeds is not None and datas[b].seeds is not None:
                shared = np.intersect1d(datas[a].seeds, datas[b].seeds)
                if len(shared):
                    notes.append(f"WARNING: {len(shared)} world seeds appear in both {a} and {b}: not held out")
    return notes


def provenance(runs: list[dict]) -> list[dict]:
    """One short record per collection run: what produced the episodes (see the collector's `runs`)."""
    out = []
    for r in runs:
        checkpoint = r.get("checkpoint") or {}
        args = r.get("args") or {}
        out.append(
            {
                "started": r.get("started"),
                "git": (r.get("git_commit") or "")[:8] + ("+dirty" if r.get("git_dirty") else ""),
                "craftax": (r.get("versions") or {}).get("craftax"),
                "eps": args.get("eps"),
                "seeds": [args.get("seed_start"), args.get("seed_end")],
                "kept": r.get("kept"),
                "finished": bool(r.get("finished")),
                "checkpoint": (checkpoint.get("sha256") or "")[:8] or None,
                "checkpoint_updates": (checkpoint.get("training") or {}).get("n_updates"),
            }
        )
    return out


def analyse(datas: dict[str, Data], horizons=HORIZONS, offsets=OFFSETS, windows=WINDOWS, primary_window=PRIMARY_WINDOW, n_boot=N_BOOT,
            view_frames=VIEW_FRAMES, use_view=True, num_tasks=None, mix=None) -> dict:  # fmt: skip
    result = {"seed_overlap": seed_overlap(datas), "files": {}}
    for label, d in datas.items():
        primary, subset = primary_episodes(d)
        entry = {"composition": composition(d), "clip_counts": clip_counts(d), "provenance": provenance(d.runs), "subset": subset}
        if d.has("achievements"):
            entry["events"] = event_durations(d, primary)
            entry["required_top_step"] = required_top_step(d, horizons=horizons, offsets=offsets)
            entry["tasks"] = task_feasibility(d, primary, offsets, num_tasks)
            entry["achievement_ladder"] = achievement_ladder(d)
        entry["spectral"] = entry["event_table"] = entry["ladder"] = entry["distance_horizon"] = None
        if d.has(*ENTITIES):
            entry["spectral"] = spectral_analysis(d, windows, primary_window, n_boot, view_frames, mix, use_view)
            masks = event_masks(d, primary)
            entry["event_table"] = event_table(masks)
            entry["clock_check"] = clock_check(entry["event_table"])
            entry["ladder"] = timescale_ladder(d, entry["spectral"], masks)
            entry["distance_horizon"] = distance_horizon(d, primary)
            entry["survival"] = survival(d)
        if d.has("action"):
            entry["chunks"] = chunk_diversity(d)
            if d.has("position"):
                entry["actions"] = action_effects(d)
                entry["action_coverage"] = action_coverage(d)
        result["files"][label] = entry
    return result


def _f(x, digits=3):
    return "-" if x is None else (f"{x:.{digits}g}" if isinstance(x, float) else str(x))


def _gap_text(g) -> str:
    if not g:
        return "-"
    text = f"{g['ratio']:.3g}x" + (" or more (the slowest entity is limited by the window)" if g["lower_bound"] else "")
    if g.get("ci"):
        text += f", bootstrap p5 to p95 {g['ci']['p5']:.3g} to {g['ci']['p95']:.3g}"
    return text + f" ({g['fastest']} fastest, {g['slowest']} slowest)"


def _gap_cell(pair) -> str:
    g = pair["all"]
    return "-" if not g else f"{g['ratio']:.3g}x" + (" (>=)" if g["lower_bound"] else "")


def _render_spectral(s: dict) -> list[str]:
    w = s["primary_window"]
    out = ["", f"### Is a hierarchy useful? The entity frequency gap ({s['subset']} episodes)", "",
           "As in H-JEPA (appendix F): the ratio of the largest to the smallest period (1 / spectral centroid) of the 7 probe entities. "
           f"Every dimension is standardised per episode, the spectrum is a Welch estimate with windows of {w} frames (hop {w // 2}, Hann), "
           f"averaged over the dimensions and pooled over the episodes: {s['episodes_used']} of {s['episodes']} episodes are long enough, "
           f"{s['segments']} segments. In the paper's six environments a gap below {GAP_THRESHOLD:g}x showed little selective abstraction "
           "(a boundary between six points, not a calibrated test); the gap says what changes fast or slowly, not what is predictable "
           "(see the distance horizon). `limited`: more than half of the power is in the two lowest bins, so the window and not the data "
           "sets the period, which then is a lower bound.", "",
           "| entity | probe group | period (steps) | 10-90% of the power in periods | limited | segments |", "|---|---|---|---|---|---|"]
    for n, e in s["entities"].items():
        b = e["band"]
        out.append(f"| {n} | {e['probe_group']} | {'>= ' if e['limited'] else ''}{_f(e['period'])} | {_f(b['fast_edge'])} to {_f(b['slow_edge'])} | "
                   f"{'yes' if e['limited'] else 'no'} | {e['segments']} |")
    for key, name in (("all", "All 7 entities"), ("without_daylight", "Without the daylight (a clock that no action changes)")):
        out += ["", f"{name}: gap {_gap_text(s['gap'][key])}."]
    out += ["", "| window | gap | without daylight | limited entities |", "|---|---|---|---|"]
    for win, v in s["sensitivity"].items():
        limited = ", ".join(n for n, x in v["entities"].items() if x["limited"]) or "-"
        out.append(f"| {win}{' (primary)' if win == w else ''} | {_gap_cell(v['gap'])} | "
                   f"{_gap_cell({'all': v['gap']['without_daylight']})} | {limited} |")
    if len(s["groups"]) > 1:
        out += ["", "Gap per source and snapshot (every episode of the file, primary window):", "",
                "| group | episodes | segments | gap | without daylight |", "|---|---|---|---|---|"]
        out += [f"| {n} | {g['episodes']} | {g['segments']} | {_gap_cell(g['gap'])} | {_gap_cell({'all': g['gap']['without_daylight']})} |"
                for n, g in s["groups"].items()]
    if s["mix"]:
        out += ["", "Gap of the mix " + ", ".join(f"{k} {v:.0%}" for k, v in s["mix"]["shares"].items()) + " (frame shares): "
                + _gap_cell(s["mix"]["gap"]) + "."]
    out.append("")
    for name, x in s["extra"].items():
        if x:
            out.append(f"Extra entity `{name}` (7x9 view from `state`, {x['episodes']} episodes): period {'>= ' if x['limited'] else ''}{_f(x['period'])}, "
                       f"10-90% of the power in periods {_f(x['band']['fast_edge'])} to {_f(x['band']['slow_edge'])}.")
    order = ", ".join(f"{g} {_f(v)}" for g, v in s["probe_group_periods"].items())
    out.append("")
    out.append(f"Mean period per probe group: {order}; " + ("in the order fast < medium < slow." if s["probe_groups_ordered"]
               else "NOT in the order fast < medium < slow: the probe groups do not describe this data."))
    if s["daylight_check"]:
        c = s["daylight_check"]
        out.append(f"Check: the daylight (a day is 300 steps) is {'' if c['limited'] else 'NOT '}limited by the window of {c['window']}"
                   + ("." if c["limited"] else ": WARNING, the column or the estimator is wrong."))
    return out


def _render_horizon(ts: dict) -> list[str]:
    lags = sorted({lag for v in ts.values() for lag in v["ratio"]})
    out = ["", "### Distance horizon: how far a distance in an entity carries the time gap", "",
           "Distance between states `lag` steps apart against unrelated states (1 = unrelated). The cost of a flat latent is such a "
           "distance: where an entity's ratio reaches 1 the cost has no signal about elapsed time beyond that lag.", "",
           "| entity | probe group | saturation lag | " + " | ".join(f"lag {l}" for l in lags) + " |", "|---|---|---|" + "---|" * len(lags)]
    return out + [f"| {n} | {v['probe_group']} | {_f(v['saturation_lag'])} | " + " | ".join(_f(v["ratio"].get(l), 2) for l in lags) + " |" for n, v in ts.items()]


def _event_row(name, s) -> str:
    q = [_f(s.get(f"p{p}")) for p in (10, 25, 50, 75, 90)]
    return (f"| {name} | {s['events']} | {s['rate']:.3g} | {s['gaps']} | " + " | ".join(q) + f" | {_f(s.get('cv'), 2)} | "
            f"{_f(s.get('censored_share'), 2)} | {regularity(s.get('cv'))} |")


def _render_events(table: dict, clocks: list[dict], subset: str) -> list[str]:
    head = ["| events | count | per frame | gaps | p10 | p25 | p50 | p75 | p90 | CV | cut off | regularity |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out = ["", f"### Inter-event times ({subset} episodes)", "",
           "An event is a step on which something in the entity changed. The gap between two events is how long a level step "
           "would have to be to contain one event; p50 is a fixed-stride candidate, p10 to p90 a band candidate. CV (std / mean of the "
           f"gaps) below {CV_REGULAR:g} is regular, above {CV_IRREGULAR:g} irregular (heuristics). `cut off`: share of gaps that the episode "
           "end interrupts (the complete gaps are then biased short).", ""] + head
    out += [_event_row(n, s) for n, s in table["entities"].items()]
    out += ["", "Vitals by event type (drops are the clocks of the environment, rises are eating, drinking and recovering):", ""] + head
    out += [_event_row(n, table["types"][n]) for n in ENTITY_EVENTS["vitals"] if n in table["types"]]
    for c in clocks:
        out.append(f"Check: the most frequent gap between `{c['event']}` events is {c['mode']} (expected {c['expected']}, {c['mode_share']:.0%} of the gaps)"
                   + ("." if c["ok"] else ": WARNING, the column or the event definition is wrong."))
    return out


def _render_ladder(ladder: dict) -> list[str]:
    out = ["", "### Time scale ladder (stride and band candidates)", "",
           "The entities in order of their scale, the median time between their events, split into clusters wherever two neighbours are at least "
           f"{tsc.CLUSTER_RATIO:g}x apart (heuristic). The spectral period cannot order the slow end (staircases, walks and slow clocks are all "
           "limited by the window and look alike), so it is shown as a cross-check. Per cluster, from the events of its entities together: the "
           "median inter-event time (a fixed stride at which a level sees about one event per step) and p10 to p90 (a band). The cumulative step of a "
           "level boundary lies between the medians of two clusters. The numbers are candidates; later experiments (strides, depth, bands) decide. "
           f"At most {ladder['levels_upper_bound']} levels have distinct content.", "",
           "| cluster | entities | median gap | p10 to p90 | CV | events | spectral period of the members | clips at T = p90 (independent) |",
           "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(ladder["clusters"], 1):
        ev, clips = c["events"], c.get("clips_at_p90")
        periods = ", ".join(f"{n} {'>= ' if v['limited'] else ''}{_f(v['period'])}" for n, v in c["spectral_periods"].items())
        out.append(f"| {i} | {', '.join(c['entities'])} | {_f(ev.get('p50'))} | {(_f(ev.get('p10')) + ' to ' + _f(ev.get('p90'))) if 'p10' in ev else '-'} | "
                   f"{_f(ev.get('cv'), 2)} ({c['regularity']}) | {ev['events']} | {periods} | "
                   f"{'T = ' + str(clips['T']) + ': ' + str(clips['clips']) + ' (' + str(clips['independent_clips']) + ')' if clips else '-'} |")
    for n, v in ladder["without_events"].items():
        out.append(f"\nNo events for `{n}` (a clock): spectral period {'>= ' if v['limited'] else ''}{_f(v['period'])} steps.")
    for b in ladder["boundaries"]:
        out.append(f"\nBetween {'+'.join(b['fast'])} and {'+'.join(b['slow'])}: a level boundary lies between {_f(b['from'])} and {_f(b['to'])} steps "
                   f"({b['ratio']:.3g}x).")
    v = ladder["verdict"]
    out += ["", f"Reading of the gap against the paper's threshold of {GAP_THRESHOLD:g}x (a heuristic): gap {_f(v['gap'])}x, "
            f"{'at least' if v['gap_at_least_threshold'] else 'below'} the threshold; its bootstrap interval is "
            f"{'above' if v['ci_above_threshold'] else 'not above'} it; the slowest spectral cluster "
            f"{'contains' if v['slow_end_has_progress'] else 'does not contain'} inventory or achievements (what the goal test depends on). "
            "A large gap is close to guaranteed here (staircases against walking), so a pass is weak evidence and a gap below the threshold "
            "is the informative outcome; whether a learned hierarchy helps is a question for the models."]
    return out


def _render_tasks(tasks: dict, subset: str) -> list[str]:
    out = ["", f"### Task feasibility ({subset} episodes, protocol A: goal = start + offset, with progress)", "",
           f"{tasks['episodes']} episodes; {tasks['possible_tasks']} have a progressing pair at some offset (one task per episode)"
           + (f", {tasks['num_tasks']} tasks wanted: {'TOO FEW, collect more expert episodes' if tasks['short'] else 'enough'}" if tasks["num_tasks"] else "") + ".", "",
           "| offset | episodes with progress | starts with progress | required set median | p90 | most often required |", "|---|---|---|---|---|---|"]
    for o, v in tasks["offsets"].items():
        out.append(f"| {o} | {v['episodes_with_progress']} | {_f(v['share_of_starts'], 2)} | {_f(v['required_median'])} | {_f(v['required_p90'])} | "
                   + ", ".join(f"{n} {s:.0%}" for n, s in v["most_required"]) + " |")
    return out


def _render_achievement_ladder(ladder: dict) -> list[str]:
    names = list(ladder)
    out = ["", "### Achievement ladder per source and snapshot", "", "Share of episodes that unlock the achievement (median frame of the first unlock).", "",
           "| achievement | " + " | ".join(f"{n} ({ladder[n]['episodes']})" for n in names) + " |", "|---|" + "---|" * len(names)]
    for a in ACHIEVEMENTS:
        cells = []
        for n in names:
            v = ladder[n]["achievements"][a]
            cells.append(f"{v['share']:.0%} ({_f(v['median_frame'])})" if v["share"] else "0%")
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    return out


def _render_survival(s: dict) -> list[str]:
    out = ["", "### Survival", "",
           f"A frame is critical when food, drink or energy is at most {CRITICAL_NEED} or health at most {CRITICAL_HEALTH} (heuristics). "
           "A critical run that ends before the episode does is a recovery; a death after a terminal critical run is a slow one "
           "(the run's length is the time from the last safe frame to the death), without one it is sudden (lava, or damage from full health).", "",
           "| source | episodes | critical frames | critical runs | recovered | deaths | after a critical run | steps of the run into death p25 / p50 / p75 / p90 |",
           "|---|---|---|---|---|---|---|---|"]
    for n, v in s.items():
        run = v["terminal_run"]
        out.append(f"| {n} | {v['episodes']} | {v['critical_share']:.1%} | {v['critical_runs']} | {_f(v['recovered_share'], 2)} | {v['deaths']} | "
                   f"{_f(v['deaths_after_critical'], 2)} | " + (" / ".join(_f(run[k]) for k in ("p25", "p50", "p75", "p90")) if run else "-") + " |")
    return out


def render(result: dict) -> str:
    out = ["# Dataset statistics", ""]
    out += [f"- {n}" for n in result["seed_overlap"]] or ["World seeds are disjoint between the files."]
    for label, e in result["files"].items():
        c = e["composition"]
        out += ["", f"## {label}", "", f"{c['episodes']} episodes, {c['frames']} frames. Length min/median/max "
                f"{_f(c['length']['min'])}/{_f(c['length']['median'])}/{_f(c['length']['max'])}."]
        if "sampled" in c:
            out.append(f"The state sections below use every k-th episode ({c['sampled']['episodes_analysed']} of {c['sampled']['episodes']}); "
                       "counts here and the clip counts use all episodes.")
        if "by_source" in c:
            out.append("Sources (episodes, share of frames): " + ", ".join(
                f"{k} {v} ({c['frame_share_by_source'][k]:.1%})" for k, v in c["by_source"].items()) + ".")
        if c.get("early_checkpoints"):
            out.append("Early checkpoints (training updates: episodes, frames): " + ", ".join(
                f"{u}: {v['episodes']}, {v['frames']}" for u, v in c["early_checkpoints"].items()) + ".")
        for name, dth in (c.get("deaths") or {}).items():
            line = f"Deaths, {name}: {dth['died']} of {dth['episodes']} episodes ({dth['died_share']:.1%})"
            if "steps_to_death" in dth:
                line += f", median {_f(dth['steps_to_death']['median'])} steps to death"
            if "causes" in dth:
                line += ", causes " + ", ".join(f"{k} {v}" for k, v in dth["causes"].items())
            out.append(line + ".")
        runs = e.get("provenance") or []
        if runs:
            out += ["", "Collected by " + str(len(runs)) + " run(s):"] + [
                f"- {r['started']}, commit {r['git'] or '-'}, craftax {_f(r['craftax'])}, eps {_f(r['eps'])}, seeds {r['seeds'][0]} to {_f(r['seeds'][1])}, "
                f"kept {_f(r['kept'])}, expert {r['checkpoint'] or '-'}" + (f" (update {r['checkpoint_updates']})" if r["checkpoint_updates"] else "")
                + ("" if r["finished"] else ", NOT FINISHED")
                for r in runs
            ]
        else:
            out += ["", "No provenance recorded (file from an older collector)."]
        if "diamond_episodes" in c:
            out.append(f"Episodes with a diamond: {c['diamond_episodes']}; achievements per episode (mean): {_f(c['achievements_per_episode_mean'])}.")

        if e.get("spectral"):
            out += _render_spectral(e["spectral"])
            out += _render_horizon(e["distance_horizon"])
        else:
            out += ["", "Spectra, events and the ladder skipped: the file lacks the derived columns (collect it with the current collector)."]
        if e.get("event_table"):
            out += _render_events(e["event_table"], e["clock_check"], e["subset"])
        if e.get("ladder"):
            out += _render_ladder(e["ladder"])

        rows = e["clip_counts"]
        by_source = list(rows[0].get("by_source", {})) if rows else []
        out += ["", "### Clips per top-level step T", "",
                "T is the product of all strides: the environment steps of one top-level step. A clip needs 3 T + 1 level-1 frames "
                "(history 3, rollout 1, window 1); episodes that ended in death are extended by T - 1 copies of the dead frame. "
                "The counts hold for any depth and any split of T into strides. `clips above 1` leaves out the sources "
                "that train level 1 only.", "",
                "| T | span | episodes long enough | clips | clips above 1 | independent clips |" + "".join(f" {n} |" for n in by_source),
                "|---|---|---|---|---|---|" + "---|" * len(by_source)]
        out += [f"| {r['T']} | {r['span']} | {r['episodes_long_enough']} | {r['clips']} | {_f(r['clips_upper'])} | {r['independent_clips']} |"
                + "".join(f" {r['by_source'][n]} |" for n in by_source) for r in rows]

        ev = e.get("events")
        if ev and ev.get("unlock_frames"):
            out += ["", "### When achievements unlock (frame of the first unlock per episode)", "",
                    "| achievement | episodes | median | p25 | p75 |", "|---|---|---|---|---|"]
            out += [f"| {n} | {v['episodes']} | {_f(v['median'])} | {_f(v['p25'])} | {_f(v['p75'])} |" for n, v in ev["unlock_frames"].items()]
            if "gap_between_unlocks" in ev:
                g = ev["gap_between_unlocks"]
                out += ["", "Steps between consecutive unlocks: " + ", ".join(f"{k} {_f(v)}" for k, v in g.items()) + "."]

        req = e.get("required_top_step")
        if req:
            hs = req["horizons"]
            out += ["", "### Top-level step T needed to reach a distance", "",
                    "A plan of H top-level steps reaches H x T environment steps, so it reaches a distance D when T >= D / H. "
                    "T is the product of all strides; this does not assume a depth or strides. "
                    + (f"Expert steps to the first diamond ({req['expert_episodes_with_diamond']} episodes): "
                       + ", ".join(f"{q} {_f(v)}" for q, v in req["steps_to_diamond"].items()) + "." if req["steps_to_diamond"]
                       else "No expert episode with a diamond in this file."), "",
                    "| distance | steps | " + " | ".join(f"T for H = {h}" for h in hs) + " |", "|---|---|" + "---|" * len(hs)]
            out += [f"| {n} | {_f(req['distances'][n])} | " + " | ".join(str(v[h]) for h in hs) + " |"
                    for n, v in req["required_top_step"].items()]
            out.append("The clip counts above say how large T can get before too few clips remain.")

        ch = e.get("chunks")
        if ch:
            groups, codes = CODEBOOK
            out += ["", "### Action chunks of length k", "", "Non-overlapping chunks. Effects are what a chunk changes (inventory, "
                    "achievements, vitals; with motion also the position). Entropy in bits: 2^bits is the effective number of distinct chunks or effects, "
                    f"the scale for the codebook size (placeholder {groups} groups x {codes} codes = {codes ** groups} combinations). Information: "
                    f"I(chunk; effect) in bits (Miller-Madow corrected) over the chunks seen at least {MIN_CHUNK_COUNT} times; coverage is the share of "
                    "windows that qualify. A chunk that says nothing about its effect (the effect depends on the state, not on the actions alone) is "
                    "a poor macro-action at that length.", "",
                    "| k | chunks | distinct chunks | chunk entropy | distinct interaction effects | interaction entropy | distinct effects with motion | effect entropy "
                    "| no effect | information (interaction) | information (with motion) | coverage |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for k, v in ch.items():
                info = v.get("information") or {}
                mi_i, mi_m = info.get("interaction"), info.get("with_motion")
                out.append(f"| {k} | {v['windows']} | {v['distinct_chunks']} | {_f(v['chunk_entropy_bits'])} | {_f(v.get('distinct_interaction_effects'))} "
                           f"| {_f(v.get('interaction_entropy_bits'))} | {_f(v.get('distinct_effects_with_motion'))} | {_f(v.get('effect_entropy_bits'))} | {_f(v.get('no_effect_share'))} "
                           f"| {_f(mi_i and mi_i['mi_bits'])} | {_f(mi_m and mi_m['mi_bits'])} | {_f(mi_m and mi_m['coverage'], 2)} |")

        ac = e.get("actions")
        if ac:
            out += ["", "### What the actions do", "", "| action | share of steps | changes the state |", "|---|---|---|"]
            out += [f"| {r['action']} | {_f(r['share'])} | {_f(r['changes_state'])} |" for r in ac]
        cov = e.get("action_coverage")
        if cov and len(cov["by_source"]) > 1:
            names = list(cov["by_source"])
            out += ["", "### Action coverage per source", "", "Steps with the action (and the share of those that changed the state).", "",
                    "| action | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
            for action in ACTIONS:
                cells = []
                for n in names:
                    v = cov["by_source"][n]["actions"][action]
                    cells.append(f"{v['steps']} ({v['changes_state'] / v['steps']:.0%})" if v["steps"] else "0")
                out.append(f"| {action} | " + " | ".join(cells) + " |")
        if cov and cov["thin"]:
            out += ["", f"Thin actions (fewer than {cov['min_steps']} steps in the whole file): {', '.join(cov['thin'])}. "
                        "If they matter for the task (crafting, placing), add data that takes them before training."]
        if e.get("tasks"):
            out += _render_tasks(e["tasks"], e["subset"])
        if e.get("achievement_ladder"):
            out += _render_achievement_ladder(e["achievement_ladder"])
        if e.get("survival"):
            out += _render_survival(e["survival"])
    return "\n".join(out) + "\n"


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def parse_mix(items) -> dict[str, float] | None:
    """`expert=0.7 early=0.3` -> {"expert": 0.7, "early": 0.3}; `noised` is the collection name of `noised_expert`."""
    if not items:
        return None
    mix = {}
    for item in items:
        name, _, share = item.partition("=")
        name = {"noised": "noised_expert"}.get(name, name)
        if name not in SOURCES.values():
            raise SystemExit(f"unknown source {name!r} in --mix (known: {', '.join(SOURCES.values())})")
        if float(share) > 0:
            mix[name] = float(share)
    return mix or None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", action="append", required=True, help="HDF5 episode file (repeat for several)")
    parser.add_argument("--label", action="append", help="name of each file (default: the file name)")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="frames read per file")
    parser.add_argument("--offsets", type=int, nargs="+", default=list(OFFSETS), help="task offsets of protocol A (experiments.yaml)")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS), help="top-level planning horizons to tabulate")
    parser.add_argument("--num-tasks", type=int, default=None, help="tasks wanted per task set (experiments.yaml): flags too few expert episodes")
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS), help="Welch windows (frames) of the frequency gap sensitivity")
    parser.add_argument("--primary-window", type=int, default=PRIMARY_WINDOW, help="window of the headline gap, the bands and the ladder")
    parser.add_argument("--bootstrap", type=int, default=N_BOOT, help="episode bootstraps of the gap")
    parser.add_argument("--view-frames", type=int, default=VIEW_FRAMES, help="frames used for the spectrum of the 7x9 block view")
    parser.add_argument("--no-view", action="store_true", help="skip the extra entities from the `state` column (terrain, view)")
    parser.add_argument("--mix", nargs="+", metavar="SOURCE=SHARE", help="the frame shares of a candidate data mix, for its gap (expert=0.7 early=0.3)")
    parser.add_argument("--out", default="analysis_report")
    args = parser.parse_args()
    labels = args.label or [Path(p).stem for p in args.data]
    if len(labels) != len(args.data):
        parser.error("--label must be given once per --data")
    mix = parse_mix(args.mix)
    result = analyse({label: Data(path, args.max_frames) for label, path in zip(labels, args.data)}, args.horizons, args.offsets,
                     windows=args.windows, primary_window=args.primary_window, n_boot=args.bootstrap, view_frames=args.view_frames,
                     use_view=not args.no_view, num_tasks=args.num_tasks, mix=mix)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_stats.md").write_text(render(result))
    (out / "dataset_stats.json").write_text(json.dumps(to_jsonable(result), indent=1))
    print(render(result))


if __name__ == "__main__":
    main()
