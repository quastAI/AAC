"""Statistics of a Craftax-Classic episode file that the design decisions depend on.

Reads the collector's HDF5 files (numpy and h5py only, no pixels) and reports:

* composition: episodes, lengths, sources (and, for early-checkpoint episodes, the checkpoints), held-out seed check between files;
* clip counts per top-level step T (the product of all strides, whatever the depth): how many training
  clips remain, with the padding of episodes that ended in death, per source and for the levels above 1;
* the top-level step T needed to reach the task offsets and the expert's time to the diamond, per planning horizon;
* event durations: when achievements unlock and the gaps between unlocks;
* time scales of the symbolic state (the frequency gap): how fast every group
  decorrelates, from the distance between states `lag` steps apart;
* action-chunk diversity per chunk length k: distinct chunks and distinct *effects*,
  which sizes the VQ codebook and the band of durations;
* what each action does: share of the data and how often it changes anything, and per source how often
  every action is taken (is the action space covered when there are no uniform random episodes?).

    python dataset_stats.py --data expert_train.h5 --label train \
        --data expert_val.h5 --label val --out report/

The sections that need the derived columns (inventory, vitals, ...) are skipped
for files written by an older collector.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

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
GROUPS = {  # state columns by how fast they change
    "position": ("fast", ["position"]),
    "facing": ("fast", ["facing"]),
    "mobs_in_view": ("fast", ["mobs_in_view"]),
    "vitals": ("medium", ["vitals"]),
    "inventory": ("slow", ["inventory"]),
    "daylight": ("slow", ["daylight"]),
    "achievements": ("slow", ["achievements"]),
}
LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
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

    def has(self, *names) -> bool:
        return all(n in self.cols for n in names)

    def episodes(self, name):
        arr = self.cols[name]
        return [arr[o : o + n] for o, n in zip(self.offsets, self.lengths)]


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


def event_durations(d: Data) -> dict:
    flags = d.episodes("achievements")
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


def _group_matrix(d: Data, columns) -> np.ndarray:
    parts = [d.cols[c].reshape(d.total, -1).astype(np.float32) for c in columns]
    x = np.concatenate(parts, axis=1)
    return (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)


def time_scales(d: Data, lags=LAGS, pairs=200_000, seed=0, min_pairs=1000) -> dict:
    """Distance between states `lag` steps apart, against unrelated states (1 = unrelated).

    A group with a small ratio at a long lag is slow. The saturation lag is the
    first lag where the ratio reaches 0.9; None if it never does within `lags`.
    Lags with fewer than `min_pairs` pairs inside episodes are left out (too noisy).
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name, (speed, columns) in GROUPS.items():
        if not d.has(*columns):
            continue
        x = _group_matrix(d, columns)
        i = rng.integers(0, d.total, pairs)
        j = rng.integers(0, d.total, pairs)
        different = d.episode[i] != d.episode[j]
        unrelated = float(np.abs(x[i[different]] - x[j[different]]).mean()) or 1.0
        ratios, changed = {}, {}
        for lag in lags:
            if lag >= d.total:
                continue
            valid = d.episode[lag:] == d.episode[:-lag]
            if valid.sum() < min_pairs:
                continue
            delta = np.abs(x[lag:] - x[:-lag])[valid]
            ratios[lag] = float(delta.mean() / unrelated)
            changed[lag] = float((delta.max(1) > 1e-6).mean())
        saturation = next((lag for lag, r in ratios.items() if r >= 0.9), None)
        out[name] = {"speed": speed, "ratio": ratios, "changed": changed, "saturation_lag": saturation}
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
        _, counts = np.unique(grams, axis=0, return_counts=True)
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
            _, c_int = np.unique(interaction, axis=0, return_counts=True)
            _, c_all = np.unique(np.concatenate([interaction, pos], axis=1), axis=0, return_counts=True)
            entry |= {
                "distinct_interaction_effects": int(len(c_int)),
                "interaction_entropy_bits": _entropy_bits(c_int),
                "distinct_effects_with_motion": int(len(c_all)),
                "effect_entropy_bits": _entropy_bits(c_all),
                "no_effect_share": float((~interaction.any(1) & ~pos.any(1)).mean()),
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


def analyse(datas: dict[str, Data], horizons=HORIZONS, offsets=OFFSETS) -> dict:
    result = {"seed_overlap": seed_overlap(datas), "files": {}}
    for label, d in datas.items():
        entry = {"composition": composition(d), "clip_counts": clip_counts(d), "provenance": provenance(d.runs)}
        if d.has("achievements"):
            entry["events"] = event_durations(d)
            entry["required_top_step"] = required_top_step(d, horizons=horizons, offsets=offsets)
        if d.has("inventory", "vitals", "facing", "daylight", "position", "mobs_in_view", "achievements"):
            entry["time_scales"] = time_scales(d)
        else:
            entry["time_scales"] = None
        if d.has("action"):
            entry["chunks"] = chunk_diversity(d)
            if d.has("position"):
                entry["actions"] = action_effects(d)
                entry["action_coverage"] = action_coverage(d)
        result["files"][label] = entry
    return result


def _f(x, digits=3):
    return "-" if x is None else (f"{x:.{digits}g}" if isinstance(x, float) else str(x))


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

        ts = e.get("time_scales")
        if ts:
            lags = sorted({lag for v in ts.values() for lag in v["ratio"]})
            out += ["", "### Time scales of the state (distance between states `lag` steps apart; 1 = unrelated)", "",
                    "| group | speed | saturation lag | " + " | ".join(f"lag {l}" for l in lags) + " |", "|---|---|---|" + "---|" * len(lags)]
            out += [f"| {n} | {v['speed']} | {_f(v['saturation_lag'])} | " + " | ".join(_f(v["ratio"].get(l), 2) for l in lags) + " |" for n, v in ts.items()]
        elif ts is None:
            out += ["", "Time scales skipped: the file lacks the derived columns (collect it with the current collector)."]

        ch = e.get("chunks")
        if ch:
            out += ["", "### Action chunks of length k", "", "Non-overlapping chunks. Effects are what a chunk changes (inventory, "
                    "achievements, vitals; with motion also the position). Entropy in bits: 2^bits is the effective number of distinct chunks or effects, "
                    "the scale for the codebook size.", "",
                    "| k | chunks | distinct chunks | chunk entropy | distinct interaction effects | interaction entropy | distinct effects with motion | effect entropy | no effect |",
                    "|---|---|---|---|---|---|---|---|---|"]
            out += [f"| {k} | {v['windows']} | {v['distinct_chunks']} | {_f(v['chunk_entropy_bits'])} | {_f(v.get('distinct_interaction_effects'))} "
                    f"| {_f(v.get('interaction_entropy_bits'))} | {_f(v.get('distinct_effects_with_motion'))} | {_f(v.get('effect_entropy_bits'))} | {_f(v.get('no_effect_share'))} |"
                    for k, v in ch.items()]

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", action="append", required=True, help="HDF5 episode file (repeat for several)")
    parser.add_argument("--label", action="append", help="name of each file (default: the file name)")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="frames read per file")
    parser.add_argument("--offsets", type=int, nargs="+", default=list(OFFSETS), help="task offsets of protocol A (experiments.yaml)")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS), help="top-level planning horizons to tabulate")
    parser.add_argument("--out", default="analysis_report")
    args = parser.parse_args()
    labels = args.label or [Path(p).stem for p in args.data]
    if len(labels) != len(args.data):
        parser.error("--label must be given once per --data")
    result = analyse({label: Data(path, args.max_frames) for label, path in zip(labels, args.data)}, args.horizons, args.offsets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_stats.md").write_text(render(result))
    (out / "dataset_stats.json").write_text(json.dumps(to_jsonable(result), indent=1))
    print(render(result))


if __name__ == "__main__":
    main()
