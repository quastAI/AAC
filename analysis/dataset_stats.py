"""Statistics of a Craftax-Classic episode file that the design decisions depend on.

Reads the collector's HDF5 files (numpy and h5py only, no pixels) and reports:

* composition: episodes, lengths, sources, held-out seed check between files;
* clip counts per depth and stride: how many training clips a level can still draw
  (E2: strides, and whether deeper levels have enough data);
* event durations: when achievements unlock and the gaps between unlocks;
* time scales of the symbolic state (the frequency gap): how fast every group
  decorrelates, from the distance between states `lag` steps apart;
* action-chunk diversity per chunk length k: distinct chunks and distinct *effects*,
  which sizes the VQ codebook and the band of durations;
* what each action does: share of the data and how often it changes anything.

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
        names = ("expert", "noised_expert", "random")
        out["by_source"] = {n: int((d.all_sources == i).sum()) for i, n in enumerate(names) if (d.all_sources == i).any()}
        out["frames_by_source"] = {n: int(lengths[d.all_sources == i].sum()) for i, n in enumerate(names) if (d.all_sources == i).any()}
        out["frame_share_by_source"] = {n: v / d.total_all for n, v in out["frames_by_source"].items()}
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
    names = {0: "expert", 1: "noised_expert", 2: "random"}
    out = {}
    for code, name in names.items():
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


def clip_counts(d: Data, strides=(2, 3, 4, 6, 8), depths=(1, 2, 3), history=3, rollout=1) -> list[dict]:
    """Clips a model of `depth` levels can draw from the file, for equal strides above level 1."""
    steps = history + rollout
    rows = []
    for depth in depths:
        for stride in (strides if depth > 1 else (1,)):
            span = steps
            for _ in range(depth - 1):
                span = (span - 1) * stride + 1
            starts = np.maximum(d.all_lengths - span + 1, 0)
            rows.append(
                {
                    "depth": depth, "stride": stride, "span": int(span),
                    "episodes_long_enough": int((d.all_lengths >= span).sum()),
                    "clips": int(starts.sum()),
                    "independent_clips": int((d.all_lengths // span).sum()),
                }
            )  # fmt: skip
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
        median = float(np.median(gaps))
        out["stride_hint"] = {
            f"{levels}_levels": int(max(2, round(median ** (1 / (levels - 1)))))
            for levels in (2, 3)
        }  # a top-level step about one median gap between unlocks
        out["stride_hint_note"] = "equal strides s with s^(levels-1) frames per top-level step close to the median gap"
    return out


def coverage(d: Data, horizons=(4, 8), depths=(1, 2, 3, 4), strides=(2, 3, 4, 5, 6, 8, 10, 12), quantiles=(50, 75, 90)) -> dict | None:
    """How far the top level of a model reaches, against the time the expert takes to the diamond.

    One step of the top level spans `stride ** (depth - 1)` environment steps (equal strides), so its
    plan of `horizon` steps reaches `horizon * stride ** (depth - 1)` steps. The rule is that the top
    level comfortably covers the typical time to the diamond (a quantile of the expert's frame of the
    first diamond). Only expert episodes (source 0) count when the file has sources.
    """
    flags = d.episodes("achievements")
    keep = np.ones(len(flags), bool) if "source" not in d.cols else d.cols["source"][d.offsets] == 0
    times = [int(np.argmax(f[:, DIAMOND_INDEX])) for f, k in zip(flags, keep) if k and f[:, DIAMOND_INDEX].any()]
    if not times:
        return None
    target = {f"p{q}": float(np.percentile(times, q)) for q in quantiles}
    rows = []
    for depth in depths:
        for stride in strides if depth > 1 else (1,):
            step = stride ** (depth - 1)
            reach = {h: h * step for h in horizons}
            covered = [q for q, t in target.items() if reach[horizons[0]] >= t]
            rows.append({"depth": depth, "stride": stride, "top_step": step, "reach": reach,
                         "covers": covered[-1] if covered else None})  # fmt: skip
    smallest = {}  # per depth and horizon: the smallest equal stride whose reach covers each quantile
    for depth in depths:
        if depth == 1:
            continue
        for h in horizons:
            smallest[f"depth{depth}_horizon{h}"] = {
                q: next(s for s in range(2, 10_000) if h * s ** (depth - 1) >= t) for q, t in target.items()
            }
    return {"episodes": len(times), "steps_to_diamond": target, "rows": rows, "smallest_stride": smallest}


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


def action_effects(d: Data) -> list[dict]:
    """Share of each action in the data, and how often it changes the state."""
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
    rows = []
    for i, name in enumerate(ACTIONS):
        m = a == i
        rows.append({"action": name, "share": float(m.mean()), "changes_state": float(changed[m].mean()) if m.any() else None})
    return rows


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


def analyse(datas: dict[str, Data]) -> dict:
    result = {"seed_overlap": seed_overlap(datas), "files": {}}
    for label, d in datas.items():
        entry = {"composition": composition(d), "clip_counts": clip_counts(d)}
        if d.has("achievements"):
            entry["events"] = event_durations(d)
            entry["coverage"] = coverage(d)
        if d.has("inventory", "vitals", "facing", "daylight", "position", "mobs_in_view", "achievements"):
            entry["time_scales"] = time_scales(d)
        else:
            entry["time_scales"] = None
        if d.has("action"):
            entry["chunks"] = chunk_diversity(d)
            if d.has("position"):
                entry["actions"] = action_effects(d)
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
        for name, dth in (c.get("deaths") or {}).items():
            line = f"Deaths, {name}: {dth['died']} of {dth['episodes']} episodes ({dth['died_share']:.1%})"
            if "steps_to_death" in dth:
                line += f", median {_f(dth['steps_to_death']['median'])} steps to death"
            if "causes" in dth:
                line += ", causes " + ", ".join(f"{k} {v}" for k, v in dth["causes"].items())
            out.append(line + ".")
        if "diamond_episodes" in c:
            out.append(f"Episodes with a diamond: {c['diamond_episodes']}; achievements per episode (mean): {_f(c['achievements_per_episode_mean'])}.")

        out += ["", "### Clips per depth and stride", "", "History 3, rollout 1. `span` is the number of level-1 frames one clip needs.", "",
                "| depth | stride | span | episodes long enough | clips | independent clips |", "|---|---|---|---|---|---|"]
        out += [f"| {r['depth']} | {r['stride']} | {r['span']} | {r['episodes_long_enough']} | {r['clips']} | {r['independent_clips']} |" for r in e["clip_counts"]]

        ev = e.get("events")
        if ev and ev.get("unlock_frames"):
            out += ["", "### When achievements unlock (frame of the first unlock per episode)", "",
                    "| achievement | episodes | median | p25 | p75 |", "|---|---|---|---|---|"]
            out += [f"| {n} | {v['episodes']} | {_f(v['median'])} | {_f(v['p25'])} | {_f(v['p75'])} |" for n, v in ev["unlock_frames"].items()]
            if "gap_between_unlocks" in ev:
                g = ev["gap_between_unlocks"]
                out += ["", "Steps between consecutive unlocks: " + ", ".join(f"{k} {_f(v)}" for k, v in g.items()) + ".",
                        f"Stride hint (equal strides): {ev['stride_hint']} ({ev['stride_hint_note']}). A hint, not a decision."]

        cov = e.get("coverage")
        if cov:
            horizons = list(cov["rows"][0]["reach"])
            out += ["", "### Reach of the top level against the time to the diamond", "",
                    f"Expert steps to the first diamond ({cov['episodes']} episodes): "
                    + ", ".join(f"{q} {_f(v)}" for q, v in cov["steps_to_diamond"].items())
                    + ". A top step spans stride^(depth-1) steps; its plan reaches horizon x that. "
                    f"`covers` is the highest quantile reached at horizon {horizons[0]}.", "",
                    "| depth | stride | steps per top-level step | " + " | ".join(f"reach, horizon {h}" for h in horizons) + " | covers |",
                    "|---|---|---|" + "---|" * len(horizons) + "---|"]
            out += [f"| {r['depth']} | {r['stride']} | {r['top_step']} | " + " | ".join(str(r["reach"][h]) for h in horizons)
                    + f" | {r['covers'] or '-'} |" for r in cov["rows"]]
            out += ["", "Smallest equal stride whose top level reaches the quantile:", "",
                    "| depth, horizon | " + " | ".join(next(iter(cov["smallest_stride"].values()))) + " |",
                    "|---|" + "---|" * len(cov["steps_to_diamond"])]
            out += [f"| {k.replace('depth', '').replace('_horizon', ', ')} | " + " | ".join(str(v) for v in q.values()) + " |"
                    for k, q in cov["smallest_stride"].items()]
            out.append("The clip counts above say whether that depth and stride still leave enough clips.")

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
    parser.add_argument("--out", default="analysis_report")
    args = parser.parse_args()
    labels = args.label or [Path(p).stem for p in args.data]
    if len(labels) != len(args.data):
        parser.error("--label must be given once per --data")
    result = analyse({label: Data(path, args.max_frames) for label, path in zip(labels, args.data)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_stats.md").write_text(render(result))
    (out / "dataset_stats.json").write_text(json.dumps(to_jsonable(result), indent=1))
    print(render(result))


if __name__ == "__main__":
    main()
