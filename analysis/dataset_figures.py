"""The dataset figures and tables of the appendix that the model design depends on, from the episode files.

Two steps, so that a figure can be restyled without the data and every number has one source:

* `compute` reads the collector's HDF5 files (numpy and h5py, no pixels; the view of the `state` column for the change
  matrix) and writes `dataset_numbers.json`: every number of every figure and table, the files' hashes, the collector's
  `runs` records and the git commit (as `dataset_tables.json` does);
* `plot` reads only that JSON and writes the figures (`fig_*.pdf`, matplotlib) and tables (`table_*.tex`, `*.csv`).

Each split is taken as one set, the way the model sees it: clips are drawn at random from all of train. Test is the
final expert only. Validation is computed like the others and only compared with train (`val_vs_train`: distances, no
figures).

* length: frames per episode (`ep_len`, L + 1 frames for L actions), per split, as a histogram and quantiles; the time
  to the diamond (first frame with `collect_diamond`) of the episodes that reach it, per split (shown for test);
* spectral: the entity frequency gap of the paper (`dataset_stats.spectral_analysis`, the 7 probe entities, Welch window
  256, window sensitivity, episode bootstrap) on the test split. The numbers are not strides: the slowest entities are
  limited by the window (the gap is a lower bound then) and the centroid also rises with fast flicker;
* achievements: achievements per episode (final frame), and per achievement the share of episodes that unlock it and the
  quantiles of the frame of its first unlock (among the episodes that do), the ladder of the expert's structure;
* change matrix: per component of the state (and the view, pooled) and per lag k, the share of frame pairs (t, t + k)
  of one episode in which it differs (net change, not "anything changed in between"). Daylight is a float and counts as
  changed above `DAYLIGHT_TOL` (a heuristic; it is also periodic with a day of 300 steps, so the share falls back near
  lag 300). The view is egocentric, so walking changes every tile: besides the egocentric rows the terrain and the mobs
  are compared in world coordinates on the tiles that both views show (needs the `position` column; only for lags up
  to `WORLD_LAG_MAX`, where two views always overlap). `half_lag` is the first lag of the grid with a share of 50%;
* clip coverage: clips (all overlapping, not independent ones) per stride, for context 1 and 5 and rollout 1 and 5, as
  in `dataset_stats.clip_counts` (a clip is `context + rollout` frames, death padding as in training);
* actions: steps per action, and per chunk length k the distinct chunks, the entropy of the chunk and its mutual
  information with the effect (`dataset_stats.chunk_diversity`), reference scales for a codebook, not settings.

    python dataset_figures.py compute --data expert_train.h5 --label train --data expert_val.h5 --label val \
        --data expert_test.h5 --label test --out ../reports/dataset/ [--sha256]
    python dataset_figures.py plot --out ../reports/dataset/

Pairs, lags and strides never cross an episode. The numbers are deterministic (seeded bootstraps and subsamples).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

import dataset_stats as ds
import dataset_tables as dt

LAGS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400)  # steps between the two frames of a pair
STRIDES = (1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400)  # frame spacing of a clip
CONTEXTS = (1, 5)  # frames of context (`history`)
ROLLOUTS = (1, 5)  # frames to predict
DAYLIGHT_TOL = 0.05  # a light change above this counts as a change (the light moves at most 0.031 per step)
VIEW_ROWS, VIEW_COLS = 7, 9  # the local view of Craftax-Classic (OBS_DIM), the player in the middle
WORLD_LAG_MAX = 6  # two views at most this many steps apart always overlap (the player moves one tile per step)
HALF = 0.5  # share of pairs that defines the half-change lag
LENGTH_QUANTILES = (10, 25, 50, 75, 90)
PAIR_CHUNK = 200_000  # frame pairs per block of the view comparison (memory)
STATE_CHUNK = 50_000  # frames per read of the `state` column
INVENTORY = ["wood", "stone", "coal", "iron", "diamond", "sapling", "wood pickaxe", "stone pickaxe", "iron pickaxe",
             "wood sword", "stone sword", "iron sword"]  # fmt: skip
MOBS = ["zombies", "cows", "skeletons", "arrows"]  # column order of `mobs_in_view`
# (group, label, column, index in the column; None: any of its columns), the rows of the change matrix from small columns
COLUMN_ROWS = (
    [("position", n, "position", i) for i, n in enumerate(("row", "column"))]
    + [("facing", "facing", "facing", 0)]
    + [("mobs in view", n, "mobs_in_view", i) for i, n in enumerate(MOBS)]
    + [("vitals", n, "vitals", i) for i, n in enumerate(ds.VITALS)]
    + [("inventory", n, "inventory", i) for i, n in enumerate(INVENTORY)]
    + [("daylight", "daylight", "daylight", 0), ("achievements", "any new unlock", "achievements", None)]
)
VIEW_LABELS = {
    "ego_any": "view, any tile", "ego_share": "view, share of tiles",
    "terrain_any": "terrain (world), any tile", "terrain_share": "terrain (world), share of tiles",
    "mobs_any": "mobs (world), any tile", "mobs_share": "mobs (world), share of tiles",
}  # fmt: skip


# --------------------------------------------------------------------------
# compute


def quantile_row(x, quantiles=LENGTH_QUANTILES) -> dict:
    x = np.asarray(x, dtype=float)
    out = {"n": int(len(x)), "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}
    out |= {f"p{q}": float(np.percentile(x, q)) for q in quantiles}
    return out


def length_numbers(d: ds.Data) -> dict:
    """Episode lengths (episode-weighted) with their quantiles; every length is kept for the histogram."""
    lengths = np.sort(d.all_lengths)
    return {"episodes": int(len(lengths)), "frames": int(lengths.sum()), "quantiles": quantile_row(lengths),
            "lengths": lengths.tolist()}  # fmt: skip


def time_to_diamond(d: ds.Data) -> dict:
    """Steps from the reset to the first frame with the diamond, of the episodes that reach it."""
    flags = d.episodes("achievements")
    steps = [int(f[:, ds.DIAMOND_INDEX].argmax()) for f in flags if f[:, ds.DIAMOND_INDEX].any()]
    if not steps:
        return {"episodes": 0, "of": len(flags)}
    return {"episodes": len(steps), "of": len(flags), "quantiles": quantile_row(steps), "steps": sorted(steps)}


def achievement_numbers(d: ds.Data) -> dict:
    """Achievements per episode (histogram over 0 to 22) and the first-unlock ladder."""
    counts = d.all_final_achievements.astype(bool).sum(1)
    first = []
    for f in d.episodes("achievements"):
        got = f.any(0)
        first.append(np.where(got, f.argmax(0), -1))
    first = np.stack(first)
    ladder = {}
    for i, name in enumerate(ds.ACHIEVEMENTS):
        got = first[:, i] >= 0
        entry = {"share": float(got.mean()), "episodes": int(got.sum())}
        if got.any():
            entry |= {f"p{q}": float(np.percentile(first[got, i], q)) for q in (10, 50, 90)}
            entry["mean"] = float(first[got, i].mean())
        ladder[name] = entry
    return {
        "per_episode": {"histogram": np.bincount(counts, minlength=ds.DIAMOND_INDEX + 4).tolist(), "mean": float(counts.mean())},
        "ladder": ladder,
    }  # fmt: skip


def action_numbers(d: ds.Data) -> dict:
    """Steps per action (frames with an action, all but the last of an episode) over the 17 actions."""
    action = d.cols["action"][:, 0]
    taken = action[~np.isnan(action)].astype(int)
    steps = np.bincount(taken, minlength=len(ds.ACTIONS))
    return {"steps": {n: int(c) for n, c in zip(ds.ACTIONS, steps)}, "share": {n: float(c / steps.sum()) for n, c in zip(ds.ACTIONS, steps)}}


def chunk_numbers(d: ds.Data) -> dict:
    """Distinct chunks, chunk entropy and the information of the chunk about its effect, per chunk length."""
    out = {}
    for k, e in ds.chunk_diversity(d).items():
        info = e.get("information", {})
        out[str(k)] = {
            "windows": e["windows"], "distinct_chunks": e["distinct_chunks"], "chunk_entropy_bits": e["chunk_entropy_bits"],
            "distinct_effects": e.get("distinct_effects_with_motion"), "effect_entropy_bits": e.get("effect_entropy_bits"),
            "information_with_motion": info.get("with_motion"), "information_interaction": info.get("interaction"),
        }  # fmt: skip
    return out


def clip_coverage(d: ds.Data, strides=STRIDES) -> dict:
    """Clips per stride for every (context, rollout), the death padding as in training."""
    out = {"episodes": int(len(d.all_lengths)), "frames": int(d.total_all), "strides": list(strides), "settings": {}}
    for context in CONTEXTS:
        for rollout in ROLLOUTS:
            rows = ds.clip_counts(d, totals=strides, history=context, rollout=rollout)
            out["settings"][f"context{context}_rollout{rollout}"] = {
                "context": context, "rollout": rollout,
                "span": [r["span"] for r in rows], "clips": [r["clips"] for r in rows],
                "episodes_long_enough": [r["episodes_long_enough"] for r in rows],
                "independent_clips": [r["independent_clips"] for r in rows],
            }  # fmt: skip
    return out


def spectral_numbers(d: ds.Data) -> dict | None:
    """The entity frequency gap of the 7 probe entities (no view), as in D0."""
    s = ds.spectral_analysis(d, use_view=False)
    if s is None:
        return None
    keep = ("subset", "episodes", "episodes_used", "segments", "primary_window", "entities", "gap", "sensitivity", "daylight_check")
    return {k: s[k] for k in keep}


# the change matrix


def pair_starts(d: ds.Data, k: int) -> np.ndarray:
    """Frames t with t + k in the same episode."""
    if k >= d.total:
        return np.zeros(0, np.int64)
    t = np.arange(d.total - k)
    return t[d.episode[:-k] == d.episode[k:]]


def column_changes(d: ds.Data, column: str, index: int | None, t: np.ndarray, k: int) -> np.ndarray:
    """Whether the component differs between frame t and t + k, per pair."""
    x = d.cols[column].reshape(d.total, -1)
    if index is None:
        return (x[t + k] != x[t]).any(1)
    if column == "daylight":
        return np.abs(x[t + k, index].astype(np.float64) - x[t, index]) > DAYLIGHT_TOL
    return x[t + k, index] != x[t, index]


def view_tiles(path, total: int) -> tuple[np.ndarray, np.ndarray]:
    """Per frame and tile of the 7x9 view from the `state` column: the block id and the mobs as a 4-bit code."""
    tiles = VIEW_ROWS * VIEW_COLS
    block = np.empty((total, tiles), np.uint8)
    mob = np.empty((total, tiles), np.uint8)
    weights = (1 << np.arange(4)).astype(np.uint8)
    with h5py.File(path, "r") as f:
        state = f["state"]
        if state.shape[1] != ds.STATE_DIM:
            raise SystemExit(f"{path}: state has {state.shape[1]} columns, expected {ds.STATE_DIM}")
        for a in range(0, total, STATE_CHUNK):
            x = state[a : min(a + STATE_CHUNK, total), : ds.VIEW_DIM].reshape(-1, tiles, ds.VIEW_CHANNELS)
            block[a : a + len(x)] = x[:, :, : ds.BLOCK_TYPES].argmax(2)
            mob[a : a + len(x)] = ((x[:, :, ds.BLOCK_TYPES :] > 0.5).astype(np.uint8) * weights).sum(2, dtype=np.uint8)
    return block, mob


def world_changes(block, mob, position, t: np.ndarray, k: int) -> dict:
    """Terrain and mobs in world coordinates over the tiles that the views of frame t and t + k share.

    The tile (i, j) of the view at t + k is the tile (i + dr, j + dc) of the view at t, with (dr, dc) the move of the
    player. Sums over the pairs: those with a changed tile, the share of changed tiles, and the overlap size.
    """
    dr = position[t + k, 0].astype(np.int64) - position[t, 0]
    dc = position[t + k, 1].astype(np.int64) - position[t, 1]
    if (np.abs(dr) >= VIEW_ROWS).any() or (np.abs(dc) >= VIEW_COLS).any():
        raise SystemExit("the player moved further than the view in a world-frame pair: lags above WORLD_LAG_MAX?")
    key = (dr + VIEW_ROWS) * 64 + (dc + VIEW_COLS)
    order = np.argsort(key, kind="stable")
    edges = np.flatnonzero(np.diff(key[order])) + 1
    sums = {n: {"any": 0, "share": 0.0} for n in ("terrain", "mobs")} | {"tiles": 0}
    for rows in np.split(order, edges):
        r, c = int(dr[rows[0]]), int(dc[rows[0]])
        i0, i1, j0, j1 = max(0, -r), min(VIEW_ROWS, VIEW_ROWS - r), max(0, -c), min(VIEW_COLS, VIEW_COLS - c)
        later, earlier = t[rows] + k, t[rows]
        tiles = (i1 - i0) * (j1 - j0)
        for name, plane in (("terrain", block), ("mobs", mob)):
            a = plane[later].reshape(-1, VIEW_ROWS, VIEW_COLS)[:, i0:i1, j0:j1]
            b = plane[earlier].reshape(-1, VIEW_ROWS, VIEW_COLS)[:, i0 + r : i1 + r, j0 + c : j1 + c]
            diff = (a != b).reshape(len(rows), -1)
            sums[name]["any"] += int(diff.any(1).sum())
            sums[name]["share"] += float(diff.sum() / tiles)
        sums["tiles"] += tiles * len(rows)
    return sums


def view_rows(d: ds.Data, block, mob, lags=LAGS) -> list[dict]:
    """The pooled view rows per lag: egocentric, and (lags up to WORLD_LAG_MAX) world-frame terrain and mobs."""
    tile = block.astype(np.uint16) * 16 + mob
    position = d.cols["position"]
    values = {key: [] for key in VIEW_LABELS}
    overlap = []
    for k in lags:
        t = pair_starts(d, k)
        n = max(len(t), 1)
        ego_any, ego_share = 0, 0.0
        for a in range(0, len(t), PAIR_CHUNK):
            diff = tile[t[a : a + PAIR_CHUNK] + k] != tile[t[a : a + PAIR_CHUNK]]
            ego_any += int(diff.any(1).sum())
            ego_share += float(diff.mean(1).sum())
        values["ego_any"].append(ego_any / n if len(t) else None)
        values["ego_share"].append(ego_share / n if len(t) else None)
        if k <= WORLD_LAG_MAX and len(t):
            acc = {"terrain": {"any": 0, "share": 0.0}, "mobs": {"any": 0, "share": 0.0}, "tiles": 0}
            for a in range(0, len(t), PAIR_CHUNK):
                part = world_changes(block, mob, position, t[a : a + PAIR_CHUNK], k)
                for name in ("terrain", "mobs"):
                    acc[name]["any"] += part[name]["any"]
                    acc[name]["share"] += part[name]["share"]
                acc["tiles"] += part["tiles"]
            for name in ("terrain", "mobs"):
                values[f"{name}_any"].append(acc[name]["any"] / n)
                values[f"{name}_share"].append(acc[name]["share"] / n)
            overlap.append(acc["tiles"] / n)
        else:
            for name in ("terrain", "mobs"):
                values[f"{name}_any"].append(None)
                values[f"{name}_share"].append(None)
            overlap.append(None)
    return [{"group": "view", "name": VIEW_LABELS[key], "share": v} for key, v in values.items()] + [
        {"group": "view overlap", "name": "mean tiles shared by the two views", "share": overlap}
    ]


def half_lag(share, lags=LAGS):
    """First lag of the grid at which the share of changed pairs reaches HALF, or None."""
    for k, s in zip(lags, share):
        if s is not None and s >= HALF:
            return k
    return None


def change_matrix(d: ds.Data, path, lags=LAGS) -> dict:
    """Share of frame pairs at every lag in which each component differs, plus the pooled view rows."""
    pairs = {str(k): int(len(pair_starts(d, k))) for k in lags}
    rows = []
    for group, name, column, index in COLUMN_ROWS:
        share = []
        for k in lags:
            t = pair_starts(d, k)
            share.append(float(column_changes(d, column, index, t, k).mean()) if len(t) else None)
        rows.append({"group": group, "name": name, "share": share})
    if d.has("position") and d.has_state():
        block, mob = view_tiles(path, d.total)
        rows += view_rows(d, block, mob, lags)
    for r in rows:
        r["half_lag"] = None if r["group"] == "view overlap" else half_lag(r["share"], lags)
    return {"lags": list(lags), "pairs": pairs, "daylight_tolerance": DAYLIGHT_TOL, "rows": rows}


# the split and the comparison with train


def split_numbers(path, spectral: bool, chunks: bool = True) -> dict:
    d = ds.Data(path, max_frames=10**12)  # every frame: the small columns fit in memory
    if d.sampled:
        raise SystemExit(f"{path}: frames were sampled")
    out = {
        "length": length_numbers(d), "time_to_diamond": time_to_diamond(d), "achievements": achievement_numbers(d),
        "change": change_matrix(d, path), "clips": clip_coverage(d), "actions": action_numbers(d),
    }  # fmt: skip
    if chunks:
        out["chunks"] = chunk_numbers(d)
    if spectral:
        out["spectral"] = spectral_numbers(d)
    return out


def ks_statistic(x, y) -> float:
    """Two-sample Kolmogorov-Smirnov distance: the largest gap between the two empirical CDFs."""
    x, y = np.sort(np.asarray(x, float)), np.sort(np.asarray(y, float))
    grid = np.concatenate([x, y])
    return float(np.abs(np.searchsorted(x, grid, side="right") / len(x) - np.searchsorted(y, grid, side="right") / len(y)).max())


def compare_with_train(split: dict, train: dict) -> dict:
    """Distances between a split and train: the figures' content as a few numbers (val should follow train)."""
    hist = lambda s: np.repeat(np.arange(len(s["achievements"]["per_episode"]["histogram"])), s["achievements"]["per_episode"]["histogram"])  # noqa: E731
    cells = [
        abs(a - b)
        for ra, rb in zip(split["change"]["rows"], train["change"]["rows"])
        for a, b in zip(ra["share"], rb["share"])
        if a is not None and b is not None
    ]
    return {
        "length_ks": ks_statistic(split["length"]["lengths"], train["length"]["lengths"]),
        "achievements_per_episode_ks": ks_statistic(hist(split), hist(train)),
        "action_share_max_abs_diff": max(abs(split["actions"]["share"][a] - train["actions"]["share"][a]) for a in ds.ACTIONS),
        "change_share_max_abs_diff": max(cells) if cells else None,
    }  # fmt: skip


def compute(data: list[str], labels: list[str], out: Path, with_sha: bool, spectral_label: str = "test") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    record = {"git": dt.git_state(), "lags": list(LAGS), "strides": list(STRIDES), "splits": {}, "files": {}}
    for path, label in zip(data, labels):
        print(f"{label}: {path}")
        record["files"][label] = dt.file_record(path, with_sha) | {"runs": dt.read_episodes(path)["runs"]}
        record["splits"][label] = split_numbers(path, spectral=label == spectral_label)
    if "train" in record["splits"]:
        record["val_vs_train"] = {k: compare_with_train(v, record["splits"]["train"]) for k, v in record["splits"].items() if k == "val"}
    (out / "dataset_numbers.json").write_text(json.dumps(ds.to_jsonable(record), indent=1))
    print(f"wrote {out / 'dataset_numbers.json'}")
    return record


# --------------------------------------------------------------------------
# tables (LaTeX, csv)


def thousands(n) -> str:
    return f"{int(round(n)):,}".replace(",", "\\,")


def tex_table(label: str, caption: str, columns: str, header: str, rows: list[str], note: str | None = None) -> str:
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{tab:{label}}}", "\\small",
             f"\\begin{{tabular}}{{{columns}}}", "\\toprule", header + " \\\\", "\\midrule"]  # fmt: skip
    lines += [r if r.startswith("\\midrule") else r + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += ["", "\\vspace{4pt}", "\\begin{minipage}{\\linewidth}", f"\\footnotesize\\raggedright {note}", "\\end{minipage}"]
    return "\n".join(lines + ["\\end{table}"]) + "\n"


def length_table(rec: dict) -> str:
    train, test = rec["splits"]["train"], rec["splits"]["test"]
    cols = ["n", "mean", "min", "p10", "p25", "p50", "p75", "p90", "max"]
    row = lambda name, q: " & ".join([name, thousands(q["n"])] + [thousands(q[c]) for c in cols[1:]])  # noqa: E731
    rows = [row("Train, episode length", train["length"]["quantiles"]), row("Test, episode length", test["length"]["quantiles"])]
    ttd = test["time_to_diamond"]
    if ttd["episodes"]:
        rows += ["\\midrule", row("Test, time to diamond", ttd["quantiles"])]
    return tex_table(
        "data-length", "Episode length and time to the diamond, in frames.", "lrrrrrrrrr",
        "& $n$ & Mean & Min & p10 & p25 & Median & p75 & p90 & Max", rows,
        "Time to diamond: steps from the reset to the first frame with the diamond, over the episodes that reach it. Episodes that reach the "
        "diamond end 500 steps after it (or earlier on death).",
    )  # fmt: skip


def spectral_table(rec: dict) -> str | None:
    s = rec["splits"]["test"].get("spectral")
    if not s:
        return None
    w = s["primary_window"]
    rows = []
    for name, e in sorted(s["entities"].items(), key=lambda kv: kv[1]["period"]):
        rows.append(f"{name.replace('_', ' ')} & {e['probe_group']} & {e['period']:.1f} & {'yes' if e['limited'] else 'no'}")
    gap = s["gap"]["all"]
    ci = gap.get("ci") or {}
    note = (f"Centroid period in frames (1 / spectral centroid, Welch window {w}, hop {w // 2}, Hann; each dimension standardised per episode), {s['episodes_used']} test "
            f"episodes. Limited: more than half of the power in the two lowest bins, so the period is a lower bound. Frequency gap (largest over smallest "
            f"period): {gap['ratio']:.3g}" + (" or more" if gap["lower_bound"] else "") +
            (f", bootstrap p5 to p95 {ci['p5']:.3g} to {ci['p95']:.3g}" if ci else "") +
            "; at windows " + ", ".join(f"{k}: {v['gap']['all']['ratio']:.3g}" for k, v in sorted(s["sensitivity"].items(), key=lambda kv: int(kv[0]))) + ".")  # fmt: skip
    return tex_table("data-spectral", "Spectral centroid period of the probe entities of the test split.", "llrl",
                     "Entity & Group & Period & Limited", rows, note)  # fmt: skip


def ladder_order(rec: dict) -> list[str]:
    ladder = rec["splits"]["train"]["achievements"]["ladder"]
    return sorted(ds.ACHIEVEMENTS, key=lambda n: (ladder[n].get("p50") is None, ladder[n].get("p50", 0.0)))


def achievement_table(rec: dict) -> str:
    train, test = (rec["splits"][s]["achievements"]["ladder"] for s in ("train", "test"))

    def cells(e):
        if "p50" not in e:
            return ["---", "---"]
        return [f"{100 * e['share']:.1f}\\%", f"{e['p50']:.0f} [{e['p10']:.0f}, {e['p90']:.0f}]"]

    rows = [" & ".join([n.replace("_", " ")] + cells(train[n]) + cells(test[n])) for n in ladder_order(rec)]
    return tex_table(
        "data-achievements", "Achievements: share of episodes that unlock each one and the frame of its first unlock.", "lrrrr",
        "& \\multicolumn{2}{c}{Train} & \\multicolumn{2}{c}{Test} \\\\\n Achievement & Share & First unlock & Share & First unlock", rows,
        "First unlock: median frame, with p10 and p90 in brackets, over the episodes that unlock it. Sorted by the train median.",
    )  # fmt: skip


def chunk_table(rec: dict) -> str:
    train, test = (rec["splits"][s].get("chunks", {}) for s in ("train", "test"))
    rows = []
    for k in sorted(train, key=int):
        cells = [k]
        for c in (train, test):
            e = c.get(k)
            info = e and e.get("information_with_motion")
            if not e:
                cells += ["---"] * 4
                continue
            cells += [thousands(e["distinct_chunks"]), f"{e['chunk_entropy_bits']:.1f}"]
            cells += [f"{info['mi_bits']:.2f}", f"{100 * info['coverage']:.0f}\\%"] if info else ["---", "---"]
        rows.append(" & ".join(cells))
    return tex_table(
        "data-chunks", "Action chunks of $k$ steps: distinct chunks, chunk entropy and information about the effect.", "lrrrrrrrr",
        "& \\multicolumn{4}{c}{Train} & \\multicolumn{4}{c}{Test} \\\\\n $k$ & Distinct & $H$ & $I$ & Cov. & Distinct & $H$ & $I$ & Cov.", rows,
        "Windows of $k$ steps without overlap (at most 500\\,000 per $k$). Distinct: chunks seen. $H$: entropy of the chunk (bits). $I$: mutual information of the chunk "
        "with the effect (bits; change of inventory, achievements, vitals and position over the window; Miller--Madow corrected). Cov.: share of windows whose chunk "
        "was seen at least 20 times, the only windows $I$ is computed on; where it is low, most chunks occur once and $I$ is not reliable. "
        "Reference scales for a codebook, not settings.",
    )  # fmt: skip


def change_csv(rec: dict, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "group", "component", "half_lag"] + [f"lag_{k}" for k in rec["lags"]])
        for label in ("train", "val", "test"):
            if label not in rec["splits"]:
                continue
            for r in rec["splits"][label]["change"]["rows"]:
                w.writerow([label, r["group"], r["name"], r["half_lag"]] + ["" if s is None else f"{s:.6f}" for s in r["share"]])
    return path


# --------------------------------------------------------------------------
# figures


COLORS = {"train": "#2a78d6", "test": "#eb6834"}  # categorical slots 1 and 2 of the reference palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(plt):
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 9, "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42, "figure.dpi": 150, "savefig.bbox": "tight",
    })  # fmt: skip


def _hist(ax, values, width, color, label, fill):
    """Share of episodes per bin as steps."""
    edges = np.arange(0, max(values) + width, width)
    h, _ = np.histogram(values, bins=edges)
    h = h / h.sum()
    if fill:
        ax.bar(edges[:-1], h, width=width, align="edge", color=color, alpha=0.35, linewidth=0, label=label)
    else:
        ax.stairs(h, edges, color=color, linewidth=1.4, label=label)


def fig_length(rec, out: Path, plt):
    train, test = rec["splits"]["train"], rec["splits"]["test"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    _hist(a, train["length"]["lengths"], 50, COLORS["train"], f"train ({train['length']['episodes']:,} episodes)", True)
    _hist(a, test["length"]["lengths"], 50, COLORS["test"], f"test ({test['length']['episodes']:,} episodes)", False)
    a.set(xlabel="episode length (frames)", ylabel="share of episodes", title="Episode length")
    a.legend(loc="upper right")
    ttd = test["time_to_diamond"]
    if ttd["episodes"]:
        _hist(b, ttd["steps"], 25, COLORS["test"], None, True)
        q = ttd["quantiles"]
        for key in ("p25", "p50", "p75"):
            b.axvline(q[key], color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
        b.text(0.97, 0.95, "dashed: p25, p50, p75\n" + ", ".join(f"{q[k]:.0f}" for k in ("p25", "p50", "p75")), transform=b.transAxes, va="top", ha="right", fontsize=7, color=MUTED)
    b.set(xlabel="steps to the diamond", ylabel="share of episodes", title=f"Time to diamond (test, {ttd['episodes']} of {ttd['of']} episodes)")
    fig.tight_layout()
    fig.savefig(out / "fig_length.pdf")
    plt.close(fig)


def fig_achievements(rec, out: Path, plt):
    order = ladder_order(rec)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    for s, off in (("train", -0.12), ("test", 0.12)):
        hist = np.array(rec["splits"][s]["achievements"]["per_episode"]["histogram"], float)
        a.step(np.arange(len(hist) + 1) - 0.5, np.append(hist / hist.sum(), 0), where="post", color=COLORS[s], label=s, linewidth=1.4)
        ladder = rec["splits"][s]["achievements"]["ladder"]
        for i, n in enumerate(order):
            e = ladder[n]
            if "p50" not in e:
                continue
            y = i + off
            b.plot([e["p10"], e["p90"]], [y, y], color=COLORS[s], linewidth=1.2, solid_capstyle="round")
            b.plot(e["p50"], y, "o", color=COLORS[s], markersize=3.5, markeredgecolor="white", markeredgewidth=0.6, label=s if i == 0 else None)
    a.set(xlabel="achievements unlocked in an episode", ylabel="share of episodes", title="Achievements per episode", xlim=(-0.5, 22.5))
    a.legend(loc="upper left")
    b.set_yticks(range(len(order)), [n.replace("_", " ") for n in order])
    b.invert_yaxis()
    b.set(xlabel="frame of the first unlock (median, p10 to p90)", title="Achievement ladder")
    b.grid(axis="y", visible=False)
    b.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_achievements.pdf")
    plt.close(fig)


def fig_change(rec, out: Path, plt):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#a9c8f0", "#2a78d6", "#12386b"])
    labels = [(r["group"], r["name"]) for r in rec["splits"]["train"]["change"]["rows"] if r["group"] != "view overlap"]
    lags = rec["lags"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 9.6), sharey=True)
    for ax, s in zip(axes, ("train", "test")):
        rows = [r for r in rec["splits"][s]["change"]["rows"] if r["group"] != "view overlap"]
        m = np.array([[np.nan if v is None else v * 100 for v in r["share"]] for r in rows])
        ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                text = "-" if np.isnan(v) else "0" if v == 0 else "<1" if v < 0.5 else f"{v:.0f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color="white" if (not np.isnan(v) and v > 55) else INK)
        ax.set_xticks(range(len(lags)), [str(k) for k in lags])
        ax.set(xlabel="lag k (steps)", title=s)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # group separators
        start = 0
        for g in dict.fromkeys(g for g, _ in labels):
            n = sum(1 for x, _ in labels if x == g)
            if start:
                ax.axhline(start - 0.5, color="white", linewidth=2.5)
            start += n
    fig.suptitle("% of frame pairs (t, t + k) of one episode in which the component changes", fontsize=9)
    axes[0].set_yticks(range(len(labels)), [f"{g}: {n}" if g in ("position", "mobs in view", "vitals", "inventory") else n for g, n in labels])
    fig.tight_layout()
    fig.savefig(out / "fig_change.pdf")
    plt.close(fig)


def fig_clips(rec, out: Path, plt):
    clips = rec["splits"]["train"]["clips"]
    strides = np.array(clips["strides"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    # a clip's frames are context + rollout: (1, 5) and (5, 1) are the same clips, so the grid is three curves
    groups = {}
    for s in clips["settings"].values():
        groups.setdefault(s["context"] + s["rollout"], []).append(s)
    for (frames, members), c in zip(sorted(groups.items()), (COLORS["train"], COLORS["test"], "#1baf7a")):
        label = f"{frames} frames: " + " or ".join(f"context {m['context']} + rollout {m['rollout']}" for m in members)
        clip = np.array(members[0]["clips"], float)
        a.plot(strides, np.where(clip > 0, clip, np.nan), color=c, linewidth=1.6, label=label)
        b.plot(strides, np.array(members[0]["episodes_long_enough"]) / clips["episodes"], color=c, linewidth=1.6, label=label)
    a.set(xscale="log", yscale="log", xlabel="stride (frames between clip frames)", ylabel="clips", title="Clips per stride (train, all overlapping clips)")
    b.set(xscale="log", xlabel="stride (frames between clip frames)", ylabel="share of episodes", title="Episodes long enough for a clip", ylim=(0, 1.02))
    a.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig_clips.pdf")
    plt.close(fig)


def fig_actions(rec, out: Path, plt):
    actions = rec["splits"]["train"]["actions"]
    names = list(actions["steps"])
    steps = np.array([actions["steps"][n] for n in names], float)
    share = np.array([actions["share"][n] for n in names])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    y = np.arange(len(names))
    ax.barh(y, np.where(steps > 0, steps, np.nan), color=COLORS["train"], height=0.6)
    for i, (n, c, p) in enumerate(zip(names, steps, share)):
        ax.text(max(c, 1) * 1.15, i, f"{c:,.0f} ({100 * p:.2g}%)" if c else "never", va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks(y, [n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set(xscale="log", xlabel="steps in train", title="Action coverage")
    ax.set_xlim(max(1, steps[steps > 0].min() / 3), steps.max() * 12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_actions.pdf")
    plt.close(fig)


def plot(out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    rec = json.loads((out / "dataset_numbers.json").read_text())
    written = []
    for name, fn in (("fig_length", fig_length), ("fig_achievements", fig_achievements), ("fig_change", fig_change), ("fig_clips", fig_clips), ("fig_actions", fig_actions)):
        fn(rec, out, plt)
        written.append(out / f"{name}.pdf")
    for name, text in (("table_length", length_table(rec)), ("table_spectral", spectral_table(rec)), ("table_achievements", achievement_table(rec)), ("table_chunks", chunk_table(rec))):
        if text:
            (out / f"{name}.tex").write_text(text)
            written.append(out / f"{name}.tex")
    written.append(change_csv(rec, out / "change_matrix.csv"))
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compute", help="episode files -> dataset_numbers.json")
    c.add_argument("--data", action="append", required=True, help="episode file (repeatable, one --label each)")
    c.add_argument("--label", action="append", required=True, help="split name: train, val or test")
    c.add_argument("--out", default="reports/dataset")
    c.add_argument("--sha256", action="store_true", help="hash the files (slow for the 8 GB train file)")
    p = sub.add_parser("plot", help="dataset_numbers.json -> figures and tables")
    p.add_argument("--out", default="reports/dataset")
    args = ap.parse_args()
    if args.command == "compute":
        if len(args.data) != len(args.label):
            raise SystemExit("--data and --label must be given the same number of times")
        if not {"train", "test"} <= set(args.label):
            raise SystemExit("the labels must include train and test")
        compute(args.data, args.label, Path(args.out), args.sha256)
    else:
        for path in plot(Path(args.out)):
            print("wrote", path)


if __name__ == "__main__":
    main()
