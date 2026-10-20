"""Matplotlib figures of the dataset section, from `dataset_numbers.json` (a dict, no data files).

Vector PDFs with embedded fonts (they stay sharp at any size in a LaTeX document, include them with
`\\includegraphics[width=\\linewidth]{...}`). Designed at `FIG_WIDTH` inches, about a text width, so that the 8 pt text
stays about the size of the body text. The two splits never share an axes: one figure each, on the same axis ranges.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .compute import JSON_NAME
from .tables import achievement_table, change_csv, chunk_table, ladder_order, length_table, spectral_table

FIG_WIDTH = 6.0  # inches
COLORS = {"train": "#2a78d6", "test": "#eb6834"}  # categorical slots 1 and 2 of the reference palette
THIRD = "#1baf7a"  # slot 3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LENGTH_BIN, DIAMOND_BIN = 50, 25  # histogram bin widths (frames)


def _style(plt):
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 9, "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
    })  # fmt: skip


def _si(x, _=None) -> str:
    """1000 -> 1K, 2500000 -> 2.5M: tick labels that need no exponent to read."""
    return f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}K" if x >= 1e3 else f"{x:g}"


def _share_hist(values, width, xmax):
    """Share of the values per bin of `width` over 0 to `xmax`."""
    edges = np.arange(0, xmax + width, width)
    counts, _ = np.histogram(values, bins=edges)
    return edges, counts / len(values)


def _ceil(x, step) -> float:
    return math.ceil(x / step) * step


def _length_limits(rec):
    """Axis ranges that train and test share: the longest episode and the tallest bin of either."""
    xmax = _ceil(max(max(rec["splits"][s]["length"]["lengths"]) for s in ("train", "test")), 500)
    ymax = max(_share_hist(rec["splits"][s]["length"]["lengths"], LENGTH_BIN, xmax)[1].max() for s in ("train", "test"))
    return xmax, ymax * 1.08


def fig_length(rec, split, plt):
    xmax, ymax = _length_limits(rec)
    info = rec["splits"][split]["length"]
    edges, share = _share_hist(info["lengths"], LENGTH_BIN, xmax)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.5))
    ax.stairs(share, edges, fill=True, color=COLORS[split], linewidth=0)
    ax.set(xlim=(0, xmax), ylim=(0, ymax), xlabel="episode length (frames)", ylabel="share of episodes",
           title=f"Episode length ({split}, {info['episodes']:,} episodes)")  # fmt: skip
    return fig


def fig_time_to_diamond(rec, plt):
    ttd = rec["splits"]["test"]["time_to_diamond"]
    xmax = _ceil(max(ttd["steps"]), 500)
    edges, share = _share_hist(ttd["steps"], DIAMOND_BIN, xmax)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.5))
    ax.stairs(share, edges, fill=True, color=COLORS["test"], linewidth=0)
    q = ttd["quantiles"]
    for key in ("p25", "p50", "p75"):
        ax.axvline(q[key], color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(0.97, 0.95, "dashed: p25, p50, p75\n" + ", ".join(f"{q[k]:.0f}" for k in ("p25", "p50", "p75")),
            transform=ax.transAxes, va="top", ha="right", fontsize=7, color=MUTED)  # fmt: skip
    ax.set(xlim=(0, xmax), xlabel="steps to the diamond", ylabel="share of episodes",
           title=f"Time to diamond (test, {ttd['episodes']} of {ttd['of']} episodes)")  # fmt: skip
    return fig


def fig_achievements(rec, split, plt):
    histograms = {s: np.array(rec["splits"][s]["achievements"]["per_episode"]["histogram"], float) for s in ("train", "test")}
    ymax = max((h / h.sum()).max() for h in histograms.values()) * 1.08
    h = histograms[split]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.5))
    ax.bar(np.arange(len(h)), h / h.sum(), width=0.8, color=COLORS[split], linewidth=0)
    n = int(h.sum())
    ax.set(xlim=(-0.5, len(h) - 0.5), ylim=(0, ymax), xlabel="achievements unlocked in an episode", ylabel="share of episodes",
           title=f"Achievements per episode ({split}, {n:,} episodes)")  # fmt: skip
    return fig


def fig_ladder(rec, plt):
    order = ladder_order(rec)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.6))
    for s, off in (("train", -0.12), ("test", 0.12)):
        ladder = rec["splits"][s]["achievements"]["ladder"]
        for i, n in enumerate(order):
            e = ladder[n]
            if "p50" not in e:
                continue
            y = i + off
            ax.plot([e["p10"], e["p90"]], [y, y], color=COLORS[s], linewidth=1.2, solid_capstyle="round")
            ax.plot(e["p50"], y, "o", color=COLORS[s], markersize=3.5, markeredgecolor="white", markeredgewidth=0.6, label=s if i == 0 else None)
    ax.set_yticks(range(len(order)), [n.replace("_", " ") for n in order])
    ax.invert_yaxis()
    ax.set(xlabel="frame of the first unlock (median, p10 to p90)", title="Achievement ladder")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    return fig


def fig_change(rec, plt):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#a9c8f0", "#2a78d6", "#12386b"])
    labels = [(r["group"], r["name"]) for r in rec["splits"]["train"]["change"]["rows"] if r["group"] != "view overlap"]
    lags = rec["lags"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH + 0.6, 9.2), sharey=True)
    for ax, s in zip(axes, ("train", "test")):
        rows = [r for r in rec["splits"][s]["change"]["rows"] if r["group"] != "view overlap"]
        m = np.array([[np.nan if v is None else v * 100 for v in r["share"]] for r in rows])
        # rectangles, not an image: stays sharp at any size
        ax.pcolormesh(np.arange(m.shape[1] + 1) - 0.5, np.arange(m.shape[0] + 1) - 0.5, np.ma.masked_invalid(m), cmap=cmap, vmin=0, vmax=100,
                      edgecolors="face", linewidth=0.3)  # fmt: skip
        ax.set_ylim(m.shape[0] - 0.5, -0.5)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                text = "-" if np.isnan(v) else "0" if v == 0 else "<1" if v < 0.5 else f"{v:.0f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=5.5, color="white" if (not np.isnan(v) and v > 55) else INK)
        n_episodes = rec["splits"][s]["length"]["episodes"]
        reach = rec["splits"][s]["change"].get("episodes", {})
        ax.set_xticks(range(len(lags)), [f"{k}\n{100 * reach[str(k)] / n_episodes:.0f}" if str(k) in reach else str(k) for k in lags], fontsize=5.5)
        ax.set(xlabel="lag k (steps); below: % of episodes\nlong enough to have a pair at k", title=s)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        start = 0  # group separators
        for g in dict.fromkeys(g for g, _ in labels):
            n = sum(1 for x, _ in labels if x == g)
            if start:
                ax.axhline(start - 0.5, color="white", linewidth=2.5)
            start += n
    fig.suptitle("% of frame pairs (t, t + k) of one episode in which the component changes", fontsize=9)
    axes[0].set_yticks(range(len(labels)), [f"{g}: {n}" if g in ("position", "mobs in view", "vitals", "inventory") else n for g, n in labels])
    fig.tight_layout()
    return fig


def fig_clips(rec, plt):
    from matplotlib.ticker import FuncFormatter, NullFormatter

    clips = rec["splits"]["train"]["clips"]
    strides = np.array(clips["strides"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.9))
    # a clip's frames are context + rollout: (1, 5) and (5, 1) are the same clips, so the grid is three curves
    groups = {}
    for s in clips["settings"].values():
        groups.setdefault(s["context"] + s["rollout"], []).append(s)
    for (frames, members), c in zip(sorted(groups.items()), (COLORS["train"], COLORS["test"], THIRD)):
        label = f"{frames} frames: " + " or ".join(f"context {m['context']} + rollout {m['rollout']}" for m in members)
        clip = np.array(members[0]["clips"], float)
        a.plot(strides, np.where(clip > 0, clip, np.nan), color=c, linewidth=1.6, label=label)
        b.plot(strides, np.array(members[0]["episodes_long_enough"]) / clips["episodes"], color=c, linewidth=1.6)
    a.set(xscale="log", yscale="log", xlabel="stride (frames between clip frames)", ylabel="clips", title="Clips per stride (train)")
    b.set(xscale="log", xlabel="stride (frames between clip frames)", ylabel="share of episodes", title="Episodes long enough for a clip", ylim=(0, 1.02))
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 400]
    for ax in (a, b):
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    a.set_yticks([1e3, 1e4, 1e5, 1e6])
    a.yaxis.set_major_formatter(FuncFormatter(_si))
    a.yaxis.set_minor_formatter(NullFormatter())
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=7)
    fig.tight_layout()
    return fig


def fig_actions(rec, plt):
    from matplotlib.ticker import FuncFormatter, NullFormatter

    actions = rec["splits"]["train"]["actions"]
    names = list(actions["steps"])
    steps = np.array([actions["steps"][n] for n in names], float)
    share = np.array([actions["share"][n] for n in names])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    y = np.arange(len(names))
    ax.barh(y, np.where(steps > 0, steps, np.nan), color=COLORS["train"], height=0.6)
    for i, (c, p) in enumerate(zip(steps, share)):
        ax.text(max(c, 1) * 1.15, i, f"{c:,.0f} ({100 * p:.2g}%)" if c else "never", va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks(y, [n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set(xscale="log", xlabel="steps in train", title="Action coverage")
    ax.set_xlim(max(1, steps[steps > 0].min() / 3), steps.max() * 12)
    ax.set_xticks([1e4, 1e5, 1e6])
    ax.xaxis.set_major_formatter(FuncFormatter(_si))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


FIGURES = {  # file name -> function of (rec, plt)
    "fig_length_train": lambda rec, plt: fig_length(rec, "train", plt),
    "fig_length_test": lambda rec, plt: fig_length(rec, "test", plt),
    "fig_time_to_diamond": fig_time_to_diamond,
    "fig_achievements_train": lambda rec, plt: fig_achievements(rec, "train", plt),
    "fig_achievements_test": lambda rec, plt: fig_achievements(rec, "test", plt),
    "fig_achievement_ladder": fig_ladder,
    "fig_change": fig_change,
    "fig_clips": fig_clips,
    "fig_actions": fig_actions,
}


def plot(out: Path) -> list[Path]:
    """Figures and tables from `out / dataset_numbers.json` into `out`; returns the files written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    rec = json.loads((out / JSON_NAME).read_text())
    written = []
    for name, fn in FIGURES.items():
        fig = fn(rec, plt)
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)
        written.append(out / f"{name}.pdf")
    for name, text in (("table_length", length_table(rec)), ("table_spectral", spectral_table(rec)), ("table_achievements", achievement_table(rec)), ("table_chunks", chunk_table(rec))):
        if text:
            (out / f"{name}.tex").write_text(text)
            written.append(out / f"{name}.tex")
    written.append(change_csv(rec, out / "change_matrix.csv"))
    return written
