"""Matplotlib figures of the dataset section, from `dataset_numbers.json` (a dict, no data files)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .compute import JSON_NAME
from .tables import achievement_table, change_csv, chunk_table, ladder_order, length_table, spectral_table

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
    """Figures and tables from `out / dataset_numbers.json` into `out`; returns the files written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    rec = json.loads((out / JSON_NAME).read_text())
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


