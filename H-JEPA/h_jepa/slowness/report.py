"""The report: every analysis of the slowness package from the stored features, as numbers (json), tables and figures.

    python -m slowness.report --run <run dir> --features <dir of slowness.encode> --out <dir> [--no-mlp]

Needs `sample.npz` and `features_<name>.npz` (names `untrained` and `epoch<N>`) from `slowness.encode` and the run's
checkpoints (for the predictor, `epoch<N>` names only). Writes `slowness_numbers.json`, `slowness_report.md` and four
figures to <out>. The reading at the end of the report is rule-based: its thresholds (`THRESHOLDS`) are heuristics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import batches as ba
from . import factors as fa
from . import variance as va

THRESHOLDS = {
    "episode_excess": 0.10,  # episode share above the null: identity carries variance
    "slow_share": 0.5,  # episode + window share of the latent
    "lag": 0.8,  # variance-weighted lag-8 correlation
    "rank_frac": 0.4,  # effective rank over the dimension
    "within_rank": 0.9,  # rank after centering per episode over the rank before
    "window_gap": 1.5,  # (window / iid SIGReg of the latent) over the same ratio of the full state
    "lost": 0.5,  # a factor is lost when its R^2 is under this fraction of the untrained latent's
    "readable": 0.3,  # ... and the untrained latent read it with at least this R^2
    "expensive": 0.15,  # cost = Var(change | action) / Var(factor)
    "rare": 0.05,  # change rate: the share of steps in which the factor changes
    "pressure": 0.02,  # the predictor's error over the variance (all axes)
    "free": 0.9,  # variance share of the free axes
}
BATCH_EPISODES = (4, 8, 16)
NAMES_ORDER = ("untrained", "epoch1", "epoch2", "epoch5", "epoch10")


def load(features: Path):
    sample = dict(np.load(features / "sample.npz"))
    feats = {}
    for path in sorted(features.glob("features_*.npz")):
        name = path.stem.removeprefix("features_")
        feats[name] = dict(np.load(path))
    order = lambda n: (n != "untrained", int("".join(c for c in n if c.isdigit()) or 0))  # noqa: E731
    return sample, {n: feats[n] for n in sorted(feats, key=order)}


def timescale(x: np.ndarray, lags=(8,)) -> dict:
    parts = va.decompose(x)
    axes = va.per_axis(x, lags=lags)
    weight = axes["variance"] / axes["variance"].sum()
    return {
        "shares": va.shares(parts),
        "null_episode": va.null_shares(x, n=5)["episode"][0],
        "erank": va.effective_rank(axes["variance"]),
        "dim": int(x.shape[-1]),
        "within_episode_rank": va.within_rank(x, "episode"),
        "within_window_rank": va.within_rank(x, "window"),
        "lag": {int(k): float((v * weight).sum()) for k, v in axes["lag_correlation"].items()},
    }


class Steps:
    """A progress bar over the analysis steps with the current step named; `tick` advances it."""

    def __init__(self, total: int):
        from tqdm import tqdm

        self.bar = tqdm(total=total, unit="step", dynamic_ncols=True)

    def tick(self, message: str | None = None, n: int = 1):
        if message:
            self.bar.set_description(message)
        self.bar.update(n)

    def close(self):
        self.bar.close()


def whitened_reference(sample: dict, dims: int) -> np.ndarray:
    """The ground-truth groups, standardised and whitened to `dims` axes: what a latent that held the whole state would
    look like to SIGReg."""
    groups = fa.build_groups(sample)
    parts = []
    for g in fa.GROUPS:
        a = groups[g]
        flat = a.reshape(-1, a.shape[-1])
        mean, std = flat.mean(0), flat.std(0)
        parts.append(np.where(std > 1e-6, (a - mean) / np.maximum(std, 1e-6), 0.0))
    z = np.concatenate(parts, -1)
    mean, axes, values = va.pca(z)
    return (((z - mean) @ axes[:, :dims]) / np.sqrt(values[:dims] + 1e-9)).astype(np.float32)


def compute(sample, feats, *, run: Path | None, mlp: bool, n_batches: int, layout: bool = False, log=print) -> dict:
    t0 = time.time()
    n, groups_n = len(feats), len(fa.GROUPS) + int(layout)
    steps = Steps(
        2 * n + (n + 1) + 2 * n * (2 if mlp else 1) + (2 * n * (1 + 2 * groups_n) if mlp else 0)
        + (n - ("untrained" in feats) if run is not None else 0)
    )
    out = {"names": list(feats), "windows": [int(v) for v in sample["rows"].shape[:3]], "scales": {}, "batches": {}}
    for name, f in feats.items():
        for feature in ("latent", "cls"):
            steps.tick(f"time scales {feature}/{name}")
            out["scales"][f"{feature}/{name}"] = timescale(f[feature])
    log(f"time scales {time.time() - t0:.0f}s")
    dims = feats[next(iter(feats))]["latent"].shape[-1]
    reference = whitened_reference(sample, dims)
    steps.tick("SIGReg of batches: whole state")
    out["batches"]["state"] = ba.compare(reference, BATCH_EPISODES, n=n_batches)
    for name, f in feats.items():
        steps.tick(f"SIGReg of batches: {name}")
        out["batches"][name] = ba.compare(f["latent"], BATCH_EPISODES, n=n_batches)
    log(f"batches {time.time() - t0:.0f}s")
    split = fa.episode_split(len(sample["episode"]))
    groups = fa.build_groups(sample, layout=layout)
    targets = fa.scalar_targets(sample)
    out["costs"] = fa.costs(targets, sample["action"])
    out["decodability"], out["attribution"] = {}, {}
    for name, f in feats.items():
        for feature in ("latent", "cls"):
            key = f"{feature}/{name}"
            steps.tick(f"factors, linear read: {key}")
            out["decodability"][key] = {"ridge": fa.decodability(f[feature], targets, split)}
            if mlp:
                steps.tick(f"factors, MLP read: {key}")
                out["decodability"][key]["mlp"] = fa.decodability(f[feature], targets, split, "mlp")
                steps.bar.set_description(f"groups -> latent, MLP: {key}")
                out["attribution"][key] = fa.attribution(f[feature], groups, split, "mlp", progress=steps.tick)
    log(f"factors {time.time() - t0:.0f}s")
    if run is not None:
        out["predictor"] = predictor(sample, feats, run, steps)
        log(f"predictor {time.time() - t0:.0f}s")
    steps.close()
    return out


def predictor(sample, feats, run: Path, steps: Steps | None = None) -> dict:
    from omegaconf import OmegaConf

    from . import encode as en
    from . import predict as pe

    cfg = OmegaConf.load(run / "config.yaml")
    num_actions, history = int(cfg.level1.wm.num_actions), int(cfg.level1.wm.history_size)
    result = {}
    for name, f in feats.items():
        if name == "untrained":  # a random predictor says nothing
            continue
        if steps:
            steps.tick(f"predictor: {name}")
        model = en.load_model(next(run.glob(f"*_epoch_{name.removeprefix('epoch')}_object.ckpt")))
        pred, target, last = pe.one_step(
            model.get_level(1), f["latent"], sample["action"], num_actions, history=history
        )
        axes = pe.per_axis_error(f["latent"], pred, target, last)
        result[name] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in axes.items()}
    return result


def reading(numbers: dict, t: dict = THRESHOLDS) -> list[str]:
    """The rule-based reading: one line per finding, each with the numbers it rests on. The last checkpoint is the one
    read; the untrained network is the control."""
    names = [n for n in numbers["names"] if n != "untrained"]
    last = names[-1]
    lines = []
    s = numbers["scales"]
    lat, cls, ctl = s[f"latent/{last}"], s[f"cls/{last}"], s["latent/untrained"]
    excess = lat["shares"]["episode"] - lat["null_episode"]
    ctl_excess = ctl["shares"]["episode"] - ctl["null_episode"]
    slow = lat["shares"]["episode"] + lat["shares"]["window"]
    lag = next(iter(lat["lag"].values()))
    if excess >= t["episode_excess"]:
        lines.append(
            f"Episode identity carries variance: its share is {excess:.2f} above the null (untrained {ctl_excess:.2f})."
        )
    else:
        lines.append(
            f"Episode identity is not what the variance is made of: its share is {excess:.2f} above the null "
            f"(CLS {cls['shares']['episode'] - cls['null_episode']:.2f}, untrained {ctl_excess:.2f}; threshold "
            f"{t['episode_excess']})."
        )
    rank_ok = lat["erank"] >= t["rank_frac"] * lat["dim"]
    within_ok = lat["within_episode_rank"] >= t["within_rank"] * lat["erank"]
    first = s[f"latent/{names[0]}"]
    falling = lat["within_episode_rank"] < first["within_episode_rank"]
    if slow >= t["slow_share"] and lag >= t["lag"]:
        lines.append(
            f"The variance is slow: {slow:.2f} of it lies between windows or episodes and the lag-8 correlation is "
            f"{lag:.2f}; the within-episode rank {'falls' if falling else 'does not fall'} over training "
            f"({first['within_episode_rank']:.0f} to {lat['within_episode_rank']:.0f})."
        )
    else:
        lines.append(f"The variance is not dominated by the slow scales (slow share {slow:.2f}, lag-8 {lag:.2f}).")
    lines.append(
        f"Rank: effective rank {lat['erank']:.0f} of {lat['dim']} ({'healthy' if rank_ok else 'low'}); "
        f"{lat['within_episode_rank']:.0f} after centering per episode "
        f"({'the dimensions are not episode-specific' if within_ok else 'some dimensions only tell episodes apart'})."
    )
    b = {r["kind"] + str(r["episodes"]): r["raw"][0] for r in numbers["batches"][last]}
    g = {r["kind"] + str(r["episodes"]): r["raw"][0] for r in numbers["batches"]["state"]}
    gap = (b["window8"] / b["iid8"]) / (g["window8"] / g["iid8"])
    lines.append(
        f"SIGReg of windows of 8 episodes is {b['window8'] / b['iid8']:.1f}x the iid value; the whole state would give "
        f"{g['window8'] / g['iid8']:.1f}x. The latent is {'slower than' if gap >= t['window_gap'] else 'about as slow as'}"
        f" the whole state (ratio of ratios {gap:.2f}, threshold {t['window_gap']})."
    )
    dec, cost = numbers["decodability"], numbers["costs"]
    lost = []
    for name, c in cost.items():
        a = dec[f"latent/untrained"]["ridge"].get(name)
        z = dec[f"latent/{last}"]["ridge"].get(name)
        if a is None or z is None or np.isnan(a) or np.isnan(z):
            continue
        if a >= t["readable"] and z < t["lost"] * a:
            lost.append((name, c["cost"], c["change_rate"]))
    if lost:
        expensive = [n for n, c, r in lost if c >= t["expensive"]]
        rare = [n for n, c, r in lost if r < t["rare"] and c < t["expensive"]]
        other = [n for n, c, r in lost if n not in expensive and n not in rare]
        lines.append(
            f"{len(lost)} factors are lost against the untrained latent (linear R^2 below {t['lost']} of it): "
            f"expensive to predict (cost >= {t['expensive']}): {', '.join(expensive) or 'none'}; rare (changes in under "
            f"{t['rare']:.0%} of the steps): {', '.join(rare) or 'none'}; neither: {', '.join(other) or 'none'}."
        )
    else:
        lines.append("No factor is lost against the untrained latent.")
    if "predictor" in numbers and last in numbers["predictor"]:
        p = numbers["predictor"][last]
        pressure = p["error_share_of_variance"]
        lines.append(
            f"The predictor's one-step error is {pressure:.3f} of the variance and {p['free_share']:.2f} of the variance "
            f"lies on free axes: "
            + (
                "the prediction loss exerts little pressure on this latent (the selection happened earlier)."
                if pressure < t["pressure"] and p["free_share"] >= t["free"]
                else "the prediction loss still pays for part of the variance."
            )
        )
    att = numbers["attribution"].get(f"latent/{last}")
    if att:
        top = max(att["only"], key=att["only"].get)
        lines.append(
            f"From the game state, the group `{top}` alone explains {att['only'][top]:.2f} of the latent's variance "
            f"(all groups: {att['all']:.2f}; held-out episodes, MLP)."
        )
    return lines


# --------------------------------------------------------------------------
# figures (light surface; one hue for the ordered epochs, grey for the untrained network)

BLUES = ["#a9c8f0", "#6fa3e3", "#3d82d9", "#1c5aa6", "#0e3a73"]
GREY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
SCALE_COLORS = {"episode": "#0e3a73", "window": "#2a78d6", "clip_means": "#8fb8ec", "clip": "#d7e6f8"}
SCALE_LABELS = {
    "episode": "between episodes",
    "window": "between windows",
    "clip_means": "within a window",
    "clip": "within a clip",
}


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def _checkpoint_colors(names):
    trained = [n for n in names if n != "untrained"]
    ramp = np.linspace(0.1, 1.0, len(trained))
    colors = {n: BLUES[min(int(round(r * (len(BLUES) - 1))), len(BLUES) - 1)] for n, r in zip(trained, ramp)}
    colors["untrained"] = GREY
    return colors


def figures(numbers: dict, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    names, colors = numbers["names"], _checkpoint_colors(numbers["names"])
    written = []

    # 1: variance by time scale
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, facecolor="#fcfcfb")
    for ax, feature in zip(axes, ("cls", "latent")):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
        left = np.zeros(len(names))
        for scale in va.LEVELS:
            v = np.array([numbers["scales"][f"{feature}/{n}"]["shares"][scale] for n in names])
            ax.barh(range(len(names)), v, left=left, height=0.6, color=SCALE_COLORS[scale], edgecolor="#fcfcfb",
                    linewidth=1.5, label=SCALE_LABELS[scale])  # fmt: skip
            left += v
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title("CLS token" if feature == "cls" else "latent (after the projector)", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("share of the variance", color=MUTED, fontsize=9)
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.04, -0.25), ncol=4, frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06, bottom=0.3)
    fig.savefig(out / "fig_timescale.png", dpi=160)
    plt.close(fig)
    written.append("fig_timescale.png")

    # 2: SIGReg under other batch compositions, as a multiple of the iid value
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor="#fcfcfb")
    _style(ax)
    kinds = [("episodes", e) for e in BATCH_EPISODES] + [("window", e) for e in BATCH_EPISODES]
    labels = [f"{'episodes' if k == 'episodes' else 'window'}\n{e} x {128 // e}" for k, e in kinds]

    def ratios(rows):
        iid = next(r["raw"][0] for r in rows if r["kind"] == "iid")
        return [next(r["raw"][0] for r in rows if r["kind"] == k and r["episodes"] == e) / iid for k, e in kinds]

    for n in [n for n in names if n != "untrained"]:
        ax.plot(range(len(kinds)), ratios(numbers["batches"][n]), color=colors[n], marker="o", markersize=5, linewidth=2, label=n)
    ax.plot(range(len(kinds)), ratios(numbers["batches"]["state"]), color=INK, linestyle="--", marker="s", markersize=5,
            linewidth=1.5, label="whole game state")  # fmt: skip
    ax.set_xticks(range(len(kinds)), labels, fontsize=8)
    ax.set_ylabel("SIGReg over the iid value", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_batches.png", dpi=160)
    plt.close(fig)
    written.append("fig_batches.png")

    # 3: the predictor's error and the copy error per principal axis, over the axis variance
    if "predictor" in numbers:
        fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor="#fcfcfb")
        _style(ax)
        for n, p in numbers["predictor"].items():
            v = np.maximum(np.array(p["variance"]), 1e-12)
            ax.plot(np.array(p["error"]) / v, color=colors[n], linewidth=2, label=f"{n}: predictor")
            if n == names[-1]:
                ax.plot(np.array(p["copy"]) / v, color=colors[n], linewidth=1.5, linestyle=":", label=f"{n}: copy the last latent")
        ax.set_yscale("log")
        ax.set_xlabel("principal axis of the latent (largest variance first)", color=MUTED, fontsize=9)
        ax.set_ylabel("squared error over the axis variance", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
        fig.tight_layout()
        fig.savefig(out / "fig_axes.png", dpi=160)
        plt.close(fig)
        written.append("fig_axes.png")

    # 4: which factors the latent reads, untrained against the last checkpoint, by what they cost
    last = names[-1]
    dec, cost = numbers["decodability"], numbers["costs"]
    rows = []
    for name, c in cost.items():
        a, z, k = (dec[f"latent/{n}"]["ridge"].get(name) for n in ("untrained", last, last))
        k = dec[f"cls/{last}"]["ridge"].get(name)
        if a is None or z is None or np.isnan(a) or np.isnan(z) or name.startswith("inv:") and c["change_rate"] < 0.003:
            continue
        rows.append((c["cost"], name, a, z, k, c["change_rate"]))
    rows.sort(reverse=True)
    rows = rows[:28]
    fig, ax = plt.subplots(figsize=(7.5, 0.26 * len(rows) + 1.2), facecolor="#fcfcfb")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    for i, (c, name, a, z, k, rate) in enumerate(rows):
        ax.plot([a, z], [i, i], color="#c9c8c3", linewidth=2, zorder=1)
        ax.scatter([a], [i], color=GREY, s=34, zorder=2, label="untrained latent" if i == 0 else None)
        ax.scatter([z], [i], color=BLUES[-2], s=34, zorder=3, label=f"latent, {last}" if i == 0 else None)
        if k is not None and not np.isnan(k):
            ax.scatter([k], [i], color="#eb6834", s=26, marker="D", zorder=2, label=f"CLS token, {last}" if i == 0 else None)
    ax.set_yticks(range(len(rows)), [f"{r[1]}  (cost {r[0]:.2f}, changes {r[5]:.0%})" for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.1, 1.02)
    ax.set_xlabel("held-out R^2 of a linear read (sorted by cost, highest first)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3)
    fig.tight_layout()
    fig.savefig(out / "fig_factors.png", dpi=160)
    plt.close(fig)
    written.append("fig_factors.png")
    return written


def markdown(numbers: dict, lines: list[str], figs: list[str]) -> str:
    names = numbers["names"]
    md = ["# Slowness analysis", "", f"Sample: {numbers['windows'][0]} episodes x {numbers['windows'][1]} windows x "
          f"{numbers['windows'][2]} frames of held-out worlds; the checkpoints: {', '.join(names)}.", "",
          "## Reading (rule-based, thresholds are heuristics)", ""]  # fmt: skip
    md += [f"- {line}" for line in lines]
    md += ["", "## Variance by time scale (share of the total)", "",
           "| | episode (null) | window | within a window | within a clip | effective rank | rank in an episode | lag 8 |",
           "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for key, s in numbers["scales"].items():
        sh = s["shares"]
        md.append(
            f"| {key} | {sh['episode']:.3f} ({s['null_episode']:.2f}) | {sh['window']:.3f} | {sh['clip_means']:.3f} | "
            f"{sh['clip']:.3f} | {s['erank']:.1f} | {s['within_episode_rank']:.1f} | {next(iter(s['lag'].values())):.2f} |"
        )
    md += ["", "## SIGReg by batch composition (latent, raw / centered)", "",
           "| | " + " | ".join(f"{r['kind']} {r['episodes']}x{r['frames_per_episode']}" for r in numbers["batches"]["state"]) + " |",
           "|---|" + "---|" * len(numbers["batches"]["state"])]  # fmt: skip
    for key, rows in numbers["batches"].items():
        md.append(f"| {'whole state' if key == 'state' else key} | " + " | ".join(f"{r['raw'][0]:.2f} / {r['centered'][0]:.2f}" for r in rows) + " |")
    if numbers["attribution"]:
        md += ["", "## R^2 of the latent from groups of game factors (held-out episodes, MLP)", "",
               "| | all | " + " | ".join(next(iter(numbers["attribution"].values()))["only"]) + " (alone) | residual episode share (null) |", "|---|---|" + "---|" * (len(next(iter(numbers["attribution"].values()))["only"]) + 1)]  # fmt: skip
        for key, a in numbers["attribution"].items():
            r = a["residual"]
            md.append(f"| {key} | {a['all']:.2f} | " + " | ".join(f"{a['only'][g]:.2f}" for g in a["only"])
                      + f" | {r['shares']['episode']:.2f} ({r['null']['episode'][0]:.2f}) |")  # fmt: skip
    if "predictor" in numbers:
        md += ["", "## Predictor error (one step, teacher-forced)", "", "| | error / variance | error / copy error | variance on free axes |", "|---|---|---|---|"]
        for n, p in numbers["predictor"].items():
            md.append(f"| {n} | {p['error_share_of_variance']:.4f} | {p['pred_over_copy']:.3f} | {p['free_share']:.3f} |")
    md += [""] + [f"![{f}]({f})" for f in figs]
    return "\n".join(md) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run", help="the run directory (checkpoints for the predictor); without it no predictor part")
    parser.add_argument("--no-mlp", action="store_true", help="skip the MLP fits (linear only)")
    parser.add_argument("--layout", action="store_true", help="add the whole view map as a group (needs many episodes)")
    parser.add_argument("--batches", type=int, default=100, help="random batches per composition")
    args = parser.parse_args()

    from cpu_threads import limit_threads

    limit_threads()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample, feats = load(Path(args.features))
    numbers = compute(sample, feats, run=Path(args.run) if args.run else None, mlp=not args.no_mlp, n_batches=args.batches, layout=args.layout)
    lines = reading(numbers)
    figs = figures(numbers, out)
    (out / "slowness_numbers.json").write_text(json.dumps(numbers, default=float))
    (out / "slowness_report.md").write_text(markdown(numbers, lines, figs))
    print("\n".join(f"- {line}" for line in lines))
    print(f"wrote {out / 'slowness_report.md'}")


if __name__ == "__main__":
    main()
