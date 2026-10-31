"""Where would SIGReg push the latent in other batch compositions? (no training)

SIGReg of a batch has a gradient with respect to every latent in it. For window batches (E episodes x K frames of one
window each) this is the pressure a window sampler would put on the encoder. It is read on the principal axes of the
latent (`variance.pca`), which are sorted into classes by what the game state explains of them (held-out episodes, MLP):

    slow         the slow group (inventory, achievements, daylight) explains the axis (R^2 above `EXPLAINED`)
    explained    the groups together explain it, the slow group alone does not (vitals, facing, the view)
    unexplained  no group explains it (the remaining half of the variance)

Per axis, over many random batches of one composition:

    batch        the variance of the axis over the whole batch (what SIGReg sees)
    within       the variance of the axis inside a group of the batch, a window for window batches (what a window gives)
    energy       the squared gradient after the batch mean's part is removed (what pushes the points apart or together)
    expand       -sum_i g_i * (z_i - batch mean) / B: the first-order drop of the loss if the axis' deviations are
                 stretched by one unit; positive = SIGReg wants more spread on that axis
    shift        the share of the gradient's energy that is the same for every point of the batch: it moves the batch
                 mean to 0, which for a window batch means moving the window's centre, against the variance between
                 windows (what the slow factors carry)

Heuristic: EXPLAINED = 0.5.
"""

from __future__ import annotations

import numpy as np
import torch

from loss import SIGReg

from . import batches as ba
from . import factors as fa

EXPLAINED = 0.5
CLASSES = ("slow", "explained", "unexplained")


def sigreg_gradient(frames: np.ndarray, seed: int, module: SIGReg | None = None):
    """The value of SIGReg for a batch (B, D) and its gradient with respect to the frames."""
    module = module or SIGReg()
    torch.manual_seed(seed)
    x = torch.from_numpy(np.ascontiguousarray(frames, np.float32)).requires_grad_(True)
    value = module(x[None])
    value.backward()
    return float(value), x.grad.numpy()


def axis_classes(r2_slow: np.ndarray, r2_all: np.ndarray) -> np.ndarray:
    """0 slow, 1 explained by some group, 2 unexplained, per axis."""
    return np.where(r2_slow > EXPLAINED, 0, np.where(r2_all > EXPLAINED, 1, 2))


def axis_pressure(x, kind, episodes, mean, axes, *, n=200, seed=0, batch=ba.BATCH, progress=None) -> dict:
    """Mean over `n` random batches of the per-axis batch and within-group variance, centred gradient energy and expansion
    value, and the shift share of the gradient energy."""
    module = SIGReg()
    rng = np.random.default_rng(seed)
    D = axes.shape[1]
    batch_var, within, energy, expand = np.zeros(D), np.zeros(D), np.zeros(D), np.zeros(D)
    shift, total, value = 0.0, 0.0, 0.0
    for i in range(n):
        frames, group = ba.draw(x, kind, episodes, rng, batch)
        v, g = sigreg_gradient(frames, seed + i, module)
        z = (frames - mean) @ axes  # principal-axis coordinates
        gp = g @ axes
        common = gp.mean(0, keepdims=True)
        dev = z - z.mean(0)
        batch_var += z.var(0)
        within += np.mean([z[group == g].var(0) for g in np.unique(group)], axis=0)
        energy += ((gp - common) ** 2).sum(0)
        expand += -(gp * dev).sum(0) / len(z)
        shift += float((common**2).sum() * len(z))
        total += float((gp**2).sum())
        value += v
        if progress:
            progress()
    return {
        "batch": batch_var / n,
        "within": within / n,
        "energy": energy / n,
        "expand": expand / n,
        "shift_share": shift / max(total, 1e-12),
        "value": value / n,
    }


def by_class(pressure: dict, classes: np.ndarray, variance: np.ndarray) -> dict:
    """Shares per axis class of the total variance, the within-window variance, the gradient energy and the positive
    expansion value."""
    out = {}
    within, energy = pressure["within"], pressure["energy"]
    expand = np.maximum(pressure["expand"], 0)
    for c, name in enumerate(CLASSES):
        m = classes == c
        out[name] = {
            "axes": int(m.sum()),
            "variance": float(variance[m].sum() / variance.sum()),
            "within": float(within[m].sum() / within.sum()),
            "energy": float(energy[m].sum() / energy.sum()),
            "expand": float(expand[m].sum() / max(expand.sum(), 1e-12)),
        }
    return out


COMPOSITIONS = (("iid", 8), ("episodes", 8), ("window", 8))
CLASS_COLORS = {"slow": "#2a78d6", "explained": "#eb6834", "unexplained": "#8a8985"}


def analyse(sample, feats, names, *, n, seed, steps) -> dict:
    """The pressure per composition and axis class for each checkpoint in `names`; `steps.tick` is called per fit and per
    batch."""
    groups = fa.build_groups(sample)
    split = fa.episode_split(len(sample["episode"]))
    out = {}
    for name in names:
        x = feats[name]["latent"]
        steps.tick(f"{name}: axes explained by the slow group")
        r2_slow, mean, axes = fa.axis_r2(x, groups, ("slow",), split, seed=seed)
        steps.tick(f"{name}: axes explained by all groups")
        r2_all, _, _ = fa.axis_r2(x, groups, tuple(groups), split, seed=seed)
        classes = axis_classes(r2_slow, r2_all)
        flat = ((x.reshape(-1, x.shape[-1]) - mean) @ axes).var(0)
        out[name] = {"r2_slow": r2_slow.tolist(), "r2_all": r2_all.tolist(), "classes": classes.tolist(),
                     "variance": flat.tolist(), "compositions": {}}  # fmt: skip
        for kind, e in COMPOSITIONS:
            steps.bar.set_description(f"{name}: SIGReg gradient, {kind} {e}x{ba.BATCH // e}")
            p = axis_pressure(x, kind, e, mean, axes, n=n, seed=seed, progress=steps.tick)
            out[name]["compositions"][f"{kind}{e}"] = {
                "classes": by_class(p, classes, flat),
                "shift_share": p["shift_share"],
                "value": p["value"],
                "batch": p["batch"].tolist(),
                "within": p["within"].tolist(),
                "energy": p["energy"].tolist(),
                "expand": p["expand"].tolist(),
            }
    return out


def markdown(result: dict) -> str:
    md = ["# SIGReg pressure by batch composition", "",
          "Axes are the principal axes of the latent, classed by what the game state explains of them (held-out episodes, "
          "MLP; heuristic threshold R^2 > 0.5). `variance` is the class's share of the latent's variance, `within` of the "
          "variance inside a window, `energy` of the gradient's energy (batch mean removed), `expand` of the positive "
          "first-order gain from stretching the axis. `shift` is the share of the gradient that moves the batch mean.", ""]  # fmt: skip
    for name, r in result.items():
        classes = np.array(r["classes"])
        md += [f"## {name}", "", f"Axes: " + ", ".join(f"{c} {int((classes == i).sum())}" for i, c in enumerate(CLASSES)), "",
               "| composition | SIGReg | shift | class | axes | variance | within | energy | expand |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for comp, v in r["compositions"].items():
            for c in CLASSES:
                k = v["classes"][c]
                md.append(f"| {comp} | {v['value']:.2f} | {v['shift_share']:.2f} | {c} | {k['axes']} | {k['variance']:.2f} | "
                          f"{k['within']:.2f} | {k['energy']:.2f} | {k['expand']:.2f} |")  # fmt: skip
        md.append("")
    return "\n".join(md)


def figure(result: dict, name: str, out):
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    r = result[name]
    comp = r["compositions"]["window8"]
    classes = np.array(r["classes"])
    colors = [CLASS_COLORS[CLASSES[c]] for c in classes]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
        ax.tick_params(colors="#52514e", labelsize=9)
    k = np.arange(len(classes))
    axes[0].plot(k, r["variance"], color="#0e3a73", linewidth=2, label="variance of the axis")
    axes[0].plot(k, comp["within"], color="#8fb8ec", linewidth=2, label="variance inside a window (batch of 8 windows)")
    axes[0].set_yscale("log")
    axes[0].scatter(k, np.full(len(k), min(comp["within"]) * 0.6), c=colors, s=14, marker="s")
    axes[0].legend(frameon=False, fontsize=8, labelcolor="#52514e")
    expand = np.maximum(np.array(comp["expand"]), 0)
    axes[1].bar(k, expand / expand.sum(), color=colors, width=1.0)
    axes[1].set_ylabel("share of the stretch SIGReg asks for", color="#52514e", fontsize=9)
    axes[1].set_xlabel("principal axis of the latent (largest variance first)", color="#52514e", fontsize=9)
    handles = [plt.Line2D([], [], marker="s", linestyle="", color=c, label=f"{n} axes") for n, c in CLASS_COLORS.items()]
    axes[1].legend(handles=handles, frameon=False, fontsize=8, labelcolor="#52514e")
    fig.suptitle(f"SIGReg on windows of 8 episodes: {name}", fontsize=10, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out / f"fig_pressure_{name}.png", dpi=160)
    plt.close(fig)


def main():
    import argparse
    import json
    from pathlib import Path

    from . import report as rp

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, help="a directory of slowness.encode")
    parser.add_argument("--out", required=True)
    parser.add_argument("--names", nargs="*", default=["epoch1", "epoch10"])
    parser.add_argument("--batches", type=int, default=200, help="random batches per composition")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from cpu_threads import limit_threads

    limit_threads()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample, feats = rp.load(Path(args.features))
    steps = rp.Steps(len(args.names) * (2 + len(COMPOSITIONS) * args.batches))
    result = analyse(sample, feats, args.names, n=args.batches, seed=args.seed, steps=steps)
    steps.close()
    (out / "pressure_numbers.json").write_text(json.dumps(result, default=float))
    (out / "pressure_report.md").write_text(markdown(result))
    for name in result:
        figure(result, name, out)
    print(markdown(result))
    print(f"wrote {out / 'pressure_report.md'}")


if __name__ == "__main__":
    main()
