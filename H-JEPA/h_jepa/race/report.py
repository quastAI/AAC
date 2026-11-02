"""The runs of the race side by side: what the latent holds over the steps of the training.

    python -m race.report --out <report.md> B2=<run dir> V3=<run dir> A3=<run dir> A4=<run dir>

Writes the markdown table (selected steps and the last one, per run) and `<report>.png` (one panel per readout, the steps
on a log scale, one line per run). A run is a run directory of the training with `race/trace.json` and `race/meta.json` (`race.evaluate`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# the factors B1 loses (mobs, ores, lava), keeps (inventory, vitals, daylight) and the near field
FACTORS = ("mob:cow", "mob:zombie", "mob:skeleton", "mob:arrow", "view:iron", "view:coal", "view:diamond", "view:lava",
           "view:grass", "vital:food", "inv:wood", "daylight")  # fmt: skip
LATENT = ("near", "rank", "dead_axes", "clip_share", "pred_over_copy")
STEPS = (0, 50, 100, 300, 1000, 2000, 5000, 15000)


def load(run: Path) -> tuple[dict, list[dict]]:
    return json.loads((run / "race" / "meta.json").read_text()), json.loads((run / "race" / "trace.json").read_text())


def value(entry: dict, key: str):
    return entry["r2"].get(key) if ":" in key or key == "daylight" else entry.get(key)


def fmt(v) -> str:
    return "–" if v is None else (f"{v:.0f}" if isinstance(v, int) or abs(v) >= 10 else f"{v:.2f}")


def table(runs: dict[str, tuple[dict, list[dict]]]) -> list[str]:
    md = ["| run | step | " + " | ".join(FACTORS + LATENT) + " |", "|---|---|" + "---|" * (len(FACTORS) + len(LATENT))]
    for name, (_, trace) in runs.items():
        steps = [e for e in trace if e["step"] in STEPS or e is trace[-1]]
        for e in steps:
            md.append(f"| {name} | {e['step']} | " + " | ".join(fmt(value(e, k)) for k in FACTORS + LATENT) + " |")
    return md


def figure(runs: dict[str, tuple[dict, list[dict]]], path: Path) -> None:
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    keys = FACTORS + LATENT
    cols = 6
    fig, axes = plt.subplots((len(keys) + cols - 1) // cols, cols, figsize=(3 * cols, 2.4 * ((len(keys) + cols - 1) // cols)))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # the first slots of a validated categorical order
    for ax, key in zip(axes.flat, keys):
        for (name, (_, trace)), color in zip(runs.items(), colors):
            pts = [(e["step"] + 1, value(e, key)) for e in trace if value(e, key) is not None]
            if pts:
                ax.plot(*zip(*pts), color=color, linewidth=2, label=name)
        ax.set_xscale("log")
        ax.set_title(key, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(color="#ecebe7", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(keys) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.supxlabel("training step + 1 (log scale; the first point is the untrained encoder)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def markdown(runs: dict[str, tuple[dict, list[dict]]], figure_name: str) -> str:
    md = ["# What the latent holds over the first steps of the training", "",
          "Linear read (ridge, held-out episodes) of the latent at each check; `near`: the four tiles next to the player; "
          "`rank`: effective rank; `dead_axes`: principal axes under a tenth of the mean variance; `clip_share`: the "
          "variance inside a clip of 4 frames; `pred_over_copy`: the one-step error over copying the last latent. Step 0 is "
          "the untrained encoder; steps are checkpoints, the epoch ends are the multiples of the steps per epoch.", ""]  # fmt: skip
    for name, (meta, trace) in runs.items():
        md.append(f"- **{name}** ({meta['run']}): encoder reads {meta['obs_key']}, stop-grad {meta['stop_grad']}, history {meta['history']}, "
                  f"SIGReg {meta['sigreg']}, {meta['steps_per_epoch']} steps per epoch, {len(trace)} checks, last step {trace[-1]['step']}")  # fmt: skip
    return "\n".join(md + [""] + table(runs) + ["", f"![{figure_name}]({figure_name})", ""])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="the markdown report; the figure is written next to it")
    p.add_argument("runs", nargs="+", metavar="NAME=DIR")
    args = p.parse_args(argv)
    runs = {}
    for item in args.runs:
        name, path = item.split("=", 1)
        runs[name] = load(Path(path))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    figure(runs, png)
    out.write_text(markdown(runs, png.name))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
