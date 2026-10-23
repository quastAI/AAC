"""Training curves of a run from its metrics CSV: the losses and the health metrics of level 1, one figure.

    python plot_training.py --run $HJEPA_HOME/ckpts/craftax/flat/seed42

stable-pretraining writes the CSV of a run to `<run>/spt/runs/<date>/<time>/<id>/metrics.csv`: every logged key is a column
and a row holds the keys of one logging call, the others are empty. The figure goes to `<run>/health/loss_curves.png` and a
copy of the CSV next to it (`health/metrics.csv`; the notebook's upload leaves `spt/` out).
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

# (title, regular expression of the columns): the keys the training logs (hjepa_forward, TrainingHealthCallback)
PANELS = [
    ("loss", r"^(fit|train|validate)/loss(_step|_epoch)?$"),
    ("prediction MSE", r"^(fit|train|validate)/pred_loss_level1(_step|_epoch)?$"),
    ("SIGReg", r"^(fit|train|validate)/sigreg_loss_level1(_step|_epoch)?$"),
    ("inverse dynamics", r"^(fit|train|validate)/idm_(loss|accuracy_loss)_level1(_step|_epoch)?$"),
    ("latent: rank, scale", r"^health(_val)?/level1/rep/(erank_frac|std_mean|dead_dim_frac)$"),
    ("dynamics: against copy and shuffled actions", r"^health(_val)?/level1/dyn/(pred_over_copy|action_lift)$"),
    ("open-loop drift (validation)", r"^health_val/level1/drift_h\d+/(ratio_last|useful_horizon)$"),
    ("learning rate", r"(^|/)(lr[-_]|hparams/lr)"),
]


def find_metrics(run: Path) -> Path:
    """The newest metrics CSV of the run (stable-pretraining's, else Lightning's CSV logger's)."""
    newest = lambda paths: sorted(paths, key=lambda p: p.stat().st_mtime)  # noqa: E731
    for found in (newest((run / "spt" / "runs").glob("*/*/*/metrics.csv")), newest(run.glob("lightning_logs/version_*/metrics.csv"))):
        if found:
            return found[-1]
    raise SystemExit(f"no metrics.csv under {run} (spt/runs/*/*/*/ or lightning_logs/version_*/)")


def read_series(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    """Per column the (steps, values) of the rows that have a value."""
    series: dict[str, tuple[list[float], list[float]]] = {}
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            step = float(row["step"]) if row.get("step") not in (None, "") else float(i)
            for key, value in row.items():
                if key in ("step", "epoch") or value in (None, ""):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                steps, values = series.setdefault(key, ([], []))
                steps.append(step)
                values.append(number)
    return series


def plot(series: dict, out: Path) -> list[str]:
    """One panel per group with data; returns the titles drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drawn = [(title, sorted(k for k in series if re.search(pattern, k))) for title, pattern in PANELS]
    drawn = [(title, keys) for title, keys in drawn if keys]
    if not drawn:
        raise SystemExit(f"none of the expected columns is in the CSV; it has: {sorted(series)[:30]}")
    cols = 2
    rows = (len(drawn) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows), squeeze=False)
    for ax, (title, keys) in zip(axes.flat, drawn):
        for key in keys:
            steps, values = series[key]
            ax.plot(steps, values, marker="." if len(steps) < 40 else None, linewidth=1, label=key)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(drawn) :]:
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return [title for title, _ in drawn]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory (ckpts/craftax/<name>/seed<N>)")
    parser.add_argument("--out", help="figure (default: <run>/health/loss_curves.png)")
    args = parser.parse_args()
    run = Path(args.run)
    metrics = find_metrics(run)
    out = Path(args.out) if args.out else run / "health" / "loss_curves.png"
    titles = plot(read_series(metrics), out)
    shutil.copy(metrics, out.parent / "metrics.csv")
    print(f"{metrics}: {', '.join(titles)}\nwrote {out}")


if __name__ == "__main__":
    main()
