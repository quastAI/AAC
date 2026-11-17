"""The Sieve report of one run: the metrics over its epochs and the readiness rules of aac_recipe.md.

    python -m sieve.report --run <run dir>      # rewrites <run>/sieve/report.md, report.json and report.png

Stage 1 is ready (the epoch to fork stage 2 from) at the first epoch E for which E and the epoch before both pass:
    probes flat     no headline MLP probe rose by more than max(0.01, half its bootstrap interval) since the epoch before
    recon flat      the held-out reconstruction (mean and worst 1%) moved by at most 3%
    no collapse     the effective rank is at least 0.8 of the best of the trained epochs; the train/held-out gaps of the
                    reconstruction and the next-latent error did not grow by more than 0.05
    (soft)          the predicted facing and faced tile beat copying the last latent (shown, not required)
Stage 2 can stop at the first epoch E for which E and the epoch before both pass: the facing, the faced tile and the mobs
no longer rise (as above), the noise moves the latent little (moves_for_noise <= 0.1) and steadily (change <= 0.02), and
the effective rank moved by at most 5%. The one-stage Sieve (`joint`, V5) can stop at the first epoch E for which E and
the epoch before both pass: no headline probe rises (as above), the noise moves the latent little and steadily (as in
stage 2), the effective rank moved by at most 5%, and the train/held-out gaps did not grow by more than 0.05 (as in
stage 1; the reconstruction is gated, so its error is not asked to be flat). Epoch 0 (the start) never qualifies. The thresholds are heuristics (THRESHOLDS),
not tuned in Craftax; the recipe warns that a dip and recovery inside an epoch is not seen at epoch checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .probe import HEADLINE

THRESHOLDS = dict(probe_rise=0.01, recon_flat=0.03, rank_keep=0.8, gap_growth=0.05, noise_low=0.1, noise_flat=0.02, rank_flat=0.05)
STAGE2_PROBES = ["facing", "faced_tile", "mob_zombie", "mob_cow", "mob_skeleton", "mob_arrow"]

SECTIONS = {
    "What the latent holds (MLP probe on held-out worlds; chance in brackets)": [f"holds_{n}" for n in HEADLINE],
    "Linear probe (what a plain distance sees)": [f"holds_{n}_linear" for n in HEADLINE] + ["linear_vs_mlp"],
    "What the predictor foresees (right where the entity changed) against copying the last latent": [
        f"{k}_{n}" for n in ("facing", "faced_tile", "mobs", "map", "inventory", "vitals") for k in ("foresees", "copy")
    ],
    "Latent and predictor": ["effective_rank", "dead_axes", "latent_error_vs_copy", "latent_error_gap", "action_lift",
                             "action_follow", "action_follow_frame", "action_follow_facing"],  # fmt: skip
    "Pixels (own decoder; tau in the units of a patch error)": ["recon_error_mean", "recon_error_worst1pct", "recon_gap",
                                                                "decoded_pred_error", "sharp_patch_share", "tau",
                                                                "recon_vs_pred_gradient", "fresh_decoder_error",
                                                                "fresh_decoder_worst1pct"],  # fmt: skip
    "What moves the latent (squared distance over that to an unrelated frame)": [
        "moves_for_facing", "moves_for_mob", "moves_for_tile", "moves_for_item", "moves_for_noise"
    ],
}


def load(run) -> list[dict]:
    out = Path(run) / "sieve"
    entries = []
    for p in out.glob("epoch_*.json"):
        if re.fullmatch(r"epoch_\d+\.json", p.name):
            entries.append(json.loads(p.read_text()))
    return sorted(entries, key=lambda e: e["epoch"])


def _num(v):
    return v if isinstance(v, (int, float)) and v is not None and not (isinstance(v, float) and math.isnan(v)) else None


def _rising(cur: dict, prev: dict, names, thr) -> list[str]:
    """The probes that rose by more than max(probe_rise, half their bootstrap interval)."""
    out = []
    for n in names:
        a, b = _num(cur.get(f"holds_{n}")), _num(prev.get(f"holds_{n}"))
        if a is None or b is None:
            continue
        ci = cur.get(f"holds_{n}_ci") or [a, a]
        tol = max(thr["probe_rise"], (ci[1] - ci[0]) / 2 if None not in ci else 0.0)
        if a - b > tol:
            out.append(f"{n} +{a - b:.3f}")
    return out


def _rel(cur, prev, key):
    a, b = _num(cur.get(key)), _num(prev.get(key))
    return None if a is None or b is None or b == 0 else abs(a - b) / abs(b)


def _gap_growth(cur, prev, thr) -> list[str]:
    out = []
    for key in ("recon_gap", "latent_error_gap"):
        a, b = _num(cur.get(key)), _num(prev.get(key))
        if a is not None and b is not None and a - b > thr["gap_growth"]:
            out.append(f"{key} grew {a - b:+.3f}")
    return out


def stage_label(stage) -> str:
    return "one stage" if stage == "joint" else f"stage {stage}"


def epoch_checks(entries: list[dict], stage, thr=THRESHOLDS) -> dict[int, dict]:
    """Per epoch (from the second): whether it passes against the epoch before, and why not."""
    out = {}
    best_rank = 0.0
    for prev, cur in zip(entries, entries[1:]):
        reasons = []
        if stage in (2, "joint"):
            rising = _rising(cur, prev, STAGE2_PROBES if stage == 2 else HEADLINE, thr)
            if rising:
                reasons.append("still rising: " + ", ".join(rising))
            noise, noise_prev = _num(cur.get("moves_for_noise")), _num(prev.get("moves_for_noise"))
            if noise is not None and (noise > thr["noise_low"] or (noise_prev is not None and abs(noise - noise_prev) > thr["noise_flat"])):
                reasons.append(f"noise moves the latent ({noise:.3f})")
            r = _rel(cur, prev, "effective_rank")
            if r is not None and r > thr["rank_flat"]:
                reasons.append(f"rank moved {r:.0%}")
            if stage == "joint":
                reasons += _gap_growth(cur, prev, thr)
        else:
            rising = _rising(cur, prev, HEADLINE, thr)
            if rising:
                reasons.append("still rising: " + ", ".join(rising))
            for key in ("recon_error_mean", "recon_error_worst1pct"):
                r = _rel(cur, prev, key)
                if r is not None and r > thr["recon_flat"]:
                    reasons.append(f"{key} moved {r:.0%}")
            rank = _num(cur.get("effective_rank")) or 0.0
            if cur["epoch"] >= 1:
                best_rank = max(best_rank, rank)
            if rank < thr["rank_keep"] * best_rank:
                reasons.append(f"rank fell to {rank:.0f} (best {best_rank:.0f})")
            reasons += _gap_growth(cur, prev, thr)
        soft = all(
            _num(cur.get(f"foresees_{n}")) is not None and cur[f"foresees_{n}"] > (_num(cur.get(f"copy_{n}")) or 0)
            for n in ("facing", "faced_tile")
        )
        out[cur["epoch"]] = {"passes": not reasons, "reasons": reasons, "beats_copy": soft}
    return out


def ready_epoch(entries: list[dict], stage, thr=THRESHOLDS) -> int | None:
    """The first epoch that passes with the epoch before it passing too (stage 1: the fork; stage 2 and joint: the stop)."""
    checks = epoch_checks(entries, stage, thr)
    epochs = [e["epoch"] for e in entries]
    for prev, cur in zip(epochs, epochs[1:]):
        if prev >= 1 and checks.get(prev, {}).get("passes") and checks[cur]["passes"]:
            return cur
    return None


def _fmt(v) -> str:
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    v = _num(v)
    if v is None:
        return "-"
    if isinstance(v, int) or abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"


def render(run) -> tuple[str, dict]:
    run = Path(run)
    entries = load(run)
    if not entries:
        return f"# Sieve report: {run.parent.name}/{run.name}\n\nNo epoch evaluated yet.\n", {}
    stage = entries[-1].get("stage", 1)
    stage = stage if stage == "joint" else int(stage)
    checks = epoch_checks(entries, stage)
    ready = ready_epoch(entries, stage)
    epochs = [e["epoch"] for e in entries]
    lines = [f"# Sieve report: {run.parent.name}/{run.name} ({stage_label(stage)})", ""]
    what = {1: "fork stage 2 from", 2: "stop stage 2 at"}.get(stage, "stop the run at")
    if ready is None:
        last = checks.get(epochs[-1])
        why = "; ".join(last["reasons"]) if last and last["reasons"] else "needs two passing epochs in a row"
        lines.append(f"**Not ready yet** (the epoch to {what}): epoch {epochs[-1]}: {why}. Extend the run if probes still rise.")
    else:
        lines.append(f"**Ready at epoch {ready}**: the proposed epoch to {what} (two passing epochs in a row; heuristics, see `sieve/report.py`).")
    lines += ["", "| epoch | passes | why not | predicts facing and faced tile better than copying |", "|---|---|---|---|"]
    for e in epochs[1:]:
        c = checks[e]
        lines.append(f"| {e} | {'yes' if c['passes'] else 'no'} | {'; '.join(c['reasons']) or '-'} | {'yes' if c['beats_copy'] else 'no'} |")
    head = "| metric | " + " | ".join(f"ep {e}" for e in epochs) + " |"
    rule = "|---|" + "---|" * len(epochs)
    last = entries[-1]
    for title, keys in SECTIONS.items():
        rows = [k for k in keys if any(k in e for e in entries)]
        if not rows:
            continue
        lines += ["", f"## {title}", "", head, rule]
        for k in rows:
            name = k
            if k.startswith("holds_") and not k.endswith("_linear") and f"chance_{k[6:]}" in last:
                name = f"{k} ({_fmt(last[f'chance_{k[6:]}'])})"
            lines.append(f"| {name} | " + " | ".join(_fmt(e.get(k)) for e in entries) + " |")
    lines += ["", "Intervals (95%, episode bootstrap) of the last epoch: "
              + ", ".join(f"{n} {_fmt(last.get(f'holds_{n}_ci'))}" for n in HEADLINE if last.get(f"holds_{n}_ci")),
              "", "Seconds per check: " + ", ".join(_fmt(e.get("seconds")) for e in entries)]  # fmt: skip
    summary = {"run": str(run), "stage": stage, "epochs": epochs, "ready_epoch": ready, "checks": checks}
    return "\n".join(lines) + "\n", summary


def figure(run, path) -> bool:
    try:
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    entries = load(run)
    if not entries:
        return False
    x = [e["epoch"] for e in entries]
    panels = [
        ("holds (MLP probe)", ["holds_mob_cow", "holds_mob_zombie", "holds_map_coal", "holds_map_iron", "holds_map_diamond",
                               "holds_facing", "holds_faced_tile", "holds_inventory"]),  # fmt: skip
        ("pixels", ["recon_error_mean", "recon_error_worst1pct", "decoded_pred_error", "fresh_decoder_error"]),
        ("latent", ["effective_rank"]),
        ("moves for (log)", ["moves_for_facing", "moves_for_mob", "moves_for_tile", "moves_for_item", "moves_for_noise"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (title, keys) in zip(axes, panels):
        for k in keys:
            y = [_num(e.get(k)) for e in entries]
            if any(v is not None for v in y):
                ax.plot(x, [math.nan if v is None else v for v in y], marker="o", label=k.replace("holds_", "").replace("moves_for_", ""))
        ax.set_title(title)
        ax.set_xlabel("epoch")
        if title.startswith("moves"):
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def write(run) -> Path:
    run = Path(run)
    out = run / "sieve"
    out.mkdir(parents=True, exist_ok=True)
    text, summary = render(run)
    (out / "report.md").write_text(text)
    (out / "report.json").write_text(json.dumps(summary, indent=1))
    figure(run, out / "report.png")
    return out / "report.md"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path)
    args = p.parse_args(argv)
    print(write(args.run).read_text())


if __name__ == "__main__":
    main()
