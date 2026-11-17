"""How many latent axes does a factor need? A check of axis alignment, no training.

Per-dimension gating or a planner's MSE distance read the latent axis by axis. That is sensible only where a game factor
(an inventory item, the vitals, daylight, facing, a mob in view, a block type in view) lives on a few axes of the
latent; if every factor is a mixture of many axes, a gate on single axes cuts across factors. For every factor this
module fits a ridge regression from the latent (fitted on training episodes, scored on held-out episodes) and measures

    r2            the held-out R^2 of the full linear fit
    k90           the fewest axes whose refit keeps 90% of that R^2 (axes ranked by the full fit's contribution
                  w_j^2 var_j), in the standard basis, in the PCA basis, and in random rotations of the latent
    pr            the participation ratio of those contributions: the effective number of axes the factor spreads over

Random rotations are the control: a rotation destroys any axis alignment and keeps the ridge fit unchanged (same R^2),
so a factor that needs as many axes in the standard basis as in a random rotation has no axis alignment. The labels
(`state`, `inventory`, `vitals`, `facing`, `daylight`) are only readout targets here, as in the probes.

    python -m sieve.concentration --checkpoint epoch0=<ckpt> --checkpoint epoch8=<ckpt> --val-file <val.h5>
                                  [--val-episodes 128] --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from probing import targets as tg

KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)
LAM = 1e-3  # ridge strength, in units of the mean eigenvalue of the training Gram matrix (heuristic)
HELD = 0.2  # a factor is analysed where the full linear fit explains at least this much on held-out episodes
KEEP = 0.9
ROTATIONS = 3


def factor_targets(part) -> tuple[np.ndarray, list[str]]:
    """The factors (frames, T) and their names, read from the sample part's label columns."""
    blocks, mobs = tg.tile_maps(torch.from_numpy(part.state))
    in_view = tg.view_concepts(blocks, mobs)["in_view"].numpy()
    facing = np.eye(4, dtype=np.float32)[np.clip(np.asarray(part.facing) - 1, 0, 3)]
    columns = [part.inventory, part.vitals, np.asarray(part.daylight)[:, None], facing,
               mobs.any(-2).numpy().astype(np.float32), in_view]  # fmt: skip
    names = ([f"inventory:{n}" for n in tg.INVENTORY_NAMES] + [f"vitals:{n}" for n in tg.VITAL_NAMES] + ["daylight"]
             + [f"facing:{n}" for n in tg.MOVES] + [f"mob:{n}" for n in tg.MOB_NAMES]
             + [f"in_view:{n}" for n in tg.IN_VIEW_BLOCKS])  # fmt: skip
    return np.concatenate([np.asarray(c, np.float32) for c in columns], 1), names


def split_by_episode(part, test_share: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of the training and the held-out frames, whole episodes apart."""
    episodes = np.unique(part.episode)
    pick = np.random.default_rng(seed).choice(episodes, max(1, int(len(episodes) * test_share)), replace=False)
    test = np.isin(part.episode, pick)
    return ~test, test


def rotation(kind: str, cov: np.ndarray, seed: int = 0) -> np.ndarray:
    """The orthogonal basis of a kind: axes (identity), pca (principal axes, largest first), rand (a random rotation)."""
    d = cov.shape[0]
    if kind == "axes":
        return np.eye(d)
    if kind == "pca":
        return np.linalg.eigh(cov)[1][:, ::-1]
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def curves(z_tr, y_tr, z_te, y_te, basis: np.ndarray, lam: float = LAM) -> dict:
    """Held-out R^2 of ridge fits on the top-k axes of `basis` for every k in KS and every factor, and the
    participation ratio of each factor's contributions. Everything runs on Gram matrices, no pass over the frames per fit."""
    mu, my = z_tr.mean(0), y_tr.mean(0)
    a, b = (z_tr - mu) @ basis, (z_te - mu) @ basis
    ya, yb = y_tr - my, y_te - my
    gram, cross = a.T @ a, a.T @ ya
    gram_te, cross_te = b.T @ b, b.T @ yb
    sst = ((y_te - y_te.mean(0)) ** 2).sum(0)
    yy = (yb**2).sum(0)
    d, t = cross.shape
    ridge = lam * np.trace(gram) / d
    weights = np.linalg.solve(gram + ridge * np.eye(d), cross)
    contribution = weights**2 * (np.diag(gram) / len(a))[:, None]
    r2 = np.zeros((len(KS), t))
    for j in range(t):
        order = np.argsort(-contribution[:, j])
        for i, k in enumerate(KS):
            s = order[: min(k, d)]
            w = np.linalg.solve(gram[np.ix_(s, s)] + ridge * np.eye(len(s)), cross[s, j])
            sse = yy[j] - 2 * w @ cross_te[s, j] + w @ gram_te[np.ix_(s, s)] @ w
            r2[i, j] = 1 - sse / max(sst[j], 1e-12)
    pr = contribution.sum(0) ** 2 / np.maximum((contribution**2).sum(0), 1e-30)
    return dict(r2=r2, pr=pr)


def axes_needed(r2: np.ndarray, keep: float = KEEP) -> np.ndarray:
    """k90 per factor from the R^2 curve (len(KS), T): the smallest k whose R^2 is at least `keep` of the full fit's."""
    full = r2[-1]
    ok = r2 >= keep * full[None]
    first = ok.argmax(0)
    return np.where(full > 0, np.array(KS)[first], np.nan).astype(float)


def analyse(z: np.ndarray, y: np.ndarray, names: list[str], train: np.ndarray, test: np.ndarray, seed: int = 0) -> dict:
    """The concentration numbers of one latent matrix (frames, d) against the factors (frames, T)."""
    z_tr, z_te, y_tr, y_te = z[train].astype(np.float64), z[test].astype(np.float64), y[train], y[test]
    cov = np.cov(z_tr, rowvar=False)
    usable = y_tr.std(0) > 1e-6
    out = {}
    runs = {"axes": [rotation("axes", cov)], "pca": [rotation("pca", cov)],
            "rand": [rotation("rand", cov, seed + i) for i in range(ROTATIONS)]}  # fmt: skip
    got = {kind: [curves(z_tr, y_tr, z_te, y_te, r) for r in rs] for kind, rs in runs.items()}
    r2_full = got["axes"][0]["r2"][-1]
    check = max(float(np.abs(c["r2"][-1] - r2_full)[usable].max()) for cs in got.values() for c in cs)  # rotation-invariant
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a factor the fit does not explain has no k90 (all NaN)
        k = {kind: np.nanmedian(np.stack([axes_needed(c["r2"]) for c in cs]), 0) for kind, cs in got.items()}
    pr = {kind: np.mean(np.stack([c["pr"] for c in cs]), 0) for kind, cs in got.items()}
    for j, name in enumerate(names):
        if not usable[j]:
            continue
        out[name] = dict(r2=float(r2_full[j]), k90_axes=float(k["axes"][j]), k90_pca=float(k["pca"][j]),
                         k90_rand=float(k["rand"][j]), pr_axes=float(pr["axes"][j]), pr_pca=float(pr["pca"][j]),
                         pr_rand=float(pr["rand"][j]))  # fmt: skip
    held = {n: v for n, v in out.items() if v["r2"] >= HELD}

    def med(key):
        return float(np.median([v[key] for v in held.values()])) if held else float("nan")

    aligned = [v["k90_axes"] < 0.5 * v["k90_rand"] for v in held.values()]
    summary = dict(held=len(held), analysed=len(out), median_k90_axes=med("k90_axes"), median_k90_pca=med("k90_pca"),
                   median_k90_rand=med("k90_rand"), median_pr_axes=med("pr_axes"), median_pr_rand=med("pr_rand"),
                   share_aligned=float(np.mean(aligned)) if aligned else float("nan"), rotation_check=check)  # fmt: skip
    return dict(factors=out, summary=summary)


def _f(v, digits=1):
    return "-" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def render(results: dict) -> str:
    """The markdown report of {label: analyse(...)}."""
    lines = ["# How many latent axes does a factor need?", "",
             "Ridge fits from the latent to each factor (training episodes), scored on held-out episodes. k90: the fewest "
             f"axes whose refit keeps {KEEP:.0%} of the full fit's R^2; standard basis, PCA basis, and the median over "
             f"{ROTATIONS} random rotations (the control: a rotation destroys axis alignment). pr: the participation ratio "
             f"of the axes' contributions. Factors with a full R^2 of at least {HELD} count as held; a smaller k90 in the "
             "standard basis than in random rotations means the factor sits on few axes.", "",
             "## Summary", "",
             "| checkpoint | held | median k90 axes | pca | random | median pr axes | random | share with k90 axes < half of random |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for label, r in results.items():
        s = r["summary"]
        lines.append(f"| {label} | {s['held']} of {s['analysed']} | {_f(s['median_k90_axes'], 0)} | {_f(s['median_k90_pca'], 0)} | "
                     f"{_f(s['median_k90_rand'], 0)} | {_f(s['median_pr_axes'])} | {_f(s['median_pr_rand'])} | "
                     f"{_f(s['share_aligned'], 2)} |")  # fmt: skip
    for label, r in results.items():
        lines += ["", f"## {label}", "", "| factor | R^2 | k90 axes | pca | random | pr axes | pca | random |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for name, v in sorted(r["factors"].items(), key=lambda kv: -kv[1]["r2"]):
            flag = "" if v["r2"] >= HELD else " (not held)"
            lines.append(f"| {name}{flag} | {_f(v['r2'], 2)} | {_f(v['k90_axes'], 0)} | {_f(v['k90_pca'], 0)} | "
                         f"{_f(v['k90_rand'], 0)} | {_f(v['pr_axes'])} | {_f(v['pr_pca'])} | {_f(v['pr_rand'])} |")  # fmt: skip
        lines.append(f"\nRotation check (the full fit's R^2 must not change with the basis): max difference "
                     f"{r['summary']['rotation_check']:.1e}.")  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv=None):
    from cpu_threads import limit_threads

    from . import check
    from . import sample as sm
    from .watch import load_model

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", action="append", required=True, help="label=path of a model checkpoint (repeat)")
    ap.add_argument("--val-file", type=Path, required=True)
    ap.add_argument("--val-episodes", type=int, default=128)
    ap.add_argument("--windows", type=int, default=4, help="windows per episode")
    ap.add_argument("--length", type=int, default=32, help="consecutive frames per window")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    limit_threads()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    part = sm._read(args.val_file, args.val_episodes, args.windows, args.length, 1)
    y, names = factor_targets(part)
    train, test = split_by_episode(part, seed=args.seed)
    print(f"[concentration] {len(part)} frames, {int(train.sum())} fit / {int(test.sum())} held out, {device}", flush=True)
    results = {}
    for item in args.checkpoint:
        label, _, path = item.partition("=")
        model = load_model(Path(path)).to(device).eval()
        z = check.encode(model.get_level(1), part.pixels, device)
        results[label] = analyse(z, y, names, train, test, seed=args.seed)
        s = results[label]["summary"]
        print(f"[concentration] {label}: {s['held']} factors held, median k90 axes {s['median_k90_axes']:.0f}, "
              f"random {s['median_k90_rand']:.0f} ({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
    (args.out / "concentration.md").write_text(render(results))
    (args.out / "concentration.json").write_text(json.dumps(results, indent=1))
    print(f"[concentration] done in {time.time() - t0:.0f} s, written to {args.out / 'concentration.md'}", flush=True)


if __name__ == "__main__":
    main()
