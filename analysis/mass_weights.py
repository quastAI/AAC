"""Sampling weights that give every content class enough pixel mass: the rule "a class is kept only if its mass m_c * D >= kappa".

    python analysis/mass_weights.py --file <craftax_expert_val.h5> --D 192 --kappa 1.5 --out reports/mass_weights

The mass of a class is the share of the dataset's pixel variance that ignoring it costs (`pixel_mass.py`). Both the cost and the frame's variance are
additive over frames, so under per-frame sampling weights w:   m_c(w) = sum_f w_f C[f, c] / sum_f w_f S[f].
C[f, c] = sum over the tiles of class c in frame f of |centroid_c - mean patch at that position|^2, S[f] = the frame's squared deviation from the mean image.

Two ways to get the classes of tiles (a tile is 7 x 7 pixels, one per map cell; the player's own tile is left out):
    clusters   k-means of the raw tile patches: no labels at all. This is what a method can use anywhere.
    labels     the simulator's block and mob of every tile (`state`): only a benchmark has them. Used here to *validate* the clusters
               and to read what the label-free weights do to the real classes.

`solve_weights` raises the weights of the frames that carry a class whose mass is under tau = kappa / D until it reaches tau (never beyond
`overshoot` x tau), with weights between 1/cap and cap and an effective sample size of at least `ess` of the frames. Writes weights.npz
(global frame index and weight), report.md and numbers.json. First order: the means of the classes and positions are those of the unweighted data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

R, C, T7 = 7, 9, 7
VIEW, CH = R * C, 21
CENTER = (R // 2) * C + C // 2
BLOCKS = ["invalid", "out_of_bounds", "grass", "water", "stone", "tree", "wood", "path", "coal", "iron", "diamond", "crafting_table", "furnace",
          "sand", "lava", "plant", "ripe_plant"]
MOBS = ["zombie", "cow", "skeleton", "arrow"]
LABELS = BLOCKS + [f"mob:{m}" for m in MOBS]
MOB_PRIORITY = [3, 2, 0, 1]  # arrow, skeleton, zombie, cow: the rarest written last wins


def tiles_of(x):
    """(T, 63, 63, 3) uint8 -> the map's tiles (T, 63, 147) float32."""
    t = x.shape[0]
    return x[:, :R * T7].reshape(t, R, T7, C, T7, 3).transpose(0, 1, 3, 2, 4, 5).reshape(t, VIEW, T7 * T7 * 3).astype(np.float32)


def label_of(state):
    """(T, 1345) -> the label of every tile (T, 63): block id, or 17 + mob id if a mob stands on it."""
    t = state.shape[0]
    tl = state[:, :VIEW * CH].reshape(t, VIEW, CH)
    lab = tl[..., :17].argmax(-1)
    mobs = tl[..., 17:] > 0.5
    for m in MOB_PRIORITY:
        lab = np.where(mobs[..., m], 17 + m, lab)
    return lab


def kmeans(x, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    xx = (x ** 2).sum(1)[:, None]
    for _ in range(iters):
        d = xx - 2 * x @ c.T + (c ** 2).sum(1)[None]
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            c[j] = x[m].mean(0) if m.any() else x[rng.integers(len(x))]
    return c


def best_tenth(c, S, share=0.1):
    """Mass of a class if only the `share` of the frames with the highest cost per unit variance were sampled (an effective sample size of `share`)."""
    top = np.argsort(-(c / S))[: max(1, int(share * len(S)))]
    return float(c[top].sum() / S[top].sum())


def ess_of(w):
    return float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))


def mass(w, Cm, S):
    return (w[:, None] * Cm).sum(0) / (w * S).sum()


def solve_weights(Cm, S, tau, cap=8.0, ess=0.5, overshoot=1.5, min_mass=1e-5, iters=400, lr=0.5):
    """Frame weights (mean 1) that lift every class with a mass in [min_mass, tau) to tau. Cm: (frames, classes), S: (frames,).

    The log-weight of a frame is a sum over the targeted classes of lam_c times how much of the class the frame carries (relative to the frame
    with the most). lam_c grows while the class is under tau; a step that would break the effective sample size is halved instead."""
    m0 = mass(np.ones(len(S)), Cm, S)
    targets = np.where((m0 >= min_mass) & (m0 < tau))[0]
    rel = Cm[:, targets] / np.maximum(Cm[:, targets].max(0), 1e-30)  # in [0, 1]
    lam = np.zeros(len(targets))
    w = np.ones(len(S))
    for _ in range(iters):
        m = mass(w, Cm, S)[targets]
        gap = np.log(np.clip(tau / np.maximum(m, 1e-30), 1e-3, 1e3))
        gap = np.where(m > overshoot * tau, np.minimum(gap, 0), np.where(m < 0.98 * tau, gap, 0))  # lift the ones below, ease the ones far above
        if not np.any(gap):
            break
        new = np.clip(lam + lr * gap, 0, None)
        cand = np.exp(np.clip(rel @ new, -20, 20))
        cand = np.clip(cand / cand.mean(), 1 / cap, cap)
        cand = cand / cand.mean()
        if ess_of(cand) < ess or cand.max() > cap * 1.0001 and False:
            lr *= 0.5
            if lr < 1e-4:
                break
            continue
        lam, w = new, cand
    return w, targets, m0


def run(path, episodes, k, D, kappa, cap, ess, out, seed=0, only=()):
    import h5py

    t0 = time.time()
    f = h5py.File(path, "r")
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    pick = np.sort(np.random.default_rng(seed).choice(len(ln), min(episodes, len(ln)), replace=False))
    rng = np.random.default_rng(seed)
    pix = (R + 2) * T7
    # ---- pass 1: mean image, position means, label means, a tile sample for k-means
    sx, sxx, n = np.zeros((pix, pix, 3)), np.zeros((pix, pix, 3)), 0
    pos_sum = np.zeros((VIEW, T7 * T7 * 3))
    lab_sum, lab_n = np.zeros((len(LABELS), T7 * T7 * 3)), np.zeros(len(LABELS))
    sample = []
    notc = np.arange(VIEW) != CENTER
    for ep in pick:
        o, m = int(off[ep]), int(ln[ep])
        x = f["pixels"][o:o + m]
        full = x.astype(np.float64)
        sx += full.sum(0)
        sxx += (full ** 2).sum(0)
        n += m
        t = tiles_of(x)
        lab = label_of(f["state"][o:o + m])
        pos_sum += t.sum(0)
        for j in range(len(LABELS)):
            s = (lab == j) & notc[None]
            if s.any():
                lab_sum[j] += t[s].sum(0)
                lab_n[j] += s.sum()
        sel = rng.random((m, VIEW)) < 0.03
        sel[:, CENTER] = False
        sample.append(t[sel])
    mu_pix = sx / n
    ss_total = float((sxx / n - mu_pix ** 2).sum() * n)
    mu_pos = (pos_sum / n).astype(np.float32)
    lab_c = (lab_sum / np.maximum(lab_n, 1)[:, None]).astype(np.float32)
    sample = np.concatenate(sample)
    cent = kmeans(sample, k, seed=seed).astype(np.float32)
    # ---- pass 2: per-frame costs of the label classes and of the clusters, the frame's variance, the cluster x label table
    cost_lab = ((lab_c[:, None, :] - mu_pos[None]) ** 2).sum(-1)  # (labels, positions)
    cost_clu = ((cent[:, None, :] - mu_pos[None]) ** 2).sum(-1)    # (clusters, positions)
    cc = (cent ** 2).sum(1)
    rows, Cl, Cc, Sf, idx = [], [], [], [], []
    table = np.zeros((k, len(LABELS)))
    for ep in pick:
        o, m = int(off[ep]), int(ln[ep])
        x = f["pixels"][o:o + m]
        t = tiles_of(x)
        lab = label_of(f["state"][o:o + m])
        d = (t ** 2).sum(-1)[..., None] - 2 * t @ cent.T + cc  # (T, 63, k)
        a = d.argmin(-1)
        Sf.append(((x.astype(np.float64) - mu_pix) ** 2).sum(axis=(1, 2, 3)))
        pos = np.broadcast_to(np.arange(VIEW), (m, VIEW))
        cl = np.zeros((m, len(LABELS)))
        cu = np.zeros((m, k))
        keep = notc[None].repeat(m, 0)
        fi = np.broadcast_to(np.arange(m)[:, None], (m, VIEW))
        np.add.at(cl, (fi[keep], lab[keep]), cost_lab[lab[keep], pos[keep]])
        np.add.at(cu, (fi[keep], a[keep]), cost_clu[a[keep], pos[keep]])
        np.add.at(table, (a[keep], lab[keep]), 1)
        Cl.append(cl)
        Cc.append(cu)
        idx.append(np.arange(o, o + m))
    Cl, Cc, S, idx = np.concatenate(Cl), np.concatenate(Cc), np.concatenate(Sf), np.concatenate(idx)
    S = S * (ss_total / S.sum())  # the frames' variances sum to the total (guards float drift)
    tau = kappa / D
    ones = np.ones(len(S))
    # ---- the label-free solution, read on both class systems
    w, targets, m0_clu = solve_weights(Cc, S, tau, cap=cap, ess=ess)
    m1_clu = mass(w, Cc, S)
    m0_lab, m1_lab = mass(ones, Cl, S), mass(w, Cl, S)
    # ---- the solution with labels (what a benchmark could do), for comparison
    wl, tl, _ = solve_weights(Cl, S, tau, cap=cap, ess=ess)
    m2_lab = mass(wl, Cl, S)
    m3_lab = None
    if only:  # what a task-aware choice of targets could do: weights solved on these label classes alone
        idx_t = [LABELS.index(c) for c in only]
        wo, _, _ = solve_weights(Cl[:, idx_t], S, tau, cap=cap, ess=ess)
        m3_lab = mass(wo, Cl, S)
    purity = float(table.max(1).sum() / table.sum())
    pj = table / table.sum()
    nmi = float(np.nansum(pj * np.log(pj / (pj.sum(1, keepdims=True) * pj.sum(0, keepdims=True) + 1e-300) + 1e-300))
                / np.sqrt(-(pj.sum(1) * np.log(pj.sum(1) + 1e-300)).sum() * -(pj.sum(0) * np.log(pj.sum(0) + 1e-300)).sum()))
    res = {"file": str(path), "frames": int(len(S)), "episodes": int(len(pick)), "D": D, "kappa": kappa, "tau": tau, "k": k, "seconds": round(time.time() - t0),
           "cluster_purity_vs_labels": purity, "cluster_nmi_vs_labels": nmi,
           "label_free": {"ess": ess_of(w), "max_weight": float(w.max()), "frames_upweighted": float((w > 1.05).mean()), "targets": int(len(targets)),
                          "clusters_below_tau": int((m0_clu < tau).sum()), "clusters_at_tau_after": int((m1_clu[targets] >= 0.9 * tau).sum())},
           "with_labels": {"ess": ess_of(wl), "max_weight": float(wl.max())},
           "label_classes": {LABELS[j]: {"mass_before": float(m0_lab[j]), "mass_label_free": float(m1_lab[j]), "mass_with_labels": float(m2_lab[j]),
                                         "best_tenth": best_tenth(Cl[:, j], S),  # sampling only the 10% of frames that carry the most of it per unit variance
                                         "frames_with": float((Cl[:, j] > 0).mean()),
                                         **({"mass_targets_only": float(m3_lab[j])} if m3_lab is not None else {})}
                             for j in np.argsort(-m0_lab) if m0_lab[j] > 0}}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "weights.npz", frame=idx, weight=w.astype(np.float32), weight_with_labels=wl.astype(np.float32))
    (out / "numbers.json").write_text(json.dumps(res, indent=1))
    (out / "report.md").write_text(report(res))
    return res


def report(r) -> str:
    only = "mass_targets_only" in next(iter(r["label_classes"].values()))
    L = ["# Mass-aware sampling weights\n",
         f"`analysis/mass_weights.py` on `{Path(r['file']).name}`: {r['episodes']} episodes, {r['frames']:,} frames, {r['seconds']} s. Rule: kept only if mass x D >= kappa; "
         f"D = {r['D']}, kappa = {r['kappa']}, so the threshold is {100 * r['tau']:.2f}% of the pixel variance. Weights between 1/cap and cap, effective sample size >= the stated share.\n",
         f"**Label-free classes:** k-means of the 7x7 tile patches, k = {r['k']}. Against the simulator's labels: purity {r['cluster_purity_vs_labels']:.2f}, normalised mutual information {r['cluster_nmi_vs_labels']:.2f}.",
         f"**Label-free weights:** effective sample size {r['label_free']['ess']:.2f}, max weight {r['label_free']['max_weight']:.1f}, {100 * r['label_free']['frames_upweighted']:.0f}% of the frames upweighted; "
         f"{r['label_free']['clusters_below_tau']} clusters under the threshold, {r['label_free']['clusters_at_tau_after']} of {r['label_free']['targets']} targeted ones at 90% of it afterwards.",
         f"**Weights solved with the labels:** effective sample size {r['with_labels']['ess']:.2f}, max weight {r['with_labels']['max_weight']:.1f}.\n",
         "## What the weights do to the real classes (mass in % of the pixel variance)\n",
         "`best 10%`: the class's mass if only the 10% of the frames that carry the most of it per unit of variance were sampled (effective sample size 0.1; each class on its own, "
         "so the classes cannot all have it at once). It is bounded by the class's share *within* a frame. `in frames`: the share of frames that contain it.\n",
         "| class | natural | label-free weights | weights with labels | " + ("targets only | " if only else "") + "best 10% | in frames | threshold |",
         "|---|---|---|---|" + ("---|" if only else "") + "---|---|---|"]
    tau = r["tau"]
    for k, v in r["label_classes"].items():
        mark = lambda x: f"{100 * x:.2f}%" + (" (kept)" if x >= tau else "")  # noqa: E731
        L.append(f"| {k} | {mark(v['mass_before'])} | {mark(v['mass_label_free'])} | {mark(v['mass_with_labels'])} | " + (f"{mark(v['mass_targets_only'])} | " if only else "") + f"{mark(v['best_tenth'])} | {100 * v['frames_with']:.0f}% | {100 * tau:.2f}% |")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=262)
    p.add_argument("--k", type=int, default=96)
    p.add_argument("--D", type=int, default=192)
    p.add_argument("--kappa", type=float, default=1.5)
    p.add_argument("--cap", type=float, default=8.0)
    p.add_argument("--ess", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default="", help="comma-separated label classes (e.g. coal,iron,mob:zombie): also solve the weights for these alone")
    p.add_argument("--out", type=Path, default=Path("reports/mass_weights"))
    a = p.parse_args()
    run(a.file, a.episodes, a.k, a.D, a.kappa, a.cap, a.ess, a.out, a.seed, tuple(c for c in a.only.split(",") if c))
    print((a.out / "report.md").read_text())


if __name__ == "__main__":
    main()
