"""How much of a Craftax-Classic frame's pixel variance each kind of content carries, and how much of its frame-to-frame change.

    python analysis/pixel_mass.py --file <craftax_expert_val.h5> --episodes 80 --out reports/pixel_mass [--traces A5=<trace.json> ...]

Why: a reconstruction or SIGReg-shaped latent pays for content by the pixels it takes (README, A5). The cost of *ignoring* a class of
content is the squared error a decoder makes when it does not know that class: the variance the class explains.

    plain frames   tiles are labelled by the block under them (from `state`); the cost of ignoring block k is
                   sum over its tiles of |mean patch of k - mean patch at that position|^2; the cost of ignoring a mob is
                   sum over its tiles of |patch - mean patch of the block under it|^2, taken as the class mean displacement;
                   the player's tile is labelled by `facing` (the facing sprite). Shares are of the total pixel variance of the frame
                   (per-pixel variance over frames, map and inventory strip), the number a decoder that predicts the mean image makes.
    differences    x_{t+1} - x_t. The view is centred on the player, so a move shifts the whole map by one tile: the plain difference is
                   mostly scrolling. The step's shift (dr, dc) in tiles is the one that minimises the difference of the aligned tiles
                   (a blocked move or a noop: 0); the aligned difference is what remains when the scroll is compensated. It is split
                   into tiles with a mob in either frame, tiles whose block changed, the player's tile, and the rest (daylight, noise).

Reads whole episodes (random, seeded) in chunks; numpy and h5py only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BLOCKS = ["invalid", "out_of_bounds", "grass", "water", "stone", "tree", "wood", "path", "coal", "iron", "diamond",
          "crafting_table", "furnace", "sand", "lava", "plant", "ripe_plant"]
MOBS = ["zombie", "cow", "skeleton", "arrow"]
R, C, T7 = 7, 9, 7  # view rows and columns, pixels per tile
CENTER = (3, 4)
VIEW = R * C
CH = 21  # state channels per tile: 17 block one-hot, 4 mob flags
SHIFTS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def tiles_of(x):
    """(T, 63, 63, 3) uint8 -> the map's tiles (T, 7, 9, 7, 7, 3) float32 and the strip (T, 14, 63, 3)."""
    t = x.shape[0]
    m = x[:, :R * T7].reshape(t, R, T7, C, T7, 3).transpose(0, 1, 3, 2, 4, 5).astype(np.float32)
    return m, x[:, R * T7:].astype(np.float32)


def state_of(s):
    t = s.shape[0]
    tl = s[:, :VIEW * CH].reshape(t, R, C, CH)
    return tl[..., :17].argmax(-1), tl[..., 17:] > 0.5  # blocks (T, 7, 9), mobs (T, 7, 9, 4)


def best_shift(a, b):
    """For consecutive tile arrays a (frame t) and b (frame t+1): (n, 7, 9, 147), the shift (dr, dc) with b[r, c] ~ a[r+dr, c+dc]."""
    n = a.shape[0]
    errs = np.full((n, len(SHIFTS)), np.inf, np.float32)
    for k, (dr, dc) in enumerate(SHIFTS):
        r0, r1, c0, c1 = max(0, -dr), min(R, R - dr), max(0, -dc), min(C, C - dc)
        d = b[:, r0:r1, c0:c1] - a[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        errs[:, k] = (d ** 2).mean(axis=(1, 2, 3))
    return errs.argmin(1)


def run(path, n_episodes, seed, out, traces):
    import h5py

    t0 = time.time()
    f = h5py.File(path, "r")
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(ln), min(n_episodes, len(ln)), replace=False))
    pix = (R + 2) * T7
    # --- accumulators
    sx, sxx = np.zeros((pix, pix, 3)), np.zeros((pix, pix, 3))  # per pixel, over frames
    n_frames = 0
    sd = sdd = 0.0
    sxd = np.zeros((pix, pix, 3))
    pos_sum, pos_cnt = np.zeros((VIEW, T7, T7, 3)), 0  # mean patch per position
    cls_sum, cls_n = np.zeros((17, T7, T7, 3)), np.zeros(17)  # block class means from tiles without a mob
    cnt_kt = np.zeros((17, VIEW))  # tiles of block k at position t (all tiles, the player's excluded)
    mob_sum, mob_n = np.zeros((4, 17, T7, T7, 3)), np.zeros((4, 17))
    pl_sum, pl_n, pl_pos = np.zeros((5, T7, T7, 3)), np.zeros(5), np.zeros((T7, T7, 3))
    # differences
    diff = {k: 0.0 for k in ("moved_plain", "still_plain", "mob", "block", "player", "rest", "strip_plain", "map_plain")}
    steps = {"moved": 0, "still": 0}
    for ep in pick:
        o, n = int(off[ep]), int(ln[ep])
        x = f["pixels"][o:o + n]
        st = f["state"][o:o + n]
        dl = f["daylight"][o:o + n, 0].astype(np.float64)
        fc = f["facing"][o:o + n, 0].astype(int)
        m, strip = tiles_of(x)
        blocks, mobs = state_of(st)
        # per-pixel moments (map + strip)
        full = x.astype(np.float64)
        sx += full.sum(0)
        sxx += (full ** 2).sum(0)
        n_frames += n
        sd += dl.sum()
        sdd += (dl ** 2).sum()
        sxd += np.einsum("t,thwc->hwc", dl, full)
        flat = m.reshape(n, VIEW, T7, T7, 3)
        pos_sum += flat.sum(0)
        pos_cnt += n
        bflat, mflat = blocks.reshape(n, VIEW), mobs.reshape(n, VIEW, 4)
        has_mob = mflat.any(-1)
        center = CENTER[0] * C + CENTER[1]
        notc = np.arange(VIEW) != center
        for k in range(17):
            sel = (bflat == k) & notc[None]
            cnt_kt[k] += sel.sum(0)
            sel_nm = sel & ~has_mob
            if sel_nm.any():
                cls_sum[k] += flat[sel_nm].sum(0)
                cls_n[k] += sel_nm.sum()
        for mi in range(4):
            sel = mflat[:, :, mi] & notc[None]
            for k in np.unique(bflat[sel]):
                s2 = sel & (bflat == k)
                mob_sum[mi, k] += flat[s2].sum(0)
                mob_n[mi, k] += s2.sum()
        for fi in range(1, 5):
            s2 = fc == fi
            if s2.any():
                pl_sum[fi] += flat[s2, center].sum(0)
                pl_n[fi] += s2.sum()
        pl_pos += flat[:, center].sum(0)
        # --- frame differences within the episode
        if n > 1:
            a, b = m[:-1], m[1:]
            plain = ((b - a) ** 2).sum(axis=(1, 2, 3, 4, 5))
            sh = best_shift(a.reshape(n - 1, R, C, -1), b.reshape(n - 1, R, C, -1))
            moved = np.array([SHIFTS[k] != (0, 0) for k in sh])
            diff["moved_plain"] += plain[moved].sum()
            diff["still_plain"] += plain[~moved].sum()
            diff["map_plain"] += plain.sum()
            diff["strip_plain"] += ((strip[1:] - strip[:-1]) ** 2).sum()
            steps["moved"] += int(moved.sum())
            steps["still"] += int((~moved).sum())
            for k, (dr, dc) in enumerate(SHIFTS):
                idx = np.where(sh == k)[0]
                if not len(idx):
                    continue
                r0, r1, c0, c1 = max(0, -dr), min(R, R - dr), max(0, -dc), min(C, C - dc)
                a_ = a[idx][:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
                b_ = b[idx][:, r0:r1, c0:c1]
                e = ((b_ - a_) ** 2).sum(axis=(-1, -2, -3))  # (k, rows, cols)
                mob_any = mobs[1:][idx][:, r0:r1, c0:c1].any(-1) | mobs[:-1][idx][:, r0 + dr:r1 + dr, c0 + dc:c1 + dc].any(-1)
                blk = blocks[1:][idx][:, r0:r1, c0:c1] != blocks[:-1][idx][:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
                pc = np.zeros(e.shape[1:], bool)
                cr, cc = CENTER[0] - r0, CENTER[1] - c0
                if 0 <= cr < pc.shape[0] and 0 <= cc < pc.shape[1]:
                    pc[cr, cc] = True
                pcm = pc[None]
                diff["player"] += e[:, pc].sum()
                diff["mob"] += e[mob_any & ~pcm].sum()
                diff["block"] += e[blk & ~mob_any & ~pcm].sum()
                diff["rest"] += e[~blk & ~mob_any & ~pcm].sum()
    diff = {k: float(v) for k, v in diff.items()}  # numpy float32 sums are not JSON
    # --- plain-frame masses
    mean_pix = sx / n_frames
    var_pix = sxx / n_frames - mean_pix ** 2
    ss_total = var_pix.sum() * n_frames  # sum of squared deviations over all frames and pixels
    ss_map = var_pix[:R * T7].sum() * n_frames
    mu_pos = pos_sum / pos_cnt  # (63, 7, 7, 3)
    m_blk = np.where(cls_n[:, None, None, None] > 0, cls_sum / np.maximum(cls_n, 1)[:, None, None, None], 0)
    masses, freq = {}, {}
    tot_tiles = cnt_kt.sum()
    for k, name in enumerate(BLOCKS):
        if cnt_kt[k].sum() == 0 or cls_n[k] == 0:
            continue
        cost = sum(cnt_kt[k, t] * ((m_blk[k] - mu_pos[t]) ** 2).sum() for t in range(VIEW) if t != CENTER[0] * C + CENTER[1])
        masses[f"block:{name}"] = cost / ss_total
        freq[f"block:{name}"] = cnt_kt[k].sum() / tot_tiles
    for mi, name in enumerate(MOBS):
        if mob_n[mi].sum() == 0:
            continue
        eff = sum(mob_sum[mi, k] - mob_n[mi, k] * m_blk[k] for k in range(17) if mob_n[mi, k] > 0) / mob_n[mi].sum()
        masses[f"mob:{name}"] = mob_n[mi].sum() * (eff ** 2).sum() / ss_total
        freq[f"mob:{name}"] = mob_n[mi].sum() / tot_tiles
    mu_c = pl_pos / n_frames
    pl_cost = sum(pl_n[fi] * ((pl_sum[fi] / pl_n[fi] - mu_c) ** 2).sum() for fi in range(1, 5) if pl_n[fi] > 0)
    masses["player:facing"] = pl_cost / ss_total
    strip_share = var_pix[R * T7:].sum() / var_pix.sum()
    dmean = sd / n_frames
    dvar = sdd / n_frames - dmean ** 2
    cov = sxd / n_frames - mean_pix * dmean
    daylight = float(((cov ** 2) / np.maximum(var_pix * dvar, 1e-12) * var_pix).sum() / var_pix.sum())
    tot_aligned = diff["mob"] + diff["block"] + diff["player"] + diff["rest"]
    res = {
        "file": str(path), "episodes": int(len(pick)), "frames": int(n_frames), "seed": seed, "seconds": round(time.time() - t0),
        "total_pixel_variance_per_pixel": float(var_pix.mean()),
        "strip_share_of_variance": float(strip_share), "map_share_of_variance": float(ss_map / ss_total),
        "daylight_linear_share_of_variance": daylight,
        "class_mass": {k: float(v) for k, v in sorted(masses.items(), key=lambda kv: -kv[1])},
        "class_tile_share": {k: float(v) for k, v in freq.items()},
        "between_class_share": float(sum(masses.values())),
        "steps_moved_share": steps["moved"] / max(1, steps["moved"] + steps["still"]),
        "plain_diff_energy_share_on_moved_steps": diff["moved_plain"] / max(1e-12, diff["moved_plain"] + diff["still_plain"]),
        "plain_diff_energy_per_step": {"moved": diff["moved_plain"] / max(1, steps["moved"]), "still": diff["still_plain"] / max(1, steps["still"])},
        "aligned_diff_share": {k: diff[k] / tot_aligned for k in ("mob", "block", "player", "rest")},
        "aligned_over_plain_energy": tot_aligned / max(1e-12, diff["map_plain"]),
        "strip_diff_energy_over_map_plain": diff["strip_plain"] / max(1e-12, diff["map_plain"]),
    }
    # --- against what a latent kept (race traces: ridge R^2 of `view:<block>` and `mob:<mob>` from the latent at a step)
    if traces:
        from scipy.stats import spearmanr  # noqa: PLC0415

        res["spearman_mass_vs_latent_r2"] = {}
        for lab, (tp, step) in traces.items():
            entries = {e["step"]: e for e in json.load(open(tp))}
            r2 = entries[step]["r2"]
            pairs = []
            for key, mass in masses.items():
                kind, name = key.split(":")
                rk = ("view:" if kind == "block" else "mob:") + name
                if kind != "player" and r2.get(rk) is not None:
                    pairs.append((key, mass, freq[key], r2[rk]))
            res["spearman_mass_vs_latent_r2"][lab] = {
                "step": step, "n": len(pairs),
                "rho_mass": float(spearmanr([p[1] for p in pairs], [p[3] for p in pairs])[0]),
                "rho_tile_share": float(spearmanr([p[2] for p in pairs], [p[3] for p in pairs])[0]),
                "pairs": [{"class": p[0], "mass": p[1], "tile_share": p[2], "latent_r2": p[3]} for p in pairs],
            }
            if step != 0 and 0 in entries:  # what is kept relative to the same run's step 0 (an untrained encoder)
                r0 = entries[0]["r2"]
                kept = [(p[0], p[1], p[3] / r0["view:" + p[0].split(":")[1] if p[0].startswith("block") else p[0]]) for p in pairs
                        if (r0.get("view:" + p[0].split(":")[1] if p[0].startswith("block") else p[0]) or 0) > 0.1]
                v = res["spearman_mass_vs_latent_r2"][lab]
                v["n_kept"] = len(kept)
                v["rho_mass_vs_kept_ratio"] = float(spearmanr([k[1] for k in kept], [k[2] for k in kept])[0])
                v["kept_ratio"] = {k[0]: float(k[2]) for k in kept}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pixel_mass.json").write_text(json.dumps(res, indent=1))
    (out / "pixel_mass.md").write_text(report(res))
    return res


def report(r) -> str:
    L = [f"# Pixel mass of Craftax-Classic content\n",
         f"`analysis/pixel_mass.py` on `{Path(r['file']).name}`: {r['episodes']} random episodes (seed {r['seed']}), {r['frames']:,} frames, {r['seconds']} s. "
         "Shares are of the total pixel variance of the frame (what a decoder that predicts the mean image gets wrong).\n",
         "## Plain frames\n",
         f"- map rows {100 * r['map_share_of_variance']:.0f}% of the variance, inventory strip {100 * r['strip_share_of_variance']:.0f}%.",
         f"- all block and mob classes together explain {100 * r['between_class_share']:.0f}%; daylight (linear, per pixel) about {100 * r['daylight_linear_share_of_variance']:.0f}%; the rest is within-class texture and noise.\n",
         "| class | tile share | share of the pixel variance |", "|---|---|---|"]
    for k, v in r["class_mass"].items():
        ts = r["class_tile_share"].get(k)
        L.append(f"| {k} | {'-' if ts is None else f'{100 * ts:.2f}%'} | {100 * v:.2f}% |")
    L += ["", "## Differences x(t+1) - x(t)\n",
          f"- the player moves (the map scrolls) in {100 * r['steps_moved_share']:.0f}% of the steps and those steps carry {100 * r['plain_diff_energy_share_on_moved_steps']:.0f}% of the plain difference energy "
          f"(per step: moved {r['plain_diff_energy_per_step']['moved']:.3g}, still {r['plain_diff_energy_per_step']['still']:.3g}).",
          f"- compensating the scroll leaves {100 * r['aligned_over_plain_energy']:.0f}% of the plain difference energy. What remains, by tile:",
          "", "| aligned difference | share |", "|---|---|"]
    for k, v in r["aligned_diff_share"].items():
        L.append(f"| {dict(mob='tiles with a mob in either frame', block='tiles whose block changed', player='the player tile (facing, sleeping)', rest='rest: daylight, noise, animation')[k]} | {100 * v:.1f}% |")
    if "spearman_mass_vs_latent_r2" in r:
        L += ["", "## Mass against what a trained latent kept\n",
              "Spearman rank correlation, per block or mob class, between its pixel mass (or tile share) and the ridge R^2 of its readout (`view:<block>`, `mob:<mob>`) from the latent at the given step of a race trace.\n",
              "`kept` = the R^2 at that step over the same run's R^2 at step 0 (untrained encoder), for classes with R^2 above 0.1 at step 0: what training removed, not what random features already carry.\n",
              "| trace | step | classes | rho (mass) | rho (tile share) | rho (mass vs kept) |", "|---|---|---|---|---|---|"]
        for lab, v in r["spearman_mass_vs_latent_r2"].items():
            kept = f"{v['rho_mass_vs_kept_ratio']:.2f} (n={v['n_kept']})" if "rho_mass_vs_kept_ratio" in v else "-"
            L.append(f"| {lab} | {v['step']} | {v['n']} | {v['rho_mass']:.2f} | {v['rho_tile_share']:.2f} | {kept} |")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("reports/pixel_mass"))
    p.add_argument("--traces", nargs="*", default=[], help="LABEL=path/to/trace.json[:step] (race traces); step defaults to 2000")
    a = p.parse_args()
    traces = {}
    for spec in a.traces:
        lab, rest = spec.split("=", 1)
        path, _, step = rest.partition(":")
        traces[lab] = (path, int(step or 2000))
    run(a.file, a.episodes, a.seed, a.out, traces)
    print((a.out / "pixel_mass.md").read_text())


if __name__ == "__main__":
    main()
