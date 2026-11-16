"""What an egocentric crop does to the pixel mass of every class, and how much of each class it leaves out of view.

    python analysis/crop_mass.py --file <craftax_expert_val.h5> --D 192 --kappa 1.5 --out reports/crop_mass

The map is 7 rows x 9 columns of 7-pixel tiles with the player at the centre. A crop of r x c tiles around the player keeps only those tiles; the mass of
class k in the crop is its cost (as in `pixel_mass.py`) over the pixel variance of the cropped tiles. The inventory strip is left out here. `coverage` is
the share of the class's tile instances that stay inside the crop: what the crop costs in content. Labels (the simulator's `state`) are used for the classes,
as in the validation part of `mass_weights.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mass_weights import CENTER, LABELS, R, C, T7, VIEW, label_of, tiles_of

CROPS = {"full 7x9": (3, 4), "7x7": (3, 3), "5x7": (2, 3), "5x5": (2, 2), "3x5": (1, 2), "3x3": (1, 1)}  # half sizes (rows, columns)


def run(path, D, kappa, out, seed=0, day=0.0):
    import h5py

    t0 = time.time()
    f = h5py.File(path, "r")
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    pix = (R + 2) * T7
    sx, n = np.zeros((pix, pix, 3)), 0
    pos_sum = np.zeros((VIEW, T7 * T7 * 3))
    lab_sum, lab_n = np.zeros((len(LABELS), T7 * T7 * 3)), np.zeros(len(LABELS))
    notc = np.arange(VIEW) != CENTER
    for ep in range(len(ln)):
        o, m = int(off[ep]), int(ln[ep])
        keep = f["daylight"][o:o + m, 0] >= day
        x = f["pixels"][o:o + m][keep]
        sx += x.astype(np.float64).sum(0)
        n += len(x)
        t = tiles_of(x)
        lab = label_of(f["state"][o:o + m][keep])
        pos_sum += t.sum(0)
        for j in range(len(LABELS)):
            s = (lab == j) & notc[None]
            if s.any():
                lab_sum[j] += t[s].sum(0)
                lab_n[j] += s.sum()
    mu_tiles = tiles_of(np.round(sx / n)[None].astype(np.uint8))[0]  # the mean image as tiles (63, 147)
    mu_tiles = tiles_of((sx / n)[None].astype(np.float32).clip(0, 255).astype(np.uint8))[0].astype(np.float64)
    mu_pos = pos_sum / n
    lab_c = lab_sum / np.maximum(lab_n, 1)[:, None]
    counts = np.zeros((len(LABELS), VIEW))
    tile_ss = np.zeros(VIEW)
    for ep in range(len(ln)):
        o, m = int(off[ep]), int(ln[ep])
        keep = f["daylight"][o:o + m, 0] >= day
        t = tiles_of(f["pixels"][o:o + m][keep]).astype(np.float64)
        lab = label_of(f["state"][o:o + m][keep])
        tile_ss += ((t - mu_tiles) ** 2).sum(axis=(0, 2))
        for j in range(len(LABELS)):
            counts[j] += ((lab == j) & notc[None]).sum(0)
    cost = ((lab_c[:, None, :] - mu_pos[None]) ** 2).sum(-1)  # (labels, positions)
    rows, cols = np.divmod(np.arange(VIEW), C)
    res = {"file": str(path), "min_daylight": day, "frames": int(n), "D": D, "kappa": kappa, "tau": kappa / D, "seconds": 0, "crops": {}}
    for name, (hr, hc) in CROPS.items():
        inside = (np.abs(rows - R // 2) <= hr) & (np.abs(cols - C // 2) <= hc)
        mass = (counts[:, inside] * cost[:, inside]).sum(1) / tile_ss[inside].sum()
        cover = counts[:, inside].sum(1) / np.maximum(counts.sum(1), 1)
        res["crops"][name] = {"tiles": int(inside.sum()), "mass": {LABELS[j]: float(mass[j]) for j in range(len(LABELS)) if counts[j].sum() > 0},
                              "coverage": {LABELS[j]: float(cover[j]) for j in range(len(LABELS)) if counts[j].sum() > 0}}
    res["seconds"] = round(time.time() - t0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "crop_mass.json").write_text(json.dumps(res, indent=1))
    (out / "crop_mass.md").write_text(report(res))
    return res


def report(r) -> str:
    tau = r["tau"]
    names = list(r["crops"])
    classes = [k for k in r["crops"][names[0]]["mass"]]
    order = sorted(classes, key=lambda k: -r["crops"][names[0]]["mass"][k])
    L = ["# Crop and pixel mass\n",
         f"`analysis/crop_mass.py` on `{Path(r['file']).name}`, {r['frames']:,} frames (daylight >= {r.get('min_daylight', 0)}), {r['seconds']} s. Mass = share of the cropped tiles' pixel variance (inventory strip left out); "
         f"threshold kappa/D = {100 * tau:.2f}% (D = {r['D']}, kappa = {r['kappa']}). `cov` = share of the class's tiles that stay in the crop.\n",
         "| class | " + " | ".join(f"{n} ({r['crops'][n]['tiles']} tiles)" for n in names) + " |", "|---|" + "---|" * len(names)]
    for k in order:
        cells = []
        for n in names:
            m, c = r["crops"][n]["mass"][k], r["crops"][n]["coverage"][k]
            cells.append(f"{100 * m:.2f}%{'*' if m >= tau else ''} (cov {100 * c:.0f}%)")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("| **classes at or above the threshold** | " + " | ".join(str(sum(r["crops"][n]["mass"][k] >= tau for k in order)) + f" of {len(order)}" for n in names) + " |")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, type=Path)
    p.add_argument("--D", type=int, default=192)
    p.add_argument("--kappa", type=float, default=1.5)
    p.add_argument("--day", type=float, default=0.0, help="keep only frames with at least this daylight (0 = all)")
    p.add_argument("--out", type=Path, default=Path("reports/crop_mass"))
    a = p.parse_args()
    run(a.file, a.D, a.kappa, a.out, day=a.day)
    print((a.out / "crop_mass.md").read_text())


if __name__ == "__main__":
    main()
