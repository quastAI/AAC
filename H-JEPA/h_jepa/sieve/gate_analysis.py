"""What stage 2's gate would prune, read from a checkpoint's own decoder and predictor (no training, no new model).

The gate of stage 2 is exp(-e / tau) per patch, e the error of the decoded predicted latent against the real next
frame. A patch whose decoded prediction misses by many tau gets no reconstruction pressure: the encoder is free to drop
it. This module computes, on the held-out frames of the metrics sample, that error for every patch and groups it by what
the patch shows (the label comes from the `state` column, read only here): the block type, a mob on it, the player's own
tile, the inventory strip; by whether the tile was already in view one step earlier, changed, or entered the view at
the edge (the player moved); and by daylight. For every group:

    n               patches in the group
    own_ok          share whose decoder error from the true latent (`e_real`) is under tau: the ceiling, a patch the
                    decoder cannot draw from the real latent cannot pass the gate whatever the predictor does
    sharp           share whose decoded prediction (`e_pred`) is under tau (the gate's own `sharp_patch_share`)
    gate@k          mean of exp(-e_pred / (k tau)) for k = 1, 2, 4, 8, 16: how much of the reconstruction the gate
                    lets through at a wider tau (a gate that stays shut at 16 tau is not a matter of tau)
    median_e/tau    the median prediction error in units of tau

`--stage2-run` also reads the probe scores of that run's first and last epoch and correlates the gate of the parent
with what stage 2 lost (a rank correlation over a dozen entities: a coarse check of the reading, not a test).

    python -m sieve.gate_analysis --parent <stage-1 *_epoch_N_object.ckpt> [--stage2-run <run>] --train-file --val-file
                                  --out <dir> [--size quick]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from probing import targets as tg

KS = (1, 2, 4, 8, 16)
MIN_PATCHES = 30  # a group with fewer patches is marked: its shares are noise
NUM_STRATA = tg.NUM_BLOCKS + tg.NUM_MOBS
PLAYER, STRIP = NUM_STRATA, NUM_STRATA + 1
STRATUM_NAMES = list(tg.BLOCK_NAMES) + list(tg.MOB_NAMES) + ["player_tile", "inventory_strip"]
KINDS = ("entering", "changed", "unchanged")  # of a patch between the frame and the next one
KEY_STRATA = ["coal", "iron", "diamond", "furnace", "crafting_table", "water", "tree", "zombie", "cow", "skeleton", "arrow"]
PROBE_OF = {  # the probe of the report -> the stratum it reads
    "mob_zombie": "zombie", "mob_cow": "cow", "mob_skeleton": "skeleton", "mob_arrow": "arrow",
    "map_coal": "coal", "map_iron": "iron", "map_diamond": "diamond", "map_lava": "lava", "map_table": "crafting_table",
    "map_furnace": "furnace", "map_water": "water", "map_tree": "tree",
}  # fmt: skip


def labels(part, t: np.ndarray, t1: np.ndarray) -> dict:
    """What every patch of the next frame shows and how it got there, for the transitions (t, t1) of a sample part.

    `stratum` (Np, 81): block id, NUM_BLOCKS + mob type where a mob stands, PLAYER for the player's tile, STRIP for the
    inventory strip. `kind` (Np, 81): 0 entering (the tile was outside the view a step earlier: the player moved),
    1 changed (another block or mob than the same world cell had a step earlier; strip: items or vitals changed),
    2 unchanged. The player has moved when the executed action is a move and the tile it enters was walkable and free
    (the probe suite's `can_move`); `shift_agreement` is the share of tiles of the overlap whose block is the same in
    both frames under that shift, a check of that rule (about 1 where the world does not change).
    """
    state = torch.from_numpy(part.state)
    blocks, mobs = tg.tile_maps(state)
    strata = tg.tile_strata(blocks, mobs)
    tt, t1t = torch.from_numpy(np.asarray(t)).long(), torch.from_numpy(np.asarray(t1)).long()
    can_move = tg.view_concepts(blocks[tt], mobs[tt])["can_move"].bool()  # (Np, 4): left, right, up, down
    action = torch.from_numpy(np.nan_to_num(part.action[t], nan=0.0)).long()
    k = (action - 1).clamp(0, 3)
    moved = (action >= 1) & (action <= 4) & can_move.gather(1, k[:, None])[:, 0]
    step = torch.tensor(tg.MOVE_STEPS)[k] * moved[:, None].long()  # (Np, 2)
    rows = (torch.arange(tg.VIEW_TILES) // tg.VIEW_COLS)[None]
    cols = (torch.arange(tg.VIEW_TILES) % tg.VIEW_COLS)[None]
    old_r, old_c = rows + step[:, :1], cols + step[:, 1:]
    inside = (old_r >= 0) & (old_r < tg.VIEW_ROWS) & (old_c >= 0) & (old_c < tg.VIEW_COLS)
    old = old_r.clamp(0, tg.VIEW_ROWS - 1) * tg.VIEW_COLS + old_c.clamp(0, tg.VIEW_COLS - 1)
    changed = strata[tt].gather(1, old) != strata[t1t]
    tile_kind = torch.where(~inside, 0, torch.where(changed, 1, 2))
    tile_stratum = strata[t1t].clone()
    player = tg.PLAYER_TILE[0] * tg.VIEW_COLS + tg.PLAYER_TILE[1]
    tile_stratum[:, player] = PLAYER
    tile_kind[:, player] = 2
    strip_changed = (
        torch.from_numpy(part.inventory[t] != part.inventory[t1]).any(-1)
        | torch.from_numpy(part.vitals[t] != part.vitals[t1]).any(-1)
    )
    strip_kind = torch.where(strip_changed, 1, 2)[:, None].expand(len(tt), tg.STRIP_PATCHES)
    stratum = torch.cat([tile_stratum, torch.full((len(tt), tg.STRIP_PATCHES), STRIP)], 1)
    kind = torch.cat([tile_kind, strip_kind], 1)
    overlap = moved[:, None] & inside
    same = (blocks[tt].gather(1, old) == blocks[t1t]) & overlap
    agreement = float(same.sum() / overlap.sum().clamp(min=1)) if bool(overlap.any()) else float("nan")
    return dict(stratum=stratum.numpy(), kind=kind.numpy(), moved=moved.numpy(), shift_agreement=agreement)


def summarise(e_pred: np.ndarray, e_real: np.ndarray, tau: float, e_diff: np.ndarray | None = None,
              joint_scale: float | None = None, floor: float = 0.1, beta: float = 1.0) -> dict:
    """The numbers of one group of patches (1-d arrays of errors). With `e_diff` also the two gates that leave the
    decoder's precision out: `rel_*` gates on what the prediction loses against the latent of the real next frame,
    max(e_pred - e_real, 0); `diff_*` on the decoded prediction against the decoded latent of the next frame (no pixels).
    With `joint_scale` also `joint_gate`, the one-stage Sieve's gate (V5): floor + (1 - floor) exp(-e_diff / (beta scale)),
    the scale the mean decoded difference of the whole sample (the training uses each batch's), and `joint_shut`, the
    share near the floor."""
    n = int(e_pred.size)
    if n == 0:
        return dict(n=0)
    out = dict(
        n=n,
        own_ok=float((e_real < tau).mean()),
        sharp=float((e_pred < tau).mean()),
        median_e_over_tau=float(np.median(e_pred) / tau),
        own_median_e_over_tau=float(np.median(e_real) / tau),
    )
    for k in KS:
        out[f"gate@{k}"] = float(np.exp(-e_pred / (k * tau)).mean())
    if e_diff is not None:
        excess = np.maximum(e_pred - e_real, 0.0)
        out["rel_sharp"] = float((excess < tau).mean())
        out["diff_sharp"] = float((e_diff < tau).mean())
        for k in KS:
            out[f"rel_gate@{k}"] = float(np.exp(-excess / (k * tau)).mean())
            out[f"diff_gate@{k}"] = float(np.exp(-e_diff / (k * tau)).mean())
        if joint_scale:
            joint = floor + (1.0 - floor) * np.exp(-e_diff / (beta * joint_scale))
            out["joint_gate"] = float(joint.mean())
            out["joint_shut"] = float((joint < floor + 0.05 * (1.0 - floor)).mean())
    return out


@torch.no_grad()
def decoded_difference(decoder, a: np.ndarray, b: np.ndarray, device, chunk: int = 1024) -> np.ndarray:
    """Patch errors (N, 81) between the decoded latents `a` and `b` (the same decoder draws both)."""
    from .decoder import patch_errors

    out = []
    for i in range(0, len(a), chunk):
        da = decoder(torch.from_numpy(a[i : i + chunk]).to(device))
        db = decoder(torch.from_numpy(b[i : i + chunk]).to(device))
        out.append(patch_errors(da, db).cpu())
    return torch.cat(out).numpy()


def analyse(model, sample, device="cpu", label: str = "model", floor: float = 0.1, beta: float = 1.0) -> dict:
    """The gate analysis of one saved `HJEPA` on the held-out frames of `sample`."""
    from . import check

    t0 = time.time()
    model = model.to(device)
    model.eval()
    jepa = model.get_level(1)
    decoder = getattr(jepa, "decoder", None)
    if decoder is None or not bool(torch.isfinite(decoder.tau)):
        raise ValueError(f"{label}: no decoder with a tau in this checkpoint (a Sieve model of V4 has one)")
    tau = float(decoder.tau)
    va = sample.val
    t, t1 = va.pairs()
    z = check.encode(jepa, va.pixels, device)
    pred = check.predict(jepa, z[t], va.action[t], device)
    e_pred = check.decode_errors(decoder, pred, va.pixels[t1], device)  # (Np, 81)
    e_real = check.decode_errors(decoder, z[t1], va.pixels[t1], device)
    e_diff = decoded_difference(decoder, pred, z[t1], device)
    lab = labels(va, t, t1)
    stratum, kind = lab["stratum"], lab["kind"]
    is_tile = stratum < NUM_STRATA  # the map tiles with a block or a mob, not the player or the strip

    scale = float(e_diff.mean())

    def group(mask):
        return summarise(e_pred[mask], e_real[mask], tau, e_diff[mask], scale, floor, beta)

    out = dict(label=label, tau=tau, size=dict(transitions=int(len(t)), patches=int(e_pred.size)),
               overall=group(np.ones_like(is_tile)), moved_share=float(lab["moved"].mean()),
               shift_agreement=lab["shift_agreement"])  # fmt: skip
    out["by_stratum"] = {STRATUM_NAMES[s]: group(stratum == s) for s in range(len(STRATUM_NAMES)) if (stratum == s).any()}
    out["by_kind"] = {KINDS[k]: group(is_tile & (kind == k)) for k in range(3)}
    out["strip_by_kind"] = {KINDS[k]: group((stratum == STRIP) & (kind == k)) for k in (1, 2)}
    out["stratum_by_kind"] = {
        name: {KINDS[k]: group((stratum == STRATUM_NAMES.index(name)) & (kind == k)) for k in range(3)}
        for name in KEY_STRATA if name in STRATUM_NAMES
    }  # fmt: skip
    day = np.repeat(va.daylight[t1][:, None], e_pred.shape[1], 1)
    edges = np.quantile(va.daylight[t1], [0.25, 0.5, 0.75])
    bins = np.digitize(day, edges)
    out["by_daylight_quartile"] = {f"q{b + 1}": group(bins == b) | dict(daylight_mean=float(day[bins == b].mean()))
                                   for b in range(4) if (bins == b).any()}  # fmt: skip
    out["seconds"] = time.time() - t0
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for v in np.unique(x):  # ties get their mean rank
        ranks[x == v] = ranks[x == v].mean()
    return ranks


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def stage2_loss(run) -> dict:
    """The change of every probe in a stage-2 run, last epoch minus its first (the parent): {probe: (first, last, epoch)}."""
    found = {}
    for p in Path(run, "sieve").glob("epoch_*.json"):
        if m := re.fullmatch(r"epoch_(\d+)\.json", p.name):
            found[int(m.group(1))] = json.loads(p.read_text())
    if len(found) < 2:
        return {}
    first, last = found[min(found)], found[max(found)]
    return {name: (first[f"holds_{name}"], last[f"holds_{name}"], max(found)) for name in PROBE_OF
            if isinstance(first.get(f"holds_{name}"), float) and isinstance(last.get(f"holds_{name}"), float)}  # fmt: skip


def compare(parent: dict, loss: dict) -> dict:
    """The gate of the parent per entity next to what stage 2 did to its probe, and the rank correlation."""
    rows = {}
    for name, (first, last, epoch) in loss.items():
        s = parent["by_stratum"].get(PROBE_OF[name])
        if s and s.get("n"):
            rows[name] = dict(stratum=PROBE_OF[name], n=s["n"], sharp=s["sharp"], gate1=s["gate@1"], own_ok=s["own_ok"],
                              first=first, last=last, change=last - first, epoch=epoch)  # fmt: skip
    if len(rows) >= 3:
        change = [r["change"] for r in rows.values()]
        rho = dict(sharp=spearman([r["sharp"] for r in rows.values()], change),
                   gate1=spearman([r["gate1"] for r in rows.values()], change), entities=len(rows))  # fmt: skip
    else:
        rho = {}
    return dict(rows=rows, rho=rho)


def _f(v, digits=2):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}f}"


def _row(name, g) -> str:
    if not g.get("n"):
        return f"| {name} | 0 |" + " - |" * (3 + len(KS))
    flag = " (few)" if g["n"] < MIN_PATCHES else ""
    gates = " | ".join(_f(g[f"gate@{k}"]) for k in KS)
    return (f"| {name}{flag} | {g['n']} | {_f(g['own_ok'])} | {_f(g['sharp'])} | {gates} | {_f(g['median_e_over_tau'], 1)} |")


HEADER = "| {} | n | own_ok | sharp | " + " | ".join(f"gate@{k}" for k in KS) + " | median e/tau |\n|---|---|---|---|" + "---|" * (len(KS) + 1)
GATES3 = ("", "rel_", "diff_")
HEADER3 = ("| {} | n | sharp abs | rel | diff | gate@1 abs | rel | diff | gate@4 abs | rel | diff | one-stage gate | near floor |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|")  # fmt: skip


def _row3(name, g) -> str:
    if not g.get("n") or "rel_sharp" not in g:
        return f"| {name} | {g.get('n', 0)} |" + " - |" * 11
    flag = " (few)" if g["n"] < MIN_PATCHES else ""
    cells = [_f(g[f"{p}sharp"]) for p in GATES3] + [_f(g[f"{p}gate@{k}"]) for k in (1, 4) for p in GATES3]
    cells += [_f(g.get("joint_gate")), _f(g.get("joint_shut"))]
    return f"| {name}{flag} | {g['n']} | " + " | ".join(cells) + " |"


def render_gates(result: dict) -> str:
    """The three gates side by side: absolute (the training's), relative, decoded difference."""
    lines = [f"# Three gates: {result['label']}", "",
             "abs: exp(-e_pred / tau), the training's gate. rel: exp(-max(e_pred - e_real, 0) / tau), what the prediction "
             "loses against the decoded latent of the real next frame. diff: exp(-|D(p) - D(z_next)|^2 / tau), the decoded "
             "prediction against the decoded latent of the next frame, no pixels. sharp = the share under tau. All three "
             "come from frames only; the labels only group the patches. one-stage gate: the V5 gate (floor + (1 - floor) "
             "exp(-diff / (beta mean diff)), the mean over the whole sample); near floor: its share of patches near the floor.", "",
             "## By what the patch shows", "", HEADER3.format("stratum")]  # fmt: skip
    order = sorted(result["by_stratum"], key=lambda n: -result["by_stratum"][n]["n"])
    lines += [_row3(n, result["by_stratum"][n]) for n in order]
    lines += ["", "## Map tiles by how the tile got there", "", HEADER3.format("kind")]
    lines += [_row3(n, g) for n, g in result["by_kind"].items()]
    lines += [_row3(f"strip, {n}", g) for n, g in result["strip_by_kind"].items()]
    lines += ["", "## The rare things, unchanged tiles only (already in view, same content)", "", HEADER3.format("stratum")]
    lines += [_row3(n, kinds["unchanged"]) for n, kinds in result["stratum_by_kind"].items()]
    lines += ["", "## By daylight", "", HEADER3.format("quartile")]
    lines += [_row3(f"{q} (daylight {g['daylight_mean']:.2f})", g) for q, g in result["by_daylight_quartile"].items()]
    return "\n".join(lines) + "\n"


def render(result: dict, comparison: dict | None = None) -> str:
    o = result["overall"]
    lines = [
        f"# What the gate prunes: {result['label']}",
        "",
        f"tau {result['tau']:.5f}; {result['size']['transitions']} held-out transitions, {result['size']['patches']} patches. "
        f"Overall: own_ok {_f(o['own_ok'])} (the decoder's error from the true latent under tau), sharp {_f(o['sharp'])} "
        f"(the decoded prediction under tau), mean gate {_f(o['gate@1'])} at tau, {_f(o['gate@4'])} at 4 tau, {_f(o['gate@16'])} at 16 tau. "
        f"The player moved in {result['moved_share']:.0%} of the transitions; shift agreement {_f(result['shift_agreement'])} "
        "(blocks that agree across the shift of a move: should be near 1).",
        "",
        "`own_ok` is the ceiling: a patch the decoder cannot draw from the true latent cannot pass the gate. `gate@k` is the "
        "mean gate at k tau. Groups marked (few) have under 30 patches.",
        "",
        "## By what the patch shows (patches of the next frame)",
        "",
        HEADER.format("stratum"),
    ]
    order = sorted(result["by_stratum"], key=lambda n: -result["by_stratum"][n]["n"])
    lines += [_row(n, result["by_stratum"][n]) for n in order]
    lines += ["", "## Map tiles by how the tile got there", "", HEADER.format("kind")]
    lines += [_row(n, g) for n, g in result["by_kind"].items()]
    lines += [_row(f"strip, {n}", g) for n, g in result["strip_by_kind"].items()]
    lines += ["", "## The rare things by how the tile got there (sharp share, n)", "",
              "| stratum | entering | changed | unchanged |", "|---|---|---|---|"]  # fmt: skip
    for name, kinds in result["stratum_by_kind"].items():
        cells = [f"{_f(kinds[k].get('sharp'))} ({kinds[k]['n']})" if kinds[k].get("n") else "- (0)" for k in KINDS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## By daylight (quartiles of the next frame's daylight value)", "", HEADER.format("quartile")]
    lines += [_row(f"{q} (daylight {g['daylight_mean']:.2f})", g) for q, g in result["by_daylight_quartile"].items()]
    if comparison and comparison["rows"]:
        lines += ["", "## What stage 2 did to the same entities (the parent's gate against the probe's change)", "",
                  "| probe | stratum | n | sharp | gate@1 | own_ok | holds first | holds last | change |",
                  "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for name, r in sorted(comparison["rows"].items(), key=lambda kv: kv[1]["sharp"]):
            lines.append(f"| {name} | {r['stratum']} | {r['n']} | {_f(r['sharp'])} | {_f(r['gate1'])} | {_f(r['own_ok'])} | "
                         f"{_f(r['first'], 3)} | {_f(r['last'], 3)} | {r['change']:+.3f} |")  # fmt: skip
        rho = comparison["rho"]
        if rho:
            lines += ["", f"Rank correlation of the parent's gate with the probe's change over {rho['entities']} entities: "
                      f"sharp {_f(rho['sharp'])}, gate@1 {_f(rho['gate1'])} (positive: the entities the gate keeps shut "
                      "are the ones that lost most; a dozen entities, a coarse check and not a test)."]  # fmt: skip
    return "\n".join(lines) + "\n"


def last_epoch_checkpoint(run) -> Path | None:
    found = {int(m.group(1)): p for p in Path(run).glob("*_object.ckpt") if (m := re.search(r"_epoch_(\d+)_object\.ckpt$", p.name))}
    return found[max(found)] if found else None


def main(argv=None):
    from . import sample as sm
    from .watch import load_model

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parent", type=Path, required=True, help="the stage-1 epoch checkpoint stage 2 forked from")
    ap.add_argument("--stage2-run", type=Path, help="the stage-2 run: its last epoch is analysed too and its probes compared")
    ap.add_argument("--parent-only", action="store_true", help="with --stage2-run: compare its probes, skip its model")
    ap.add_argument("--train-file", type=Path, help="needed for the cached metrics sample (not with --val-episodes)")
    ap.add_argument("--val-file", type=Path, required=True)
    ap.add_argument("--val-episodes", type=int, help="read only this many held-out episodes (4 windows x 32 frames each), no cache")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", default="full", choices=["full", "quick"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--gate-floor", type=float, default=0.1, help="the one-stage gate's floor (V5)")
    ap.add_argument("--gate-beta", type=float, default=1.0, help="the one-stage gate's beta (V5)")
    args = ap.parse_args(argv)

    from cpu_threads import limit_threads

    limit_threads()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.val_episodes:
        import types

        sample = types.SimpleNamespace(val=sm._read(args.val_file, args.val_episodes, 4, 32, 1))  # the val part's seed
    else:
        if args.train_file is None:
            ap.error("--train-file is needed without --val-episodes")
        sample = sm.load(args.train_file, args.val_file, size=args.size)
    print(f"[gate] sample {len(sample.val)} held-out frames, {device}", flush=True)
    gate = dict(floor=args.gate_floor, beta=args.gate_beta)
    label = f"{args.parent.name} (stage-1 parent)" if args.stage2_run else args.parent.name
    results = [analyse(load_model(args.parent), sample, device, label=label, **gate)]
    comparison = None
    if args.stage2_run:
        last = last_epoch_checkpoint(args.stage2_run)
        if last is not None and not args.parent_only:
            results.append(analyse(load_model(last), sample, device, label=f"{last.name} (end of stage 2)", **gate))
        comparison = compare(results[0], stage2_loss(args.stage2_run))
    text = "\n".join(render(r, comparison if i == 0 else None) for i, r in enumerate(results))
    text += "\n" + "\n".join(render_gates(r) for r in results)
    (args.out / "gate_analysis.md").write_text(text)
    (args.out / "gate_analysis.json").write_text(json.dumps(dict(results=results, comparison=comparison), indent=1))
    print(f"[gate] done in {sum(r['seconds'] for r in results):.0f} s, written to {args.out / 'gate_analysis.md'}", flush=True)


if __name__ == "__main__":
    main()
