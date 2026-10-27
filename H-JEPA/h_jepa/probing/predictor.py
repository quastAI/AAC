"""Block B: what the predictor does with what the latent holds.

- change test: open-loop rollouts with the true actions, scored only where the entity changed (copying the last latent
  is wrong there by construction) and, apart, where it did not (a predictor must not invent changes);
- events: one step after place, make and collect (`do` on the faced block), successful and failed presses apart, the
  inventory change and the faced tile's change (mined stone turns into path, a placed table appears);
- mobs: does a one-step prediction move a mob that was already there (arrows fly straight), against copying?
- monotonicity: along the expert's own path, does the latent distance to a goal frame fall (encoded, and predicted with
  the expert's actions, the planner's cost)?
- loss attribution: which transitions carry the one-step prediction loss;
- the inverse dynamics per action.
"""

from __future__ import annotations

import numpy as np
import torch

import health

from . import readout as ro
from . import targets as tg

HORIZONS = (1, 2, 4, 8, 16, 32, 64)
MIN_EVENTS = 10  # fewer successful presses: no event score
RECOVER_MIN_GAP = 0.05  # the encoded latent must beat copying by this much on changed windows for `recovered`
TOLERANCE = {"daylight": 0.05}  # a level counts as changed / right within this; counts within 0.5 (rounding)


def view(split: dict, x: torch.Tensor) -> torch.Tensor:
    return x.reshape(split["clips"], split["T"], *x.shape[1:])


@torch.no_grad()
def open_loop(
    jepa, emb: torch.Tensor, act: torch.Tensor, history: int, steps: int, device, chunk: int = 512
) -> torch.Tensor:
    """Open-loop predictions of `steps` frames after the `history` context latents of each window, with the true
    actions: emb (N, history + steps, D) (only the first `history` are used), act (N, history + steps, A) ->
    (N, steps, D). The action of frame t-1 leads to frame t."""
    out = []
    with health.eval_mode(jepa.predictor):
        for a in range(0, len(emb), chunk):
            roll = emb[a : a + chunk, :history].to(device)
            u = act[a : a + chunk].to(device)
            preds = []
            for k in range(1, steps + 1):
                t = history + k - 1
                step = jepa.predict(roll[:, -history:], u[:, t - history : t])[:, -1:]
                roll = torch.cat([roll, step], dim=1)
                preds.append(step[:, 0].float().cpu())
            out.append(torch.stack(preds, 1))
    return torch.cat(out) if out else torch.zeros(0, steps, emb.shape[-1])


def windows(split: dict, history: int, length: int, every: int):
    """Window starts every `every` frames of each clip: (clip, first frame) of windows of `history + length` frames."""
    T = split["T"]
    starts = list(range(0, T - history - length + 1, every))
    if length < 1 or not starts:
        return None, None
    clip = torch.arange(split["clips"]).repeat(len(starts))
    first = torch.tensor(starts).repeat_interleave(split["clips"])
    return clip, first


def _gather(x: torch.Tensor, clip: torch.Tensor, first: torch.Tensor, length: int) -> torch.Tensor:
    """x (clips, T, ...) -> (windows, length, ...) starting at `first`."""
    idx = first[:, None] + torch.arange(length)[None]
    return x[clip[:, None], idx]


# --------------------------------------------------------------------------
# change test


def _correct(spec: dict, name: str, value: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    if spec["kind"] == "cls":
        return value.argmax(1, keepdim=True) == truth.long()
    if spec["kind"] == "bin":
        return (value > 0) == (truth > 0.5)
    return (value - truth).abs() < TOLERANCE.get(name, 0.5)


def _changed(spec: dict, name: str, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    if spec["kind"] in ("cls", "bin"):
        return before != after
    return (after - before).abs() > TOLERANCE.get(name, 0.5)


def change_scores(spec: dict, name: str, pred, copy, enc, before, after) -> dict:
    """Per column and pooled: accuracy of the predicted, copied and encoded readout on the windows where the entity
    changed and where it did not; `recovered` = (predicted - copy) / (encoded - copy) on changed windows, the share of
    the encoded latent's advantage over copying that the prediction gets; `shown` (counts and levels) = the predicted
    signed change as a share of the encoded one."""
    changed = _changed(spec, name, before, after)
    ok = {src: _correct(spec, name, v, after) for src, v in (("pred", pred), ("copy", copy), ("enc", enc))}
    names = spec["names"] if spec["kind"] != "cls" else spec["names"][:1]
    columns = {}
    for j, col in enumerate(names):
        columns[col] = _change_entry(changed[:, j], {s: o[:, j] for s, o in ok.items()})
        if spec["kind"] == "reg":
            sign = torch.sign(after[:, j] - before[:, j])[changed[:, j]]
            enc_move = ((enc[:, j] - copy[:, j])[changed[:, j]] * sign).sum()
            pred_move = ((pred[:, j] - copy[:, j])[changed[:, j]] * sign).sum()
            columns[col]["shown"] = (
                float(pred_move / enc_move) if changed[:, j].sum() >= MIN_EVENTS and enc_move > 0 else None
            )
    pooled = _change_entry(changed.flatten(), {s: o.flatten() for s, o in ok.items()})
    return {"columns": columns, **pooled}


def _change_entry(changed: torch.Tensor, ok: dict) -> dict:
    n_c, n_s = int(changed.sum()), int((~changed).sum())
    entry = {"changed": n_c, "same": n_s}
    for s, o in ok.items():
        entry[f"{s}_changed"] = float(o[changed].float().mean()) if n_c else None
        entry[f"{s}_same"] = float(o[~changed].float().mean()) if n_s else None
    gap = None if not n_c else entry["enc_changed"] - entry["copy_changed"]
    entry["recovered"] = (
        (entry["pred_changed"] - entry["copy_changed"]) / gap if gap is not None and gap >= RECOVER_MIN_GAP else None
    )
    return entry


def change_test(
    fitted: dict,
    val: dict,
    jepa,
    *,
    history: int,
    horizons=HORIZONS,
    every: int = 8,
    device="cpu",
    boot: int = 0,
    seed: int = 0,
) -> dict:
    """The latent's ridge probes (fitted on encoded latents of the training worlds) applied to open-loop predictions
    k steps ahead on the held-out worlds, scored where the entity changed between the last context frame and frame k
    and where it did not; intervals of the pooled accuracy on changed windows (episode bootstrap)."""
    emb, act = view(val, val["frames"]["latent"]), view(val, val["act"])
    kmax = min(max(horizons), val["T"] - history)
    horizons = [k for k in horizons if k <= kmax]
    clip, first = windows(val, history, kmax, every)
    if clip is None:
        return {"history": history, "by_k": {}}
    pred = open_loop(
        jepa,
        _gather(emb, clip, first, history + kmax),
        _gather(act, clip, first, history + kmax),
        history,
        kmax,
        device,
    )
    last = first + history - 1
    copy = emb[clip, last]
    groups = val["clip_episode"][clip].numpy()
    by_k = {}
    for k in horizons:
        enc = emb[clip, last + k]
        out = {}
        for name, f in fitted.items():
            spec = tg.TARGETS[name]
            y = view(val, val["y"][name])
            fp, fc, fe = f["predict"](pred[:, k - 1]), f["predict"](copy), f["predict"](enc)
            before, after = y[clip, last], y[clip, last + k]
            entry = change_scores(spec, name, fp, fc, fe, before, after)
            if boot and entry["changed"]:
                changed = _changed(spec, name, before, after)
                right = _correct(spec, name, fp, after)

                def stat(idx, changed=changed, right=right):
                    c = changed[idx]
                    return float(right[idx][c].float().mean()) if c.any() else None

                entry["pred_changed_interval"] = ro.episode_bootstrap(stat, groups, boot, seed)
            out[name] = entry
        by_k[k] = out
    return {"history": history, "every": every, "windows": len(clip), "by_k": by_k}


# --------------------------------------------------------------------------
# events


def _event_names(val: dict, t0: int) -> tuple[list, torch.Tensor, torch.Tensor]:
    """Name of the event of every awake step (clip b, frame t) for t >= t0 (place/make by action; collect by `do` on
    the faced block), with the clip and frame indices."""
    action, sleeping = view(val, val["action"]), view(val, val["y"]["sleeping"][:, 0] > 0.5)
    blocks, facing = view(val, val["y"]["blocks"]), view(val, val["y"]["facing_raw"])
    T = val["T"]
    b, t = torch.meshgrid(torch.arange(val["clips"]), torch.arange(t0, T - 1), indexing="ij")
    b, t = b.flatten(), t.flatten()
    a = action[b, t]
    faced = blocks[b, t, tg.faced_index(facing[b, t])]
    names = []
    for ai, fb in zip(a.tolist(), faced.tolist()):
        if ai < 0:
            names.append(None)
        elif tg.ACTION_NAMES[ai] in tg.PLACE_MAKE:
            names.append(tg.ACTION_NAMES[ai])
        elif ai == tg.DO and tg.BLOCK_NAMES[fb] in tg.COLLECT:
            names.append(tg.COLLECT[tg.BLOCK_NAMES[fb]][0])
        else:
            names.append(None)
    keep = torch.tensor([n is not None for n in names], dtype=torch.bool) & ~sleeping[b, t]
    return [n for n, k in zip(names, keep.tolist()) if k], b[keep], t[keep]


EVENT_COLUMN = {**tg.PLACE_MAKE, **{v[0]: v[1:] for v in tg.COLLECT.values()}}  # event -> (column group, column, sign)


def event_group(name: str) -> str:
    return name.split("_")[0] if name.split("_")[0] in ("place", "make") else "collect"


@torch.no_grad()
def one_step(jepa, emb, act, b, t, history: int, device) -> torch.Tensor:
    """The predicted latent of frame t + 1 from the `history` latents up to frame t (teacher-forced, as in training)."""
    offsets = torch.arange(-history + 1, 1)
    ctx, ctx_act = emb[b[:, None], t[:, None] + offsets], act[b[:, None], t[:, None] + offsets]
    out = []
    with health.eval_mode(jepa.predictor):
        for a in range(0, len(ctx), 512):
            out.append(jepa.predict(ctx[a : a + 512].to(device), ctx_act[a : a + 512].to(device))[:, -1].float().cpu())
    return torch.cat(out) if out else torch.zeros(0, emb.shape[-1])


def event_probes(fitted: dict, val: dict, jepa, decoder, *, history: int, device="cpu") -> dict:
    """One-step predictions after the place, make and collect events (awake steps), successful and failed presses
    apart. For the column the event changes: the signed change of the latent's ridge probe from the latent before the
    step to the predicted one and to the encoded next one; `shown` is the predicted success-minus-failure gap over the
    encoded one (1 as a real frame, 0 no change predicted). Where the faced tile changes (mining, placing), the share of
    predicted, encoded-next and copied latents whose map decoder reads the new block at the faced tile."""
    names, b, t = _event_names(val, history - 1)
    if not names:
        return {}
    emb, act = view(val, val["frames"]["latent"]), view(val, val["act"])
    ys = {g: view(val, val["y"][g]) for g in ("inventory", "vitals")}
    blocks, facing = view(val, val["y"]["blocks"]), view(val, val["y"]["facing_raw"])
    pred = one_step(jepa, emb, act, b, t, history, device)
    before_z, after_z = emb[b, t], emb[b, t + 1]
    group = [EVENT_COLUMN[n][0] for n in names]
    column = torch.tensor([EVENT_COLUMN[n][1] for n in names])
    sign = torch.tensor([float(EVENT_COLUMN[n][2]) for n in names])
    rows = torch.arange(len(names))
    succeeded = torch.zeros(len(names), dtype=torch.bool)
    d_pred, d_enc = torch.zeros(len(names)), torch.zeros(len(names))
    for g in ("inventory", "vitals"):
        here = torch.tensor([x == g for x in group])
        if not here.any():
            continue
        probe = fitted[g]["predict"]
        col, sg, bb, tt = column[here], sign[here], b[here], t[here]
        r = torch.arange(int(here.sum()))
        start = probe(before_z[here])[r, col]
        d_pred[here] = sg * (probe(pred[here])[r, col] - start)
        d_enc[here] = sg * (probe(after_z[here])[r, col] - start)
        succeeded[here] = sg * (ys[g][bb, tt + 1, col] - ys[g][bb, tt, col]) > 0
    faced = tg.faced_index(facing[b, t])
    old, new = blocks[b, t, faced], blocks[b, t + 1, faced]
    tile_changed = new != old
    reads = {}
    if decoder is not None:
        for src, z in (("pred", pred), ("enc", after_z), ("copy", before_z)):
            reads[src] = decoder(z)["blocks"][rows, faced].argmax(-1) == new

    def entry(mask):
        s, f = mask & succeeded, mask & ~succeeded
        enc = d_enc[s].mean().item() if s.sum() >= MIN_EVENTS else None
        ps = d_pred[s].mean().item() if s.sum() >= MIN_EVENTS else None
        ef = d_enc[f].mean().item() if f.any() else None
        pf = d_pred[f].mean().item() if f.any() else None
        gap = None if enc is None else enc - (ef or 0.0)
        shown = None if gap is None or ps is None or gap < 0.1 else (ps - (pf or 0.0)) / gap
        e = {
            "succeeded": int(s.sum()),
            "failed": int(f.sum()),
            "encoded": enc,
            "encoded_failed": ef,
            "pred": ps,
            "pred_failed": pf,
            "shown": shown,
        }
        changed = s & tile_changed
        e["tile_changes"] = int(changed.sum())
        for src, r in reads.items():
            e[f"tile_{src}"] = float(r[changed].float().mean()) if changed.sum() >= MIN_EVENTS else None
        return e

    name_of = np.array(names)
    out = {n: entry(torch.from_numpy(name_of == n)) for n in dict.fromkeys(names)}
    for grp in tg.EVENT_GROUPS:
        mask = torch.tensor([event_group(n) == grp for n in names])
        if mask.any():
            out[f"{grp} (all)"] = entry(mask)
    return out


# --------------------------------------------------------------------------
# the faced tile


FACED_CAUSES = ("move", "turn", "blocked move", "faced tile changed", "unchanged")


def faced_causes(val: dict, b: torch.Tensor, t: torch.Tensor) -> tuple[list, torch.Tensor, torch.Tensor]:
    """For the transitions t -> t + 1 of clips b: what changed the faced tile (FACED_CAUSES), the true faced block at
    t + 1, and the oracle: the next faced block read from the true map at t (the tile it will be, if the previous view
    shows it), right or wrong. The oracle is what a latent holding the whole previous view could know; it fails where an
    event changes the tile (mined stone turns into path) or the tile enters from outside the view."""
    pos, facing = view(val, val["y"]["position"]), view(val, val["y"]["facing_raw"])
    blocks, action = view(val, val["y"]["blocks"]), view(val, val["action"])
    now = blocks[b, t, tg.faced_index(facing[b, t])]
    nxt = blocks[b, t + 1, tg.faced_index(facing[b, t + 1])]
    moved = (pos[b, t + 1] != pos[b, t]).any(-1)
    turned = facing[b, t + 1] != facing[b, t]
    a = action[b, t]
    is_move = (a >= tg.ACTION_NAMES.index("left")) & (a <= tg.ACTION_NAMES.index("down"))
    cause = [
        "move" if m else "turn" if tr else "blocked move" if mv else "faced tile changed" if n != o else "unchanged"
        for m, tr, mv, n, o in zip(moved.tolist(), turned.tolist(), is_move.tolist(), nxt.tolist(), now.tolist())
    ]
    step = torch.tensor([tg.FACING_STEP[k] for k in (1, 2, 3, 4)])[facing[b, t + 1].clamp(1, 4) - 1]
    world = pos[b, t + 1] + step  # the next faced tile in world coordinates (view row = position[0])
    row = world[:, 0] - pos[b, t, 0] + tg.PLAYER_TILE[0]
    col = world[:, 1] - pos[b, t, 1] + tg.PLAYER_TILE[1]
    inside = (row >= 0) & (row < tg.VIEW_ROWS) & (col >= 0) & (col < tg.VIEW_COLS)
    index = row.clamp(0, tg.VIEW_ROWS - 1) * tg.VIEW_COLS + col.clamp(0, tg.VIEW_COLS - 1)
    oracle = inside & (blocks[b, t, index] == nxt)
    return cause, nxt, oracle


def faced_by_cause(fitted: dict, val: dict, jepa, *, history: int, device="cpu") -> dict:
    """One step ahead, the faced block read by the latent's ridge probe from the predicted latent, the real next latent
    and the copied last latent, split by what changed it (faced_causes), next to the oracle that reads it from the true
    previous view. Where the real next latent is right and the prediction is not, the predictor misses what the view
    showed; where the oracle fails too, the tile could not be known from the view."""
    if "facing_block" not in fitted:
        return {}
    emb, act = view(val, val["frames"]["latent"]), view(val, val["act"])
    action = view(val, val["action"])
    b, t = torch.meshgrid(torch.arange(val["clips"]), torch.arange(history - 1, val["T"] - 1), indexing="ij")
    b, t = b.flatten(), t.flatten()
    keep = action[b, t] >= 0
    b, t = b[keep], t[keep]
    if not len(b):
        return {}
    cause, nxt, oracle = faced_causes(val, b, t)
    probe = fitted["facing_block"]["predict"]
    pred = one_step(jepa, emb, act, b, t, history, device)
    right = {src: probe(z).argmax(1) == nxt for src, z in (("pred", pred), ("enc", emb[b, t + 1]), ("copy", emb[b, t]))}
    cause = np.array(cause)
    out = {"transitions": len(b)}
    for name in ("all",) + FACED_CAUSES:
        mask = torch.ones(len(b), dtype=torch.bool) if name == "all" else torch.from_numpy(cause == name)
        if not mask.any():
            continue
        out[name] = {
            "transitions": int(mask.sum()),
            "share": float(mask.float().mean()),
            "oracle": float(oracle[mask].float().mean()),
            **{src: float(r[mask].float().mean()) for src, r in right.items()},
        }
    return out


SWAP_FACED = ("stone", "iron", "coal", "tree", "path", "grass", "water", "crafting_table")
SWAP_COLUMNS = ("wood", "stone", "coal", "iron")


@torch.no_grad()
def faced_swap(
    model, val: dict, sprites: dict, fitted: dict, *, normalize, device="cpu", seed: int = 0, min_frames: int = 10
) -> dict:
    """Does the predictor use the faced tile? Bright held-out frames whose player holds a stone pickaxe (stone, coal and
    iron can be mined): paste each block type at the faced tile, encode the edited frame, predict one step with `do`
    (history 1, as the planner rolls out) and read the change of the latent's inventory probe from the edited frame to
    the prediction. If the change does not depend on the pasted block, the predictor ignores the faced tile. Also
    the share of edited frames whose latent reads the pasted block at the faced tile (does the encoder see the
    paste?)."""
    from .collect import paste
    from .encoder import encode_pixels

    swap = val.get("swap") or {}
    if not swap or not len(swap.get("frame", [])) or not {"inventory", "facing_block"} <= set(fitted):
        return {}
    frames, pixels = swap["frame"], swap["pixels"]
    tile = tg.faced_index(val["y"]["facing_raw"][frames])
    rows = torch.arange(len(frames))
    holds = val["y"]["inventory"][frames, tg.INVENTORY_NAMES.index("stone_pickaxe")] >= 1
    keep = holds & ~val["y"]["mobs"][frames][rows, tile].any(-1)
    out = {"frames": int(keep.sum()), "blocks": {}}
    if keep.sum() < min_frames:
        return out
    pixels, tile = pixels[keep], tile[keep]
    jepa = model.get_level(1)
    do = torch.nn.functional.one_hot(torch.tensor(tg.DO), len(tg.ACTION_NAMES)).float().to(device)
    inventory, faced = fitted["inventory"]["predict"], fitted["facing_block"]["predict"]
    columns = {n: tg.INVENTORY_NAMES.index(n) for n in SWAP_COLUMNS}
    g = torch.Generator().manual_seed(seed)
    libs = sprites.get("blocks", {})
    for name in ("unedited",) + SWAP_FACED:
        c = None if name == "unedited" else tg.BLOCK_NAMES.index(name)
        if c is not None and not libs.get(c):
            continue
        edited = (
            pixels
            if c is None
            else torch.stack(
                [
                    paste(pixels[j], tile[j], libs[c][int(torch.randint(len(libs[c]), (1,), generator=g))])
                    for j in range(len(pixels))
                ]
            )
        )
        lat, _ = encode_pixels(model, edited, normalize, device)
        preds = []
        with health.eval_mode(jepa.predictor):
            for a in range(0, len(lat), 512):
                z = lat[a : a + 512].to(device)[:, None]
                act = jepa.action_embed(do[None, None].expand(len(z), 1, -1))
                preds.append(jepa.predict(z, act)[:, -1].float().cpu())
        delta = inventory(torch.cat(preds)) - inventory(lat)
        entry = {n: float(delta[:, j].mean()) for n, j in columns.items()}
        if c is not None:
            entry["reads_pasted"] = float((faced(lat).argmax(1) == c).float().mean())
        out["blocks"][name] = entry
    return out


# --------------------------------------------------------------------------
# mobs


def mob_dynamics(val: dict, jepa, decoder, *, history: int, device="cpu", chunk: int = 2048) -> dict:
    """One step ahead, for each mob type: are the tiles where it stands at t + 1 scored above the empty tiles by the map
    decoder of the predicted latent, of the encoded next latent and of the copied last latent? Split into mobs that were
    within one tile at t (arrows fly straight, zombies walk: predictable) and new ones (spawned or entered the view)."""
    if decoder is None:
        return {}
    mobs, pos = view(val, val["y"]["mobs"]), view(val, val["y"]["position"])
    emb, act = view(val, val["frames"]["latent"]), view(val, val["act"])
    T = val["T"]
    b, t = torch.meshgrid(torch.arange(val["clips"]), torch.arange(history - 1, T - 1), indexing="ij")
    b, t = b.flatten(), t.flatten()
    keep = mobs[b, t + 1].any(-1).any(-1)
    b, t = b[keep], t[keep]
    if not len(b):
        return {}
    pred = one_step(jepa, emb, act, b, t, history, device)
    scores = {
        src: decoder(z)["mobs"].reshape(len(b), tg.VIEW_TILES, tg.NUM_MOBS)
        for src, z in (("pred", pred), ("enc", emb[b, t + 1]), ("copy", emb[b, t]))
    }
    rel = torch.stack(
        torch.meshgrid(
            torch.arange(tg.VIEW_ROWS) - tg.PLAYER_TILE[0],
            torch.arange(tg.VIEW_COLS) - tg.PLAYER_TILE[1],
            indexing="ij",
        ),
        -1,
    ).reshape(tg.VIEW_TILES, 2)
    scored = torch.arange(tg.VIEW_TILES) != tg.PLAYER_INDEX
    out = {}
    for k, name in enumerate(tg.MOB_NAMES):
        now, nxt = mobs[b, t, :, k], mobs[b, t + 1, :, k]
        cont = torch.zeros_like(nxt)
        for a in range(0, len(b), chunk):
            w0 = pos[b[a : a + chunk], t[a : a + chunk]][:, None] + rel[None]
            w1 = pos[b[a : a + chunk], t[a : a + chunk] + 1][:, None] + rel[None]
            dist = (w1[:, :, None] - w0[:, None, :]).abs().sum(-1)  # (n, 63 at t+1, 63 at t)
            near = (dist <= 1) & now[a : a + chunk, None, :]
            cont[a : a + chunk] = near.any(-1) & nxt[a : a + chunk]
        new = nxt & ~cont
        empty = ~nxt
        entry = {"continuing": int(cont[:, scored].sum()), "new": int(new[:, scored].sum())}
        for kind, positive in (("continuing", cont), ("new", new)):
            for src, s in scores.items():
                pos_s, neg_s = s[:, :, k][positive & scored], s[:, :, k][empty & scored]
                y = torch.cat([torch.ones(len(pos_s)), torch.zeros(len(neg_s))]).numpy()
                entry[f"{kind}_{src}"] = ro.auroc(y, torch.cat([pos_s, neg_s]).numpy())
        out[name] = entry
    return out


# --------------------------------------------------------------------------
# monotonicity toward a goal frame


def monotonicity(val: dict, jepa, *, history: int, offsets=(8, 16, 32, 64), every: int = 8, device="cpu") -> dict:
    """Goal = the latent of frame t + K of an expert clip (source 0 when the data says). Along the clip's own frames
    and along the open-loop predictions with its actions: the rank correlation of the squared distance to the goal with
    the steps left (1 = it falls all the way), the share of steps where it falls, and for the predictions the final
    distance over the start distance (below 1: the expert's actions are predicted to approach the goal). Windows whose
    goal changes the inventory apart."""
    emb, act, inv = view(val, val["frames"]["latent"]), view(val, val["act"]), view(val, val["y"]["inventory"])
    expert = None
    if val.get("source") is not None:
        expert = view(val, val["source"])[:, 0] == 0
    out = {"expert_only": expert is not None, "offsets": {}}
    for K in offsets:
        clip, first = windows(val, history, K, every)
        if clip is None:
            continue
        if expert is not None:
            keep = expert[clip]
            clip, first = clip[keep], first[keep]
            if not len(clip):
                continue
        last = first + history - 1
        path = _gather(emb, clip, last, K + 1)  # frames t .. t + K
        goal = path[:, -1:]
        d_enc = (path - goal).pow(2).sum(-1)  # (W, K + 1)
        pred = open_loop(
            jepa, _gather(emb, clip, first, history + K), _gather(act, clip, first, history + K), history, K, device
        )
        d_pred = torch.cat([d_enc[:, :1], (pred - goal).pow(2).sum(-1)], 1)
        left = torch.arange(K, -1, -1).float()
        inventory_goal = (inv[clip, last + K] != inv[clip, last]).any(-1)
        entry = {"windows": len(clip), "inventory_goal": int(inventory_goal.sum())}
        for src, d in (("encoded", d_enc), ("predicted", d_pred)):
            rho = torch.tensor([ro.spearman(row.numpy(), left.numpy()) or float("nan") for row in d])
            falls = (d[:, 1:] < d[:, :-1]).float().mean(1)
            for part, mask in (
                ("all", torch.ones(len(d), dtype=torch.bool)),
                ("inventory goal", inventory_goal),
                ("other goal", ~inventory_goal),
            ):
                if not mask.any():
                    continue
                r = rho[mask]
                entry.setdefault(part, {})[src] = {
                    "rho_median": float(np.nanmedian(r.numpy())) if (~torch.isnan(r)).any() else None,
                    "rho_above_half": float((r > 0.5).float().mean()),
                    "falls": float(falls[mask].mean()),
                }
                if src == "predicted":
                    entry[part][src]["final_over_start"] = float((d[mask, -1] / d[mask, 0].clamp_min(1e-12)).median())
        out["offsets"][K] = entry
    return out


# --------------------------------------------------------------------------
# loss attribution


def loss_attribution(val: dict, jepa, *, history: int, device="cpu") -> dict:
    """The one-step prediction error (teacher-forced, the training's history, per-dimension MSE as in the loss) of
    every held-out transition t -> t + 1, by what happened: share of transitions, share of the total loss, mean loss
    over the overall mean, and the predictor's error over copying the last latent."""
    emb, act = view(val, val["frames"]["latent"]), view(val, val["act"])
    action, sleeping = view(val, val["action"]), view(val, val["y"]["sleeping"][:, 0] > 0.5)
    blocks, facing, mobs = view(val, val["y"]["blocks"]), view(val, val["y"]["facing_raw"]), view(val, val["y"]["mobs"])
    ys = {g: view(val, val["y"][g]) for g in ("inventory", "vitals")}
    T = val["T"]
    b, t = torch.meshgrid(torch.arange(val["clips"]), torch.arange(history - 1, T - 1), indexing="ij")
    b, t = b.flatten(), t.flatten()
    valid = action[b, t] >= 0
    b, t = b[valid], t[valid]
    if not len(b):
        return {}
    pred = one_step(jepa, emb, act, b, t, history, device)
    err = (pred - emb[b, t + 1]).pow(2).mean(-1)
    copy = (emb[b, t] - emb[b, t + 1]).pow(2).mean(-1)
    a = action[b, t]
    faced = blocks[b, t, tg.faced_index(facing[b, t])]
    category = []
    for i, (ai, fb) in enumerate(zip(a.tolist(), faced.tolist())):
        name = tg.ACTION_NAMES[ai]
        if sleeping[b[i], t[i]]:
            category.append("asleep")
            continue
        event = (
            name
            if name in tg.PLACE_MAKE
            else tg.COLLECT[tg.BLOCK_NAMES[fb]][0]
            if ai == tg.DO and tg.BLOCK_NAMES[fb] in tg.COLLECT
            else None
        )
        if event is not None:
            g, col, sign = EVENT_COLUMN[event]
            ok = sign * (ys[g][b[i], t[i] + 1, col] - ys[g][b[i], t[i], col]) > 0
            category.append(f"{event_group(event)} (succeeded)" if ok else "press without effect")
        elif name in tg.MOVES:
            category.append("move")
        elif name == "do":
            category.append("do (other)")
        else:
            category.append(name)
    category = np.array(category)
    total, mean = float(err.sum()), float(err.mean())
    rows = {}

    def row(mask):
        mask = torch.as_tensor(mask)
        if not mask.any():
            return None
        return {
            "transitions": int(mask.sum()),
            "share": float(mask.float().mean()),
            "loss_share": float(err[mask].sum()) / total if total > 0 else None,
            "relative_loss": float(err[mask].mean()) / mean if mean > 0 else None,
            "pred_over_copy": float(err[mask].mean() / copy[mask].mean().clamp_min(1e-12)),
        }

    for c in sorted(set(category.tolist()), key=lambda c: -(category == c).sum()):
        rows[c] = row(category == c)
    flags = {
        "arrow in view": mobs[b, t + 1, :, tg.MOB_NAMES.index("arrow")].any(-1),
        "any mob in view": mobs[b, t + 1].any(-1).any(-1),
    }
    overlapping = {k: row(v) for k, v in flags.items() if v.any()}
    return {
        "transitions": len(err),
        "mean_loss": mean,
        "mean_copy": float(copy.mean()),
        "rows": rows,
        "overlapping": overlapping,
    }


# --------------------------------------------------------------------------
# inverse dynamics


@torch.no_grad()
def idm_per_action(val: dict, jepa, device) -> dict | None:
    """Inverse-dynamics accuracy per action over awake steps; the place and make actions also by outcome."""
    loss_fn = getattr(jepa, "idm_loss_fn", None)
    if loss_fn is None or not hasattr(loss_fn, "num_classes"):
        return None
    emb = view(val, val["frames"]["latent"])
    action, sleeping = view(val, val["action"])[:, :-1], view(val, val["y"]["sleeping"][:, 0] > 0.5)[:, :-1]
    with health.eval_mode(loss_fn):
        logits = torch.cat(
            [
                loss_fn.idm(
                    emb[a : a + 256, :-1].flatten(0, 1).to(device), emb[a : a + 256, 1:].flatten(0, 1).to(device), None
                )
                .float()
                .cpu()
                for a in range(0, len(emb), 256)
            ]
        )
    guess = logits.reshape(-1, loss_fn.num_slots, loss_fn.num_classes)[:, 0].argmax(-1)
    target, asleep = action.flatten(), sleeping.flatten()
    inventory = view(val, val["y"]["inventory"])
    valid = target >= 0
    awake = valid & ~asleep
    correct = guess == target

    def mean(x):
        return float(x.float().mean()) if len(x) else None

    out = {
        "awake": {"steps": int(awake.sum()), "accuracy": mean(correct[awake])},
        "asleep": {"steps": int((valid & asleep).sum()), "accuracy": mean(correct[valid & asleep])},
        "actions": {},
    }
    if awake.any():
        out["awake"]["majority"] = torch.bincount(target[awake], minlength=loss_fn.num_classes).max().item() / int(
            awake.sum()
        )
    change = (inventory[:, 1:] - inventory[:, :-1]).flatten(0, 1)  # aligned with the actions
    for i, name in enumerate(tg.ACTION_NAMES):
        here = awake & (target == i)
        entry = {"steps": int(here.sum()), "accuracy": mean(correct[here])}
        if name in tg.PLACE_MAKE and here.any():
            _, column, sign = tg.PLACE_MAKE[name]
            ok = sign * change[:, column] > 0
            entry["succeeded"] = {"steps": int((here & ok).sum()), "accuracy": mean(correct[here & ok])}
            entry["failed"] = {"steps": int((here & ~ok).sum()), "accuracy": mean(correct[here & ~ok])}
        out["actions"][name] = entry
    return out
