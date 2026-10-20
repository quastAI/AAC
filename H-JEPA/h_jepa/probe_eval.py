"""Probing: what do the latents of each level encode of the symbolic state, and what do predictions keep of it?

Closed-form ridge probes (no training loop, deterministic) from the latents of
every level to symbolic targets of Craftax-Classic, fitted on training worlds
and scored on held-out worlds. Targets are grouped by how fast they change:

    fast    position, facing direction, mobs in view, blocks in view, block faced
    medium  health, food, drink, energy, sleeping
    slow    inventory, achievements, daylight

The blocks in view (tiles of each block type), the block the player faces and the
sleeping flag are read from the `state` column, when the data has it.

Scores are comparable across targets: R^2 for counts and levels, 2*AUROC - 1 for
binary flags, and (accuracy - majority) / (1 - majority) for classes (facing
direction, block faced). The same probes on a randomly initialised encoder of
the same architecture are the control: what a probe finds there is architecture
and data, the difference is what training added.

Level 1 also gets, on the held-out worlds:

    rollouts  the level-1 probes, fitted on encoded latents, applied unchanged to
              open-loop predicted latents k steps ahead (the true actions, the
              planner's history), against copying the last context latent and
              against the encoded latent of the same frame.
    events    one-step predictions after the place and make actions, split into
              successful and failed presses: does the predicted latent show the
              inventory change that the encoded next frame shows?
    idm       the inverse-dynamics accuracy per action, over awake steps (the game
              ignores every action while the player sleeps).

Needs a dataset with the derived columns of the collector (inventory, vitals,
facing, daylight, position, mobs_in_view). Usage:

    python probe_eval.py --run <run dir> --train-dataset X --val-dataset Y
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import health
from eval_tasks import ACHIEVEMENT_NAMES

TARGETS = {
    "position": dict(group="fast", kind="reg", names=["row", "col"]),
    "facing": dict(group="fast", kind="cls", names=["direction"], classes=4),
    "mobs_in_view": dict(group="fast", kind="reg", names=["zombies", "cows", "skeletons", "arrows"]),
    "vitals": dict(group="medium", kind="reg", names=["health", "food", "drink", "energy"]),
    "inventory": dict(
        group="slow", kind="reg",
        names=["wood", "stone", "coal", "iron", "diamond", "sapling", "wood_pickaxe", "stone_pickaxe",
               "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"],
    ),  # fmt: skip
    "daylight": dict(group="slow", kind="reg", names=["light"]),
    "achievements": dict(group="slow", kind="bin", names=ACHIEVEMENT_NAMES),
}
COLUMNS = list(TARGETS)  # dataset columns the probes need
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)  # relative to the number of samples

# Layout of the `state` column (as in analysis/dataset_section/data.py): the 7x9 view, 21 channels per tile (the block
# type one-hot, then zombie, cow, skeleton, arrow), then inventory, vitals, direction, light; the last entry is the
# sleeping flag. Block types in Craftax-Classic order.
BLOCK_NAMES = [
    "invalid", "out_of_bounds", "grass", "water", "stone", "tree", "wood", "path", "coal", "iron", "diamond",
    "crafting_table", "furnace", "sand", "lava", "plant", "ripe_plant",
]  # fmt: skip
VIEW_ROWS, VIEW_COLS, VIEW_CHANNELS = 7, 9, 21
STATE_DIM = 1345
PLAYER_TILE = (VIEW_ROWS // 2, VIEW_COLS // 2)
FACING_STEP = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # `facing` -> the neighbouring tile it looks at
STATE_TARGETS = {
    "view_blocks": dict(group="fast", kind="reg", names=BLOCK_NAMES),  # tiles of each block type in the view
    "facing_block": dict(group="fast", kind="cls", names=["block"], classes=len(BLOCK_NAMES)),
    "sleeping": dict(group="medium", kind="bin", names=["sleeping"]),
}

ACTION_NAMES = [
    "noop", "left", "right", "up", "down", "do", "sleep", "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
    "make_iron_sword",
]  # fmt: skip
# The place and make actions and how their success shows in the inventory (column, +1 rises, -1 falls), as `SUCCESS`
# in analysis/dataset_section/actions.py.
EVENTS = {
    "place_stone": (1, -1), "place_table": (0, -1), "place_furnace": (1, -1), "place_plant": (5, -1),
    "make_wood_pickaxe": (6, 1), "make_stone_pickaxe": (7, 1), "make_iron_pickaxe": (8, 1),
    "make_wood_sword": (9, 1), "make_stone_sword": (10, 1), "make_iron_sword": (11, 1),
}  # fmt: skip
HORIZONS = (1, 2, 4, 8, 16, 32, 64)  # rollout steps at which the probes are scored
MIN_EVENTS = 10  # fewer successful presses: no event score


# --------------------------------------------------------------------------
# probes


def ridge_fit(x: torch.Tensor, y: torch.Tensor, lam: float):
    """Standardise x, fit y ~ x with an intercept; returns the predict function."""
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    xb = torch.cat([(x - mean) / std, torch.ones(len(x), 1)], dim=1).double()
    reg = lam * len(x) * torch.eye(xb.shape[1], dtype=torch.double)
    reg[-1, -1] = 0  # not the intercept
    w = torch.linalg.solve(xb.T @ xb + reg, xb.T @ y.double())

    def predict(z: torch.Tensor) -> torch.Tensor:
        zb = torch.cat([(z - mean) / std, torch.ones(len(z), 1)], dim=1).double()
        return (zb @ w).float()

    return predict


def fit_with_validation(x, y, seed=0):
    """Pick the ridge strength on a split of the training data, then refit on all of it."""
    n = len(x)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    cut = int(0.8 * n)
    a, b = perm[:cut], perm[cut:]
    best, best_err = LAMBDAS[0], float("inf")
    for lam in LAMBDAS:
        err = (ridge_fit(x[a], y[a], lam)(x[b]) - y[b]).pow(2).mean().item()
        if err < best_err:
            best, best_err = lam, err
    return ridge_fit(x, y, best), best


def r2_columns(y: torch.Tensor, pred: torch.Tensor) -> list[float | None]:
    var = y.var(0, unbiased=False)
    mse = (y - pred).pow(2).mean(0)
    return [None if v < 1e-8 else float(1 - m / v) for v, m in zip(var, mse)]


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    """Mann-Whitney AUROC; None unless both classes occur."""
    positives = y > 0.5
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos < 5 or n_neg < 5:
        return None
    ranks = np.argsort(np.argsort(score)) + 1.0
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_target(spec: dict, ztr, ytr, seed=0) -> dict:
    """The probe of one target, fitted on the training split: {"predict", "lambda"}."""
    kind = spec["kind"]
    if kind == "cls":
        y = torch.nn.functional.one_hot(ytr[:, 0].long(), spec["classes"]).float()
    else:
        y = ytr if kind == "reg" else ytr * 2 - 1
    predict, lam = fit_with_validation(ztr, y, seed)
    return {"predict": predict, "lambda": lam}


def score_target(spec: dict, fitted: dict, z, y) -> dict:
    """Column scores and the target score of a fitted probe on (z, y)."""
    kind = spec["kind"]
    pred = fitted["predict"](z)
    if kind == "cls":
        k = spec["classes"]
        accuracy = (pred.argmax(1) == y[:, 0].long()).float().mean().item()
        majority = torch.bincount(y[:, 0].long(), minlength=k).max().item() / len(y)
        score = (accuracy - majority) / max(1 - majority, 1e-6)
        return {"columns": {spec["names"][0]: score}, "score": score, "lambda": fitted["lambda"], "group": spec["group"],
                "accuracy": accuracy, "majority": majority}  # fmt: skip
    if kind == "reg":
        columns = r2_columns(y, pred)
    else:
        columns = []
        for j in range(y.shape[1]):
            a = auroc(y[:, j].numpy(), pred[:, j].numpy())
            columns.append(None if a is None else 2 * a - 1)
    valid = [c for c in columns if c is not None]
    return {
        "columns": dict(zip(spec["names"], columns)),
        "score": float(np.mean(valid)) if valid else None,
        "lambda": fitted["lambda"],
        "group": spec["group"],
    }


def probe_target(spec: dict, ztr, ytr, zva, yva, seed=0) -> dict:
    """Fit on the training split, score on the held-out one. Returns column scores and the target score."""
    return score_target(spec, fit_target(spec, ztr, ytr, seed), zva, yva)


def fit_targets(z, y: dict, targets: dict, seed=0) -> dict:
    return {name: fit_target(spec, z, y[name], seed) for name, spec in targets.items() if name in y}


def score_targets(fitted: dict, z, y: dict, targets: dict) -> dict:
    """Per target and group: {"targets": {name: result}, "groups": {group: mean score}}."""
    scored = {name: score_target(targets[name], f, z, y[name]) for name, f in fitted.items()}
    groups = {}
    for g in ("fast", "medium", "slow"):
        scores = [t["score"] for n, t in scored.items() if targets[n]["group"] == g and t["score"] is not None]
        groups[g] = float(np.mean(scores)) if scores else None
    return {"targets": scored, "groups": groups}


def probe_levels(train: dict, val: dict, targets: dict | None = None) -> dict:
    """train/val: {level: (Z, {column: Y})}. Scores per level, target and group."""
    targets = targets or TARGETS
    return {level: score_targets(fit_targets(*train[level], targets), *val[level], targets) for level in train}


# --------------------------------------------------------------------------
# collecting latents and aligned targets


def frame_of_step(level: int, steps: torch.Tensor, cfg) -> torch.Tensor:
    """Level-1 frame whose latent chunk ends the level-`level` step `steps`."""
    index = steps
    for n in range(level, 1, -1):
        lv = cfg[f"level{n}"]
        index = index * int(lv.stride) + int(lv.get("window_size", 1)) - 1
    return index


def common_frames(model, cfg, num_frames: int, thin: int) -> torch.Tensor:
    """Level-1 frames at which every level has a latent, so all levels are probed on the same states.

    Probing each level on its own steps gave the levels different numbers of samples (33 / 9 / 3
    per clip for strides of 4) and different frames, which confounds the comparison between them
    (RQ2) with sample size, and ridge probes on 192 dimensions need many samples.
    """
    common = None
    for n in range(1, model.num_levels + 1):
        steps = torch.arange(num_frames)
        frames = frame_of_step(n, steps, cfg)
        frames = frames[frames < num_frames]
        common = frames if common is None else common[torch.isin(common, frames)]
    if not len(common):
        raise ValueError("the levels have no level-1 frame in common (check stride and window_size)")
    return common[::thin]


def state_targets(state: torch.Tensor, facing: torch.Tensor) -> dict:
    """The targets read from `state` (..., 1345) and `facing` (..., 1; 1 to 4): blocks in view, block faced, sleeping."""
    lead = state.shape[:-1]
    tiles = state[..., : VIEW_ROWS * VIEW_COLS * VIEW_CHANNELS].reshape(*lead, VIEW_ROWS, VIEW_COLS, VIEW_CHANNELS)
    block = tiles[..., : len(BLOCK_NAMES)].argmax(-1)  # (..., rows, cols)
    counts = torch.nn.functional.one_hot(block, len(BLOCK_NAMES)).sum((-3, -2)).float()
    direction = facing[..., 0].long().clamp(1, 4)
    dr = torch.tensor([FACING_STEP[d][0] for d in (1, 2, 3, 4)])[direction - 1]
    dc = torch.tensor([FACING_STEP[d][1] for d in (1, 2, 3, 4)])[direction - 1]
    flat = block.reshape(*lead, VIEW_ROWS * VIEW_COLS)
    faced = flat.gather(-1, ((PLAYER_TILE[0] + dr) * VIEW_COLS + PLAYER_TILE[1] + dc)[..., None]).float()
    return {"view_blocks": counts, "facing_block": faced, "sleeping": (state[..., STATE_DIM - 1 :] > 0.5).float()}


def frame_targets(batch: dict, targets: dict) -> dict:
    """{target: (B, T, k)} on the CPU for every frame of the batch's level-1 clips."""
    y = {}
    for c in targets:
        if c in STATE_TARGETS:
            continue
        col = batch[f"{c}_level1"].float().cpu()
        y[c] = col - 1 if c == "facing" else col  # directions 1..4 -> classes 0..3
    if any(c in STATE_TARGETS for c in targets):
        derived = state_targets(batch["state_level1"].float().cpu(), batch["facing_level1"].cpu())
        y.update({c: derived[c] for c in targets if c in STATE_TARGETS})
    return y


def has_state(cache_dir, name) -> bool:
    """Does the dataset have the `state` column of the collector (for the targets read from it)?"""
    import h5py

    from stable_worldmodel.data.utils import get_cache_dir

    try:
        with h5py.File(Path(cache_dir or get_cache_dir()) / f"{name}.h5", "r") as f:
            return "state" in f and f["state"].shape[1] == STATE_DIM
    except OSError:
        return False


@torch.no_grad()
def collect(model, loader, cfg, device, normalize, thin: int = 1, targets: dict | None = None):
    targets = targets or TARGETS
    levels = range(1, model.num_levels + 1)
    jepas = [model.get_level(n) for n in levels]
    saved = [j.target_length for j in jepas]
    was_training = model.training
    for j in jepas:
        j.target_length = None
    model.eval()
    z = {n: [] for n in levels}
    y = {n: {c: [] for c in targets} for n in levels}
    try:
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            normalize(batch)
            out = model.encode_hierarchical(batch)
            common = common_frames(model, cfg, batch["pixels_level1"].shape[1], thin)
            cols = frame_targets(batch, targets)
            for n in levels:
                emb = out[f"embed_{n}"].float().cpu()
                steps = torch.arange(emb.shape[1])
                frames = frame_of_step(n, steps, cfg)
                keep = torch.isin(frames, common)
                z[n].append(emb[:, steps[keep]].flatten(0, 1))
                for c in targets:
                    y[n][c].append(cols[c][:, frames[keep]].flatten(0, 1))
    finally:
        for j, t in zip(jepas, saved):
            j.target_length = t
        model.train(was_training)
    return {n: (torch.cat(z[n]), {c: torch.cat(y[n][c]) for c in targets}) for n in levels}


@torch.no_grad()
def collect_clips(model, loader, cfg, device, normalize, targets: dict) -> dict:
    """Whole level-1 clips: latents (B, T, D), action embeddings, action indices (B, T; -1 none), the sleeping flag and
    the targets of every frame, on the CPU."""
    from training_monitor import health_idm_target

    jepa = model.get_level(1)
    saved, was_training = jepa.target_length, model.training
    jepa.target_length = None
    model.eval()
    emb, act, action, sleeping, y = [], [], [], [], {c: [] for c in targets}
    try:
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            normalize(batch)
            out = model.encode_hierarchical(batch)
            emb.append(out["embed_1"].float().cpu())
            act.append(out["action_1"].float().cpu())
            index = health_idm_target(out, 1, cfg)
            action.append(index[..., 0].cpu() if index is not None else torch.full(emb[-1].shape[:2], -1))
            cols = frame_targets(batch, targets)
            for c in targets:
                y[c].append(cols[c])
            if "state_level1" in batch:
                sleeping.append(batch["state_level1"][..., STATE_DIM - 1].cpu() > 0.5)
            else:  # without `state` every step counts as awake
                sleeping.append(torch.zeros(emb[-1].shape[:2], dtype=torch.bool))
    finally:
        jepa.target_length = saved
        model.train(was_training)
    return {
        "emb": torch.cat(emb), "act": torch.cat(act), "action": torch.cat(action), "sleeping": torch.cat(sleeping),
        "y": {c: torch.cat(v) for c, v in y.items()},
    }  # fmt: skip


# --------------------------------------------------------------------------
# level 1: rollouts, events, inverse dynamics


@torch.no_grad()
def rollout_latents(jepa, emb, act, history: int, horizons, every: int, device, chunk: int = 512):
    """Open-loop predictions with the true actions, from windows starting every `every` frames of each clip.

    Returns ({k: predicted latents}, last context latents, {k: encoded latents of the target frames}, clip index,
    target frame of k=0 (the last context frame)); the windows are the same for every k.
    """
    B, T, _ = emb.shape
    kmax = min(max(horizons), T - history)
    horizons = [k for k in horizons if k <= kmax]
    starts = list(range(0, T - history - kmax + 1, every))
    if kmax < 1 or not starts:
        return {}, None, {}, None, None
    clip = torch.arange(B).repeat(len(starts))
    last = torch.tensor(starts).repeat_interleave(B) + history - 1
    windows = torch.cat([emb[:, s : s + history + kmax] for s in starts])
    actions = torch.cat([act[:, s : s + history + kmax] for s in starts])
    pred = {k: [] for k in horizons}
    with health.eval_mode(jepa.predictor):
        for a in range(0, len(windows), chunk):
            w, u = windows[a : a + chunk].to(device), actions[a : a + chunk].to(device)
            roll = w[:, :history]
            for k in range(1, kmax + 1):
                t = history + k - 1  # the action of frame t-1 leads to frame t
                step = jepa.predict(roll[:, -history:], u[:, t - history : t])[:, -1:]
                roll = torch.cat([roll, step], dim=1)
                if k in pred:
                    pred[k].append(step[:, 0].float().cpu())
    pred = {k: torch.cat(v) for k, v in pred.items()}
    true = {k: windows[:, history - 1 + k] for k in horizons}
    return pred, windows[:, history - 1], true, clip, last


def rollout_probes(fitted: dict, clips: dict, jepa, targets: dict, *, history, horizons, every, device) -> dict:
    """Scores of the encoded-latent probes on predicted latents k steps ahead, on copying the last context latent and
    on the encoded latents of the same frames."""
    pred, copy, true, clip, last = rollout_latents(jepa, clips["emb"], clips["act"], history, horizons, every, device)
    by_k = {}
    for k, z in pred.items():
        y = {c: v[clip, last + k] for c, v in clips["y"].items()}
        by_k[k] = {
            "pred": score_targets(fitted, z, y, targets),
            "copy": score_targets(fitted, copy, y, targets),
            "encoded": score_targets(fitted, true[k], y, targets),
        }
    return {"history": history, "every": every, "n": 0 if clip is None else len(clip), "by_k": by_k}


def _mean_or_none(x: torch.Tensor, n_min: int = 1):
    return float(x.mean()) if len(x) >= n_min else None


@torch.no_grad()
def event_probes(fitted: dict, clips: dict, jepa, *, history, device) -> dict:
    """One-step predictions after the place and make actions (awake steps), successful and failed presses apart.

    For the item the action changes, the signed change of the inventory probe from the latent before the step: to the
    predicted latent (`pred`) and to the encoded next latent (`encoded`, what a real frame shows). `shown` is the gap
    between successful and failed presses in the prediction, as a share of the same gap in the encoded latents: 1 for a
    predictor that shows the change as a real frame does, 0 for one that predicts no change.
    """
    emb, act, action, sleeping = clips["emb"], clips["act"], clips["action"], clips["sleeping"]
    inventory = clips["y"]["inventory"]
    B, T, _ = emb.shape
    t = torch.arange(history - 1, T - 1)
    b, t = torch.meshgrid(torch.arange(B), t, indexing="ij")
    b, t = b.flatten(), t.flatten()
    names = [ACTION_NAMES[i] if i >= 0 else None for i in action[b, t].tolist()]
    keep = torch.tensor([n in EVENTS for n in names], dtype=torch.bool) & ~sleeping[b, t]
    b, t = b[keep], t[keep]
    names = [n for n, k in zip(names, keep.tolist()) if k]
    if not names:
        return {}
    column = torch.tensor([EVENTS[n][0] for n in names])
    sign = torch.tensor([float(EVENTS[n][1]) for n in names])
    succeeded = sign * (inventory[b, t + 1, column] - inventory[b, t, column]) > 0
    offsets = torch.arange(-history + 1, 1)
    ctx, ctx_act = emb[b[:, None], t[:, None] + offsets], act[b[:, None], t[:, None] + offsets]
    with health.eval_mode(jepa.predictor):
        pred = torch.cat([jepa.predict(ctx[a : a + 512].to(device), ctx_act[a : a + 512].to(device))[:, -1].float().cpu()
                          for a in range(0, len(ctx), 512)])  # fmt: skip
    probe = fitted["inventory"]["predict"]
    rows = torch.arange(len(names))
    before = probe(emb[b, t])[rows, column]
    d_pred = sign * (probe(pred)[rows, column] - before)
    d_enc = sign * (probe(emb[b, t + 1])[rows, column] - before)

    def entry(mask):
        s, f = mask & succeeded, mask & ~succeeded
        enc, ps = _mean_or_none(d_enc[s], MIN_EVENTS), _mean_or_none(d_pred[s], MIN_EVENTS)
        ef, pf = _mean_or_none(d_enc[f]), _mean_or_none(d_pred[f])
        gap = None if enc is None else enc - (ef or 0.0)
        shown = None if gap is None or ps is None or gap < 0.1 else (ps - (pf or 0.0)) / gap
        return {"succeeded": int(s.sum()), "failed": int(f.sum()), "encoded": enc, "encoded_failed": ef, "pred": ps,
                "pred_failed": pf, "shown": shown}  # fmt: skip

    name_of = np.array(names)
    out = {n: entry(torch.from_numpy(name_of == n)) for n in EVENTS if (name_of == n).any()}
    for group in ("place", "make"):
        out[f"{group} (all)"] = entry(torch.from_numpy(np.char.startswith(name_of, group)))
    return out


@torch.no_grad()
def idm_per_action(clips: dict, jepa, device) -> dict | None:
    """Inverse-dynamics accuracy per action over awake steps; the place and make actions also by outcome."""
    loss_fn = getattr(jepa, "idm_loss_fn", None)
    if loss_fn is None or not hasattr(loss_fn, "num_classes"):
        return None
    emb, action, sleeping = clips["emb"], clips["action"][:, :-1], clips["sleeping"][:, :-1]
    with health.eval_mode(loss_fn):
        logits = torch.cat([
            loss_fn.idm(emb[a : a + 256, :-1].flatten(0, 1).to(device), emb[a : a + 256, 1:].flatten(0, 1).to(device), None)
            .float().cpu() for a in range(0, len(emb), 256)
        ])  # fmt: skip
    guess = logits.reshape(-1, loss_fn.num_slots, loss_fn.num_classes)[:, 0].argmax(-1)
    target, asleep = action.flatten(), sleeping.flatten()
    inventory = clips["y"]["inventory"]
    valid = target >= 0
    awake = valid & ~asleep
    correct = guess == target
    out = {"awake": {"steps": int(awake.sum()), "accuracy": _mean_or_none(correct[awake].float())},
           "asleep": {"steps": int((valid & asleep).sum()), "accuracy": _mean_or_none(correct[valid & asleep].float())},
           "actions": {}}  # fmt: skip
    if awake.any():
        out["awake"]["majority"] = torch.bincount(target[awake], minlength=loss_fn.num_classes).max().item() / int(awake.sum())
    change = inventory[:, 1:] - inventory[:, :-1]  # (B, T-1, 12), aligned with the actions
    change = change.flatten(0, 1)
    for i, name in enumerate(ACTION_NAMES):
        here = awake & (target == i)
        entry = {"steps": int(here.sum()), "accuracy": _mean_or_none(correct[here].float())}
        if name in EVENTS and here.any():
            column, sign = EVENTS[name]
            ok = sign * change[:, column] > 0
            entry["succeeded"] = {"steps": int((here & ok).sum()), "accuracy": _mean_or_none(correct[here & ok].float())}
            entry["failed"] = {"steps": int((here & ~ok).sum()), "accuracy": _mean_or_none(correct[here & ~ok].float())}
        out["actions"][name] = entry
    return out


# --------------------------------------------------------------------------
# the whole analysis


def run_probing(
    model, cfg, cache_dir, *, transform, normalize, device, frames=129, max_clips=256, thin=1,
    control=True, batch_size=16, seed=0, rollouts=True, horizons=HORIZONS, rollout_history=None, rollout_every=8,
):  # fmt: skip
    """Probe every level on held-out worlds, with an untrained control; level 1 also on rollouts, after the place and
    make actions and with the IDM per action. Returns (summary, findings)."""
    from training_monitor import build_health_loader

    names = {"train": cfg.data.dataset.name, "val": cfg.data.dataset.val_name}
    use_state = all(has_state(cache_dir, name) for name in names.values())
    targets = {**TARGETS, **STATE_TARGETS} if use_state else dict(TARGETS)
    columns = ("pixels", "action", *COLUMNS) + (("state",) if use_state else ())
    loaders = {}
    for split, name in names.items():
        loader, used = build_health_loader(
            cfg, cache_dir, transform, frames=frames, batch_size=batch_size, max_clips=max_clips,
            seed=seed, name=name, columns=columns,
        )  # fmt: skip
        if loader is None:
            raise RuntimeError(f"no clips of the {split} dataset {name!r} for probing")
        loaders[split] = loader
    train = collect(model, loaders["train"], cfg, device, normalize, thin, targets)
    val = collect(model, loaders["val"], cfg, device, normalize, thin, targets)
    summary = {
        "levels": probe_levels(train, val, targets),
        "n_train": {n: len(train[n][0]) for n in train},
        "n_val": {n: len(val[n][0]) for n in val},
        "dim": {n: train[n][0].shape[1] for n in train},
        "state_targets": use_state,
    }
    if control:
        from hjepa_utils import create_world_model

        with torch.random.fork_rng(devices=[]):  # a seeded control, without reseeding the caller's RNG
            torch.manual_seed(seed)
            random_model = create_world_model(cfg)[0]
        random_model = random_model.to(device)
        random_levels = probe_levels(
            collect(random_model, loaders["train"], cfg, device, normalize, thin, targets),
            collect(random_model, loaders["val"], cfg, device, normalize, thin, targets),
            targets,
        )
        for n, entry in summary["levels"].items():
            entry["control_groups"] = random_levels[n]["groups"]
            for name, t in entry["targets"].items():
                t["control"] = random_levels[n]["targets"][name]["score"]
    if rollouts:
        jepa = model.get_level(1)
        history = int(rollout_history or cfg.get("health", {}).get("plan_history", 1))
        fitted = fit_targets(*train[1], targets)
        clips = collect_clips(model, loaders["val"], cfg, device, normalize, targets)
        summary["rollout"] = rollout_probes(fitted, clips, jepa, targets, history=history, horizons=horizons,
                                            every=rollout_every, device=device)  # fmt: skip
        summary["events"] = event_probes(fitted, clips, jepa, history=history, device=device)
        summary["idm"] = idm_per_action(clips, jepa, device)
    return summary, health.diagnose_probing(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory with config.yaml and *_object.ckpt")
    parser.add_argument("--cache-dir", default=os.environ.get("HJEPA_HOME"))
    parser.add_argument("--train-dataset"), parser.add_argument("--val-dataset")
    parser.add_argument("--frames", type=int, default=129)
    parser.add_argument("--max-clips", type=int, default=256)
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--no-rollouts", action="store_true", help="skip the rollouts, events and IDM of level 1")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS), help="rollout steps to score")
    parser.add_argument("--rollout-history", type=int, default=None, help="context latents (default: health.plan_history)")
    parser.add_argument("--rollout-every", type=int, default=8, help="frames between rollout starts in a clip")
    args = parser.parse_args()

    from omegaconf import OmegaConf

    from data import Compose, OneHotColumn, discrete_num_actions
    from main_hjepa import _normalize_uint8_images_on_device

    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    if args.train_dataset:
        cfg.data.dataset.name = args.train_dataset
    if args.val_dataset:
        cfg.data.dataset.val_name = args.val_dataset
    ckpt = sorted(run.glob("*_object.ckpt"), key=lambda p: p.stat().st_mtime)[-1]
    model = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = model if hasattr(model, "num_levels") else model.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    num_actions = discrete_num_actions(cfg)
    transform = Compose(OneHotColumn(num_actions, "action", "action")) if num_actions else Compose()
    summary, findings = run_probing(
        model, cfg, Path(args.cache_dir), transform=transform, normalize=_normalize_uint8_images_on_device,
        device=device, frames=args.frames, max_clips=args.max_clips, control=not args.no_control,
        rollouts=not args.no_rollouts, horizons=tuple(args.horizons), rollout_history=args.rollout_history,
        rollout_every=args.rollout_every,
    )  # fmt: skip
    out = run / "health"
    out.mkdir(exist_ok=True)
    (out / "probe_report.md").write_text(health.render_probing_report(summary, findings))
    (out / "probe_summary.json").write_text(
        json.dumps({"summary": health.to_jsonable(summary), "findings": health.findings_to_json(findings)}, indent=1)
    )
    print(health.findings_block(findings))
    print(f"wrote {out / 'probe_report.md'}")


if __name__ == "__main__":
    main()
