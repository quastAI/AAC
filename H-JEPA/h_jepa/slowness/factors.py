"""Ground truth against the latent: which game factors the latent's variance is made of, and what they cost to predict.

The factors come from the simulator columns (the world model never sees them). Four groups that do not overlap:

    slow     inventory, achievements, daylight
    medium   vitals, sleeping
    facing   the direction the player looks
    view     the composition of the 7x9 view: tiles of each block type and mobs of each type in it
    near     the four tiles next to the player (block type, one-hot)

The whole map (63 tiles one-hot, 1,323 inputs) overfits: with 148 training episodes its held-out R^2 of the latent is
negative, also in the MLP. `view` and `near` are the low-dimensional parts of it.

`attribution`: the latent z (all 192 dimensions) is the target and the groups are the inputs, fitted on training episodes
and scored on held-out episodes. R^2 is variance-weighted: sum of the squared errors over the dimensions over the sum of
the variances, so a dimension counts as much as its variance. A group's unique share is what R^2 loses when the group is
left out of the inputs. What no group explains is the residual; its episode share above the null is the part of the
variance that is the episode and nothing the game state says.

`decodability`: the reverse direction per factor (ridge from z), for the checkpoints against each other.
`costs`: per factor how much of its change one step later the action explains: Var(f_{t+1} - f_t | action) / Var(f),
next to how often it changes. A feature the loss drops is either expensive (high cost) or rare (a low change rate).
"""

from __future__ import annotations

import numpy as np
import torch

from probing import targets as tg

from . import variance as va

GROUPS = ("slow", "medium", "facing", "view", "near")
NEIGHBOURS = tuple(tg.PLAYER_INDEX + d for d in (-tg.VIEW_COLS, tg.VIEW_COLS, -1, 1))
RIDGE = 1e-3  # penalty per sample on standardised inputs (a heuristic, not tuned)


def build_groups(cols: dict, layout: bool = False) -> dict[str, np.ndarray]:
    """The groups as float32 (E, W, L, k) from the sample's columns; `layout=True` adds the whole map (63 tiles one-hot,
    1,323 inputs), which needs many episodes: with 148 it overfits."""
    shape = cols["state"].shape[:3]
    state = torch.from_numpy(cols["state"].reshape(-1, tg.STATE_DIM)).float()
    blocks, mobs = tg.tile_maps(state)
    one_hot = torch.nn.functional.one_hot(blocks, tg.NUM_BLOCKS)
    view = torch.cat([one_hot.sum(-2).float(), mobs.sum(-2).float()], -1)
    near = one_hot[:, list(NEIGHBOURS)].flatten(1).float()
    facing = torch.nn.functional.one_hot((torch.from_numpy(cols["facing"][..., 0].reshape(-1)).long() - 1).clamp(0, 3), 4)
    flat = lambda name: cols[name].reshape(-1, cols[name].shape[-1]).astype(np.float32)  # noqa: E731
    groups = {
        "slow": np.concatenate([flat("inventory"), flat("achievements"), flat("daylight")], -1),
        "medium": np.concatenate([flat("vitals"), (state[:, tg.STATE_DIM - 1 :] > 0.5).numpy().astype(np.float32)], -1),
        "facing": facing.numpy().astype(np.float32),
        "view": view.numpy(),
        "near": near.numpy(),
    }
    if layout:
        groups["layout"] = torch.cat([one_hot.flatten(1).float(), mobs.flatten(1).float()], -1).numpy()
    return {k: v.reshape(*shape, v.shape[-1]) for k, v in groups.items()}


def scalar_targets(cols: dict) -> dict[str, np.ndarray]:
    """Named per-frame factors (E, W, L) for decodability and cost: the inventory items, vitals, daylight, the mobs in
    view and the tiles of each block type in view."""
    shape = cols["state"].shape[:3]
    state = torch.from_numpy(cols["state"].reshape(-1, tg.STATE_DIM)).float()
    blocks, mobs = tg.tile_maps(state)
    out = {}
    for i, name in enumerate(tg.INVENTORY_NAMES):
        out[f"inv:{name}"] = cols["inventory"][..., i].astype(np.float32)
    for i, name in enumerate(tg.VITAL_NAMES):
        out[f"vital:{name}"] = cols["vitals"][..., i].astype(np.float32)
    out["daylight"] = cols["daylight"][..., 0].astype(np.float32)
    for k, name in enumerate(tg.MOB_NAMES):
        out[f"mob:{name}"] = mobs[..., k].sum(-1).numpy().astype(np.float32).reshape(shape)
    counts = torch.nn.functional.one_hot(blocks, tg.NUM_BLOCKS).sum(-2).numpy().astype(np.float32)
    for c, name in enumerate(tg.BLOCK_NAMES):
        out[f"view:{name}"] = counts[:, c].reshape(shape)
    return {k: v for k, v in out.items() if v.std() > 0}


def episode_split(n: int, test: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    cut = int(round(n * (1 - test)))
    return np.sort(order[:cut]), np.sort(order[cut:])


def standardise(train: np.ndarray, *others: np.ndarray):
    """Z-scores with the training mean and standard deviation; constant columns become 0."""
    mean, std = train.mean(0), train.std(0)
    scale = np.where(std > 1e-6, 1 / np.maximum(std, 1e-6), 0.0)
    return tuple(((a - mean) * scale).astype(np.float32) for a in (train, *others))


def ridge(x_tr, y_tr, x_te, alpha: float = RIDGE) -> np.ndarray:
    """Closed-form ridge with an intercept; the prediction for x_te."""
    x_mean, y_mean = x_tr.mean(0), y_tr.mean(0)
    xc = (x_tr - x_mean).astype(np.float64)
    gram = xc.T @ xc + alpha * len(xc) * np.eye(xc.shape[1])
    beta = np.linalg.solve(gram, xc.T @ (y_tr - y_mean).astype(np.float64))
    return (x_te - x_mean) @ beta + y_mean


ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def ridge_cv(x_tr, y_tr, x_te, alphas=ALPHAS) -> np.ndarray:
    """Ridge whose penalty is chosen on the last fifth of the training rows (rows are ordered by episode, so that is a
    block of held-out episodes), then refitted on all training rows."""
    cut = int(len(x_tr) * 0.8)
    scores = [r2_weighted(y_tr[cut:], ridge(x_tr[:cut], y_tr[:cut], x_tr[cut:], a)) for a in alphas]
    return ridge(x_tr, y_tr, x_te, alphas[int(np.argmax(scores))])


def mlp(x_tr, y_tr, x_te, *, hidden: int = 256, epochs: int = 12, lr: float = 1e-3, seed: int = 0) -> np.ndarray:
    """A one-hidden-layer network on the raw target scale (a dimension keeps the weight of its variance)."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    x, y = torch.from_numpy(x_tr), torch.from_numpy(y_tr).float()
    y_mean = y.mean(0)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(x) + 511) // 512)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=gen).split(512):
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(x[idx]), y[idx] - y_mean).backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        return (net(torch.from_numpy(x_te)) + y_mean).numpy()


def r2_weighted(y: np.ndarray, pred: np.ndarray) -> float:
    """1 - (sum over dimensions of the squared errors) / (sum of the variances)."""
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def _inputs(groups: dict, names, tr, te):
    flat = lambda a, idx: a[idx].reshape(-1, a.shape[-1])  # noqa: E731
    parts = [standardise(flat(groups[n], tr), flat(groups[n], te)) for n in names]
    return np.concatenate([p[0] for p in parts], -1), np.concatenate([p[1] for p in parts], -1)


def attribution(z: np.ndarray, groups: dict, split, method: str = "ridge", seed: int = 0, progress=None) -> dict:
    """R^2 of z from all groups, from each group alone and without each group (held-out episodes), the unique share of
    each group and the time-scale shares of the residual of the full fit (`residual`) against its null. `progress` is
    called after each fit (1 + 2 per group)."""
    tr, te = split
    names = tuple(groups)
    fit = ridge_cv if method == "ridge" else (lambda a, b, c: mlp(a, b, c, seed=seed))
    y_tr, y_te = z[tr].reshape(-1, z.shape[-1]), z[te].reshape(-1, z.shape[-1])

    def run(names):
        x_tr, x_te = _inputs(groups, names, tr, te)
        pred = fit(x_tr, y_tr, x_te)
        if progress:
            progress()
        return pred, r2_weighted(y_te, pred)

    pred, full = run(names)
    out = {"groups": list(names), "all": full, "only": {}, "without": {}, "unique": {}}
    for g in names:
        out["only"][g] = run((g,))[1]
        out["without"][g] = run(tuple(n for n in names if n != g))[1]
        out["unique"][g] = full - out["without"][g]
    resid = (y_te - pred).reshape(*z[te].shape)
    out["residual"] = {
        "shares": va.shares(va.decompose(resid)),
        "null": va.null_shares(resid, n=5),
        "variance_share": float(((y_te - pred) ** 2).sum() / ((y_te - y_te.mean(0)) ** 2).sum()),
    }
    return out


def axis_r2(z: np.ndarray, groups: dict, names, split, method: str = "mlp", seed: int = 0):
    """R^2 of each principal axis of z (largest variance first) from the groups `names`, on held-out episodes.
    Returns the R^2 (D,), the pooled mean and the axes (D, D) of `variance.pca`."""
    tr, te = split
    mean, axes, _ = va.pca(z)
    p = (np.asarray(z, np.float64) - mean) @ axes
    x_tr, x_te = _inputs(groups, names, tr, te)
    y_tr, y_te = p[tr].reshape(-1, p.shape[-1]).astype(np.float32), p[te].reshape(-1, p.shape[-1])
    pred = ridge_cv(x_tr, y_tr, x_te) if method == "ridge" else mlp(x_tr, y_tr, x_te, seed=seed)
    sst = ((y_te - y_te.mean(0)) ** 2).sum(0)
    return 1 - ((y_te - pred) ** 2).sum(0) / np.maximum(sst, 1e-12), mean, axes


def decodability(z: np.ndarray, targets: dict, split, method: str = "ridge", seed: int = 0) -> dict[str, float]:
    """R^2 per factor from z, fitted on training episodes, scored on held-out ones (NaN where the factor does not vary
    in the held-out episodes). `ridge` fits each factor alone; `mlp` fits all factors in one network on standardised
    targets (one fit instead of one per factor), which reads what a linear map cannot."""
    tr, te = split
    x_tr, x_te = standardise(z[tr].reshape(-1, z.shape[-1]), z[te].reshape(-1, z.shape[-1]))
    names = list(targets)
    y_tr = np.stack([targets[n][tr].reshape(-1) for n in names], -1)
    y_te = np.stack([targets[n][te].reshape(-1) for n in names], -1)
    if method == "mlp":
        y_tr_std, y_te_std = standardise(y_tr, y_te)
        pred = mlp(x_tr, y_tr_std, x_te, seed=seed)
        y_te = y_te_std
    else:
        pred = np.concatenate([ridge_cv(x_tr, y_tr[:, [k]], x_te) for k in range(len(names))], -1)
    sst = ((y_te - y_te.mean(0)) ** 2).sum(0)
    r2 = 1 - ((y_te - pred) ** 2).sum(0) / np.where(sst > 0, sst, np.nan)
    return dict(zip(names, map(float, r2)))


def costs(targets: dict, action: np.ndarray) -> dict[str, dict]:
    """Per factor, inside windows: the variance of the factor, how often it changes in one step, the variance of the
    change and the part of it the action does not explain, Var(change | action) / Var(factor)."""
    act = action[..., :-1, 0] if action.ndim == 4 else action[..., :-1]
    act = np.nan_to_num(act, nan=-1).astype(np.int64)
    out = {}
    for name, f in targets.items():
        d = (f[..., 1:] - f[..., :-1])
        resid = np.zeros_like(d, np.float64)
        for a in np.unique(act):
            m = act == a
            resid[m] = d[m] - d[m].mean()
        out[name] = {
            "variance": float(f.var()),
            "change_rate": float((d != 0).mean()),
            "change_variance": float(d.var()),
            "cost": float(resid.var() / max(f.var(), 1e-12)),
        }
    return out
