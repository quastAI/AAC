"""Readouts and scores: closed-form ridge probes, trained probes (softmax and MLP, several heads at once), and intervals
from an episode bootstrap.

Scores are comparable across targets: R^2 for counts and levels, 2*AUROC - 1 for flags, (accuracy - majority) /
(1 - majority) for classes. Ridge probes are deterministic and cheap: they are the linear readout of the state targets.
The trained probes read the tile maps (17 block classes per tile, 4 mob flags per tile) and give the nonlinear readout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LAMBDAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)  # ridge strengths, relative to the number of samples


# --------------------------------------------------------------------------
# ridge probes


def ridge_fit(x: torch.Tensor, y: torch.Tensor, lam: float):
    """Standardise x, fit y ~ x with an intercept; returns the predict function."""
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    xb = torch.cat([(x - mean) / std, torch.ones(len(x), 1)], dim=1).double()
    reg = lam * len(x) * torch.eye(xb.shape[1], dtype=torch.double)
    reg[-1, -1] = 0  # not the intercept
    w = torch.linalg.solve(xb.T @ xb + reg, xb.T @ y.double())

    def predict(z: torch.Tensor) -> torch.Tensor:
        zb = torch.cat([(z.float() - mean) / std, torch.ones(len(z), 1)], dim=1).double()
        return (zb @ w).float()

    return predict


def _holdout(n: int, groups, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """80/20 split of the training rows, by group (e.g. clip) when given, so that the held-out rows are not neighbours
    of training rows."""
    g = torch.Generator().manual_seed(seed)
    if groups is not None:
        groups = torch.as_tensor(groups)
        ids = torch.unique(groups)
        if len(ids) >= 5:
            held = ids[torch.randperm(len(ids), generator=g)[: max(1, len(ids) // 5)]]
            b = torch.isin(groups, held)
            return torch.nonzero(~b)[:, 0], torch.nonzero(b)[:, 0]
    perm = torch.randperm(n, generator=g)
    cut = int(0.8 * n)
    return perm[:cut], perm[cut:]


def fit_with_validation(x, y, seed=0, groups=None):
    """Pick the ridge strength on a held-out part of the training data, then refit on all of it.

    Returns (predict, lambda, at_edge): `at_edge` says the best strength is the smallest or largest of the grid, so the
    grid may be too narrow."""
    a, b = _holdout(len(x), groups, seed)
    best, best_err = LAMBDAS[0], float("inf")
    for lam in LAMBDAS:
        err = (ridge_fit(x[a], y[a], lam)(x[b]) - y[b]).pow(2).mean().item()
        if err < best_err:
            best, best_err = lam, err
    return ridge_fit(x, y, best), best, best in (LAMBDAS[0], LAMBDAS[-1])


def fit_target(spec: dict, z, y, seed=0, groups=None) -> dict:
    """The ridge probe of one target: {"predict", "lambda", "at_edge"}; classes and flags are fitted on one-hot and
    +-1."""
    kind = spec["kind"]
    if kind == "cls":
        y = F.one_hot(y[:, 0].long(), spec["classes"]).float()
    elif kind == "bin":
        y = y * 2 - 1
    predict, lam, edge = fit_with_validation(z, y, seed, groups)
    return {"predict": predict, "lambda": lam, "at_edge": edge}


def fit_targets(z, y: dict, targets: dict, seed=0, groups=None) -> dict:
    return {name: fit_target(spec, z, y[name], seed, groups) for name, spec in targets.items() if name in y}


# --------------------------------------------------------------------------
# scores


def r2_columns(y: torch.Tensor, pred: torch.Tensor) -> list[float | None]:
    var = y.var(0, unbiased=False)
    mse = (y - pred).pow(2).mean(0)
    return [None if v < 1e-8 else float(1 - m / v) for v, m in zip(var, mse)]


def auroc(y, score, weight=None, min_n: int = 5) -> float | None:
    """Mann-Whitney AUROC, ties counted half; `weight` per sample (the token sample's natural weights). None unless
    both classes have `min_n` samples."""
    y = np.asarray(y) > 0.5
    s = np.asarray(score, dtype=np.float64)
    w = np.ones(len(s)) if weight is None else np.asarray(weight, dtype=np.float64)
    if int(y.sum()) < min_n or int((~y).sum()) < min_n:
        return None
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    _, start = np.unique(s, return_index=True)
    pos = np.add.reduceat(np.where(y, w, 0.0), start)
    neg = np.add.reduceat(np.where(y, 0.0, w), start)
    below = np.cumsum(neg) - neg
    total = pos.sum() * neg.sum()
    return None if total <= 0 else float((pos * (below + 0.5 * neg)).sum() / total)


def score_prediction(spec: dict, pred: torch.Tensor, y: torch.Tensor) -> dict:
    """Column scores and the target score of a prediction (values, flag scores, or class scores) against y."""
    kind = spec["kind"]
    if kind == "cls":
        k = spec["classes"]
        truth = y[:, 0].long()
        accuracy = (pred.argmax(1) == truth).float().mean().item()
        majority = torch.bincount(truth, minlength=k).max().item() / max(len(y), 1)
        score = (accuracy - majority) / max(1 - majority, 1e-6)
        return {
            "columns": {spec["names"][0]: score},
            "score": score,
            "group": spec["group"],
            "accuracy": accuracy,
            "majority": majority,
        }
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
        "group": spec["group"],
    }


def group_means(scored: dict, targets: dict) -> dict:
    """Mean target score per group (fast, medium, slow)."""
    out = {}
    for g in ("fast", "medium", "slow"):
        scores = [t["score"] for n, t in scored.items() if targets[n]["group"] == g and t["score"] is not None]
        out[g] = float(np.mean(scores)) if scores else None
    return out


def score_predictions(preds: dict, y: dict, targets: dict) -> dict:
    """Per target and group: {"targets": {name: result}, "groups": {group: mean score}}."""
    scored = {name: score_prediction(targets[name], p, y[name]) for name, p in preds.items()}
    return {"targets": scored, "groups": group_means(scored, targets)}


def score_targets(fitted: dict, z, y: dict, targets: dict) -> dict:
    return score_predictions({name: f["predict"](z) for name, f in fitted.items()}, y, targets)


def class_recalls(y: torch.Tensor, pred: torch.Tensor, classes: int, weight=None) -> tuple[np.ndarray, np.ndarray]:
    """Samples per class (unweighted) and recall per class (weighted when `weight` is given; nan without samples)."""
    y, pred = y.long().flatten(), pred.long().flatten()
    w = torch.ones(len(y)) if weight is None else torch.as_tensor(weight, dtype=torch.float32).flatten()
    n = torch.bincount(y, minlength=classes).numpy()
    total = torch.bincount(y, weights=w, minlength=classes)
    hit = torch.bincount(y, weights=w * (pred == y).float(), minlength=classes)
    recall = torch.where(total > 0, hit / total.clamp_min(1e-12), torch.full_like(total, float("nan")))
    return n, recall.numpy()


def balanced_accuracy(n: np.ndarray, recall: np.ndarray, min_n: int = 20) -> float | None:
    """Mean recall over the classes with at least `min_n` samples (every class counts the same)."""
    keep = (n >= min_n) & ~np.isnan(recall)
    return float(recall[keep].mean()) if keep.any() else None


def confusion(y: torch.Tensor, pred: torch.Tensor, classes: int, weight=None) -> np.ndarray:
    """(true, predicted) counts, weighted when `weight` is given."""
    y, pred = y.long().flatten(), pred.long().flatten()
    w = None if weight is None else torch.as_tensor(weight, dtype=torch.float64).flatten()
    flat = torch.bincount(y * classes + pred, weights=w, minlength=classes * classes)
    return flat.reshape(classes, classes).double().numpy()


def spearman(a, b) -> float | None:
    """Rank correlation (ties get their mean rank); None with fewer than 3 pairs or a constant side."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if len(a) < 3:
        return None

    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x), dtype=np.float64)
        for v in np.unique(x):
            tie = x == v
            r[tie] = r[tie].mean()
        return r

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------
# episode bootstrap


def episode_bootstrap(stat, groups, n: int = 200, seed: int = 0, level: float = 0.95):
    """Percentile interval of `stat(indices)` over resamples of whole episodes: the frames of an episode are not
    independent, so frame-level intervals are too narrow. `stat` returns a float, a 1-d array or None. Returns
    (low, high) of the same shape (nan where undefined), or None for n = 0 or no groups."""
    groups = np.asarray(groups)
    if n <= 0 or len(groups) == 0:
        return None
    ids, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        pick = rng.integers(0, len(ids), len(ids))
        v = stat(np.concatenate([members[i] for i in pick]))
        values.append(None if v is None else np.asarray(v, dtype=np.float64))
    shape = next((v.shape for v in values if v is not None), None)
    if shape is None:
        return None
    stack = np.stack([np.full(shape, np.nan) if v is None else v for v in values])
    tail = (1 - level) / 2 * 100
    with np.errstate(all="ignore"), _quiet_nan():
        low, high = np.nanpercentile(stack, tail, axis=0), np.nanpercentile(stack, 100 - tail, axis=0)
    if low.ndim == 0:
        return float(low), float(high)
    return low, high


class _quiet_nan:
    """Silences numpy's all-nan slice warning (columns without samples stay nan)."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


# --------------------------------------------------------------------------
# trained probes with several heads


@dataclass
class Head:
    name: str
    kind: str  # "cls": softmax over `classes` per slot; "bin": a logit per slot; "reg": a value per slot
    slots: int
    classes: int = 1

    @property
    def width(self) -> int:
        return self.slots * (self.classes if self.kind == "cls" else 1)


def make_net(kind: str, dim: int, out: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(dim, out)
    return nn.Sequential(
        nn.Linear(dim, 512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, out),
    )


def class_weights(y: torch.Tensor, classes: int, weight=None) -> torch.Tensor:
    """Inverse frequency of each class among the scored entries of `y` (-1 = not scored), scaled so that the present
    classes average to one per class (0 for an absent class)."""
    valid = y >= 0
    w = torch.ones_like(y, dtype=torch.float32) if weight is None else weight
    counts = torch.bincount(y[valid], weights=w[valid].float(), minlength=classes).float()
    present = (counts > 0).sum().clamp_min(1)
    return torch.where(counts > 0, counts.sum() / (present * counts.clamp_min(1e-12)), torch.zeros_like(counts))


def train_heads(
    kind: str,
    x: torch.Tensor,
    targets: dict,
    heads: list[Head],
    *,
    epochs: int = 30,
    steps: int | None = None,
    balanced: bool = False,
    sample_weight=None,
    seed: int = 0,
    device: str = "cpu",
    batch: int = 512,
    lr: float = 1e-3,
):
    """Fit a softmax probe ("linear") or an MLP with several heads on standardised features; returns predict(z) ->
    {head: output}: class logits (N, slots, classes), flag logits (N, slots), values (N, slots).

    targets: {head: (N, slots)}, long class indices for "cls" (-1 = not scored), floats for "bin" and "reg" (nan = not
    scored). `balanced`: a "cls" head weights each class by the inverse of its training frequency, a "bin" head weights
    the positives of each slot by negatives / positives, so rare classes count as much as common ones. `sample_weight`
    multiplies every sample's loss (the natural weights of a stratified sample). `steps`, when given, replaces `epochs`
    by a budget of optimizer steps (whole epochs, at least that many steps), so that small sets are not undertrained."""
    torch.manual_seed(seed)
    x = x.float()
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    xs = ((x - mean) / std).to(device)
    sw = torch.ones(len(x)) if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32)
    prepared, extra = {}, {}
    for h in heads:
        t = targets[h.name].reshape(len(x), h.slots)
        if h.kind == "cls":
            t = t.long()
            cw = class_weights(t, h.classes, sw[:, None].expand_as(t)) if balanced else torch.ones(h.classes)
            extra[h.name] = cw.to(device)
        elif h.kind == "bin":
            t = t.float()
            valid = ~torch.isnan(t)
            pos = ((t > 0.5) & valid).float().sum(0)
            neg = ((t <= 0.5) & valid).float().sum(0)
            extra[h.name] = (neg / pos.clamp_min(1)).clamp(max=1e4).to(device) if balanced else None
        else:
            t = t.float()
            mu, sd = torch.nanmean(t, 0), torch.sqrt(torch.nanmean((t - torch.nanmean(t, 0)) ** 2, 0)).clamp_min(1e-6)
            extra[h.name] = (mu, sd)
            t = (t - mu) / sd
        prepared[h.name] = t.to(device)
    sw = sw.to(device)
    net = make_net(kind, x.shape[1], sum(h.width for h in heads)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)

    def split(out):
        parts, a = {}, 0
        for h in heads:
            chunk = out[:, a : a + h.width]
            a += h.width
            parts[h.name] = chunk.reshape(len(out), h.slots, h.classes) if h.kind == "cls" else chunk
        return parts

    if steps is not None:
        epochs = max(1, math.ceil(steps / max(1, math.ceil(len(xs) / batch))))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xs), device=device)
        for i in range(0, len(xs), batch):
            idx = perm[i : i + batch]
            parts = split(net(xs[idx]))
            w = sw[idx]
            loss = 0.0
            for h in heads:
                t, p = prepared[h.name][idx], parts[h.name]
                if h.kind == "cls":
                    valid = t >= 0
                    ce = F.cross_entropy(
                        p.reshape(-1, h.classes), t.clamp_min(0).reshape(-1), weight=extra[h.name], reduction="none"
                    ).reshape(t.shape)
                    ww = w[:, None] * valid
                elif h.kind == "bin":
                    valid = ~torch.isnan(t)
                    ce = F.binary_cross_entropy_with_logits(
                        p, torch.nan_to_num(t), pos_weight=extra[h.name], reduction="none"
                    )
                    ww = w[:, None] * valid
                else:
                    valid = ~torch.isnan(t)
                    ce = (p - torch.nan_to_num(t)).pow(2)
                    ww = w[:, None] * valid
                loss = loss + (ce * ww).sum() / ww.sum().clamp_min(1e-12)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    @torch.no_grad()
    def predict(z: torch.Tensor) -> dict:
        outs = []
        for i in range(0, len(z), 4096):
            outs.append(net(((z[i : i + 4096].float() - mean) / std).to(device)).float().cpu())
        parts = split(torch.cat(outs) if outs else torch.zeros(0, sum(h.width for h in heads)))
        for h in heads:
            if h.kind == "reg":
                mu, sd = extra[h.name]
                parts[h.name] = parts[h.name] * sd + mu
        return parts

    return predict
