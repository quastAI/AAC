"""Probing: what do the latents of each level encode of the symbolic state?

Closed-form ridge probes (no training loop, deterministic) from the latents of
every level to symbolic targets of Craftax-Classic, fitted on training worlds
and scored on held-out worlds. Targets are grouped by how fast they change:

    fast    position, facing direction, mobs in view
    medium  health, food, drink, energy
    slow    inventory, achievements, daylight

Scores are comparable across targets: R^2 for counts and levels, 2*AUROC - 1 for
binary flags, and (accuracy - majority) / (1 - majority) for the facing
direction. The same probes on a randomly initialised encoder of the same
architecture are the control: what a probe finds there is architecture and data,
the difference is what training added.

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
COLUMNS = list(TARGETS)
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)  # relative to the number of samples


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


def probe_target(spec: dict, ztr, ytr, zva, yva, seed=0) -> dict:
    """Fit on the training split, score on the held-out one. Returns column scores and the target score."""
    kind = spec["kind"]
    if kind == "cls":
        k = spec["classes"]
        onehot = torch.nn.functional.one_hot(ytr[:, 0].long(), k).float()
        predict, lam = fit_with_validation(ztr, onehot, seed)
        accuracy = (predict(zva).argmax(1) == yva[:, 0].long()).float().mean().item()
        majority = torch.bincount(yva[:, 0].long(), minlength=k).max().item() / len(yva)
        score = (accuracy - majority) / max(1 - majority, 1e-6)
        return {"columns": {"direction": score}, "score": score, "lambda": lam, "accuracy": accuracy, "majority": majority}
    predict, lam = fit_with_validation(ztr, ytr if kind == "reg" else ytr * 2 - 1, seed)
    pred = predict(zva)
    if kind == "reg":
        columns = r2_columns(yva, pred)
    else:
        columns = []
        for j in range(yva.shape[1]):
            a = auroc(yva[:, j].numpy(), pred[:, j].numpy())
            columns.append(None if a is None else 2 * a - 1)
    valid = [c for c in columns if c is not None]
    return {
        "columns": dict(zip(spec["names"], columns)),
        "score": float(np.mean(valid)) if valid else None,
        "lambda": lam,
    }


def probe_levels(train: dict, val: dict) -> dict:
    """train/val: {level: (Z, {column: Y})}. Scores per level, target and group."""
    out = {}
    for level in train:
        ztr, ytr = train[level]
        zva, yva = val[level]
        targets = {name: probe_target(spec, ztr, ytr[name], zva, yva[name]) for name, spec in TARGETS.items()}
        groups = {}
        for g in ("fast", "medium", "slow"):
            scores = [t["score"] for n, t in targets.items() if TARGETS[n]["group"] == g and t["score"] is not None]
            groups[g] = float(np.mean(scores)) if scores else None
        out[level] = {"targets": targets, "groups": groups}
    return out


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


@torch.no_grad()
def collect(model, loader, cfg, device, normalize, thin: int = 1):
    levels = range(1, model.num_levels + 1)
    jepas = [model.get_level(n) for n in levels]
    saved = [j.target_length for j in jepas]
    was_training = model.training
    for j in jepas:
        j.target_length = None
    model.eval()
    z = {n: [] for n in levels}
    y = {n: {c: [] for c in COLUMNS} for n in levels}
    try:
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            normalize(batch)
            out = model.encode_hierarchical(batch)
            common = common_frames(model, cfg, batch["pixels_level1"].shape[1], thin)
            for n in levels:
                emb = out[f"embed_{n}"].float().cpu()
                steps = torch.arange(emb.shape[1])
                frames = frame_of_step(n, steps, cfg)
                keep = torch.isin(frames, common)
                z[n].append(emb[:, steps[keep]].flatten(0, 1))
                for c in COLUMNS:
                    col = batch[f"{c}_level1"].float().cpu()
                    if c == "facing":
                        col = col - 1  # directions 1..4 -> classes 0..3
                    y[n][c].append(col[:, frames[keep]].flatten(0, 1))
    finally:
        for j, t in zip(jepas, saved):
            j.target_length = t
        model.train(was_training)
    return {n: (torch.cat(z[n]), {c: torch.cat(y[n][c]) for c in COLUMNS}) for n in levels}


# --------------------------------------------------------------------------
# the whole analysis


def run_probing(
    model, cfg, cache_dir, *, transform, normalize, device, frames=129, max_clips=256, thin=1,
    control=True, batch_size=16, seed=0,
):  # fmt: skip
    """Probe every level on held-out worlds, with an untrained control. Returns (summary, findings)."""
    from training_monitor import build_health_loader

    columns = ("pixels", "action", *COLUMNS)
    loaders = {}
    for split, name in (("train", cfg.data.dataset.name), ("val", cfg.data.dataset.val_name)):
        loader, used = build_health_loader(
            cfg, cache_dir, transform, frames=frames, batch_size=batch_size, max_clips=max_clips,
            seed=seed, name=name, columns=columns,
        )  # fmt: skip
        if loader is None:
            raise RuntimeError(f"no clips of the {split} dataset {name!r} for probing")
        loaders[split] = loader
    train = collect(model, loaders["train"], cfg, device, normalize, thin)
    val = collect(model, loaders["val"], cfg, device, normalize, thin)
    summary = {
        "levels": probe_levels(train, val),
        "n_train": {n: len(train[n][0]) for n in train},
        "n_val": {n: len(val[n][0]) for n in val},
        "dim": {n: train[n][0].shape[1] for n in train},
    }
    if control:
        from hjepa_utils import create_world_model

        with torch.random.fork_rng(devices=[]):  # a seeded control, without reseeding the caller's RNG
            torch.manual_seed(seed)
            random_model = create_world_model(cfg)[0]
        random_model = random_model.to(device)
        random_levels = probe_levels(
            collect(random_model, loaders["train"], cfg, device, normalize, thin),
            collect(random_model, loaders["val"], cfg, device, normalize, thin),
        )
        for n, entry in summary["levels"].items():
            entry["control_groups"] = random_levels[n]["groups"]
            for name, t in entry["targets"].items():
                t["control"] = random_levels[n]["targets"][name]["score"]
    return summary, health.diagnose_probing(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory with config.yaml and *_object.ckpt")
    parser.add_argument("--cache-dir", default=os.environ.get("HJEPA_HOME"))
    parser.add_argument("--train-dataset"), parser.add_argument("--val-dataset")
    parser.add_argument("--frames", type=int, default=129)
    parser.add_argument("--max-clips", type=int, default=256)
    parser.add_argument("--no-control", action="store_true")
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
