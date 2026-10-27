"""The probe suite of a trained model (see probing/__init__.py): runs at the end of training (health.probe_at_end) and
as the `probe` jobs of experiments/run.py.

    python probe_suite.py --run <run dir> [--checkpoint <ckpt>] [--out <dir>] [--train-dataset X --val-dataset Y]

Writes <out>/probe_report.md and <out>/probe_summary.json (default <run>/health). Needs the `state` column in both
datasets. The run time on an A100 is not measured yet (estimate 20 to 30 minutes with 512 clips per split: about 60
trained probes of 4000 steps, the sprite swaps, the rollouts).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import health
from probing import encoder as enc
from probing import predictor as pr
from probing import targets as tg
from probing.collect import clip_episodes, collect_split
from probing.report import render
from probing.verdict import entity_verdicts


def has_columns(cache_dir, name, columns) -> list[str]:
    """The columns of `columns` the dataset lacks (`state` must have the collector's width)."""
    import h5py

    from stable_worldmodel.data.utils import get_cache_dir

    try:
        with h5py.File(Path(cache_dir or get_cache_dir()) / f"{name}.h5", "r") as f:
            missing = [c for c in columns if c not in f]
            if "state" in f and f["state"].shape[1] != tg.STATE_DIM:
                missing.append("state (wrong width)")
            return missing
    except OSError:
        return ["(dataset not readable)"]


def git_commit() -> str:
    """The commit of the code that ran the suite (with a mark for uncommitted changes), or 'unknown'."""
    import subprocess

    here = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if commit.returncode:
        return "unknown"
    return commit.stdout.strip() + (" (uncommitted changes)" if dirty.stdout.strip() else "")


def facts(train: dict, val: dict) -> dict:
    """Base rates of the probe clips: block types per view tile, mobs per frame and tile, events per 1000 awake
    steps."""
    shares = enc.tile_shares(train, val)
    mobs = {}
    scored = torch.arange(tg.VIEW_TILES) != tg.PLAYER_INDEX
    for k, name in enumerate(tg.MOB_NAMES):
        frames = sum(int(s["y"]["mobs"][..., k].any(-1).sum()) for s in (train, val))
        tiles = sum(int(s["y"]["mobs"][:, scored, k].sum()) for s in (train, val))
        total = sum(len(s["y"]["mobs"]) for s in (train, val))
        mobs[name] = {"frames": frames / max(total, 1), "tiles": tiles / max(total * int(scored.sum()), 1)}
    counts, awake = {}, 0
    for s in (train, val):
        names, b, t = pr._event_names(s, 0)
        view = lambda x: pr.view(s, x)  # noqa: E731
        action, sleeping = view(s["action"]), view(s["y"]["sleeping"][:, 0] > 0.5)
        awake += int(((action[:, :-1] >= 0) & ~sleeping[:, :-1]).sum())
        ys = {g: view(s["y"][g]) for g in ("inventory", "vitals")}
        for n, bi, ti in zip(names, b.tolist(), t.tolist()):
            g, col, sign = pr.EVENT_COLUMN[n]
            ok = sign * (ys[g][bi, ti + 1, col] - ys[g][bi, ti, col]) > 0
            c = counts.setdefault(n, [0, 0])
            c[0 if ok else 1] += 1
    events = {
        n: {"succeeded": 1000 * c[0] / max(awake, 1), "failed": 1000 * c[1] / max(awake, 1)}
        for n, c in sorted(counts.items())
    }
    return {"tile_share": shares.tolist(), "mobs": mobs, "events": events, "awake_steps": awake}


def merge_sprites(*libraries) -> dict:
    out = {"blocks": {}, "mobs": {}}
    for lib in libraries:
        for kind in ("blocks", "mobs"):
            for key, crops in lib.get(kind, {}).items():
                out[kind].setdefault(key, []).extend(crops)
    return out


def compat_levels(state: dict, upper: dict) -> dict:
    """Per level: groups, the untrained control and the per-target scores (what compare_runs and the findings read).
    With upper levels, every level on the frames they share; level 1 alone: all frames."""
    ridge = state["ridge"]
    level1 = {"groups": ridge["latent"]["groups"], "control_groups": ridge["control"]["groups"], "targets": {}}
    for n, t in ridge["latent"]["targets"].items():
        level1["targets"][n] = {**t, "control": ridge["control"]["targets"][n]["score"]}
    if not upper:
        return {1: level1}
    levels = {n: {"groups": e["groups"], "targets": e["targets"]} for n, e in upper.items()}
    levels[1]["control_groups"] = level1["control_groups"]
    return levels


def run_suite(
    model,
    cfg,
    cache_dir,
    *,
    transform,
    normalize,
    device,
    frames=129,
    max_clips=512,
    batch_size=16,
    seed=0,
    steps=4000,
    layers=(3, 6, 9),
    token_cap=4000,
    strip_every=4,
    swap_frames=512,
    boot=200,
    horizons=pr.HORIZONS,
    every=8,
    offsets=(8, 16, 32, 64),
    spectrum_ks=(4, 8, 16, 32, 64, 128),
    min_n=20,
    checkpoint=None,
    log=None,
):
    """The whole suite on a trained model. Returns (summary, findings). `checkpoint` is only recorded in the report."""
    from hjepa_utils import create_world_model
    from training_monitor import build_health_loader

    log = log or (lambda msg: print(f"[probe suite] {msg}", flush=True))
    start = time.time()
    names = {"train": cfg.data.dataset.name, "val": cfg.data.dataset.val_name}
    columns = list(tg.COLUMNS)
    for split, name in names.items():
        missing = has_columns(cache_dir, name, columns)
        if missing:
            raise RuntimeError(f"the {split} dataset {name!r} lacks {missing}: collect it with the current collector")
    optional = [c for c in tg.OPTIONAL_COLUMNS if all(not has_columns(cache_dir, n, [c]) for n in names.values())]
    with torch.random.fork_rng(devices=[]):  # a seeded control, without reseeding the caller's RNG
        torch.manual_seed(seed)
        control = create_world_model(cfg)[0].to(device)
    data = {}
    for split, name in names.items():
        loader, used = build_health_loader(
            cfg,
            cache_dir,
            transform,
            frames=frames,
            batch_size=batch_size,
            max_clips=max_clips,
            seed=seed,
            name=name,
            columns=columns + optional,
            disjoint=True,
        )
        if loader is None:
            raise RuntimeError(f"no clips of the {split} dataset {name!r} for probing")
        data[split] = collect_split(
            model,
            control,
            loader,
            cfg,
            device,
            normalize,
            episodes=clip_episodes(loader),
            layers=layers,
            token_cap=token_cap,
            strip_every=strip_every,
            swap_frames=swap_frames if split == "val" else 0,
            seed=seed,
        )
        log(f"collected {split}: {data[split]['clips']} clips of {used} frames ({time.time() - start:.0f} s)")
    del control
    train, val = data["train"], data["val"]
    jepa = model.get_level(1)
    train_history = int(cfg.level1.wm.history_size)
    plan_history = int(cfg.get("health", {}).get("plan_history", 1))
    summary = {
        "settings": {
            "clips (train / held-out)": f"{train['clips']} / {val['clips']}",
            "frames per clip": val["T"],
            "episodes (train / held-out)": " / ".join(str(len(np.unique(s["clip_episode"]))) for s in (train, val)),
            "tile sample (train / held-out)": " / ".join(str(len(s["tokens"].get("block", []))) for s in (train, val)),
            "ViT layers": ", ".join(str(layer) for layer in val["layers"]) or "-",
            "probe steps": steps,
            "bootstrap": boot,
            "history (training / planning)": f"{train_history} / {plan_history}",
            "min_samples": min_n,
            "commit": git_commit(),
            "checkpoint": str(checkpoint) if checkpoint else "the model in memory (end of training)",
            "device": str(device),
        }
    }
    summary["facts"] = facts(train, val)

    summary["state"], fitted = enc.state_probes(train, val, steps=steps, device=device, boot=boot, seed=seed)
    summary["upper_levels"] = enc.upper_levels(train, val, seed)
    summary["levels"] = compat_levels(summary["state"], summary["upper_levels"])
    log(f"state probes ({time.time() - start:.0f} s)")
    summary["maps"], decoders = enc.map_decode(
        train, val, steps=steps, device=device, boot=boot, min_n=min_n, seed=seed
    )
    log(f"map decode ({time.time() - start:.0f} s)")
    summary["tokens"] = enc.token_probes(train, val, steps=steps, device=device, boot=boot, min_n=min_n, seed=seed)
    log(f"token probes ({time.time() - start:.0f} s)")
    summary["strip"] = enc.strip_chain(train, val, seed)
    sprites = merge_sprites(train["sprites"], val["sprites"])
    shares = enc.tile_shares(train, val)
    summary["causes"] = enc.causes(summary["maps"], summary["tokens"], shares, enc.sprite_distances(sprites), min_n)
    summary["spectrum"] = enc.spectrum(
        train, val, shares, ks=spectrum_ks, steps=steps, device=device, min_n=min_n, seed=seed
    )
    log(f"strip, causes, spectrum ({time.time() - start:.0f} s)")
    summary["swaps"] = enc.swap_sensitivity(
        model, val, sprites, decoders, normalize=normalize, device=device, boot=boot, seed=seed
    )
    summary["distance"] = enc.distance_sensitivity(val, enc.step_scale(val, "latent"))
    log(f"swaps, distance ({time.time() - start:.0f} s)")
    summary["cross"] = enc.cross_probes(train, val, steps=steps, device=device, boot=boot, min_n=min_n, seed=seed)
    summary["displacement"] = enc.displacement(train, val, steps=steps, device=device, seed=seed)
    log(f"facing-relative tiles, displacement ({time.time() - start:.0f} s)")

    summary["loss"] = pr.loss_attribution(val, jepa, history=train_history, device=device)
    summary["change"] = {
        h: pr.change_test(
            fitted, val, jepa, history=h, horizons=horizons, every=every, device=device, boot=boot, seed=seed
        )
        for h in sorted({plan_history, train_history})
    }
    summary["change_plan"] = summary["change"][plan_history]
    summary["events"] = pr.event_probes(
        fitted, val, jepa, decoders.get("latent mlp"), history=train_history, device=device
    )
    summary["mobs"] = pr.mob_dynamics(val, jepa, decoders.get("latent mlp"), history=train_history, device=device)
    summary["monotonicity"] = pr.monotonicity(
        val, jepa, history=plan_history, offsets=offsets, every=every, device=device
    )
    summary["idm"] = pr.idm_per_action(val, jepa, device)
    summary["faced"] = {
        "by_cause": pr.faced_by_cause(fitted, val, jepa, history=train_history, device=device),
        "swap": pr.faced_swap(model, val, sprites, fitted, normalize=normalize, device=device, seed=seed),
    }
    log(f"predictor ({time.time() - start:.0f} s)")

    summary["verdicts"] = entity_verdicts(summary, min_n)
    summary["n_train"] = {1: len(train["frames"]["latent"])}
    summary["n_val"] = {1: len(val["frames"]["latent"])}
    summary["dim"] = {1: int(train["frames"]["latent"].shape[1])}
    summary["settings"]["run time (s)"] = round(time.time() - start)
    return summary, health.diagnose_probing(summary)


def write(out: Path, summary: dict, findings) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_report.md").write_text(render(summary, findings))
    (out / "probe_summary.json").write_text(
        json.dumps({"summary": health.to_jsonable(summary), "findings": health.findings_to_json(findings)}, indent=1)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory with config.yaml and *_object.ckpt")
    parser.add_argument("--checkpoint", help="model to probe (default: the newest *_object.ckpt)")
    parser.add_argument("--out", help="report directory (default: <run>/health)")
    parser.add_argument("--cache-dir", default=os.environ.get("HJEPA_HOME"))
    parser.add_argument("--train-dataset"), parser.add_argument("--val-dataset")
    parser.add_argument("--frames", type=int, default=129)
    parser.add_argument(
        "--max-clips", type=int, default=512, help="clips per split (disjoint, whole clips of `frames`)"
    )
    parser.add_argument("--steps", type=int, default=4000, help="optimizer steps of every trained probe")
    parser.add_argument(
        "--layers", type=int, nargs="*", default=[3, 6, 9], help="ViT layers whose patch tokens are probed"
    )
    parser.add_argument("--token-cap", type=int, default=4000, help="sampled tiles per block or mob type and split")
    parser.add_argument("--swap-frames", type=int, default=512, help="bright held-out frames for the sprite swaps")
    parser.add_argument("--boot", type=int, default=200, help="episode bootstrap resamples (0 = no intervals)")
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(pr.HORIZONS), help="rollout steps of the change test"
    )
    args = parser.parse_args()

    from cpu_threads import limit_threads

    limit_threads()

    from omegaconf import OmegaConf

    from data import Compose, OneHotColumn, discrete_num_actions
    from main_hjepa import _normalize_uint8_images_on_device

    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    if args.train_dataset:
        cfg.data.dataset.name = args.train_dataset
    if args.val_dataset:
        cfg.data.dataset.val_name = args.val_dataset
    ckpt = (
        Path(args.checkpoint)
        if args.checkpoint
        else sorted(run.glob("*_object.ckpt"), key=lambda p: p.stat().st_mtime)[-1]
    )
    model = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = model if hasattr(model, "num_levels") else model.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    num_actions = discrete_num_actions(cfg)
    transform = Compose(OneHotColumn(num_actions, "action", "action")) if num_actions else Compose()
    summary, findings = run_suite(
        model,
        cfg,
        Path(args.cache_dir),
        transform=transform,
        normalize=_normalize_uint8_images_on_device,
        device=device,
        frames=args.frames,
        max_clips=args.max_clips,
        steps=args.steps,
        layers=tuple(args.layers),
        token_cap=args.token_cap,
        swap_frames=args.swap_frames,
        boot=args.boot,
        horizons=tuple(args.horizons),
        checkpoint=ckpt,
    )
    out = Path(args.out) if args.out else run / "health"
    write(out, summary, findings)
    print(health.findings_block(findings))
    print(f"wrote {out / 'probe_report.md'}")


if __name__ == "__main__":
    main()
