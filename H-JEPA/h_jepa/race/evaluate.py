"""Read the latent of every checkpoint of a run on the fixed held-out sample (`race.readout`).

    python -m race.evaluate --run <run dir> --val-file <val.h5> [--max-val-episodes N]

The checkpoints are the dense ones (`<name>_step_<N>_object.ckpt`, `utils.DenseCheckpoints`, step 0 = the untrained model) and
the epochs' (`<name>_epoch_<N>_object.ckpt`, at step N x steps per epoch). Writes `<run>/race/trace.json` (one entry per step,
written as they come; steps already in the file are skipped, so a rerun resumes; `trace_done` at the end) and `<run>/race/meta.json` (what `race.report`
prints about the run). The run's `config.yaml` says what the encoder reads (`obs_key`), the history and the number of actions.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from . import readout

STEP = re.compile(r"_step_(\d+)_object\.ckpt$")
EPOCH = re.compile(r"_epoch_(\d+)_object\.ckpt$")


def log(msg: str) -> None:
    print(f"[race] {msg}", flush=True)


def checkpoints(run: Path, steps_per_epoch: int) -> dict[int, Path]:
    """The model files of a run by training step (a dense file wins over an epoch file of the same step)."""
    found = {}
    for path in sorted(run.glob("*_object.ckpt")):
        if m := EPOCH.search(path.name):
            found[int(m.group(1)) * steps_per_epoch] = path
    for path in sorted(run.glob("*_object.ckpt")):
        if m := STEP.search(path.name):
            found[int(m.group(1))] = path
    return dict(sorted(found.items()))


def load_model(path: Path):
    model = torch.load(path, map_location="cpu", weights_only=False)
    return model if hasattr(model, "num_levels") else model.model


def evaluate(run: Path, val_file: Path, max_val_episodes=None, device=None, seed: int = 0) -> list[dict]:
    from omegaconf import OmegaConf

    from cpu_threads import limit_threads

    limit_threads()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(run / "config.yaml")
    obs_key = str(cfg.get("obs_key", "pixels"))
    history, num_actions = int(cfg.level1.wm.history_size), int(cfg.level1.wm.num_actions)
    dense = json.loads((run / "dense_checkpoints.json").read_text()) if (run / "dense_checkpoints.json").exists() else {}
    per_epoch = int(dense.get("steps_per_epoch", 0)) or int(cfg.get("steps_per_epoch", 0))
    files = checkpoints(run, per_epoch)
    if not files:
        raise FileNotFoundError(f"no *_object.ckpt in {run}")
    out = run / "race"
    out.mkdir(exist_ok=True)
    trace_path = out / "trace.json"
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
    done = {e["step"] for e in trace}
    (out / "meta.json").write_text(json.dumps({
        "run": run.name, "obs_key": obs_key, "history": history, "steps_per_epoch": per_epoch,
        "stop_grad": bool(cfg.level1.wm.get("detach_pred_target", False)), "sigreg": float(cfg.level1.loss.embed.sigreg.weight),
        "dense": dense, "val_file": str(val_file), "checks": list(files),
    }, indent=1))  # fmt: skip
    t0 = time.time()
    sample = readout.Sample(val_file, obs_key=obs_key, seed=seed, max_episodes=max_val_episodes)
    log(f"{len(files)} checkpoints, {sample.shape[0]} held-out episodes ({time.time() - t0:.0f} s), obs {obs_key}, {device}")
    for step, path in files.items():
        if step in done:
            continue
        t0 = time.time()
        entry = {"step": step, "epoch": step / max(per_epoch, 1), **readout.check(load_model(path), sample, history, num_actions, device)}
        trace.append(entry)
        trace.sort(key=lambda e: e["step"])
        trace_path.write_text(json.dumps(trace))
        r2 = entry["r2"]
        pick = " ".join(f"{k.split(':')[-1]} {r2[k]:.2f}" for k in ("mob:cow", "mob:zombie", "view:iron", "view:coal") if r2.get(k) is not None)
        log(f"step {step} ({time.time() - t0:.0f} s): rank {entry['rank']:.0f}, near {entry['near']:.2f}, {pick}, pred/copy {entry['pred_over_copy']:.2f}")  # fmt: skip
    (out / "trace_done").write_text(f"{len(trace)} checks\n")  # trace.json grows during the run: this marks it finished
    return trace


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path, help="the run directory (config.yaml and the *_object.ckpt files)")
    p.add_argument("--val-file", required=True, type=Path, help="the held-out file the sample is drawn from")
    p.add_argument("--max-val-episodes", type=int, help="the first N eligible episodes only (a smoke run)")
    p.add_argument("--device")
    args = p.parse_args(argv)
    evaluate(args.run, args.val_file, args.max_val_episodes, args.device)


if __name__ == "__main__":
    main()
