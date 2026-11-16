"""The metrics worker: evaluates every epoch checkpoint of a run as it appears, next to the training on the same GPU.

    python -m sieve.watch --run <run dir> --train-file <train.h5> --val-file <val.h5> [--bank <bank.npz>]
                          [--until-epoch N] [--training-pid-file <pid>] [--once] [--size full|quick]

Reads `<name>_epoch_<N>_object.ckpt` (epoch 0 is the model the run started from), writes `<run>/sieve/epoch_<N>.json` per
checkpoint and `<run>/sieve/report.md` (+ `.png`, `report.json`) after each; a rerun skips the epochs done. It stops after
epoch `--until-epoch`, after one pass with `--once`, or when the training process of `--training-pid-file` has ended and
nothing new is left. The run's `config.yaml` says the stage (stage 2 adds the fresh decoder). Progress lines start with
`[sieve]`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch

EPOCH = re.compile(r"_epoch_(\d+)_object\.ckpt$")
SETTLE = 20.0  # seconds a checkpoint file must be untouched before it is read (torch.save is not atomic)


def log(msg: str) -> None:
    print(f"[sieve] {msg}", flush=True)


def checkpoints(run: Path) -> dict[int, Path]:
    found = {int(m.group(1)): p for p in run.glob("*_object.ckpt") if (m := EPOCH.search(p.name))}
    return dict(sorted(found.items()))


def _alive(pid_file) -> bool:
    if pid_file is None:
        return True
    try:
        os.kill(int(Path(pid_file).read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def load_model(path: Path):
    model = torch.load(path, map_location="cpu", weights_only=False)
    return model if hasattr(model, "get_level") else model.model


def _summary(m: dict) -> str:
    keys = ("holds_mob_cow", "holds_mob_zombie", "holds_map_iron", "holds_map_coal", "holds_facing", "holds_faced_tile")
    parts = [f"{k[6:]} {m[k]:.2f}" for k in keys if isinstance(m.get(k), float)]
    for k in ("effective_rank", "recon_error_mean", "moves_for_noise"):
        if isinstance(m.get(k), float):
            parts.append(f"{k} {m[k]:.3g}")
    return ", ".join(parts)


def watch(run, train_file, val_file, bank=None, until_epoch=None, pid_file=None, once=False, size="full", poll=30.0,
          device=None) -> list[int]:  # fmt: skip
    from omegaconf import OmegaConf

    from cpu_threads import limit_threads

    from . import check, report
    from . import counterfactuals as cf
    from . import sample as sm

    limit_threads()
    run = Path(run)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = run / "sieve"
    while not (run / "config.yaml").exists():  # the training writes it once its data is built
        if once or not _alive(pid_file):
            raise FileNotFoundError(f"{run}/config.yaml")
        time.sleep(poll)
    cfg = OmegaConf.load(run / "config.yaml")
    sieve_cfg = cfg.level1.get("sieve", None)
    stage = int(sieve_cfg.stage) if sieve_cfg is not None else 0
    t0 = time.time()
    sample = sm.load(train_file, val_file, size=size)
    bank_arrays = cf.load(bank) if bank and Path(bank).exists() else None
    log(f"{run.name}: stage {stage}, sample {len(sample.train)} training and {len(sample.val)} held-out frames "
        f"({time.time() - t0:.0f} s), counterfactuals {'yes' if bank_arrays else 'no'}, {device}")  # fmt: skip
    out.mkdir(exist_ok=True)
    evaluated = []
    settle = 0.0 if once else SETTLE  # a one-off pass reads files of a finished training
    while True:
        done = {int(m.group(1)) for p in out.glob("epoch_*.json") if (m := re.search(r"epoch_(\d+)\.json$", p.name))}
        todo = [(e, p) for e, p in checkpoints(run).items() if e not in done and time.time() - p.stat().st_mtime >= settle]
        for epoch, path in todo:
            t1 = time.time()
            try:
                model = load_model(path)
            except Exception as error:  # a file still being written: the next poll reads it
                log(f"epoch {epoch}: not readable yet ({error!r})")
                continue
            budget = sm.SIZES[size]
            metrics = check.evaluate(model, sample, bank_arrays, sieve_cfg, device, fresh=stage == 2,
                                     probe_steps=budget["probe_steps"], fresh_steps=budget["fresh_steps"],
                                     log=lambda step, e=epoch: log(f"epoch {e}: {step}"))  # fmt: skip
            metrics.update(epoch=epoch, stage=stage, checkpoint=path.name)
            tmp = out / f"epoch_{epoch}.json.tmp"
            tmp.write_text(json.dumps(metrics, indent=1))
            tmp.replace(out / f"epoch_{epoch}.json")
            report.write(run)
            evaluated.append(epoch)
            log(f"epoch {epoch} done in {time.time() - t1:.0f} s: {_summary(metrics)}")
            done.add(epoch)
        finished = until_epoch is not None and all(e in done for e in range(int(until_epoch) + 1))
        if once or finished or (not todo and not _alive(pid_file) and not _pending(run, done)):
            break
        time.sleep(poll)
    report.write(run)
    log(f"finished: {len(done)} epochs evaluated, report {out / 'report.md'}")
    return evaluated


def _pending(run: Path, done: set) -> bool:
    return any(e not in done for e in checkpoints(run))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--train-file", required=True, type=Path)
    p.add_argument("--val-file", required=True, type=Path)
    p.add_argument("--bank", type=Path, help="the counterfactual frames (sieve.counterfactuals); without: no moves_for_* metrics")
    p.add_argument("--until-epoch", type=int, help="stop once this epoch is evaluated")
    p.add_argument("--training-pid-file", type=Path, help="stop when this process has ended and nothing is left to read")
    p.add_argument("--once", action="store_true", help="one pass over the checkpoints there are")
    p.add_argument("--size", choices=["full", "quick"], default="full")
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--device")
    args = p.parse_args(argv)
    watch(args.run, args.train_file, args.val_file, args.bank, args.until_epoch, args.training_pid_file, args.once,
          args.size, args.poll, args.device)  # fmt: skip


if __name__ == "__main__":
    main()
