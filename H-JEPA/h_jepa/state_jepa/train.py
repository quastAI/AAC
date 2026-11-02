"""Train the MLP-JEPA on the state (A3; A4 with `--stop-grad`) and read the latent at many steps of the first epoch.

    python -m state_jepa.train --train-file <train.h5> --val-file <val.h5> --out <run dir> [--stop-grad] [--epochs 1]

Writes to <run dir>: `trace.json` (the readouts of `trace.check` at every check step, written as they come; step 0 is the
untrained encoder), `losses.csv` (the training losses every `--log-every` steps), `config.json` and `model.pt` (the
final weights). B1's settings where they carry over: batch 128, clips of 4 frames (history 3), latent 192, SIGReg 0.09,
IDM 0.1, AdamW with weight decay 0.001, gradient norm clipped at 1, linear warm-up then cosine decay over the run. Not
B1's: the learning rate (1e-3, an MLP; B1's 5e-5 is the ViT's) and the dead frame that B1 pads after a death.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from . import data, trace
from .model import Config, StateJEPA

CHECKS = (0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000)  # dense where the race would be, then every `--check-every`


def log(msg: str) -> None:
    print(f"[state_jepa] {msg}", flush=True)


def check_steps(total: int, early=CHECKS, every: int = 1000) -> list[int]:
    """The steps at which the latent is read: the early ones, then every `every` steps, and the last step."""
    return sorted({s for s in early if s <= total} | set(range(0, total + 1, every)) | {total})


def schedule(step: int, total: int, warmup: float = 0.01) -> float:
    """The learning-rate factor: linear warm-up over the first `warmup` of the steps, then cosine to 0."""
    w = max(1, round(warmup * total))
    if step < w:
        return (step + 1) / w
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - w) / max(1, total - w))))


def describe(entry: dict) -> str:
    r2 = entry["r2"]
    pick = " ".join(f"{k.split(':')[-1]} {r2[k]:.2f}" for k in ("mob:cow", "mob:zombie", "view:iron", "view:coal") if r2.get(k) is not None)
    return f"rank {entry['rank']:.0f}, dead {entry['dead_axes']}, near {entry['near']:.2f}, {pick}, pred/copy {entry['pred_over_copy']:.2f}"


def train(args) -> list[dict]:
    from cpu_threads import limit_threads

    limit_threads()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(history=args.history, latent_dim=args.latent_dim, hidden=args.hidden, sigreg=args.sigreg, idm=args.idm,
                 stop_grad=args.stop_grad)  # fmt: skip
    clip = cfg.history + 1

    t0 = time.time()
    table = data.load(args.train_file, sources=args.sources, max_episodes=args.max_episodes)
    starts = data.clip_starts(table, clip)
    per_epoch = len(starts) // args.batch_size
    total = args.max_steps if args.max_steps is not None else max(1, round(args.epochs * per_epoch))
    sample = trace.Sample(args.val_file, windows=args.val_windows, length=args.val_length, max_episodes=args.max_val_episodes)
    log(f"{len(table.action)} training frames, {len(starts)} clips ({per_epoch} steps per epoch), {total} steps; "
        f"{sample.state.shape[0]} held-out episodes in the sample; {time.time() - t0:.0f} s")  # fmt: skip

    model = StateJEPA(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: schedule(s, total))
    gen = torch.Generator().manual_seed(args.seed)
    checks = check_steps(total, every=args.check_every)
    (out / "config.json").write_text(json.dumps({"model": cfg.to_dict(), "args": vars(args) | {"out": str(out)},
                                                 "steps": total, "steps_per_epoch": per_epoch, "checks": checks}, indent=1))  # fmt: skip

    history, rows, running = [], [], {}
    order, cursor = torch.randperm(len(starts), generator=gen), 0
    t0 = time.time()
    for step in range(total + 1):
        if step in checks:
            entry = {"step": step, "epoch": step / max(per_epoch, 1), **trace.check(model, sample)}
            history.append(entry)
            (out / "trace.json").write_text(json.dumps(history))
            log(f"step {step}/{total} ({time.time() - t0:.0f} s): {describe(entry)}")
        if step == total:
            break
        if cursor + args.batch_size > len(order):  # a new epoch: a new order of the clips
            order, cursor = torch.randperm(len(starts), generator=gen), 0
        x, action = data.batch(table, starts[order[cursor : cursor + args.batch_size]], clip)
        cursor += args.batch_size
        losses = model.losses(x, action)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v)
        if (step + 1) % args.log_every == 0:
            rows.append({"step": step + 1, "lr": sched.get_last_lr()[0], **{k: v / args.log_every for k, v in running.items()}})
            running = {}
            if (step + 1) % (args.log_every * 10) == 0:
                last = rows[-1]
                log(f"step {step + 1}/{total}: loss {last['loss']:.3f} (pred {last['pred']:.4f}, sigreg {last['sigreg']:.2f}, "
                    f"idm {last['idm']:.3f}, accuracy {last['idm_accuracy']:.2f}), {time.time() - t0:.0f} s")  # fmt: skip

    if rows:
        with open(out / "losses.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict()}, out / "model.pt")
    log(f"done in {time.time() - t0:.0f} s: {out}")
    return history


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True, help="the held-out sample the latent is read on")
    p.add_argument("--out", required=True, help="the run directory")
    p.add_argument("--stop-grad", action="store_true", help="A4: detach the prediction target")
    p.add_argument("--epochs", type=float, default=1.0, help="passes over the training clips (1 = B1's first epoch)")
    p.add_argument("--max-steps", type=int, help="a fixed number of steps instead (a smoke run)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--latent-dim", type=int, default=192)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--sigreg", type=float, default=0.09)
    p.add_argument("--idm", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--sources", type=int, nargs="+", default=list(data.SOURCES))
    p.add_argument("--max-episodes", type=int, help="the first N training episodes only (a smoke run)")
    p.add_argument("--max-val-episodes", type=int, help="the first N eligible held-out episodes only (a smoke run)")
    p.add_argument("--val-windows", type=int, default=4)
    p.add_argument("--val-length", type=int, default=64)
    p.add_argument("--check-every", type=int, default=1000, help="steps between checks after the early ones")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    train(parser().parse_args(argv))


if __name__ == "__main__":
    main()
