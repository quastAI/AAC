"""Is the block the agent faces in the latent? A linear and a nonlinear probe on the encoded latents of a checkpoint.

    python tile_probe.py --run <run> --checkpoint <run>/flat_epoch_2_object.ckpt --out <run>/health/tile_probe_epoch_2

probe_eval.py reads the faced block with a ridge probe (about 69% accuracy at epoch 2, 36% for the majority class). This
trains a softmax probe and a small MLP on the same latents (clips of the training worlds, scored on the held-out worlds,
a fixed number of epochs, nothing tuned on the held-out clips) and prints the accuracy and, per block type, the recall and
the most frequent mistake. If the MLP is far above the linear probe, the block is in the latent but not linearly readable;
if both stay about equal, the latent does not hold it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import probe_eval as pe


def make_net(kind: str, dim: int, classes: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(dim, classes)
    return nn.Sequential(
        nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, classes)
    )  # fmt: skip


def train_probe(kind: str, ztr, ytr, zva, classes: int, epochs: int = 30, seed: int = 0, batch: int = 512):
    """Fit a softmax probe ("linear") or an MLP on the standardised latents; returns the predicted classes (train, val)."""
    torch.manual_seed(seed)
    mean, std = ztr.mean(0), ztr.std(0).clamp_min(1e-6)
    xtr, xva = (ztr - mean) / std, (zva - mean) / std
    net = make_net(kind, xtr.shape[1], classes)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            F.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(xtr).argmax(-1), net(xva).argmax(-1)


def per_class(yva: torch.Tensor, preds: dict, names: list[str], min_n: int = 20) -> list[dict]:
    """Per block type of the held-out worlds (at least `min_n` samples): count, recall of each probe, and the most
    frequent wrong prediction of the last probe."""
    rows = []
    last = list(preds.values())[-1]
    for c in range(len(names)):
        mask = yva == c
        n = int(mask.sum())
        if n < min_n:
            continue
        wrong = last[mask][last[mask] != c]
        top = torch.bincount(wrong, minlength=len(names)).argmax().item() if len(wrong) else None
        rows.append(
            {
                "block": names[c],
                "n": n,
                **{k: float((p[mask] == c).float().mean()) for k, p in preds.items()},
                "mistake": None if top is None else names[top],
                "mistake_share": 0.0 if top is None else float((wrong == top).float().sum() / n),
            }
        )
    return rows


def render(summary: dict, rows: list[dict]) -> str:
    lines = [
        "# Faced block: linear and nonlinear probe",
        "",
        f"Samples (train / held-out): {summary['n_train']} / {summary['n_val']}; majority class on the held-out worlds: "
        f"{summary['majority']:.2f}.",
        "",
        "| probe | train accuracy | held-out accuracy |",
        "|---|---|---|",
        *(f"| {k} | {a['train']:.2f} | {a['val']:.2f} |" for k, a in summary["accuracy"].items()),
        "",
        "Per block type of the held-out worlds (recall of each probe; the mistake is the MLP's most frequent wrong answer,"
        " as a share of all samples of the type):",
        "",
        "| block | samples | " + " | ".join(summary["accuracy"]) + " | most frequent mistake |",
        "|---|---|" + "---|" * len(summary["accuracy"]) + "---|",
    ]
    for r in rows:
        recalls = " | ".join(f"{r[k]:.2f}" for k in summary["accuracy"])
        mistake = "-" if r["mistake"] is None else f"{r['mistake']} ({r['mistake_share']:.2f})"
        lines.append(f"| {r['block']} | {r['n']} | {recalls} | {mistake} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory with config.yaml")
    parser.add_argument("--checkpoint", required=True, help="model to probe, e.g. an epoch checkpoint")
    parser.add_argument("--out", help="report directory (default: <run>/health/tile_probe)")
    parser.add_argument("--cache-dir", default=os.environ.get("HJEPA_HOME"))
    parser.add_argument("--frames", type=int, default=129)
    parser.add_argument("--max-clips", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    from cpu_threads import limit_threads

    limit_threads()

    from omegaconf import OmegaConf

    from data import Compose, OneHotColumn, discrete_num_actions
    from main_hjepa import _normalize_uint8_images_on_device
    from training_monitor import build_health_loader

    run, cache_dir = Path(args.run), Path(args.cache_dir)
    cfg = OmegaConf.load(run / "config.yaml")
    names = {"train": cfg.data.dataset.name, "val": cfg.data.dataset.val_name}
    assert all(pe.has_state(cache_dir, n) for n in names.values()), "the datasets need the `state` column"
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model if hasattr(model, "num_levels") else model.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    num_actions = discrete_num_actions(cfg)
    transform = Compose(OneHotColumn(num_actions, "action", "action")) if num_actions else Compose()
    targets = {"facing_block": pe.STATE_TARGETS["facing_block"]}
    columns = ("pixels", "action", *pe.COLUMNS, "state")
    data = {}
    for split, name in names.items():
        loader, _ = build_health_loader(
            cfg, cache_dir, transform, frames=args.frames, batch_size=16, max_clips=args.max_clips, seed=0, name=name, columns=columns
        )  # fmt: skip
        z, y = pe.collect(model, loader, cfg, device, _normalize_uint8_images_on_device, 1, targets)[1]
        data[split] = (z, y["facing_block"][:, 0].long())
    del model
    (ztr, ytr), (zva, yva) = data["train"], data["val"]
    classes = len(pe.BLOCK_NAMES)
    preds, accuracy = {}, {}
    for kind in ("linear", "mlp"):
        ptr, pva = train_probe(kind, ztr, ytr, zva, classes, epochs=args.epochs)
        preds[kind] = pva
        accuracy[kind] = {"train": float((ptr == ytr).float().mean()), "val": float((pva == yva).float().mean())}
    summary = {
        "n_train": len(ytr),
        "n_val": len(yva),
        "majority": float(torch.bincount(yva, minlength=classes).max() / len(yva)),
        "accuracy": accuracy,
    }
    report = render(summary, per_class(yva, preds, pe.BLOCK_NAMES))
    out = Path(args.out) if args.out else run / "health" / "tile_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tile_probe.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
