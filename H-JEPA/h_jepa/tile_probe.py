"""Is the block the agent faces in the latent, and where is it lost? Linear and nonlinear probes on four features.

    python tile_probe.py --run <run> --checkpoint <run>/flat_epoch_2_object.ckpt --out <run>/health/tile_probe_epoch_2

probe_eval.py reads the faced block from the latent with a ridge probe (about 69% accuracy at epoch 2, 36% for the
majority class). This trains a softmax probe and a small MLP (clips of the training worlds, scored on the held-out
worlds, a fixed number of epochs, nothing tuned on the held-out clips) on four features of the same frames:

- latent: the model's latent (the ViT's CLS token after the projector), what the planner compares;
- CLS: the ViT's CLS token before the projector;
- mean patches: the mean of the ViT's patch tokens of the map (7 x 9 tiles; one 7-pixel patch per tile);
- faced patch: the ViT's patch token at the faced tile.

If the faced patch reads the block and the latent does not, the encoder sees the tile and the summary drops it; if CLS
reads it and the latent does not, the projector drops it. Per block type: the recall of each feature (MLP).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import probe_eval as pe

FEATURES = {
    "latent": "the latent (CLS after the projector)",
    "cls": "CLS before the projector",
    "mean_patches": "mean of the map's patch tokens",
    "faced_patch": "patch token at the faced tile",
}
PATCHES_PER_ROW = pe.VIEW_COLS  # 63 x 63 pixels, 7-pixel patches: 9 x 9, the 7 rows of the map on top


def faced_patch_index(direction: torch.Tensor) -> torch.Tensor:
    """Token index (after the CLS token) of the tile the agent faces; `direction` is 1 to 4 (probe_eval.FACING_STEP)."""
    d = direction.long().clamp(1, 4)
    dr = torch.tensor([pe.FACING_STEP[k][0] for k in (1, 2, 3, 4)], device=d.device)[d - 1]
    dc = torch.tensor([pe.FACING_STEP[k][1] for k in (1, 2, 3, 4)], device=d.device)[d - 1]
    return 1 + (pe.PLAYER_TILE[0] + dr) * PATCHES_PER_ROW + pe.PLAYER_TILE[1] + dc


def features_of(hidden: torch.Tensor, latent: torch.Tensor, direction: torch.Tensor) -> dict:
    """The four features of frames from the ViT's tokens `hidden` (N, 1 + patches, D), their latents (N, D') and the
    facing directions (N,)."""
    index = faced_patch_index(direction)
    map_patches = hidden[:, 1 : 1 + pe.VIEW_ROWS * PATCHES_PER_ROW]
    return {
        "latent": latent,
        "cls": hidden[:, 0],
        "mean_patches": map_patches.mean(1),
        "faced_patch": hidden[torch.arange(len(hidden)), index],
    }


@torch.no_grad()
def collect_features(model, loader, cfg, device, normalize) -> tuple[dict, torch.Tensor]:
    """The four features and the faced block of every frame of the clips, on the CPU."""
    jepa = model.get_level(1)
    vit = jepa.encoder.encoder
    captured = []
    hook = vit.register_forward_hook(lambda module, args, output: captured.append(output.last_hidden_state.detach()))
    saved, was_training = jepa.target_length, model.training
    jepa.target_length = None
    model.eval()
    feats = {name: [] for name in FEATURES}
    blocks = []
    try:
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            normalize(batch)
            captured.clear()
            out = model.encode_hierarchical(batch)
            latent = out["embed_1"].float().flatten(0, 1)
            hidden = torch.cat(captured).float()
            assert len(hidden) == len(latent), "the ViT ran on a different number of frames than the latents"
            direction = batch["facing_level1"][..., 0].flatten(0, 1)
            for name, f in features_of(hidden, latent, direction).items():
                feats[name].append(f.cpu())
            blocks.append(pe.frame_targets(batch, {"facing_block": pe.STATE_TARGETS["facing_block"]})["facing_block"].flatten(0, 1))
    finally:
        hook.remove()
        jepa.target_length = saved
        model.train(was_training)
    return {name: torch.cat(v) for name, v in feats.items()}, torch.cat(blocks)[:, 0].long()


def make_net(kind: str, dim: int, classes: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(dim, classes)
    return nn.Sequential(
        nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, classes)
    )  # fmt: skip


def train_probe(kind: str, ztr, ytr, zva, classes: int, epochs: int = 30, seed: int = 0, batch: int = 512):
    """Fit a softmax probe ("linear") or an MLP on the standardised features; returns the predicted classes (train, val)."""
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


def per_class(yva: torch.Tensor, preds: dict, names: list[str], min_n: int = 50, mistake_of: str | None = None):
    """Per block type of the held-out worlds with at least `min_n` samples: count, recall of each prediction set, and the
    most frequent wrong answer of `mistake_of` (default: the last). Returns (rows, {skipped type: count})."""
    rows, skipped = [], {}
    chosen = preds[mistake_of] if mistake_of else list(preds.values())[-1]
    for c in range(len(names)):
        mask = yva == c
        n = int(mask.sum())
        if n == 0:
            continue
        if n < min_n:
            skipped[names[c]] = n
            continue
        wrong = chosen[mask][chosen[mask] != c]
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
    return rows, skipped


def render(summary: dict, rows: list[dict], skipped: dict, columns: list[str]) -> str:
    lines = [
        "# Faced block: where is it lost?",
        "",
        f"Samples (train / held-out): {summary['n_train']} / {summary['n_val']}; majority class on the held-out worlds: "
        f"{summary['majority']:.2f}.",
        "",
        "| feature | linear train | linear held-out | MLP train | MLP held-out |",
        "|---|---|---|---|---|",
    ]
    for feature, a in summary["accuracy"].items():
        lines.append(
            f"| {FEATURES.get(feature, feature)} | {a['linear']['train']:.2f} | {a['linear']['val']:.2f} | "
            f"{a['mlp']['train']:.2f} | {a['mlp']['val']:.2f} |"
        )
    note = ""
    if skipped:
        note = " Left out, too few held-out samples: " + ", ".join(f"{k} ({v})" for k, v in skipped.items()) + "."
    lines += [
        "",
        "Per block type of the held-out worlds: recall of the MLP on each feature; the mistake is the most frequent "
        f"wrong answer of the first feature, as a share of all samples of the type.{note}",
        "",
        "| block | samples | " + " | ".join(columns) + " | most frequent mistake |",
        "|---|---|" + "---|" * len(columns) + "---|",
    ]
    for r in rows:
        recalls = " | ".join(f"{r[k]:.2f}" for k in columns)
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
    parser.add_argument("--min-samples", type=int, default=50, help="block types with fewer held-out samples are left out of the table")
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
    assert model.num_levels == 1, "the patch-token features are for the flat model (one level)"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    num_actions = discrete_num_actions(cfg)
    transform = Compose(OneHotColumn(num_actions, "action", "action")) if num_actions else Compose()
    columns = ("pixels", "action", *pe.COLUMNS, "state")
    data = {}
    for split, name in names.items():
        loader, _ = build_health_loader(
            cfg, cache_dir, transform, frames=args.frames, batch_size=16, max_clips=args.max_clips, seed=0, name=name, columns=columns
        )  # fmt: skip
        data[split] = collect_features(model, loader, cfg, device, _normalize_uint8_images_on_device)
    del model
    (ftr, ytr), (fva, yva) = data["train"], data["val"]
    classes = len(pe.BLOCK_NAMES)
    accuracy, mlp_preds = {}, {}
    for feature in FEATURES:
        accuracy[feature] = {}
        for kind in ("linear", "mlp"):
            ptr, pva = train_probe(kind, ftr[feature], ytr, fva[feature], classes, epochs=args.epochs)
            accuracy[feature][kind] = {"train": float((ptr == ytr).float().mean()), "val": float((pva == yva).float().mean())}
            if kind == "mlp":
                mlp_preds[feature] = pva
    summary = {
        "n_train": len(ytr),
        "n_val": len(yva),
        "majority": float(torch.bincount(yva, minlength=classes).max() / len(yva)),
        "accuracy": accuracy,
    }
    rows, skipped = per_class(yva, mlp_preds, pe.BLOCK_NAMES, min_n=args.min_samples, mistake_of="latent")
    report = render(summary, rows, skipped, list(FEATURES))
    out = Path(args.out) if args.out else run / "health" / "tile_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tile_probe.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
