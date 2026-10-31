"""Encode the nested sample with several checkpoints and an untrained control, and store the features once.

    python -m slowness.encode --run <run dir> --data <val.h5> --out <dir> [--epochs 1 2 5 10] [--windows 4 --length 64]

Writes <out>/sample.npz (episodes, windows, rows, time steps, the factor columns) and <out>/features_<name>.npz with the
CLS token before the projector and the latent of every frame, shaped (E, W, L, D). The names are `epoch<N>` and
`untrained`. Later analyses read these files only, so a rerun of an analysis does not encode again.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import sample as sm


def load_model(path):
    model = torch.load(path, map_location="cpu", weights_only=False)
    return model if hasattr(model, "num_levels") else model.model


def untrained_model(cfg, seed: int = 0):
    """A freshly initialised model of the run's configuration (the control of the probe suite uses the same seed)."""
    from hjepa_utils import create_world_model

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return create_world_model(cfg)[0]


def find_checkpoints(run: Path, epochs) -> dict[str, Path]:
    """`epoch<N>` for the saved epochs asked for (the last epoch's file equals the run's final model)."""
    found = {}
    for n in epochs:
        hits = sorted(run.glob(f"*_epoch_{n}_object.ckpt"))
        if not hits:
            raise FileNotFoundError(f"no *_epoch_{n}_object.ckpt in {run}")
        found[f"epoch{n}"] = hits[0]
    return found


def encode_frames(
    model, pixels: np.ndarray, normalize, device: str, chunk: int = 256, desc: str | None = None
) -> dict[str, np.ndarray]:
    """CLS token and latent of uint8 frames (N, 63, 63, 3); `desc` shows a progress bar over the frames."""
    from probing.encoder import encode_pixels
    from tqdm import tqdm

    model = model.to(device)
    latent, cls = [], []
    step = chunk * 32
    with tqdm(total=len(pixels), desc=desc, disable=desc is None, unit="frame", unit_scale=True, leave=False) as bar:
        for a in range(0, len(pixels), step):
            x = torch.from_numpy(np.ascontiguousarray(pixels[a : a + step].transpose(0, 3, 1, 2)))
            with torch.no_grad():
                lat, c = encode_pixels(model, x, normalize, device, chunk=chunk)
            latent.append(lat.numpy())
            cls.append(c.numpy())
            bar.update(len(x))
    return {"latent": np.concatenate(latent), "cls": np.concatenate(cls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory with config.yaml and the *_object.ckpt files")
    parser.add_argument("--data", required=True, help="the validation episode file (.h5)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, nargs="*", default=[1, 2, 5, 10])
    parser.add_argument("--windows", type=int, default=sm.WINDOWS)
    parser.add_argument("--length", type=int, default=sm.LENGTH)
    parser.add_argument("--max-episodes", type=int, help="only the first N eligible episodes (a smoke run)")
    parser.add_argument("--random-episodes", type=int, help="a random subset of N eligible episodes (a big file is ordered)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()

    from cpu_threads import limit_threads

    limit_threads()

    import h5py
    from main_hjepa import _normalize_uint8_images_on_device
    from omegaconf import OmegaConf

    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.load(run / "config.yaml")

    with h5py.File(args.data, "r") as f:
        lengths, offsets = f["ep_len"][:], f["ep_offset"][:]
    sample = sm.nested_sample(lengths, offsets, windows=args.windows, length=args.length, seed=args.seed, max_episodes=args.max_episodes, random_episodes=args.random_episodes)
    t0 = time.time()
    columns = sm.read_columns(args.data, sample, ("pixels",) + sm.FACTORS, progress=True)
    print(f"read {sample['rows'].size} frames of {len(sample['episode'])} episodes in {time.time() - t0:.0f} s", flush=True)
    pixels = columns.pop("pixels")
    shape = sample["rows"].shape
    np.savez(out / "sample.npz", **sample, **columns)

    models = {name: path for name, path in find_checkpoints(run, args.epochs).items()}
    for name in [*models, "untrained"]:
        target = out / f"features_{name}.npz"
        if target.exists():
            print(f"{name}: {target} exists, skipped", flush=True)
            continue
        model = untrained_model(cfg, args.seed) if name == "untrained" else load_model(models[name])
        t0 = time.time()
        feats = encode_frames(model, pixels.reshape(-1, *pixels.shape[3:]), _normalize_uint8_images_on_device, device, args.chunk, desc=f"encode {name}")
        np.savez(target, **{k: v.reshape(*shape, -1) for k, v in feats.items()})
        print(f"{name}: {len(pixels.reshape(-1, *pixels.shape[3:]))} frames in {time.time() - t0:.0f} s on {device}", flush=True)


if __name__ == "__main__":
    main()
