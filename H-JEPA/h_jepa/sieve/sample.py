"""The fixed frames every Sieve metric reads, drawn once and cached.

Two parts, each episodes x windows x consecutive frames (`slowness.sample.nested_sample`: windows spread over an
episode, one contiguous read each):
    train   training worlds: the probes and the fresh decoder are fitted here
    val     held-out worlds: every score
A transition is a frame and the next one inside a window (the action between them is known). The labels (`state`
and the columns next to it) are only read by the probes, never by a model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = ("pixels", "action", "state", "inventory", "vitals", "facing", "daylight")


@dataclass
class Part:
    """One part of the sample, flattened to frames (N = episodes x windows x length)."""

    pixels: np.ndarray  # (N, 63, 63, 3) uint8
    action: np.ndarray  # (N,) float, the action taken at the frame; NaN at an episode's last frame
    state: np.ndarray  # (N, 1345) float32
    inventory: np.ndarray  # (N, 12)
    vitals: np.ndarray  # (N, 4)
    facing: np.ndarray  # (N,) 1..4
    daylight: np.ndarray  # (N,)
    episode: np.ndarray  # (N,) the file's episode index
    length: int  # frames per window

    def __len__(self) -> int:
        return len(self.pixels)

    def pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Indices (t, t + 1) of the transitions inside the windows with a known action."""
        n = len(self)
        t = np.arange(n)
        keep = (t % self.length != self.length - 1) & ~np.isnan(self.action)
        return t[keep], t[keep] + 1


def _read(path, episodes: int, windows: int, length: int, seed: int) -> Part:
    import h5py

    from slowness import sample as sm

    with h5py.File(Path(path), "r") as f:
        lengths, offsets = f["ep_len"][:], f["ep_offset"][:]
    layout = sm.nested_sample(lengths, offsets, windows=windows, length=length, seed=seed, random_episodes=episodes)
    cols = sm.read_columns(path, layout, COLUMNS)
    flat = {k: v.reshape(-1, *v.shape[3:]) for k, v in cols.items() if k in COLUMNS}
    E, W, L = layout["rows"].shape
    return Part(
        pixels=flat["pixels"],
        action=flat["action"].reshape(-1).astype(np.float32),
        state=flat["state"].astype(np.float32),
        inventory=flat["inventory"].astype(np.float32),
        vitals=flat["vitals"].astype(np.float32),
        facing=flat["facing"].reshape(-1).astype(np.int64),
        daylight=flat["daylight"].reshape(-1).astype(np.float32),
        episode=np.repeat(layout["episode"], W * L),
        length=L,
    )


@dataclass
class Sample:
    train: Part
    val: Part


SIZES = {  # episodes x windows x frames per window, and the training steps of the probes and of the fresh decoder
    "full": dict(train=192, val=128, windows=4, length=32, probe_steps=1500, fresh_steps=3000),  # 24,576 + 16,384 frames
    "quick": dict(train=8, val=8, windows=2, length=16, probe_steps=100, fresh_steps=100),  # a plumbing check only
}


def load(train_file, val_file, size: str = "full", seed: int = 0, cache_dir=None) -> Sample:
    """The sample of the two files; cached as `sieve_sample_<hash>.npz` in `cache_dir` (default: next to the val file)."""
    s = SIZES[size]
    key = hashlib.sha1(f"{Path(train_file).name}|{Path(val_file).name}|{size}|{seed}".encode()).hexdigest()[:10]
    cache = Path(cache_dir or Path(val_file).parent) / f"sieve_sample_{key}.npz"
    if cache.exists():
        z = np.load(cache)
        parts = {}
        for name in ("train", "val"):
            fields = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(name + "/")}
            parts[name] = Part(**{k: v for k, v in fields.items() if k != "length"}, length=int(fields["length"]))
        return Sample(**parts)
    sample = Sample(
        train=_read(train_file, s["train"], s["windows"], s["length"], seed),
        val=_read(val_file, s["val"], s["windows"], s["length"], seed + 1),
    )
    arrays = {f"{name}/{k}": np.asarray(v) for name in ("train", "val") for k, v in vars(getattr(sample, name)).items()}
    tmp = cache.with_name(cache.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(cache)
    return sample
