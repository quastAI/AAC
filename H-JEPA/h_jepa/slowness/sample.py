"""The nested sample: E episodes x W windows x L consecutive frames.

Every eligible episode (length >= W * L) is cut into W equal segments and one window of L consecutive frames starts at a
random place inside each segment, so the windows of an episode never overlap and are spread over its whole length. One
sample serves every analysis: between episodes (the E axis), between windows of an episode (the W axis, minutes apart),
within a window (the L axis, frames apart) and, as the first four frames of a window, within a training clip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WINDOWS = 4
LENGTH = 64
CLIP = 4  # frames of a training clip (history 3 + 1)

# the columns of the ground-truth analysis (the factors) next to the pixels
FACTORS = ("state", "inventory", "vitals", "daylight", "facing", "achievements", "action", "position", "step_idx")
EPISODE_COLUMNS = ("source", "dead")


def nested_sample(
    lengths, offsets, *, windows: int = WINDOWS, length: int = LENGTH, seed: int = 0, max_episodes: int | None = None
) -> dict:
    """The sample of a file with the episode lengths and the row of every episode's first frame.

    Returns `episode` (E,) the file's episode indices, `start` (E, W) the first frame of each window within its episode,
    `rows` (E, W, L) the rows of the file and `t` (E, W, L) the frame's time step within its episode.
    """
    lengths, offsets = np.asarray(lengths, np.int64), np.asarray(offsets, np.int64)
    episode = np.flatnonzero(lengths >= windows * length)[:max_episodes]
    if not len(episode):
        raise ValueError(f"no episode has {windows * length} frames: lower `windows` or `length`")
    rng = np.random.default_rng(seed)
    start = np.empty((len(episode), windows), np.int64)
    for i, ep in enumerate(episode):
        edges = np.linspace(0, lengths[ep], windows + 1).astype(np.int64)
        for w in range(windows):
            hi = edges[w + 1] - length  # the window must end inside its segment
            start[i, w] = rng.integers(edges[w], hi + 1)
    t = start[:, :, None] + np.arange(length)
    return {"episode": episode, "start": start, "rows": offsets[episode][:, None, None] + t, "t": t}


def read_columns(path, sample: dict, columns) -> dict:
    """The columns of the file for the sample, each as (E, W, L, ...). Windows are contiguous, so each is one slice."""
    import h5py

    rows = sample["rows"]
    out = {}
    with h5py.File(Path(path), "r") as f:
        for name in columns:
            data = f[name]
            arr = np.empty(rows.shape + data.shape[1:], data.dtype)
            for i in range(rows.shape[0]):
                for w in range(rows.shape[1]):
                    a = int(rows[i, w, 0])
                    arr[i, w] = data[a : a + rows.shape[2]]
            out[name] = arr
        for name in EPISODE_COLUMNS:  # one value per episode: the first frame's (source) or the last's (dead)
            if name in f:
                ep = sample["episode"]
                offsets, lengths = f["ep_offset"][:][ep], f["ep_len"][:][ep]
                out[name] = f[name][offsets if name == "source" else offsets + lengths - 1]
    return out


def spread_frames(sample: dict, per_episode: int) -> np.ndarray:
    """(E, per_episode) indices into the flattened W * L frames of each episode: evenly spread over its windows, for
    analyses that want frames far apart (the between-episode share) and not a dense run."""
    total = sample["rows"].shape[1] * sample["rows"].shape[2]
    return np.linspace(0, total - 1, per_episode).round().astype(np.int64)[None, :].repeat(len(sample["episode"]), 0)
