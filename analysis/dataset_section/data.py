"""Constants of Craftax-Classic episode files and `Split`, the small columns of one file as one set."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie",
    "collect_stone", "place_stone", "eat_plant", "defeat_skeleton",
    "make_stone_pickaxe", "make_stone_sword", "wake_up", "place_furnace",
    "collect_coal", "collect_iron", "collect_diamond", "make_iron_pickaxe",
    "make_iron_sword",
]  # fmt: skip
DIAMOND_INDEX = ACHIEVEMENTS.index("collect_diamond")
ACTIONS = [
    "noop", "left", "right", "up", "down", "do", "sleep", "place_stone", "place_table",
    "place_furnace", "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword", "make_iron_sword",
]  # fmt: skip
VITALS = ("health", "food", "drink", "energy")  # column order of `vitals`
INVENTORY = [
    "wood", "stone", "coal", "iron", "diamond", "sapling", "wood pickaxe", "stone pickaxe", "iron pickaxe",
    "wood sword", "stone sword", "iron sword",
]  # fmt: skip
MOBS = ("zombies", "cows", "skeletons", "arrows")  # column order of `mobs_in_view`
# layout of the `state` column (the renderer): the 7x9 view as a block one-hot plus mob flags per tile, then the rest
VIEW_ROWS, VIEW_COLS = 7, 9
VIEW_TILES = VIEW_ROWS * VIEW_COLS
VIEW_CHANNELS, BLOCK_TYPES = 21, 17
VIEW_DIM = VIEW_TILES * VIEW_CHANNELS
STATE_DIM = 1345


class Split:
    """The small columns of one episode file, every frame, with the episode of each frame.

    Frames are contiguous per episode in file order (`ep_offset` is the cumulative `ep_len`). Columns the file lacks are
    missing from `cols`; `has` asks. `final_achievements` and `dead` are read at the last frame of each episode.
    """

    COLUMNS = ("action", "achievements", "inventory", "vitals", "facing", "daylight", "position", "mobs_in_view")

    def __init__(self, path):
        self.path = str(path)
        with h5py.File(path, "r") as f:
            self.lengths = f["ep_len"][:].astype(np.int64)
            offsets = f["ep_offset"][:].astype(np.int64)
            if not np.array_equal(offsets, np.concatenate([[0], np.cumsum(self.lengths)[:-1]])):
                raise SystemExit(f"{path}: episodes are not stored contiguously in file order")
            self.total = int(self.lengths.sum())
            self.cols = {c: f[c][: self.total] for c in self.COLUMNS if c in f}
            last = offsets + self.lengths - 1
            self.dead = f["dead"][last].astype(bool) if "dead" in f else np.zeros(len(self.lengths), bool)
            self.runs = json.loads(f.attrs["runs"]) if "runs" in f.attrs else []
            self.state_ok = "state" in f and f["state"].shape[1] == STATE_DIM
        self.offsets = offsets
        self.n_episodes = len(self.lengths)
        self.episode = np.repeat(np.arange(self.n_episodes), self.lengths)
        self.final_achievements = self.cols["achievements"][last].astype(bool) if "achievements" in self.cols else None

    def has(self, *names) -> bool:
        return all(n in self.cols for n in names)

    def episodes(self, name) -> list[np.ndarray]:
        """The rows of a column, one array per episode."""
        arr = self.cols[name]
        return [arr[o : o + n] for o, n in zip(self.offsets, self.lengths)]


def file_record(path, with_sha: bool) -> dict:
    p = Path(path)
    st = p.stat()
    rec = {"path": str(p), "bytes": st.st_size, "mtime": int(st.st_mtime)}
    if with_sha:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        rec["sha256"] = h.hexdigest()
    return rec


def git_state() -> dict:
    here = Path(__file__).resolve().parents[1]  # analysis/
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", str(here)], cwd=here, text=True).strip())
        return {"commit": commit, "analysis_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "analysis_dirty": None}
