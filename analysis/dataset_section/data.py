"""Constants of Craftax-Classic episode files and `Split`, the small columns of one file as one set."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import NamedTuple

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
SLEEPING_COLUMN = STATE_DIM - 1  # the last entry of `state`: the player is asleep
NOOP = 0
PLAYER_TILE = (VIEW_ROWS // 2, VIEW_COLS // 2)  # (row, column) of the player in the view
FACING_STEP = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # `facing` -> the neighbouring tile it looks at (left, right, up, down)
CREATURES = 0b0111  # bits of a tile's mob code: zombie, cow, skeleton (the arrows are the fourth bit)
STATE_CHUNK = 50_000  # frames per read of the `state` column


class View(NamedTuple):
    """The `state` column reduced to what the analysis needs, per frame."""

    block: np.ndarray  # (frames, 63) uint8: block id of every tile of the view
    mob: np.ndarray  # (frames, 63) uint8: 4-bit code of the mobs on every tile (zombie, cow, skeleton, arrow)
    sleeping: np.ndarray  # (frames,) bool


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
        self._view = None
        self._executed = None

    def has(self, *names) -> bool:
        return all(n in self.cols for n in names)

    def view(self) -> View:
        """The 7x9 view and the sleeping flag from the `state` column, read once (needs `state_ok`)."""
        if self._view is None:
            block = np.empty((self.total, VIEW_TILES), np.uint8)
            mob = np.empty((self.total, VIEW_TILES), np.uint8)
            sleeping = np.empty(self.total, bool)
            weights = (1 << np.arange(4)).astype(np.uint8)
            with h5py.File(self.path, "r") as f:
                state = f["state"]
                for a in range(0, self.total, STATE_CHUNK):
                    x = state[a : min(a + STATE_CHUNK, self.total)]
                    tiles = x[:, :VIEW_DIM].reshape(-1, VIEW_TILES, VIEW_CHANNELS)
                    block[a : a + len(x)] = tiles[:, :, :BLOCK_TYPES].argmax(2)
                    mob[a : a + len(x)] = ((tiles[:, :, BLOCK_TYPES:] > 0.5).astype(np.uint8) * weights).sum(2, dtype=np.uint8)
                    sleeping[a : a + len(x)] = x[:, SLEEPING_COLUMN] > 0.5
            self._view = View(block, mob, sleeping)
        return self._view

    def executed_action(self) -> np.ndarray:
        """The `action` column as the game executed it, (frames,) float with NaN where there is no action.

        While the player is asleep Craftax replaces every action by a noop, so a step taken asleep is a noop. Files
        relabeled with purejaxql/scripts/relabel_asleep.py (and collected since) hold this already; older files hold
        the policy's choice, and the rule is applied here, so both give the same numbers. Without the `state` column
        (the sleeping flag) the column is used as it is.
        """
        if self._executed is None:
            action = self.cols["action"][:, 0].astype(np.float32)
            if self.state_ok:
                action = action.copy()
                action[self.view().sleeping & ~np.isnan(action)] = NOOP
            self._executed = action
        return self._executed

    def asleep_steps(self) -> int:
        """Steps taken while asleep (all noop by the game); 0 without the `state` column."""
        if not self.state_ok:
            return 0
        return int((self.view().sleeping & ~np.isnan(self.cols["action"][:, 0])).sum())

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
