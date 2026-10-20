"""The change matrix: per state component and lag k, the share of frame pairs (t, t + k) of one episode in which it differs.

Net change (the value differs between the two frames), not "anything changed in between". Daylight is a float and counts
as changed above `DAYLIGHT_TOL` (a heuristic; it is also periodic with a day of 300 steps, so the share falls back near
lag 300). The view is egocentric, so walking changes every tile: besides the pooled egocentric rows, terrain and mobs are
compared in world coordinates on the tiles that both views show (needs `position`), for lags up to `WORLD_LAG_MAX` where
two views always overlap. `half_lag` is the first lag of the grid with a share of 50%.
"""

from __future__ import annotations

import h5py
import numpy as np

from .data import (BLOCK_TYPES, INVENTORY, MOBS, STATE_DIM, VIEW_CHANNELS, VIEW_COLS, VIEW_DIM, VIEW_ROWS, VIEW_TILES,
                   VITALS, Split)

LAGS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400, 600, 800)  # steps between the two frames of a pair
DAYLIGHT_TOL = 0.05  # a light change above this counts as a change (the light moves at most 0.031 per step)
WORLD_LAG_MAX = 6  # two views at most this many steps apart always overlap (the player moves one tile per step)
HALF = 0.5  # share of pairs that defines the half-change lag
PAIR_CHUNK = 200_000  # frame pairs per block of the view comparison (memory)
STATE_CHUNK = 50_000  # frames per read of the `state` column
# (group, label, column, index in the column; None: any of its columns): the rows read from small columns
COLUMN_ROWS = (
    [("position", n, "position", i) for i, n in enumerate(("row", "column"))]
    + [("facing", "facing", "facing", 0)]
    + [("mobs in view", n, "mobs_in_view", i) for i, n in enumerate(MOBS)]
    + [("vitals", n, "vitals", i) for i, n in enumerate(VITALS)]
    + [("inventory", n, "inventory", i) for i, n in enumerate(INVENTORY)]
    + [("daylight", "daylight", "daylight", 0), ("achievements", "any new unlock", "achievements", None)]
)
VIEW_LABELS = {
    "ego_any": "view, any tile", "ego_share": "view, share of tiles",
    "terrain_any": "terrain (world), any tile", "terrain_share": "terrain (world), share of tiles",
    "mobs_any": "mobs (world), any tile", "mobs_share": "mobs (world), share of tiles",
}  # fmt: skip
OVERLAP_ROW = "mean tiles shared by the two views"


def pair_starts(s: Split, k: int) -> np.ndarray:
    """Frames t with t + k in the same episode."""
    if k >= s.total:
        return np.zeros(0, np.int64)
    t = np.arange(s.total - k)
    return t[s.episode[:-k] == s.episode[k:]]


def column_changes(s: Split, column: str, index: int | None, t: np.ndarray, k: int) -> np.ndarray:
    """Whether the component differs between frame t and t + k, per pair."""
    x = s.cols[column].reshape(s.total, -1)
    if index is None:
        return (x[t + k] != x[t]).any(1)
    if column == "daylight":
        return np.abs(x[t + k, index].astype(np.float64) - x[t, index]) > DAYLIGHT_TOL
    return x[t + k, index] != x[t, index]


def view_tiles(path, total: int) -> tuple[np.ndarray, np.ndarray]:
    """Per frame and tile of the 7x9 view from the `state` column: the block id and the mobs as a 4-bit code."""
    block = np.empty((total, VIEW_TILES), np.uint8)
    mob = np.empty((total, VIEW_TILES), np.uint8)
    weights = (1 << np.arange(4)).astype(np.uint8)
    with h5py.File(path, "r") as f:
        state = f["state"]
        if state.shape[1] != STATE_DIM:
            raise SystemExit(f"{path}: state has {state.shape[1]} columns, expected {STATE_DIM}")
        for a in range(0, total, STATE_CHUNK):
            x = state[a : min(a + STATE_CHUNK, total), :VIEW_DIM].reshape(-1, VIEW_TILES, VIEW_CHANNELS)
            block[a : a + len(x)] = x[:, :, :BLOCK_TYPES].argmax(2)
            mob[a : a + len(x)] = ((x[:, :, BLOCK_TYPES:] > 0.5).astype(np.uint8) * weights).sum(2, dtype=np.uint8)
    return block, mob


def world_changes(block, mob, position, t: np.ndarray, k: int) -> dict:
    """Terrain and mobs in world coordinates over the tiles that the views of frame t and t + k share.

    The tile (i, j) of the view at t + k is the tile (i + dr, j + dc) of the view at t, with (dr, dc) the move of the
    player. Sums over the pairs: those with a changed tile, the share of changed tiles, and the overlap size.
    """
    dr = position[t + k, 0].astype(np.int64) - position[t, 0]
    dc = position[t + k, 1].astype(np.int64) - position[t, 1]
    if (np.abs(dr) >= VIEW_ROWS).any() or (np.abs(dc) >= VIEW_COLS).any():
        raise SystemExit("the player moved further than the view in a world-frame pair: lags above WORLD_LAG_MAX?")
    key = (dr + VIEW_ROWS) * 64 + (dc + VIEW_COLS)
    order = np.argsort(key, kind="stable")
    edges = np.flatnonzero(np.diff(key[order])) + 1
    sums = {n: {"any": 0, "share": 0.0} for n in ("terrain", "mobs")} | {"tiles": 0}
    for rows in np.split(order, edges):
        r, c = int(dr[rows[0]]), int(dc[rows[0]])
        i0, i1, j0, j1 = max(0, -r), min(VIEW_ROWS, VIEW_ROWS - r), max(0, -c), min(VIEW_COLS, VIEW_COLS - c)
        later, earlier = t[rows] + k, t[rows]
        tiles = (i1 - i0) * (j1 - j0)
        for name, plane in (("terrain", block), ("mobs", mob)):
            a = plane[later].reshape(-1, VIEW_ROWS, VIEW_COLS)[:, i0:i1, j0:j1]
            b = plane[earlier].reshape(-1, VIEW_ROWS, VIEW_COLS)[:, i0 + r : i1 + r, j0 + c : j1 + c]
            diff = (a != b).reshape(len(rows), -1)
            sums[name]["any"] += int(diff.any(1).sum())
            sums[name]["share"] += float(diff.sum() / tiles)
        sums["tiles"] += tiles * len(rows)
    return sums


def view_rows(s: Split, block, mob, lags=LAGS) -> list[dict]:
    """The pooled view rows per lag: egocentric, and (lags up to WORLD_LAG_MAX) world-frame terrain and mobs."""
    tile = block.astype(np.uint16) * 16 + mob
    position = s.cols["position"]
    values = {key: [] for key in VIEW_LABELS}
    overlap = []
    for k in lags:
        t = pair_starts(s, k)
        n = max(len(t), 1)
        ego_any, ego_share = 0, 0.0
        for a in range(0, len(t), PAIR_CHUNK):
            diff = tile[t[a : a + PAIR_CHUNK] + k] != tile[t[a : a + PAIR_CHUNK]]
            ego_any += int(diff.any(1).sum())
            ego_share += float(diff.mean(1).sum())
        values["ego_any"].append(ego_any / n if len(t) else None)
        values["ego_share"].append(ego_share / n if len(t) else None)
        if k <= WORLD_LAG_MAX and len(t):
            acc = {"terrain": {"any": 0, "share": 0.0}, "mobs": {"any": 0, "share": 0.0}, "tiles": 0}
            for a in range(0, len(t), PAIR_CHUNK):
                part = world_changes(block, mob, position, t[a : a + PAIR_CHUNK], k)
                for name in ("terrain", "mobs"):
                    acc[name]["any"] += part[name]["any"]
                    acc[name]["share"] += part[name]["share"]
                acc["tiles"] += part["tiles"]
            for name in ("terrain", "mobs"):
                values[f"{name}_any"].append(acc[name]["any"] / n)
                values[f"{name}_share"].append(acc[name]["share"] / n)
            overlap.append(acc["tiles"] / n)
        else:
            for name in ("terrain", "mobs"):
                values[f"{name}_any"].append(None)
                values[f"{name}_share"].append(None)
            overlap.append(None)
    rows = [{"group": "view", "name": VIEW_LABELS[key], "share": v} for key, v in values.items()]
    return rows + [{"group": "view overlap", "name": OVERLAP_ROW, "share": overlap}]


def half_lag(share, lags=LAGS):
    """First lag of the grid at which the share of changed pairs reaches HALF, or None."""
    for k, v in zip(lags, share):
        if v is not None and v >= HALF:
            return k
    return None


def change_matrix(s: Split, lags=LAGS) -> dict:
    """Share of frame pairs at every lag in which each component differs, plus the pooled view rows."""
    pairs = {str(k): int(len(pair_starts(s, k))) for k in lags}
    rows = []
    for group, name, column, index in COLUMN_ROWS:
        share = []
        for k in lags:
            t = pair_starts(s, k)
            share.append(float(column_changes(s, column, index, t, k).mean()) if len(t) else None)
        rows.append({"group": group, "name": name, "share": share})
    if s.has("position") and s.state_ok:
        block, mob = view_tiles(s.path, s.total)
        rows += view_rows(s, block, mob, lags)
    for r in rows:
        r["half_lag"] = None if r["group"] == "view overlap" else half_lag(r["share"], lags)
    episodes = {str(k): int((s.lengths > k).sum()) for k in lags}  # episodes with a pair at lag k: the larger lags rest on the long ones
    return {"lags": list(lags), "pairs": pairs, "episodes": episodes, "daylight_tolerance": DAYLIGHT_TOL, "rows": rows}
