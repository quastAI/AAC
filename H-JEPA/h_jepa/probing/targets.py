"""What the probes read: the symbolic state of a Craftax-Classic frame, from the dataset columns and the `state` column.

The frame is 63 x 63 pixels: the 7 x 9 map on top (one 7-pixel tile per ViT patch), the inventory strip (2 tile rows)
below. The `state` column holds the same view as 21 channels per tile (the block type one-hot, then zombie, cow,
skeleton, arrow), then inventory, vitals, direction, light, and the sleeping flag last. View coordinates: tile (r, c)
is the world cell player_position + (r - 3, c - 4) (the renderer's local position).
"""

from __future__ import annotations

import torch

from eval_tasks import ACHIEVEMENT_NAMES

BLOCK_NAMES = [
    "invalid",
    "out_of_bounds",
    "grass",
    "water",
    "stone",
    "tree",
    "wood",
    "path",
    "coal",
    "iron",
    "diamond",
    "crafting_table",
    "furnace",
    "sand",
    "lava",
    "plant",
    "ripe_plant",
]
MOB_NAMES = ["zombie", "cow", "skeleton", "arrow"]
INVENTORY_NAMES = [
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "sapling",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
]
VITAL_NAMES = ["health", "food", "drink", "energy"]
ACTION_NAMES = [
    "noop",
    "left",
    "right",
    "up",
    "down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]
MOVES = ("left", "right", "up", "down")
DO = ACTION_NAMES.index("do")

VIEW_ROWS, VIEW_COLS, VIEW_CHANNELS = 7, 9, 21
VIEW_TILES = VIEW_ROWS * VIEW_COLS
NUM_BLOCKS, NUM_MOBS = len(BLOCK_NAMES), len(MOB_NAMES)
STATE_DIM = 1345
TILE = 7  # pixels per tile, the ViT's patch size
GRID_COLS = 9  # 9 x 9 patches: the 7 map rows, then the 2 rows of the inventory strip
STRIP_ROWS = 2
STRIP_PATCHES = STRIP_ROWS * GRID_COLS
PLAYER_TILE = (VIEW_ROWS // 2, VIEW_COLS // 2)
PLAYER_INDEX = PLAYER_TILE[0] * VIEW_COLS + PLAYER_TILE[1]
FACING_STEP = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # `facing` -> the neighbouring tile it looks at
# a tile with several mobs counts for the rarest (the stratum of the token sample)
MOB_PRIORITY = [MOB_NAMES.index(m) for m in ("arrow", "skeleton", "zombie", "cow")]

# Probe targets of a frame, grouped by how fast they change. Absolute position is not a target: the view is
# egocentric, a frame does not show where in the world it is (it scored below zero, also untrained). Mobs are scored as
# presence (2*AUROC-1): counts are almost always 0, an R^2 on them says little. Where each mob is: the tile maps.
TARGETS = {
    "facing": dict(group="fast", kind="cls", names=["direction"], classes=4),
    "mobs_present": dict(group="fast", kind="bin", names=MOB_NAMES),
    "view_blocks": dict(group="fast", kind="reg", names=BLOCK_NAMES),  # tiles of each block type in the view
    "facing_block": dict(group="fast", kind="cls", names=["block"], classes=NUM_BLOCKS),
    "vitals": dict(group="medium", kind="reg", names=VITAL_NAMES),
    "sleeping": dict(group="medium", kind="bin", names=["sleeping"]),
    "inventory": dict(group="slow", kind="reg", names=INVENTORY_NAMES),
    "daylight": dict(group="slow", kind="reg", names=["light"]),
    "achievements": dict(group="slow", kind="bin", names=ACHIEVEMENT_NAMES),
}
GROUPS = ("fast", "medium", "slow")

# View concepts (group "view", not part of the fast/medium/slow means): the questions the game asks of the 7x9 view,
# computed from the block and mob maps. Rings are Chebyshev distances from the player (1 to 4), 5 = not in view.
# Walkable as in the game: a move succeeds into grass, sand, path and lava (lava kills), not into the solid blocks
# (water, stone, tree, the ores, table, furnace, plants), a mob or the edge; crafting needs a table or furnace among
# the 8 surrounding tiles (CLOSE_BLOCKS of Craftax).
IN_VIEW_BLOCKS = [
    "water",
    "tree",
    "stone",
    "path",
    "sand",
    "coal",
    "iron",
    "diamond",
    "crafting_table",
    "furnace",
    "lava",
    "plant",
    "ripe_plant",
]
NEAREST_BLOCKS = ["water", "tree", "stone", "coal", "iron", "diamond", "crafting_table", "furnace", "lava"]
WALKABLE_BLOCKS = ["grass", "sand", "path", "lava"]
MOVE_NAMES = ["left", "right", "up", "down"]
MOVE_STEPS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # (row, column) of the neighbour a move would enter
NEAR_NAMES = ["near_table", "near_furnace", "lava_adjacent", "zombie_within_2", "arrow_in_line"]
NOT_IN_VIEW = 5.0
TARGETS.update(
    {
        "in_view": dict(group="view", kind="bin", names=IN_VIEW_BLOCKS),
        "nearest": dict(group="view", kind="reg", names=NEAREST_BLOCKS),
        "can_move": dict(group="view", kind="bin", names=MOVE_NAMES),
        "near": dict(group="view", kind="bin", names=NEAR_NAMES),
    }
)
# The tiles around the player in the player's frame (facing-relative): up to 3 ahead, and the left, right and behind
# neighbour. The faced tile is ahead1. A tile outside the view has no label (-1).
CROSS_NAMES = ["ahead1", "ahead2", "ahead3", "left1", "right1", "behind1"]
COLUMNS = ("pixels", "action", "achievements", "inventory", "vitals", "facing", "daylight", "position", "state")
OPTIONAL_COLUMNS = ("source",)  # 0 = the final expert: tasks come from it

# One-step events: the item each action changes and how a success shows (column, +1 rises, -1 falls), as `SUCCESS` in
# analysis/dataset_section/actions.py. Collecting is `do` on the faced block; whether the faced tile itself changes
# (mined stone becomes path, a placed table appears) is read from the next frame, not listed here.
PLACE_MAKE = {
    "place_stone": ("inventory", 1, -1),
    "place_table": ("inventory", 0, -1),
    "place_furnace": ("inventory", 1, -1),
    "place_plant": ("inventory", 5, -1),
    "make_wood_pickaxe": ("inventory", 6, 1),
    "make_stone_pickaxe": ("inventory", 7, 1),
    "make_iron_pickaxe": ("inventory", 8, 1),
    "make_wood_sword": ("inventory", 9, 1),
    "make_stone_sword": ("inventory", 10, 1),
    "make_iron_sword": ("inventory", 11, 1),
}
COLLECT = {  # `do` facing this block: (event, column group, column, sign)
    "tree": ("collect_wood", "inventory", 0, 1),
    "stone": ("collect_stone", "inventory", 1, 1),
    "coal": ("collect_coal", "inventory", 2, 1),
    "iron": ("collect_iron", "inventory", 3, 1),
    "diamond": ("collect_diamond", "inventory", 4, 1),
    "grass": ("collect_sapling", "inventory", 5, 1),
    "water": ("drink_water", "vitals", 2, 1),
    "ripe_plant": ("eat_plant", "vitals", 1, 1),
}
EVENT_GROUPS = ("place", "make", "collect")


def tile_rings() -> torch.Tensor:
    """Distance of each of the 63 view tiles from the player (Chebyshev, 0 for the player's tile; 1 to 4)."""
    r = torch.arange(VIEW_ROWS)[:, None].expand(VIEW_ROWS, VIEW_COLS)
    c = torch.arange(VIEW_COLS)[None, :].expand(VIEW_ROWS, VIEW_COLS)
    return torch.maximum((r - PLAYER_TILE[0]).abs(), (c - PLAYER_TILE[1]).abs()).flatten()


RINGS = (1, 2, 3, 4)


def faced_index(facing: torch.Tensor) -> torch.Tensor:
    """View tile index (0 to 62) of the tile the player faces; `facing` is 1 to 4."""
    d = facing.long().clamp(1, 4)
    dr = torch.tensor([FACING_STEP[k][0] for k in (1, 2, 3, 4)], device=d.device)[d - 1]
    dc = torch.tensor([FACING_STEP[k][1] for k in (1, 2, 3, 4)], device=d.device)[d - 1]
    return (PLAYER_TILE[0] + dr) * VIEW_COLS + PLAYER_TILE[1] + dc


def tile_maps(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The block type (..., 63) and the mob flags (..., 63, 4) of every view tile, from `state` (..., 1345)."""
    lead = state.shape[:-1]
    tiles = state[..., : VIEW_TILES * VIEW_CHANNELS].reshape(*lead, VIEW_TILES, VIEW_CHANNELS)
    return tiles[..., :NUM_BLOCKS].argmax(-1), tiles[..., NUM_BLOCKS:] > 0.5


def tile_strata(blocks: torch.Tensor, mobs: torch.Tensor) -> torch.Tensor:
    """Stratum of each tile for the token sample: NUM_BLOCKS + mob type if a mob stands on it, else its block type."""
    stratum = blocks.clone()
    for m in reversed(MOB_PRIORITY):  # the rarest mob is written last and wins
        stratum = torch.where(mobs[..., m], torch.full_like(stratum, NUM_BLOCKS + m), stratum)
    return stratum


def _tile_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Row and column offsets from the player of the 63 view tiles."""
    r = torch.arange(VIEW_ROWS)[:, None].expand(VIEW_ROWS, VIEW_COLS).flatten() - PLAYER_TILE[0]
    c = torch.arange(VIEW_COLS)[None, :].expand(VIEW_ROWS, VIEW_COLS).flatten() - PLAYER_TILE[1]
    return r, c


def view_concepts(blocks: torch.Tensor, mobs: torch.Tensor) -> dict:
    """The view-concept targets of block maps (..., 63) and mob maps (..., 63, 4): `in_view` (13 flags), `nearest`
    (ring of the nearest tile of 9 types, 5 if none), `can_move` (4 flags), `near` (5 flags). The player's own tile is
    left out."""
    ring = tile_rings().to(blocks.device)
    r, c = (t.to(blocks.device) for t in _tile_grid())
    around = ring >= 1
    ids = {n: BLOCK_NAMES.index(n) for n in BLOCK_NAMES}
    in_view = torch.stack([((blocks == ids[n]) & around).any(-1) for n in IN_VIEW_BLOCKS], -1).float()
    nearest = torch.stack(
        [
            torch.where((blocks == ids[n]) & around, ring, torch.full_like(ring, int(NOT_IN_VIEW))).amin(-1)
            for n in NEAREST_BLOCKS
        ],
        -1,
    ).float()
    any_mob = mobs.any(-1)
    walkable = torch.zeros_like(blocks, dtype=torch.bool)
    for n in WALKABLE_BLOCKS:
        walkable |= blocks == ids[n]
    free = walkable & ~any_mob
    moves = []
    for dr, dc in MOVE_STEPS:
        at = ((r == dr) & (c == dc)).nonzero()[0, 0]
        moves.append(free[..., at])
    can_move = torch.stack(moves, -1).float()
    close = ring == 1
    near = torch.stack(
        [
            ((blocks == ids["crafting_table"]) & close).any(-1),
            ((blocks == ids["furnace"]) & close).any(-1),
            ((blocks == ids["lava"]) & close).any(-1),
            (mobs[..., MOB_NAMES.index("zombie")] & (ring <= 2) & around).any(-1),
            (mobs[..., MOB_NAMES.index("arrow")] & ((r == 0) | (c == 0)) & around).any(-1),
        ],
        -1,
    ).float()
    return {"in_view": in_view, "nearest": nearest, "can_move": can_move, "near": near}


def facing_cross(blocks: torch.Tensor, facing: torch.Tensor) -> torch.Tensor:
    """Block ids (..., 6) of the CROSS_NAMES tiles in the player's frame, from block maps (..., 63) and `facing` (...;
    1 to 4); -1 where a tile lies outside the 7x9 view (none of the 6 does: 3 rows and 4 columns lie on each side)."""
    d = facing.long().clamp(1, 4) - 1
    step = torch.tensor([FACING_STEP[k] for k in (1, 2, 3, 4)], device=blocks.device)[d]  # (..., 2) = (dr, dc)
    dr, dc = step[..., 0], step[..., 1]
    offsets = [(k * dr, k * dc) for k in (1, 2, 3)] + [(-dc, dr), (dc, -dr), (-dr, -dc)]  # left, right, behind
    out = []
    for orow, ocol in offsets:
        row, col = PLAYER_TILE[0] + orow, PLAYER_TILE[1] + ocol
        ok = (row >= 0) & (row < VIEW_ROWS) & (col >= 0) & (col < VIEW_COLS)
        index = (row.clamp(0, VIEW_ROWS - 1) * VIEW_COLS + col.clamp(0, VIEW_COLS - 1))[..., None]
        out.append(torch.where(ok, blocks.gather(-1, index)[..., 0], torch.full_like(row, -1)))
    return torch.stack(out, -1)


def frame_targets(batch: dict) -> dict:
    """Per frame of the batch's level-1 clips, on the CPU: the probe targets (B, T, k) of TARGETS, and the tile maps
    `blocks` (B, T, 63), `mobs` (B, T, 63, 4), the `cross` tiles (B, T, 6), the raw `facing` (B, T; 1 to 4) and
    `position` (B, T, 2)."""
    col = {
        c: batch[f"{c}_level1"].cpu()
        for c in ("achievements", "inventory", "vitals", "facing", "daylight", "position", "state")
    }
    state = col["state"].float()
    blocks, mobs = tile_maps(state)
    facing = col["facing"][..., 0].long()
    faced = faced_index(facing)
    return {
        "facing": (facing - 1).clamp(0, 3)[..., None].float(),  # directions 1..4 -> classes 0..3
        "mobs_present": mobs.any(-2).float(),
        "view_blocks": torch.nn.functional.one_hot(blocks, NUM_BLOCKS).sum(-2).float(),
        "facing_block": blocks.gather(-1, faced[..., None]).float(),
        "vitals": col["vitals"].float(),
        "sleeping": (state[..., STATE_DIM - 1 :] > 0.5).float(),
        "inventory": col["inventory"].float(),
        "daylight": col["daylight"].float(),
        "achievements": col["achievements"].float(),
        **view_concepts(blocks, mobs),
        "cross": facing_cross(blocks, facing),
        "blocks": blocks,
        "mobs": mobs,
        "facing_raw": facing,
        "position": col["position"].long(),
    }
