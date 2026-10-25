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


def frame_targets(batch: dict) -> dict:
    """Per frame of the batch's level-1 clips, on the CPU: the probe targets (B, T, k) of TARGETS, and the tile maps
    `blocks` (B, T, 63), `mobs` (B, T, 63, 4), the raw `facing` (B, T; 1 to 4) and `position` (B, T, 2)."""
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
        "blocks": blocks,
        "mobs": mobs,
        "facing_raw": facing,
        "position": col["position"].long(),
    }
