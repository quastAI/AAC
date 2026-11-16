"""Frames re-rendered from stored simulator states after one edit: what moves the latent (the `moves_for_*` metrics).

    python -m sieve.counterfactuals --val-file <val.h5> --out <bank.npz> [--states 256] [--seed 0]

Built once per file on the CPU (JAX, Craftax 1.6.1, the env's own renderer), then every checkpoint only encodes the
frames. Each stored `full_state` of a held-out frame is unpacked, edited and rendered again:
    facing   the player turns (direction d -> the next one)
    mob      a cow is added on a free grass tile in view
    tile     the faced tile becomes another block (stone, tree, water, coal, iron, diamond or a table)
    item     one inventory item (one under 9) goes up by one: the inventory strip changes
    noise    a night frame drawn again with another `state_rng`: the same state with another static (Craftax draws the
             night's static from it, the only random part of a frame)
The edits use day frames (light >= 0.7, no static); the noise uses night frames (light < 0.35, the static is strong). The
unedited render must equal the stored frame (`render_match` in the bank's meta, 1.0 for the dataset of the collector).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EDITS = ("facing", "mob", "tile", "item")
DAY, NIGHT = 0.7, 0.35
STEP = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # the player's direction -> the faced tile (row, column)
VIEW_ROWS, VIEW_COLS = 7, 9
GRASS = 2
NEW_TILES = (4, 5, 3, 8, 9, 10, 11)  # stone, tree, water, coal, iron, diamond, crafting table


def _render(state) -> np.ndarray:
    from stable_worldmodel.envs.craftax.env import _sim

    return np.asarray(_sim().observe(state)[2])


def _inside(pos, shape) -> bool:
    return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]


def edit(state, kind: str, rng: np.random.Generator):
    """The state after one edit, or None when this state allows none of that kind."""
    if kind == "facing":
        d = int(state.player_direction)
        return state.replace(player_direction=np.asarray(d % 4 + 1, dtype=np.asarray(state.player_direction).dtype))
    if kind == "mob":
        cows = state.cows
        free_slots = np.flatnonzero(~np.asarray(cows.mask, bool))
        if not len(free_slots):
            return None
        player = np.asarray(state.player_position)
        tiles = []
        for r in range(VIEW_ROWS):
            for c in range(VIEW_COLS):
                pos = player + np.array([r - VIEW_ROWS // 2, c - VIEW_COLS // 2])
                if (r, c) != (VIEW_ROWS // 2, VIEW_COLS // 2) and _inside(pos, state.map.shape):
                    if int(state.map[pos[0], pos[1]]) == GRASS and not bool(state.mob_map[pos[0], pos[1]]):
                        tiles.append(pos)
        if not tiles:
            return None
        pos = tiles[rng.integers(len(tiles))]
        slot = int(free_slots[0])
        position, mask, health = np.array(cows.position), np.array(cows.mask), np.array(cows.health)
        position[slot], mask[slot], health[slot] = pos, True, 3
        mob_map = np.array(state.mob_map)
        mob_map[pos[0], pos[1]] = True
        return state.replace(cows=cows.replace(position=position, mask=mask, health=health), mob_map=mob_map)
    if kind == "tile":
        pos = np.asarray(state.player_position) + np.array(STEP[int(state.player_direction)])
        if not _inside(pos, state.map.shape):
            return None
        current = int(state.map[pos[0], pos[1]])
        choices = [b for b in NEW_TILES if b != current]
        world = np.array(state.map)
        world[pos[0], pos[1]] = choices[rng.integers(len(choices))]
        return state.replace(map=world)
    if kind == "item":
        inv = state.inventory
        fields = [k for k, v in vars(inv).items() if int(v) < 9]
        if not fields:
            return None
        name = fields[rng.integers(len(fields))]
        value = np.asarray(getattr(inv, name))
        return state.replace(inventory=inv.replace(**{name: np.asarray(value + 1, dtype=value.dtype)}))
    if kind == "noise":
        key = np.asarray(state.state_rng)
        return state.replace(state_rng=rng.integers(0, 2**32, key.shape, dtype=np.uint64).astype(key.dtype))
    raise ValueError(kind)


def build(full_states: np.ndarray, pixels: np.ndarray, states: int = 256, seed: int = 0) -> tuple[dict, dict]:
    """The bank from candidate frames (their `full_state` and stored pixels). Returns the arrays and the meta."""
    from stable_worldmodel.envs.craftax.env import unpack_state

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(full_states))
    day, night = [], []
    for i in order:
        s = unpack_state(full_states[i])
        light = float(s.light_level)
        if light >= DAY and len(day) < states:
            day.append((i, s))
        elif light < NIGHT and len(night) < states // 2:
            night.append((i, s))
        if len(day) >= states and len(night) >= states // 2:
            break
    out = {"orig": [], **{k: [] for k in EDITS}, **{f"valid_{k}": [] for k in EDITS}, "noise_orig": [], "noise_alt": []}
    match = []
    for i, s in day:
        frame = _render(s)
        match.append(bool(np.array_equal(frame, pixels[i])))
        out["orig"].append(frame)
        for kind in EDITS:
            edited = edit(s, kind, rng)
            out[f"valid_{kind}"].append(edited is not None)
            out[kind].append(frame if edited is None else _render(edited))
    for i, s in night:
        frame = _render(s)
        match.append(bool(np.array_equal(frame, pixels[i])))
        out["noise_orig"].append(frame)
        out["noise_alt"].append(_render(edit(s, "noise", rng)))
    arrays = {k: np.asarray(v, dtype=bool if k.startswith("valid_") else np.uint8) for k, v in out.items()}
    for k in ("noise_orig", "noise_alt"):
        if not len(arrays[k]):
            arrays[k] = np.zeros((0, *pixels.shape[1:]), np.uint8)
    changed = {k: float(np.mean([(a != b).any() for a, b in zip(arrays["orig"], arrays[k])])) if len(arrays["orig"]) else 0.0 for k in EDITS}
    changed["noise"] = float(np.mean([(a != b).any() for a, b in zip(arrays["noise_orig"], arrays["noise_alt"])])) if len(arrays["noise_orig"]) else 0.0
    meta = {"day_states": len(day), "night_states": len(night), "render_match": float(np.mean(match)) if match else float("nan"),
            "valid": {k: int(arrays[f"valid_{k}"].sum()) for k in EDITS}, "frames_changed": changed, "seed": seed}  # fmt: skip
    return arrays, meta


def build_from_file(val_file, out, states: int = 256, seed: int = 0, episodes: int = 96, windows: int = 4, length: int = 16) -> dict:
    """Candidate frames from a nested sample of the file (contiguous reads), the bank written to `out` (npz + .json)."""
    import h5py

    from slowness import sample as sm

    with h5py.File(Path(val_file), "r") as f:
        lengths, offsets = f["ep_len"][:], f["ep_offset"][:]
    episodes = min(episodes, int((lengths >= windows * length).sum()))
    layout = sm.nested_sample(lengths, offsets, windows=windows, length=length, seed=seed, random_episodes=episodes)
    cols = sm.read_columns(val_file, layout, ("full_state", "pixels"))
    full, pixels = cols["full_state"].reshape(-1, cols["full_state"].shape[-1]), cols["pixels"].reshape(-1, *cols["pixels"].shape[3:])
    arrays, meta = build(full, pixels, states=states, seed=seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out)
    meta["val_file"] = str(val_file)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return meta


def load(path) -> dict:
    with np.load(Path(path)) as z:
        return {k: z[k] for k in z.files}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-file", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--states", type=int, default=256, help="day states for the edits (half as many night states for the noise)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    meta = build_from_file(args.val_file, args.out, args.states, args.seed)
    print(f"[sieve] counterfactual bank: {json.dumps(meta)}", flush=True)
    if meta["render_match"] < 0.99:
        print("[sieve] WARNING: the unedited renders differ from the stored frames (another Craftax version or renderer?)", flush=True)


if __name__ == "__main__":
    main()
