"""The compact state probe: one linear head and one MLP head, each fitted to the whole state at once, on the latent of the
training worlds, scored on the validation worlds. Fixed steps and seed, so checkpoints compare. Labels only to read.

Targets (from the `state` column and the columns next to it):
    regression       inventory (12), vitals (4), daylight
    presence         each mob type somewhere in view (4)
    classes          facing (4), the faced tile's block (17)
    view map         the block of each of the 62 view tiles (the player's own tile left out) and the mobs on it
Absolute position is not a target: the view is egocentric (the probe suite found it unreadable, also untrained).

The view map and the faced tile are very unequal (grass about 40% of the tiles, diamond 0.05%): their cross-entropy weighs
each class by count^-0.5 of the training targets (at most 32 times the most common class), the mob flags by
sqrt(negatives / positives) (at most 32), so that a rare class is not ignored by the probe. Scores:
    R^2 (regression; 0 = the mean), accuracy (facing; the majority share is given), balanced accuracy (the faced tile, the
    view map, presence: the mean recall over the classes present; chance = 1 / classes), recall per block type on the map.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from probing import targets as tg

MAP_TILES = [i for i in range(tg.VIEW_TILES) if i != tg.PLAYER_INDEX]
REG = ["inventory:" + n for n in tg.INVENTORY_NAMES] + ["vitals:" + n for n in tg.VITAL_NAMES] + ["daylight"]
NB, NM, NT = tg.NUM_BLOCKS, tg.NUM_MOBS, len(MAP_TILES)
SLICES = {}
_o = 0
for _name, _width in (("reg", len(REG)), ("mobs", NM), ("facing", 4), ("faced", NB), ("map", NT * NB), ("map_mobs", NT * NM)):
    SLICES[_name] = slice(_o, _o + _width)
    _o += _width
OUT_DIM = _o
STEPS, BATCH, SEED = 1500, 1024, 0
MAX_RATIO = 32.0
# the entities of the report; every map block and every mob type on the map are in the JSON as well
HEADLINE = ["inventory", "vitals", "daylight", "facing", "faced_tile", "mob_zombie", "mob_cow", "mob_skeleton", "mob_arrow",
            "map", "map_coal", "map_iron", "map_diamond", "map_lava", "map_table", "map_furnace", "map_water", "map_tree"]  # fmt: skip
SHORT = {"crafting_table": "table"}


def targets(part) -> dict:
    """The probe's targets of every frame of a sample part (numpy)."""
    state = torch.from_numpy(part.state)
    blocks, mobs = tg.tile_maps(state)
    facing = torch.from_numpy(part.facing).long()
    faced = blocks.gather(-1, tg.faced_index(facing)[..., None])[..., 0]
    return {
        "reg": np.concatenate([part.inventory, part.vitals, part.daylight[:, None]], 1).astype(np.float32),
        "mobs": mobs.any(-2).numpy().astype(np.float32),
        "facing": (facing - 1).clamp(0, 3).numpy(),
        "faced": faced.numpy(),
        "map": blocks[:, MAP_TILES].numpy(),
        "map_mobs": mobs[:, MAP_TILES].numpy().astype(np.float32),
    }


def class_weights(labels: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(labels.reshape(-1), minlength=classes).astype(np.float64)
    w = np.where(counts > 0, np.maximum(counts, 1) ** -0.5, 0.0)
    floor = w[counts > 0].min()
    w = np.where(counts > 0, np.minimum(w, floor * MAX_RATIO), 1.0)
    return torch.tensor(w / w[counts > 0].mean(), dtype=torch.float32)


def pos_weights(flags: np.ndarray) -> torch.Tensor:
    pos = flags.reshape(-1, flags.shape[-1]).sum(0)
    neg = flags.reshape(-1, flags.shape[-1]).shape[0] - pos
    return torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1.0, MAX_RATIO), dtype=torch.float32)


class Probe:
    """The fitted linear or MLP head with the standardisation of its inputs and of the regression targets."""

    def __init__(self, kind: str, z_train: np.ndarray, y_train: dict, device="cpu", steps=STEPS, seed=SEED):
        self.device = device
        self.z_mean, self.z_std = z_train.mean(0), z_train.std(0) + 1e-6
        self.r_mean, self.r_std = y_train["reg"].mean(0), y_train["reg"].std(0) + 1e-6
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        d = z_train.shape[1]
        self.net = (nn.Linear(d, OUT_DIM) if kind == "linear" else nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, OUT_DIM))).to(device)
        self.w_faced = class_weights(y_train["faced"], NB).to(device)
        self.w_map = class_weights(y_train["map"], NB).to(device)
        self.pw_mobs = pos_weights(y_train["mobs"]).to(device)
        self.pw_map_mobs = pos_weights(y_train["map_mobs"]).to(device)
        x = torch.from_numpy((z_train - self.z_mean) / self.z_std).float()
        y = {k: torch.from_numpy(v) for k, v in y_train.items()}
        y["reg"] = (y["reg"] - torch.from_numpy(self.r_mean)) / torch.from_numpy(self.r_std)
        opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3 if kind == "linear" else 1e-3, weight_decay=1e-4)
        n = len(x)
        for _ in range(steps):
            idx = torch.randint(0, n, (min(BATCH, n),), generator=g)
            loss = self._loss(self.net(x[idx].to(device)), {k: v[idx].to(device) for k, v in y.items()})
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        self.net.eval()

    def _loss(self, out, y):
        s = SLICES
        loss = F.mse_loss(out[:, s["reg"]], y["reg"].float())
        loss = loss + F.binary_cross_entropy_with_logits(out[:, s["mobs"]], y["mobs"], pos_weight=self.pw_mobs)
        loss = loss + F.cross_entropy(out[:, s["facing"]], y["facing"].long())
        loss = loss + F.cross_entropy(out[:, s["faced"]], y["faced"].long(), weight=self.w_faced)
        loss = loss + F.cross_entropy(out[:, s["map"]].reshape(-1, NB), y["map"].reshape(-1).long(), weight=self.w_map)
        mm = out[:, s["map_mobs"]].reshape(-1, NT, NM)
        return loss + F.binary_cross_entropy_with_logits(mm, y["map_mobs"], pos_weight=self.pw_map_mobs)

    @torch.no_grad()
    def predict(self, z: np.ndarray) -> dict:
        """Decoded predictions: regression values in the targets' units, classes, presence flags."""
        x = torch.from_numpy((z - self.z_mean) / self.z_std).float()
        out = torch.cat([self.net(x[a : a + 4096].to(self.device)).float().cpu() for a in range(0, len(x), 4096)])
        s = SLICES
        return {
            "reg": out[:, s["reg"]].numpy() * self.r_std + self.r_mean,
            "mobs": (out[:, s["mobs"]] > 0).numpy(),
            "facing": out[:, s["facing"]].argmax(-1).numpy(),
            "faced": out[:, s["faced"]].argmax(-1).numpy(),
            "map": out[:, s["map"]].reshape(-1, NT, NB).argmax(-1).numpy(),
            "map_mobs": (out[:, s["map_mobs"]].reshape(-1, NT, NM) > 0).numpy(),
        }


# ------------------------------------------------------------------------------------------------------- scores


def r2(y, p) -> float:
    """Mean R^2 over the columns that vary."""
    y, p = y.reshape(len(y), -1).astype(np.float64), p.reshape(len(p), -1).astype(np.float64)
    var = y.var(0)
    keep = var > 1e-9
    if not keep.any():
        return float("nan")
    return float((1 - ((y - p) ** 2).mean(0)[keep] / var[keep]).mean())


def balanced_accuracy(y, p) -> float:
    y, p = y.reshape(-1), p.reshape(-1)
    classes = np.unique(y)
    return float(np.mean([(p[y == c] == c).mean() for c in classes])) if len(classes) else float("nan")


def recall(y, p, c) -> float:
    y, p = y.reshape(-1), p.reshape(-1)
    hit = y == c
    return float((p[hit] == c).mean()) if hit.any() else float("nan")


def scores(pred: dict, y: dict) -> dict:
    """Every entity's score on the frames given."""
    reg = y["reg"]
    out = {
        "inventory": r2(reg[:, :12], pred["reg"][:, :12]),
        "vitals": r2(reg[:, 12:16], pred["reg"][:, 12:16]),
        "daylight": r2(reg[:, 16:], pred["reg"][:, 16:]),
        "facing": float((pred["facing"] == y["facing"]).mean()),
        "faced_tile": balanced_accuracy(y["faced"], pred["faced"]),
        "map": balanced_accuracy(y["map"], pred["map"]),
    }
    for m, name in enumerate(tg.MOB_NAMES):
        out[f"mob_{name}"] = balanced_accuracy(y["mobs"][:, m].astype(int), pred["mobs"][:, m].astype(int))
        out[f"map_mob_{name}"] = balanced_accuracy(y["map_mobs"][..., m].astype(int), pred["map_mobs"][..., m].astype(int))
    for b, name in enumerate(tg.BLOCK_NAMES):
        out[f"map_{SHORT.get(name, name)}"] = recall(y["map"], pred["map"], b)
    return out


def chance(y: dict) -> dict:
    """What a constant guess scores: R^2 0, the majority share for facing, 1 / classes for a balanced accuracy, 0.5 for presence."""
    out = {"inventory": 0.0, "vitals": 0.0, "daylight": 0.0}
    out["facing"] = float(np.bincount(y["facing"], minlength=4).max() / len(y["facing"]))
    out["faced_tile"] = 1.0 / max(len(np.unique(y["faced"])), 1)
    out["map"] = 1.0 / max(len(np.unique(y["map"])), 1)
    for name in tg.MOB_NAMES:
        out[f"mob_{name}"] = out[f"map_mob_{name}"] = 0.5
    for name in tg.BLOCK_NAMES:
        out[f"map_{SHORT.get(name, name)}"] = 0.0  # recall of a guess that never names the class
    return out


def bootstrap(pred: dict, y: dict, episode: np.ndarray, names, reps: int = 100, seed: int = 0) -> dict:
    """95% interval of each named score over episodes (episodes drawn with replacement)."""
    rng = np.random.default_rng(seed)
    eps = np.unique(episode)
    rows = {e: np.flatnonzero(episode == e) for e in eps}
    draws = {n: [] for n in names}
    for _ in range(reps):
        idx = np.concatenate([rows[e] for e in rng.choice(eps, len(eps))])
        s = scores({k: v[idx] for k, v in pred.items()}, {k: v[idx] for k, v in y.items()})
        for n in names:
            draws[n].append(s[n])
    with warnings.catch_warnings():  # a class absent from every draw (a rare ore) has no interval: NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return {n: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for n, v in draws.items()}


def skill(score: float, base: float) -> float:
    """The score above chance, as a share of the room above chance."""
    return (score - base) / max(1.0 - base, 1e-9)


def foresees(pred_next: dict, pred_copy: dict, y_t: dict, y_next: dict) -> dict:
    """The probe on the predicted latent against the probe on the last latent (copying), both scored against frame t+1,
    only where the entity changed between t and t+1. Counts and vitals are right when the rounded value is."""
    out = {}

    def add(name, hit_next, hit_copy):
        n = int(hit_next.size)
        out[f"changed_{name}"] = n
        out[f"foresees_{name}"] = float(hit_next.mean()) if n else float("nan")
        out[f"copy_{name}"] = float(hit_copy.mean()) if n else float("nan")

    for name, cols in (("inventory", slice(0, 12)), ("vitals", slice(12, 16))):
        changed = y_next["reg"][:, cols] != y_t["reg"][:, cols]
        truth = y_next["reg"][:, cols][changed]
        add(name, np.round(pred_next["reg"][:, cols][changed]) == truth, np.round(pred_copy["reg"][:, cols][changed]) == truth)
    for name, key in (("facing", "facing"), ("faced_tile", "faced"), ("mobs", "mobs"), ("map", "map")):
        changed = y_next[key] != y_t[key]
        truth = y_next[key][changed]
        add(name, pred_next[key][changed] == truth, pred_copy[key][changed] == truth)
    return out
