"""Chunked behaviour cloning on the simulator state: which entities of the state do the expert's next actions need?

An MLP reads the symbolic observation of frame t (the `state` column, the observation the expert acted on: the 7x9 view as
block one-hot and mob flags per tile, then inventory, vitals, facing, daylight and the sleeping flag) and predicts the
expert's next `chunk` actions a_t .. a_t+chunk-1 (cross-entropy per step, the first prediction is the action of frame t).
No pixels, no encoder and no history: the expert is recurrent, so part of what it does is memory and stays unexplained.
The state is simulator ground truth: this is an analysis, the world models never see it.

Importance of an entity (a block type, a mob type, a ring of tiles, one tile, an inventory item, ...): its columns are
replaced by those of another random frame of the held-out set (all columns of one entity together, so a tile stays one-hot)
and the rise of the cross-entropy is read as the share of the model's skill that is lost; the skill is what the model
gains over the marginal action distribution of the training set. Permuting measures reliance, not necessity: an entity
that another one duplicates shows little. `--retrain` therefore trains the model again without the coarse groups and the
most important entities (their columns zero in training and in evaluation).

    python bc.py --chunk 4 8 16 --distance --out $HJEPA_HOME/reports

`--distance` adds a second task on the same pipeline, temporal distance: the states of frame t and of the frame k later
(k from 1 to 64 steps, one value per frame) in, which k out. What it needs of the state, next to what the actions need,
is what a temporal-distance term would ask of a latent beyond a behaviour-cloning term. Daylight is left out of it: the
light level is a clock and would give k away.

`--inverse 4 8` adds multi-step inverse dynamics, the counterpart of the IDM and of the behaviour-cloning task: the states
of frame t and of frame t+k in, the k actions between them out, each step its own distribution (many action sequences
lead to the same state, so the target is not deterministic and the cross-entropy has a floor above zero; the skill is
read against the marginal action distribution, as for the other tasks).

Needs `craftax_expert_train.h5` and `craftax_expert_val.h5` (columns `state`, `action`, `ep_len`, `source`); tasks and
training of the world models use the final expert's episodes (`source` 0), so does this.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from probing import targets as tg

NUM_ACTIONS = len(tg.ACTION_NAMES)
NUM_BLOCKS, NUM_MOBS, TILES, CHANNELS = tg.NUM_BLOCKS, tg.NUM_MOBS, tg.VIEW_TILES, tg.VIEW_CHANNELS
VIEW_DIM = TILES * CHANNELS
# the 22 values after the view: inventory (/10), vitals (/10), facing (one-hot), daylight, sleeping
INVENTORY, VITALS = VIEW_DIM, VIEW_DIM + len(tg.INVENTORY_NAMES)
FACING = VITALS + len(tg.VITAL_NAMES)
DAYLIGHT, SLEEPING = FACING + len(tg.FACING_STEP), FACING + len(tg.FACING_STEP) + 1
assert SLEEPING + 1 == tg.STATE_DIM
MOB_BITS = torch.tensor([1 << m for m in range(NUM_MOBS)], dtype=torch.uint8)


def log(msg: str) -> None:
    print(f"[bc] {msg}", flush=True)


# --- data ----------------------------------------------------------------------------------------------------------


@dataclass
class StateTable:
    """The kept frames of a file in file order, the observation stored compactly: per tile the block id and the 4 mob
    bits (126 bytes of the 5.3 kB of floats) plus the 22 values after the view. `features` rebuilds the observation."""

    block: torch.Tensor  # (n, 63) uint8
    mob: torch.Tensor  # (n, 63) uint8, bit m = a mob of type m on the tile
    tail: torch.Tensor  # (n, 22) float32
    action: torch.Tensor  # (n,) int8, -1 where the frame has no action (the last of an episode)
    episode: torch.Tensor  # (n,) int32, the episode of the file

    def to(self, device) -> "StateTable":
        return StateTable(*(t.to(device) for t in (self.block, self.mob, self.tail, self.action, self.episode)))

    def features(self, rows: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """The (len(rows), 1345) observation of the frames; `keep` (1345 bool/float) zeroes the columns it holds 0 for."""
        blocks = F.one_hot(self.block[rows].long(), NUM_BLOCKS)
        mobs = ((self.mob[rows].unsqueeze(-1) & MOB_BITS.to(self.mob.device)) > 0).long()
        x = torch.cat([torch.cat([blocks, mobs], -1).flatten(1).float(), self.tail[rows]], 1)
        return x if keep is None else x * keep


def load_table(path, sources=(0,), action_column="action", max_episodes=None, block=50_000) -> StateTable:
    """The `state` and the actions of the episodes of the given `source`s (the final expert: 0). `max_episodes` keeps the
    first ones only (a smoke run). The column is read in blocks of `block` frames, blocks without a kept frame are skipped."""
    with h5py.File(path, "r") as f:
        ep_len = f["ep_len"][:].astype(np.int64)
        total = int(ep_len.sum())
        episode = np.repeat(np.arange(len(ep_len), dtype=np.int32), ep_len)
        keep = np.isin(f["source"][:total], list(sources)) if "source" in f else np.ones(total, bool)
        if max_episodes:
            keep &= np.isin(episode, np.unique(episode[keep])[:max_episodes])
        if not keep.any():
            raise SystemExit(f"{path}: no frame of source {sources}")
        if f["state"].shape[1] != tg.STATE_DIM:
            raise SystemExit(f"{path}: state has {f['state'].shape[1]} columns, expected {tg.STATE_DIM}")
        action = np.nan_to_num(f[action_column][:total, 0].astype(np.float32), nan=-1)[keep].astype(np.int8)
        blocks, mobs, tails = [], [], []
        for a in range(0, total, block):
            k = keep[a : a + block]
            if not k.any():
                continue
            x = f["state"][a : a + block][k]
            tiles = x[:, :VIEW_DIM].reshape(-1, TILES, CHANNELS)
            blocks.append(tiles[:, :, :NUM_BLOCKS].argmax(2).astype(np.uint8))
            mobs.append(((tiles[:, :, NUM_BLOCKS:] > 0.5) * (1 << np.arange(NUM_MOBS))).sum(2).astype(np.uint8))
            tails.append(x[:, VIEW_DIM:].astype(np.float32))
    return StateTable(
        *(torch.from_numpy(np.concatenate(v)) for v in (blocks, mobs, tails)),
        torch.from_numpy(action),
        torch.from_numpy(episode[keep]),
    )


def chunk_starts(table: StateTable, chunk: int) -> torch.Tensor:
    """Frames t whose next `chunk` actions are all in the frame's episode (a window never crosses an episode)."""
    episode, action = table.episode.cpu().numpy(), table.action.cpu().numpy()
    t = np.arange(max(len(action) - chunk + 1, 0))
    bad = np.concatenate([[0], np.cumsum(action < 0)])
    ok = (episode[t + chunk - 1] == episode[t]) & (bad[t + chunk] - bad[t] == 0)
    return torch.from_numpy(t[ok])


def chunk_targets(table: StateTable, rows: torch.Tensor, chunk: int) -> torch.Tensor:
    """(len(rows), chunk) action indices a_t .. a_t+chunk-1."""
    return table.action[rows[:, None] + torch.arange(chunk, device=rows.device)].long()


# --- tasks: what the model reads and predicts ---------------------------------------------------------------------


class ActionChunk:
    """The expert's next `chunk` actions from the state of frame t."""

    classes, dim = NUM_ACTIONS, tg.STATE_DIM

    def __init__(self, chunk: int):
        self.chunk = self.steps = chunk
        self.label, self.file, self.title = f"chunk {chunk}", f"bc_entities_chunk{chunk}", f"BC entities, chunk {chunk}"
        self.what = f"A model reads the state of one frame and predicts the expert's next {chunk} actions."

    def starts(self, table):
        return chunk_starts(table, self.chunk)

    def targets(self, table, rows):
        return chunk_targets(table, rows, self.chunk)

    def features(self, table, rows, keep=None):
        return table.features(rows, keep)

    def columns(self):
        return entities()


class Distance:
    """Temporal distance: the state of frame t and of the frame k later in, which k out (classes of `KS`, one k per frame,
    fixed by a hash of the frame, so that a frame always gets the same pair). The daylight of both frames is zero."""

    KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    steps, classes, dim = 1, len(KS), 2 * tg.STATE_DIM
    label, file, title = "distance", "bc_entities_distance", "Temporal distance entities"
    what = (
        "A model reads the state of frame t and of the frame k later (k from 1 to 64 steps, 12 classes, one k per frame) "
        "and predicts k; an entity comes from another random pair, in both frames. Daylight is left out: the light level "
        "is a clock."
    )

    def __init__(self):
        self._off = {}

    def index(self, rows):
        return ((rows * 2654435761) % 2**32) % len(self.KS)

    def starts(self, table):
        episode = table.episode.cpu().numpy()
        t = np.arange(len(episode))
        k = np.array(self.KS)[self.index(torch.from_numpy(t)).numpy()]
        later = np.minimum(t + k, len(t) - 1)
        return torch.from_numpy(t[(t + k < len(t)) & (episode[later] == episode[t])])

    def targets(self, table, rows):
        return self.index(rows)[:, None]

    def features(self, table, rows, keep=None):
        device = rows.device
        if device not in self._off:
            off = torch.ones(self.dim, device=device)
            off[[DAYLIGHT, tg.STATE_DIM + DAYLIGHT]] = 0.0
            self._off[device] = off
        k = torch.as_tensor(self.KS, device=device)[self.index(rows)]
        x = torch.cat([table.features(rows), table.features(rows + k)], 1) * self._off[device]
        return x if keep is None else x * keep

    def columns(self):
        return {
            n: np.concatenate([c, c + tg.STATE_DIM]) for n, c in entities().items() if "daylight" not in n
        }  # the entity in both frames


class InverseChunk:
    """Multi-step inverse dynamics: the states of frame t and of frame t+k in, the k actions between them out, each step
    its own categorical (no sequence is the one answer: the target is not deterministic). An entity is replaced in both
    frames."""

    classes, dim = NUM_ACTIONS, 2 * tg.STATE_DIM

    def __init__(self, k: int):
        self.k = self.steps = k
        self.label, self.file, self.title = (
            f"inverse {k}",
            f"bc_entities_inverse{k}",
            f"Multi-step inverse dynamics, {k} steps",
        )
        self.what = (
            f"A model reads the state of frame t and of frame t+{k} and predicts the {k} actions between them, each step "
            "as its own distribution (many action sequences lead to the same state: the target is not deterministic); "
            "an entity comes from another random pair, in both frames."
        )

    def starts(self, table):
        episode, action = table.episode.cpu().numpy(), table.action.cpu().numpy()
        t = np.arange(max(len(action) - self.k, 0))
        bad = np.concatenate([[0], np.cumsum(action < 0)])  # the last frame of an episode has no action, and needs none
        ok = (episode[t + self.k] == episode[t]) & (bad[t + self.k] - bad[t] == 0)
        return torch.from_numpy(t[ok])

    def targets(self, table, rows):
        return chunk_targets(table, rows, self.k)

    def features(self, table, rows, keep=None):
        x = torch.cat([table.features(rows), table.features(rows + self.k)], 1)
        return x if keep is None else x * keep

    def columns(self):
        return {n: np.concatenate([c, c + tg.STATE_DIM]) for n, c in entities().items()}


def as_task(task):
    """A task, or an int: the chunk length of the action task."""
    return ActionChunk(task) if isinstance(task, int) else task


# --- entities ------------------------------------------------------------------------------------------------------


def entities() -> dict[str, np.ndarray]:
    """What the state is made of, name -> its columns. `block:`, `mob:`, `ring:` and `tile:` cut the view three ways (by
    type, by distance from the player, by place), the rest are the values after the view; `group:` are the coarse parts
    that the retraining leaves out one by one. A ring is the Chebyshev distance from the player, ring 0 the player's tile."""
    tile = np.arange(TILES)[:, None] * CHANNELS + np.arange(CHANNELS)[None, :]  # (63, 21) column of a tile's channel
    row, col = np.divmod(np.arange(TILES), tg.VIEW_COLS)
    dr, dc = row - tg.PLAYER_TILE[0], col - tg.PLAYER_TILE[1]
    ring = np.maximum(abs(dr), abs(dc))
    out = {f"block:{n}": tile[:, c].ravel() for c, n in enumerate(tg.BLOCK_NAMES) if n != "invalid"}
    out |= {f"mob:{n}": tile[:, NUM_BLOCKS + m].ravel() for m, n in enumerate(tg.MOB_NAMES)}
    out |= {f"ring:{r}": tile[ring == r].ravel() for r in range(ring.max() + 1)}
    out |= {f"tile:{a:+d},{b:+d}": tile[t] for t, (a, b) in enumerate(zip(dr, dc))}
    out |= {f"inventory:{n}": np.array([INVENTORY + i]) for i, n in enumerate(tg.INVENTORY_NAMES)}
    out |= {f"vitals:{n}": np.array([VITALS + i]) for i, n in enumerate(tg.VITAL_NAMES)}
    out |= {
        "state:facing": np.arange(FACING, DAYLIGHT),
        "state:daylight": np.array([DAYLIGHT]),
        "state:sleeping": np.array([SLEEPING]),
    }
    out |= {
        "group:view": np.arange(VIEW_DIM),
        "group:inventory": np.arange(INVENTORY, VITALS),
        "group:vitals": np.arange(VITALS, FACING),
        "group:facing": out["state:facing"],
        "group:daylight": out["state:daylight"],
        "group:sleeping": out["state:sleeping"],
    }
    return out


def column_mask(names, entity_columns: dict[str, np.ndarray], dim: int = tg.STATE_DIM) -> torch.Tensor:
    """(dim,) float, 0 on the columns of the named entities."""
    keep = torch.ones(dim)
    for n in names:
        keep[torch.from_numpy(entity_columns[n])] = 0.0
    return keep


# --- model and training --------------------------------------------------------------------------------------------


class ChunkBC(nn.Module):
    """MLP from the observation of a frame to the logits of `chunk` steps of classes, (batch, chunk, classes): the
    next actions, or one step of temporal distances."""

    def __init__(self, chunk, hidden=512, depth=3, dropout=0.1, classes=NUM_ACTIONS, dim=tg.STATE_DIM):
        super().__init__()
        dims = [dim] + [hidden] * depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers, nn.Linear(hidden, chunk * classes))
        self.chunk, self.classes = chunk, classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(len(x), self.chunk, self.classes)


def step_losses(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy (batch, chunk) and whether the argmax is right, per step."""
    ce = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view_as(y)
    return ce, logits.argmax(-1) == y


@torch.no_grad()
def evaluate(model, table, rows, chunk, keep=None, batch_size=8192) -> tuple[np.ndarray, np.ndarray]:
    """Mean cross-entropy and accuracy per step over the windows `rows`."""
    model.eval()
    task = as_task(chunk)
    ce = torch.zeros(task.steps, device=rows.device)
    ok = torch.zeros(task.steps, device=rows.device)
    for a in range(0, len(rows), batch_size):
        r = rows[a : a + batch_size]
        c, o = step_losses(model(task.features(table, r, keep)), task.targets(table, r))
        ce += c.sum(0)
        ok += o.sum(0)
    return (ce / len(rows)).cpu().numpy(), (ok / len(rows)).cpu().numpy()


def marginal_log_probs(table: StateTable, rows: torch.Tensor, chunk) -> torch.Tensor:
    """(steps, classes) log of the class frequencies per step: the model that sees nothing (a little smoothed)."""
    task = as_task(chunk)
    y = task.targets(table, rows)
    counts = torch.stack([torch.bincount(y[:, k], minlength=task.classes) for k in range(task.steps)]).float() + 1.0
    return (counts / counts.sum(1, keepdim=True)).log()


def fit(
    train,
    val,
    chunk,
    keep=None,
    epochs=8,
    batch_size=4096,
    lr=1e-3,
    weight_decay=0.01,
    hidden=512,
    depth=3,
    dropout=0.1,
    seed=0,
    tag="",
):
    """Train a ChunkBC on the windows of `train`, keep the epoch with the lowest validation cross-entropy."""
    device = train.block.device
    task = as_task(chunk)
    torch.manual_seed(seed)
    model = ChunkBC(task.steps, hidden, depth, dropout, task.classes, task.dim).to(device)
    starts, val_starts = task.starts(train).to(device), task.starts(val).to(device)
    batch_size = min(batch_size, len(starts))
    per_epoch = len(starts) // batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * per_epoch, pct_start=0.1)
    best, state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(starts), device=device)
        total = 0.0
        for i in range(per_epoch):
            rows = starts[order[i * batch_size : (i + 1) * batch_size]]
            loss = F.cross_entropy(
                model(task.features(train, rows, keep)).flatten(0, 1), task.targets(train, rows).flatten()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() if i % 20 == 0 else 0.0
        ce, acc = evaluate(model, val, val_starts, task, keep)
        log(
            f"{tag}{task.label} epoch {epoch + 1}/{epochs}: train ce {total / len(range(0, per_epoch, 20)):.4f}, "
            f"val ce {ce.mean():.4f}, val accuracy {acc.mean():.4f} (step 1: {acc[0]:.4f})"
        )
        if ce.mean() < best:
            best, state = float(ce.mean()), {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model


# --- importance ----------------------------------------------------------------------------------------------------


@torch.no_grad()
def entity_importance(
    model, table, rows, chunk, columns, marginal, keep=None, repeats=3, boot=200, seed=0, batch_size=8192
):
    """Per entity of `columns`, the rise of the cross-entropy and the fall of the accuracy per step when its columns come
    from another random frame, as sums per episode (one pass over `rows`, `repeats` partners per batch), and from them
    the share of the model's skill that is lost with an episode bootstrap. Columns of an entity that do not vary in `rows`
    give exactly 0 and are listed as constant."""
    model.eval()
    task = as_task(chunk)
    device = rows.device
    _, ep = torch.unique(table.episode.to(device)[rows], return_inverse=True)
    episodes, names = int(ep.max()) + 1, list(columns)
    masks = torch.zeros(len(names), task.dim, dtype=torch.bool, device=device)
    for m, n in enumerate(names):
        masks[m, torch.from_numpy(columns[n]).to(device)] = True
    shape = (episodes, task.steps)
    base_ce, base_ok, marg_ce = (torch.zeros(shape, device=device) for _ in range(3))
    d_ce, d_ok = (torch.zeros(len(names), *shape, device=device) for _ in range(2))
    count = torch.zeros(episodes, device=device)
    low = torch.full((task.dim,), float("inf"), device=device)
    high = -low
    gen = torch.Generator(device=device).manual_seed(seed)
    order = torch.randperm(
        len(rows), device=device, generator=gen
    )  # batches mix the episodes: a partner is another scene
    for a in range(0, len(rows), batch_size):
        sel = order[a : a + batch_size]
        r, e = rows[sel], ep[sel]
        y = task.targets(table, r)
        x = task.features(table, r, keep)
        ce0, ok0 = step_losses(model(x), y)
        count.index_add_(0, e, torch.ones(len(e), device=device))
        base_ce.index_add_(0, e, ce0)
        base_ok.index_add_(0, e, ok0.float())
        marg_ce.index_add_(0, e, -marginal.gather(1, y.T).T)
        low, high = torch.minimum(low, x.amin(0)), torch.maximum(high, x.amax(0))
        for _ in range(repeats):
            partner = x.roll(int(torch.randint(1, max(len(sel), 2), (1,), generator=gen, device=device)), 0)
            for m in range(len(names)):
                ce, ok = step_losses(model(torch.where(masks[m], partner, x)), y)
                d_ce[m].index_add_(0, e, ce - ce0)
                d_ok[m].index_add_(0, e, ok.float() - ok0.float())
    d_ce, d_ok = d_ce / repeats, d_ok / repeats

    n_e = count.cpu().numpy()
    ce_e, marg_e = base_ce.mean(1).cpu().numpy(), marg_ce.mean(1).cpu().numpy()  # per episode, summed over frames
    delta_e = d_ce.mean(2).cpu().numpy()  # (entities, episodes)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(episodes, np.full(episodes, 1 / episodes), size=boot) if boot else np.ones((1, episodes))
    n_b = weights @ n_e
    skill_b = (weights @ marg_e - weights @ ce_e) / n_b  # nats the model gains over the marginal, per resample
    constant = ((high - low) == 0).cpu().numpy()
    skill = (marg_e.sum() - ce_e.sum()) / n_e.sum()
    out = {}
    for m, n in enumerate(names):
        delta = delta_e[m].sum() / n_e.sum()
        lost_b = (weights @ delta_e[m]) / n_b / skill_b
        out[n] = {
            "delta_ce": float(delta),
            "skill_lost": float(delta / skill),
            "skill_lost_ci": [float(np.percentile(lost_b, 2.5)), float(np.percentile(lost_b, 97.5))] if boot else None,
            "delta_ce_steps": (d_ce[m].sum(0) / count.sum()).cpu().tolist(),
            "delta_accuracy": float(d_ok[m].mean(1).sum().item() / count.sum().item()),
            "constant": bool(constant[columns[n]].all()),
        }
    return {
        "frames": int(n_e.sum()),
        "episodes": episodes,
        "skill": float(skill),
        "ce": float(ce_e.sum() / n_e.sum()),
        "marginal_ce": float(marg_e.sum() / n_e.sum()),
        "ce_steps": (base_ce.sum(0) / count.sum()).cpu().tolist(),
        "accuracy_steps": (base_ok.sum(0) / count.sum()).cpu().tolist(),
        "entities": out,
    }


def retrain_without(train, val, chunk, columns, names, result, **fit_kw) -> dict:
    """Train the model again without each of `names` (its columns zero in training and evaluation); the skill lost is the
    rise of the validation cross-entropy over the model with everything, in shares of its skill (one seed, no interval)."""
    task = as_task(chunk)
    rows = task.starts(val).to(val.block.device)
    out = {}
    for n in names:
        keep = column_mask([n], columns, task.dim).to(val.block.device)
        model = fit(train, val, task, keep=keep, tag=f"without {n}: ", **fit_kw)
        ce, acc = evaluate(model, val, rows, task, keep)
        out[n] = {
            "ce": float(ce.mean()),
            "accuracy": float(acc.mean()),
            "skill_lost": float((ce.mean() - result["ce"]) / result["skill"]),
        }
    return out


# --- report --------------------------------------------------------------------------------------------------------


def _ci(v) -> str:
    return "" if v is None else f"{v[0]:.2f} to {v[1]:.2f}"


def report(result: dict, chunk) -> str:
    """The markdown report of one task: the model, then every family of entities by the share of skill lost."""
    task = as_task(chunk)
    ent, steps = result["entities"], task.steps
    head = "| entity | skill lost | 95% interval | Δ ce | Δ accuracy |"
    rule = "|---|---|---|---|---|"
    if steps > 1:
        head += f" Δ ce, step 1 | Δ ce, step {steps} |"
        rule += "---|---|"
    lines = [
        f"# {task.title}",
        "",
        f"{task.what} {result['frames']} validation samples of {result['episodes']} episodes (the final expert's, "
        f"held-out worlds). Cross-entropy {result['ce']:.3f} nats against {result['marginal_ce']:.3f} for the marginal "
        f"distribution of the target: the model gains {result['skill']:.3f} nats ('skill'). Accuracy per step: "
        + ", ".join(f"{a:.2f}" for a in result["accuracy_steps"])
        + ". Skill lost: the rise of the cross-entropy when the entity's columns come from another frame, in shares of "
        "the skill (permutation: reliance, not necessity); interval: episode bootstrap. Constant: the columns never vary.",
        "",
    ]
    for family, title in (
        ("group", "Coarse parts"),
        ("block", "Block types in the view (all tiles)"),
        ("mob", "Mobs in the view (all tiles)"),
        ("ring", "Distance from the player (all channels)"),
        ("inventory", "Inventory"),
        ("vitals", "Vitals"),
        ("state", "Facing, daylight, sleeping"),
    ):
        rows = sorted(((n, v) for n, v in ent.items() if n.split(":")[0] == family), key=lambda p: -p[1]["skill_lost"])
        lines += [f"## {title}", "", head, rule]
        for n, v in rows:
            tag = " (constant)" if v["constant"] else ""
            line = (
                f"| {n.split(':', 1)[1]}{tag} | {v['skill_lost']:.3f} | {_ci(v['skill_lost_ci'])} | {v['delta_ce']:.4f} | "
                f"{v['delta_accuracy']:+.4f} |"
            )
            if steps > 1:
                line += f" {v['delta_ce_steps'][0]:.4f} | {v['delta_ce_steps'][-1]:.4f} |"
            lines.append(line)
        lines.append("")
    lines += [
        "## Skill lost by place (the player stands at the centre, up is up)",
        "",
        "| | " + " | ".join(f"{c:+d}" for c in range(-(tg.VIEW_COLS // 2), tg.VIEW_COLS // 2 + 1)) + " |",
        "|---|" + "---|" * tg.VIEW_COLS,
    ]
    for r in range(-(tg.VIEW_ROWS // 2), tg.VIEW_ROWS // 2 + 1):
        cells = [
            f"{ent[f'tile:{r:+d},{c:+d}']['skill_lost']:.3f}"
            for c in range(-(tg.VIEW_COLS // 2), tg.VIEW_COLS // 2 + 1)
        ]
        lines.append(f"| {r:+d} | " + " | ".join(cells) + " |")
    if result.get("retrain"):
        lines += [
            "",
            "## Trained again without the entity",
            "",
            "Skill lost: the rise of the validation cross-entropy over the model with everything, in shares of its skill "
            "(one seed, no interval).",
            "",
            "| entity | skill lost | permutation | ce | accuracy |",
            "|---|---|---|---|---|",
        ]
        for n, v in sorted(result["retrain"].items(), key=lambda p: -p[1]["skill_lost"]):
            lines.append(
                f"| {n} | {v['skill_lost']:.3f} | {ent[n]['skill_lost']:.3f} | {v['ce']:.4f} | {v['accuracy']:.4f} |"
            )
    return "\n".join(lines) + "\n"


# --- entry point ---------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    home = os.environ.get("HJEPA_HOME", ".")
    parser.add_argument("--train-file", default=f"{home}/craftax_expert_train.h5")
    parser.add_argument(
        "--val-file", default=f"{home}/craftax_expert_val.h5", help="held out: early stopping and importance"
    )
    parser.add_argument("--out", default=f"{home}/reports", help="bc_entities_<task>.md and .json are written here")
    parser.add_argument(
        "--chunk", type=int, nargs="*", default=[4, 8, 16], help="actions predicted per frame (none: --chunk alone)"
    )
    parser.add_argument("--distance", action="store_true", help="also the temporal-distance task (see above)")
    parser.add_argument(
        "--inverse", type=int, nargs="*", default=[], help="also multi-step inverse dynamics over this many steps"
    )
    parser.add_argument(
        "--sources", type=int, nargs="+", default=[0], help="`source` of the episodes (0: the final expert)"
    )
    parser.add_argument(
        "--actions",
        default="action",
        choices=["action", "action_policy"],
        help="the executed action (noop while asleep, what training uses) or the policy's choice",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--repeats", type=int, default=3, help="random partner frames per batch and entity")
    parser.add_argument("--boot", type=int, default=200, help="episode bootstrap resamples (0 = no intervals)")
    parser.add_argument(
        "--retrain-top", type=int, default=6, help="entities (besides the groups) trained again without; 0: groups only"
    )
    parser.add_argument("--no-retrain", action="store_true", help="the permutation importance only")
    parser.add_argument("--max-episodes", type=int, help="episodes per file (a smoke run)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tasks = [ActionChunk(c) for c in args.chunk] + ([Distance()] if args.distance else [])
    tasks += [InverseChunk(k) for k in args.inverse]
    if not tasks:
        parser.error("no task: give --chunk lengths, --distance and/or --inverse lengths")

    from cpu_threads import limit_threads

    limit_threads()
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train, val = (
        load_table(p, args.sources, args.actions, args.max_episodes).to(device)
        for p in (args.train_file, args.val_file)
    )
    log(
        f"loaded {len(train.action)} train and {len(val.action)} validation frames in {time.time() - t0:.0f} s ({device})"
    )
    fit_kw = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, hidden=args.hidden, depth=args.depth, seed=args.seed
    )
    for task in tasks:
        t0 = time.time()
        columns = task.columns()
        model = fit(train, val, task, **fit_kw)
        rows = task.starts(val).to(device)
        marginal = marginal_log_probs(train, task.starts(train).to(device), task)
        result = entity_importance(
            model, val, rows, task, columns, marginal, repeats=args.repeats, boot=args.boot, seed=args.seed
        )
        log(f"{task.label}: the importance of {len(columns)} entities in {time.time() - t0:.0f} s")
        if not args.no_retrain:
            # not the entities that repeat a group (state:) or that a one-hot leaves recoverable (block:, a single
            # channel: the tile without it is the tile of that type), not the cuts of the view (ring:, tile:)
            skip = ("group:", "tile:", "ring:", "block:", "state:")
            candidates = [(n, v["skill_lost"]) for n, v in result["entities"].items()]
            top = [
                n
                for n, _ in sorted(
                    (c for c in candidates if not c[0].startswith(skip) and not result["entities"][c[0]]["constant"]),
                    key=lambda p: -p[1],
                )[: args.retrain_top]
            ]
            groups = [n for n in columns if n.startswith("group:")]
            result["retrain"] = retrain_without(train, val, task, columns, groups + top, result, **fit_kw)
        result["args"] = vars(args) | {"task": task.label}
        (out / f"{task.file}.json").write_text(json.dumps(result, indent=1))
        (out / f"{task.file}.md").write_text(report(result, task))
        log(f"{task.label} done in {time.time() - t0:.0f} s: {out / f'{task.file}.md'}")


if __name__ == "__main__":
    main()
