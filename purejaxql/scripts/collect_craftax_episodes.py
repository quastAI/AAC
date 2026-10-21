"""Collect Craftax-Classic episodes into an HDF5 file for world-model training.

The expert (PQN-RNN, trained on the symbolic observation) acts; pixels are
rendered from the same environment state. Episodes are filtered on the diamond
achievement (index 19) unless --no-require-diamond is given.

The output follows the flat layout read by H-JEPA's `HDF5Dataset`: one array per
column with all episodes concatenated, plus `ep_offset` / `ep_len`. Each episode
has L+1 frames o_0..o_L and L actions; `action[t]` is the action taken in frame
`t` (the last one is NaN), as in the other stable-worldmodel datasets.

Columns (N = total frames):
    pixels        (N, 63, 63, 3)  uint8
    state         (N, 1345)       float32  symbolic observation
    action        (N, 1)          float32  the action the game executed, index 0..16, NaN in the last frame: the
                                           policy's action, except noop (0) while the player is asleep (Craftax
                                           replaces every action by a noop then; see `executed_actions`)
    action_policy (N, 1)          float32  the action the policy chose, NaN in the last frame
    reward        (N,)            float32  reward of the transition into this frame (0 in frame 0)
    achievements  (N, 22)         uint8    cumulative flags, Craftax-Classic Achievement order
    full_state    (N, 20952)      uint8    snapshot of the complete env state (see pack_state)
    inventory     (N, 12)         int16    wood, stone, coal, iron, diamond, sapling, then
                                           wood/stone/iron pickaxe, wood/stone/iron sword
    vitals        (N, 4)          int8     health, food, drink, energy
    facing        (N, 1)          int8     player direction, 1 left, 2 right, 3 up, 4 down
    daylight      (N, 1)          float32  light level
    position      (N, 2)          int16    player (row, col) on the 64x64 map
    mobs_in_view  (N, 4)          int8     zombies, cows, skeletons, arrows inside the 7x9 view
    dead          (N,)            uint8    1 where the player is dead (health 0 or in lava): the last frame of an
                                           episode that ended in death, else 0
    step_idx      (N,)            int32
    ep_idx        (N,)            int32
    seed          (N,)            int64    world seed of the episode
    source        (N,)            int8     0 expert (final checkpoint, diamond episodes), 1 noised expert,
                                           2 uniform random, 3 early checkpoint (a policy of an earlier snapshot)
    policy_updates (N,)           int32    training updates of the checkpoint that acted (the `_u<updates>` of a
                                           snapshot name), -1 when not known (uniform random, or a final policy file)

`full_state` lets the planning environment start from any frame. Episodes are also
replayable from `seed` and the action sequence (`action` or `action_policy`: they differ only where the game
ignores the action). With
    reset_key, step_key, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
the reset uses `reset_key` and the transition at step t (t = state.timestep before
the step) uses `jax.random.fold_in(step_key, t)`. Anything that re-simulates an
episode must follow this convention.

Examples:
    # expert episodes that reach diamond, ending 500 steps after the diamond
    python scripts/collect_craftax_episodes.py --out data/expert.h5 \
        --checkpoint models/Craftax-Classic-Symbolic-v1/pqn_rnn_..._vmap0.safetensors \
        --num-episodes 1000 --seed-start 0

    # uniform-random episodes (no checkpoint, no diamond filter)
    python scripts/collect_craftax_episodes.py --out data/random.h5 --eps 1.0 \
        --no-require-diamond --num-episodes 1000 --seed-start 5000000
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT, OBS_DIM, Achievement, BlockType
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
from craftax.craftax_env import make_craftax_env_from_name

from purejaxql.pqn_rnn_craftax import RNNQNetwork
from purejaxql.utils.save_load import load_params

# The episode contract (split of PRNGKey(seed), fold_in per step) must not depend on the default of
# the installed jax: jax >= 0.5 switched it and would generate other worlds.
jax.config.update("jax_threefry_partitionable", False)

ENV_NAME = "Craftax-Classic-Symbolic-v1"
DIAMOND = Achievement.COLLECT_DIAMOND.value
NUM_ACHIEVEMENTS = len(Achievement)
SOURCE_EXPERT, SOURCE_NOISED, SOURCE_RANDOM, SOURCE_EARLY = 0, 1, 2, 3
SOURCES = {"expert": SOURCE_EXPERT, "noised": SOURCE_NOISED, "random": SOURCE_RANDOM, "early": SOURCE_EARLY}
ROLLOUT_CHUNK = 250
NOOP = 0
SLEEPING = -1  # the last entry of the symbolic observation (`state`): the player is asleep
REPLAY_CHUNK = 500


CHECKPOINT_NAME = re.compile(r"_vmap\d+(?:_u(?P<updates>\d+))?\.safetensors$")


def config_path_of(checkpoint):
    """`<name>_config.yaml` of a final policy file or a snapshot (`..._vmap0.safetensors`, `..._vmap0_u0000381.safetensors`)."""
    path = str(checkpoint)
    if not CHECKPOINT_NAME.search(path):
        raise ValueError(f"cannot derive the config of {path}; pass --config")
    return CHECKPOINT_NAME.sub("_config.yaml", path)


def policy_updates_of(checkpoint):
    """Training updates of a snapshot (`_u<updates>` in its name); -1 for a final policy file or no checkpoint."""
    match = CHECKPOINT_NAME.search(str(checkpoint)) if checkpoint else None
    return int(match.group("updates")) if match and match.group("updates") else -1


def load_expert(checkpoint, config_path, action_dim):
    """Return (network, variables) for a checkpoint written by pqn_rnn_craftax.py (final file or snapshot)."""
    if config_path is None:
        config_path = config_path_of(checkpoint)
    cfg = OmegaConf.load(config_path)
    if cfg["ENV_NAME"] != ENV_NAME:
        raise ValueError(f"Checkpoint was trained on {cfg['ENV_NAME']}, expected {ENV_NAME}.")
    variables = load_params(checkpoint)
    if "batch_stats" not in variables:
        raise ValueError(
            f"{checkpoint} has no batch_stats (saved by an older version of "
            "pqn_rnn_craftax.py, which dropped them); retrain the expert."
        )
    network = RNNQNetwork(
        action_dim=action_dim,
        hidden_size=cfg.get("HIDDEN_SIZE", 128),
        num_layers=cfg.get("NUM_LAYERS", 2),
        num_rnn_layers=cfg.get("NUM_RNN_LAYERS", 1),
        norm_type=cfg["NORM_TYPE"],
        norm_input=cfg.get("NORM_INPUT", False),
        add_last_action=cfg.get("ADD_LAST_ACTION", False),
    )
    return network, {"params": variables["params"], "batch_stats": variables["batch_stats"]}


def eps_greedy(key, q_vals, eps):
    """Same rule as the training script, for one environment."""
    key_a, key_e = jax.random.split(key)
    random_action = jax.random.randint(key_a, (), 0, q_vals.shape[-1])
    return jnp.where(jax.random.uniform(key_e) < eps, random_action, jnp.argmax(q_vals))


def make_rollout(env, env_params, network, variables, eps, num_envs):
    """Batched rollout that records actions and, for the replay check, rewards, achievement counts, positions and health."""
    fold_in = jax.vmap(jax.random.fold_in, in_axes=(0, None))

    def init(seeds):
        keys = jax.vmap(lambda s: jax.random.split(jax.random.PRNGKey(s), 3))(seeds)
        reset_keys, step_keys, policy_keys = keys[:, 0], keys[:, 1], keys[:, 2]
        obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        hs = network.initialize_carry(num_envs) if network is not None else ()
        neg = jnp.full((num_envs,), -1, dtype=jnp.int32)
        last_action = jnp.zeros((num_envs,), dtype=jnp.int32)
        carry = (hs, obs, last_action, state, jnp.int32(0), neg, neg)
        return carry, (step_keys, policy_keys)

    @jax.jit
    def run_chunk(carry, step_keys, policy_keys):
        def step(carry, _):
            hs, obs, last_action, state, t, diamond_t, done_t = carry
            if network is None:
                q_vals = jnp.zeros((num_envs, env.num_actions))
            else:
                # Same call as the training sample phase; episodes never auto-reset
                # here, so `done` is False and the hidden state is never re-initialised.
                hs, q_vals = network.apply(
                    variables,
                    hs,
                    obs[None],
                    jnp.zeros((1, num_envs), dtype=bool),
                    last_action[None],
                    train=False,
                )
                q_vals = q_vals[0]
            action = jax.vmap(eps_greedy, in_axes=(0, 0, None))(
                fold_in(policy_keys, t), q_vals, eps
            )
            obs, state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                fold_in(step_keys, t), state, action, env_params
            )
            ach = state.achievements
            got_diamond = ach[:, DIAMOND] & (done_t < 0)  # ignore steps after the episode ended
            diamond_t = jnp.where((diamond_t < 0) & got_diamond, t, diamond_t)
            done_t = jnp.where((done_t < 0) & done, t, done_t)
            carry = (hs, obs, action, state, t + 1, diamond_t, done_t)
            return carry, (
                action.astype(jnp.int8), reward, ach.sum(-1).astype(jnp.int8),
                state.player_position.astype(jnp.int16), state.player_health.astype(jnp.int8),
            )  # fmt: skip

        return jax.lax.scan(step, carry, None, length=ROLLOUT_CHUNK)

    def rollout(seeds, max_steps, stop_at_diamond, tail_steps):
        carry, (step_keys, policy_keys) = init(jnp.asarray(seeds))
        chunks, t_run = [], 0
        while t_run < max_steps:
            carry, out = run_chunk(carry, step_keys, policy_keys)
            chunks.append(jax.tree_util.tree_map(np.asarray, out))
            t_run += ROLLOUT_CHUNK
            diamond_t, done_t = np.asarray(carry[5]), np.asarray(carry[6])
            finished = done_t >= 0
            if stop_at_diamond:
                finished |= (diamond_t >= 0) & (t_run - 1 >= diamond_t + tail_steps)
            if finished.all():
                break
        actions, rewards, ach_counts, positions, health = (np.concatenate(x) for x in zip(*chunks))
        length = np.minimum(np.where(done_t >= 0, done_t + 1, t_run), max_steps)
        has_diamond = (diamond_t >= 0) & (diamond_t < length)
        if stop_at_diamond:
            length = np.where(has_diamond, np.minimum(length, diamond_t + 1 + tail_steps), length)
        return dict(
            actions=actions, rewards=rewards, ach_counts=ach_counts, positions=positions, health=health,
            length=length, has_diamond=has_diamond, diamond_t=diamond_t,
        )  # fmt: skip

    return rollout


def pack_state(state):
    """Lossless flat byte view of an env state: the leaves in tree order, as raw bytes.

    The planning environment unpacks it with the leaf shapes and dtypes of a freshly
    reset state, so both sides only depend on the pinned Craftax version.
    """
    parts = []
    for leaf in jax.tree_util.tree_leaves(state):
        leaf = jnp.asarray(leaf)
        if leaf.dtype == jnp.bool_:
            parts.append(leaf.astype(jnp.uint8).reshape(-1))
        else:
            parts.append(jax.lax.bitcast_convert_type(leaf, jnp.uint8).reshape(-1))
    return jnp.concatenate(parts)


def derived_columns(state):
    """Symbolic quantities of a state for probing and data analysis (all small)."""
    inventory = jnp.stack(jax.tree_util.tree_leaves(state.inventory)).astype(jnp.int16)
    vitals = jnp.stack(
        [state.player_health, state.player_food, state.player_drink, state.player_energy]
    ).astype(jnp.int8)

    def in_view(mobs):
        delta = mobs.position - state.player_position
        inside = (jnp.abs(delta[:, 0]) <= OBS_DIM[0] // 2) & (jnp.abs(delta[:, 1]) <= OBS_DIM[1] // 2)
        return (mobs.mask & inside).sum()

    mobs = jnp.stack([in_view(m) for m in (state.zombies, state.cows, state.skeletons, state.arrows)])
    # the player died: health used up, or walked into lava (which leaves the health as it was)
    in_lava = state.map[state.player_position[0], state.player_position[1]] == BlockType.LAVA.value
    dead = (state.player_health <= 0) | in_lava
    return (
        inventory,
        vitals,
        state.player_direction.astype(jnp.int8)[None],
        state.light_level.astype(jnp.float32)[None],
        state.player_position.astype(jnp.int16),
        mobs.astype(jnp.int8),
        dead.astype(jnp.uint8),
    )


def make_replay(env, env_params):
    """Re-simulate one episode from (seed, actions) and render pixels from its states."""
    render = make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_AGENT)

    def frame(state):
        pixels = jnp.clip(jnp.round(render(state)), 0, 255).astype(jnp.uint8)
        return (pixels, state.achievements.astype(jnp.uint8), pack_state(state), *derived_columns(state))

    @jax.jit
    def reset(seed):
        reset_key, step_key, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
        obs, state = env.reset(reset_key, env_params)
        return obs, state, step_key, *frame(state)

    @jax.jit
    def run_chunk(state, step_key, t0, actions):
        def step(carry, action):
            state, t = carry
            obs, state, reward, _, _ = env.step(
                jax.random.fold_in(step_key, t), state, action, env_params
            )
            return (state, t + 1), (obs, reward, *frame(state))

        (state, _), out = jax.lax.scan(step, (state, t0), actions)
        return state, out

    def replay(seed, actions):
        length = len(actions)
        obs0, state, step_key, *frame0 = reset(seed)
        padded = np.zeros(-(-length // REPLAY_CHUNK) * REPLAY_CHUNK, dtype=np.int32)
        padded[:length] = actions
        first = (obs0, np.zeros(1, np.float32), *frame0)
        columns = [[np.asarray(x)[None]] if i != 1 else [x] for i, x in enumerate(first)]
        for t0 in range(0, len(padded), REPLAY_CHUNK):
            state, out = run_chunk(state, step_key, t0, jnp.asarray(padded[t0 : t0 + REPLAY_CHUNK]))
            for acc, x in zip(columns, out):
                acc.append(np.asarray(x))
        obs, reward, *rest = (np.concatenate(c)[: length + 1] for c in columns)
        names = ("pixels", "achievements", "full_state", "inventory", "vitals", "facing",
                 "daylight", "position", "mobs_in_view", "dead")  # fmt: skip
        return dict(state=obs.astype(np.float32), reward=reward.astype(np.float32), **dict(zip(names, rest)))

    return replay


class EpisodeWriter:
    """Append episodes to an HDF5 file in the stable-worldmodel flat layout."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(path, "a")
        self.initialized = "ep_len" in self.f
        if self.initialized:
            self._drop_unfinished_rows()

    def _drop_unfinished_rows(self):
        """Cut rows that an interrupted run wrote after its last finished episode.

        An episode counts once its ep_offset and ep_len entries exist, and those are written last.
        Columns longer than that hold the rows of an unfinished episode; a shorter one is damage
        that cannot be repaired here.
        """
        if self.num_episodes:
            rows = int(self.f["ep_offset"][-1] + self.f["ep_len"][-1])
        else:
            rows = 0
        for key in self.f:
            if key in ("ep_offset", "ep_len"):
                continue
            have = int(self.f[key].shape[0])
            if have < rows:
                raise RuntimeError(f"{key} has {have} rows but the finished episodes cover {rows}")
            if have > rows:
                print(f"{key}: dropping {have - rows} rows of an unfinished episode")
                self.f[key].resize(rows, axis=0)

    @property
    def num_episodes(self):
        return int(self.f["ep_len"].shape[0]) if self.initialized else 0

    def existing_seeds(self):
        if not self.initialized or self.num_episodes == 0:
            return set()
        offsets = self.f["ep_offset"][:]
        return set(int(s) for s in self.f["seed"][offsets])

    def _create(self, ep):
        for key, data in ep.items():
            shape = (0,) + data.shape[1:]
            big = data.ndim >= 2
            self.f.create_dataset(
                key,
                shape=shape,
                maxshape=(None,) + data.shape[1:],
                dtype=data.dtype,
                chunks=((100,) if big else (1000,)) + data.shape[1:],
                compression="gzip" if big else None,
                compression_opts=4 if big else None,
            )
        self.f.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)
        self.f.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        self.initialized = True

    def append(self, ep):
        ep = dict(ep, ep_idx=np.full(len(ep["step_idx"]), self.num_episodes, dtype=np.int32))
        if not self.initialized:
            self._create(ep)
        columns = set(self.f.keys()) - {"ep_offset", "ep_len"}
        if columns != set(ep):
            raise ValueError(f"Column mismatch with existing file: {sorted(columns ^ set(ep))}")
        n = len(ep["step_idx"])
        start = int(self.f["ep_offset"][-1] + self.f["ep_len"][-1]) if self.num_episodes else 0
        for key, data in ep.items():
            ds = self.f[key]
            if ds.shape[1:] != data.shape[1:] or ds.dtype != data.dtype:
                raise ValueError(f"{key}: {data.shape}/{data.dtype} does not match {ds.shape}/{ds.dtype}")
            ds.resize(start + n, axis=0)
            ds[start : start + n] = data
        for key, value in (("ep_offset", start), ("ep_len", n)):
            self.f[key].resize(self.f[key].shape[0] + 1, axis=0)
            self.f[key][-1] = value
        self.f.flush()

    def runs(self):
        """Provenance: one record per collection run that touched this file (file attribute `runs`)."""
        return json.loads(self.f.attrs["runs"]) if "runs" in self.f.attrs else []

    def start_run(self, record):
        runs = self.runs() + [dict(record, finished=False)]
        self.f.attrs["runs"] = json.dumps(runs)
        self.f.flush()
        return len(runs) - 1

    def finish_run(self, index, outcome):
        runs = self.runs()
        runs[index].update(outcome, finished=True)
        self.f.attrs["runs"] = json.dumps(runs)
        self.f.flush()

    def close(self):
        self.f.close()


def _git(*args):
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).parent), *args], capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def provenance(args):
    """What produced the episodes of this run: code, versions, expert checkpoint, flags."""
    record = {
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "args": {k: v for k, v in vars(args).items()},
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", str(Path(__file__).parents[1]))),
        "versions": {p: _version(p) for p in ("craftax", "jax", "jaxlib", "flax", "numpy", "h5py")},
        "python": sys.version.split()[0],
        "jax_backend": jax.default_backend(),
    }
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        record["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        sidecar = checkpoint.with_suffix(".json")  # a snapshot of the chunked training: updates, steps, test metrics
        if sidecar.exists():
            record["checkpoint"]["training"] = json.loads(sidecar.read_text())
    return record


def executed_actions(policy, state):
    """The actions the game executed: noop wherever the player is asleep in the frame the action is taken in.

    `craftax_step` starts with `action = select(state.is_sleeping, NOOP, action)`, so an action chosen while asleep has
    no effect, and replaying the noop gives the same episode. policy: (N, 1) float with NaN where there is no action;
    state: (N, 1345), the symbolic observation of the same frames.
    """
    out = np.array(policy, dtype=np.float32, copy=True)
    asleep = (np.asarray(state)[:, SLEEPING] > 0.5) & ~np.isnan(out[:, 0])
    out[asleep, 0] = NOOP
    return out


def build_episode(seed, actions, rendered, source, policy_updates=-1):
    length = len(actions)
    action_policy = np.full((length + 1, 1), np.nan, dtype=np.float32)
    action_policy[:length, 0] = actions
    return dict(
        pixels=rendered["pixels"],
        state=rendered["state"],
        action=executed_actions(action_policy, rendered["state"]),
        action_policy=action_policy,
        reward=rendered["reward"],
        achievements=rendered["achievements"],
        full_state=rendered["full_state"],
        inventory=rendered["inventory"],
        vitals=rendered["vitals"],
        facing=rendered["facing"],
        daylight=rendered["daylight"],
        position=rendered["position"],
        mobs_in_view=rendered["mobs_in_view"],
        dead=rendered["dead"],
        step_idx=np.arange(length + 1, dtype=np.int32),
        seed=np.full(length + 1, seed, dtype=np.int64),
        source=np.full(length + 1, source, dtype=np.int8),
        policy_updates=np.full(length + 1, policy_updates, dtype=np.int32),
    )  # fmt: skip


def check_replay(seed, length, rendered, rollout, i):
    """The replay must reproduce the rollout; otherwise episodes are not replayable."""
    ok = np.allclose(rendered["reward"][1:], rollout["rewards"][:length, i], atol=1e-6)
    ok &= np.array_equal(rendered["achievements"][1:].sum(-1), rollout["ach_counts"][:length, i])
    # the player's path and health also depend on the mobs, so a divergence of the random
    # stream shows here even when no reward or achievement changes
    ok &= np.array_equal(rendered["position"][1:], rollout["positions"][:length, i])
    ok &= np.array_equal(rendered["vitals"][1:, 0], rollout["health"][:length, i])
    if not ok:
        raise RuntimeError(
            f"Replay of seed={seed} differs from the rollout; the environment "
            "is not deterministic in (seed, actions), so episodes would not be replayable."
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="HDF5 file; created, or appended to if it exists")
    p.add_argument("--checkpoint", help="expert .safetensors (not needed for --eps 1.0)")
    p.add_argument("--config", help="expert config yaml (default: derived from the checkpoint name)")
    p.add_argument("--num-episodes", type=int, default=None, help="episodes to add to the file")
    p.add_argument("--target-frames", type=int, default=None,
                   help="stop once this run has added this many frames (the last episode is finished, so "
                        "the run overshoots by less than one episode); with --num-episodes, whichever comes first")  # fmt: skip
    p.add_argument("--seed-start", type=int, required=True, help="first world seed; seeds already in the file are skipped")
    p.add_argument("--seed-end", type=int, default=None,
                   help="stop before this seed (exclusive): the end of the range of this split and source, "
                        "so a low diamond rate cannot run into the seeds of another split")  # fmt: skip
    p.add_argument("--eps", type=float, default=0.0, help="epsilon-greedy noise: 0 expert, 1 uniform random")
    p.add_argument("--source", choices=list(SOURCES), default=None,
                   help="label of the episodes (the `source` column). Default from --eps: 0 expert, 1 noised expert, "
                        "uniform random. `early` marks the rollouts of an earlier snapshot (--checkpoint .._u<updates>.safetensors), "
                        "usually with --eps 0 --no-require-diamond")  # fmt: skip
    p.add_argument("--num-envs", type=int, default=64, help="episodes simulated in parallel")
    p.add_argument("--max-steps", type=int, default=10000, help="episode horizon (Craftax-Classic limit is 10000)")
    p.add_argument("--require-diamond", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-at-diamond", action=argparse.BooleanOptionalAction, default=True,
                   help="end the episode when the diamond is collected (plus --tail-steps)")  # fmt: skip
    p.add_argument("--tail-steps", type=int, default=500,
                   help="steps kept after the diamond, as a margin for long planning horizons "
                        "(the episode still ends early on death or at --max-steps)")  # fmt: skip
    return p.parse_args()


def main():
    args = parse_args()
    if args.num_episodes is None and args.target_frames is None:
        raise SystemExit("give --num-episodes or --target-frames")
    if args.eps < 1.0 and args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --eps 1.0")
    default = "expert" if args.eps == 0 else "random" if args.eps >= 1 else "noised"
    name = args.source or default
    if (name == "random") != (args.eps >= 1.0) or (name == "expert" and args.eps != 0):
        raise SystemExit(f"--source {name} does not match --eps {args.eps}")
    source = SOURCES[name]
    policy_updates = policy_updates_of(args.checkpoint)

    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    env_params = env.default_params
    network, variables = (None, None)
    if args.eps < 1.0:
        network, variables = load_expert(args.checkpoint, args.config, env.num_actions)

    rollout = make_rollout(env, env_params, network, variables, args.eps, args.num_envs)
    replay = make_replay(env, env_params)
    writer = EpisodeWriter(args.out)
    run_index = writer.start_run(dict(provenance(args), source=name))
    skip, seed, kept, tried, t0 = writer.existing_seeds(), args.seed_start, 0, 0, time.time()

    exhausted, frames = False, 0

    def done():
        return (args.num_episodes is not None and kept >= args.num_episodes) or (
            args.target_frames is not None and frames >= args.target_frames
        )

    while not done() and not exhausted:
        seeds = []
        while len(seeds) < args.num_envs:
            if seed not in skip:
                seeds.append(seed)
            seed += 1
        result = rollout(np.array(seeds, dtype=np.int32), args.max_steps, args.stop_at_diamond, args.tail_steps)
        keep = result["has_diamond"] if args.require_diamond else np.ones(len(seeds), dtype=bool)
        if args.seed_end is not None:  # the batch is full-size: seeds past the range are simulated but dropped
            keep = keep & (np.asarray(seeds) < args.seed_end)
            exhausted = seed >= args.seed_end
        tried += len(seeds)
        for i in np.flatnonzero(keep):
            if done():
                break
            length = int(result["length"][i])
            actions = result["actions"][:length, i].astype(np.int32)
            rendered = replay(seeds[i], actions)
            check_replay(seeds[i], length, rendered, result, i)
            writer.append(build_episode(seeds[i], actions, rendered, source, policy_updates))
            kept += 1
            frames += length + 1  # L actions, L + 1 frames
        print(
            f"[{time.time() - t0:7.1f}s] tried {tried} | kept {kept}"
            + (f"/{args.num_episodes}" if args.num_episodes is not None else "")
            + f" | {frames}" + (f"/{args.target_frames}" if args.target_frames is not None else "") + " frames | "
            f"diamond rate {result['has_diamond'].mean():.2%} | "
            f"mean length {result['length'].mean():.0f} | next seed {seed}",
            flush=True,
        )
    if not done():
        print(f"WARNING: seeds [{args.seed_start}, {args.seed_end}) are used up: kept {kept} episodes, {frames} frames")
    total = writer.num_episodes
    writer.finish_run(
        run_index,
        {"kept": kept, "tried": tried, "frames": frames, "next_seed": seed, "seconds": round(time.time() - t0)},
    )
    writer.close()
    print(f"Wrote {kept} episodes to {args.out} ({total} in file)")


if __name__ == "__main__":
    main()
