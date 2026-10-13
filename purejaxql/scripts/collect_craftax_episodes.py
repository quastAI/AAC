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
    action        (N, 1)          float32  action index 0..16, NaN in the last frame
    reward        (N,)            float32  reward of the transition into this frame (0 in frame 0)
    achievements  (N, 22)         uint8    cumulative flags, Craftax-Classic Achievement order
    full_state    (N, 20952)      uint8    snapshot of the complete env state (see pack_state)
    step_idx      (N,)            int32
    ep_idx        (N,)            int32
    seed          (N,)            int64    world seed of the episode
    source        (N,)            int8     0 expert, 1 noised expert, 2 uniform random

`full_state` lets the planning environment start from any frame. Episodes are also
replayable from `seed` and the action sequence. With
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
import re
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT, Achievement
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
from craftax.craftax_env import make_craftax_env_from_name

from purejaxql.pqn_rnn_craftax import RNNQNetwork
from purejaxql.utils.save_load import load_params

ENV_NAME = "Craftax-Classic-Symbolic-v1"
DIAMOND = Achievement.COLLECT_DIAMOND.value
NUM_ACHIEVEMENTS = len(Achievement)
SOURCE_EXPERT, SOURCE_NOISED, SOURCE_RANDOM = 0, 1, 2
ROLLOUT_CHUNK = 250
REPLAY_CHUNK = 500


def load_expert(checkpoint, config_path, action_dim):
    """Return (network, variables) for a checkpoint written by pqn_rnn_craftax.py."""
    if config_path is None:
        config_path = re.sub(r"_vmap\d+\.safetensors$", "_config.yaml", str(checkpoint))
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
    """Batched rollout that records actions, rewards and achievement counts only."""
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
            return carry, (action.astype(jnp.int8), reward, ach.sum(-1).astype(jnp.int8))

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
        actions, rewards, ach_counts = (np.concatenate(x) for x in zip(*chunks))
        length = np.minimum(np.where(done_t >= 0, done_t + 1, t_run), max_steps)
        has_diamond = (diamond_t >= 0) & (diamond_t < length)
        if stop_at_diamond:
            length = np.where(has_diamond, np.minimum(length, diamond_t + 1 + tail_steps), length)
        return dict(
            actions=actions, rewards=rewards, ach_counts=ach_counts,
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


def make_replay(env, env_params):
    """Re-simulate one episode from (seed, actions) and render pixels from its states."""
    render = make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_AGENT)

    def frame(state):
        pixels = jnp.clip(jnp.round(render(state)), 0, 255).astype(jnp.uint8)
        return pixels, state.achievements.astype(jnp.uint8), pack_state(state)

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
        obs0, state, step_key, pix0, ach0, full0 = reset(seed)
        padded = np.zeros(-(-length // REPLAY_CHUNK) * REPLAY_CHUNK, dtype=np.int32)
        padded[:length] = actions
        first = (obs0, np.zeros(1, np.float32), pix0, ach0, full0)
        columns = [[np.asarray(x)[None]] if i != 1 else [x] for i, x in enumerate(first)]
        for t0 in range(0, len(padded), REPLAY_CHUNK):
            state, out = run_chunk(state, step_key, t0, jnp.asarray(padded[t0 : t0 + REPLAY_CHUNK]))
            for acc, x in zip(columns, out):
                acc.append(np.asarray(x))
        obs, reward, pixels, ach, full = (np.concatenate(c)[: length + 1] for c in columns)
        return dict(
            pixels=pixels, state=obs.astype(np.float32), reward=reward.astype(np.float32),
            achievements=ach, full_state=full,
        )  # fmt: skip

    return replay


class EpisodeWriter:
    """Append episodes to an HDF5 file in the stable-worldmodel flat layout."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(path, "a")
        self.initialized = "ep_len" in self.f

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
            ds.resize(ds.shape[0] + n, axis=0)
            ds[-n:] = data
        for key, value in (("ep_offset", start), ("ep_len", n)):
            self.f[key].resize(self.f[key].shape[0] + 1, axis=0)
            self.f[key][-1] = value
        self.f.flush()

    def close(self):
        self.f.close()


def build_episode(seed, actions, rendered, source):
    length = len(actions)
    action = np.full((length + 1, 1), np.nan, dtype=np.float32)
    action[:length, 0] = actions
    return dict(
        pixels=rendered["pixels"],
        state=rendered["state"],
        action=action,
        reward=rendered["reward"],
        achievements=rendered["achievements"],
        full_state=rendered["full_state"],
        step_idx=np.arange(length + 1, dtype=np.int32),
        seed=np.full(length + 1, seed, dtype=np.int64),
        source=np.full(length + 1, source, dtype=np.int8),
    )  # fmt: skip


def check_replay(seed, length, rendered, rollout, i):
    """The replay must reproduce the rollout; otherwise episodes are not replayable."""
    ok = np.allclose(rendered["reward"][1:], rollout["rewards"][:length, i], atol=1e-6)
    ok &= np.array_equal(rendered["achievements"][1:].sum(-1), rollout["ach_counts"][:length, i])
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
    p.add_argument("--num-episodes", type=int, required=True, help="episodes to add to the file")
    p.add_argument("--seed-start", type=int, required=True, help="first world seed; seeds already in the file are skipped")
    p.add_argument("--eps", type=float, default=0.0, help="epsilon-greedy noise: 0 expert, 1 uniform random")
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
    if args.eps < 1.0 and args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --eps 1.0")
    source = SOURCE_EXPERT if args.eps == 0 else SOURCE_RANDOM if args.eps >= 1 else SOURCE_NOISED

    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    env_params = env.default_params
    network, variables = (None, None)
    if args.eps < 1.0:
        network, variables = load_expert(args.checkpoint, args.config, env.num_actions)

    rollout = make_rollout(env, env_params, network, variables, args.eps, args.num_envs)
    replay = make_replay(env, env_params)
    writer = EpisodeWriter(args.out)
    skip, seed, kept, tried, t0 = writer.existing_seeds(), args.seed_start, 0, 0, time.time()

    while kept < args.num_episodes:
        seeds = []
        while len(seeds) < args.num_envs:
            if seed not in skip:
                seeds.append(seed)
            seed += 1
        result = rollout(np.array(seeds, dtype=np.int32), args.max_steps, args.stop_at_diamond, args.tail_steps)
        keep = result["has_diamond"] if args.require_diamond else np.ones(len(seeds), dtype=bool)
        tried += len(seeds)
        for i in np.flatnonzero(keep)[: args.num_episodes - kept]:
            length = int(result["length"][i])
            actions = result["actions"][:length, i].astype(np.int32)
            rendered = replay(seeds[i], actions)
            check_replay(seeds[i], length, rendered, result, i)
            writer.append(build_episode(seeds[i], actions, rendered, source))
            kept += 1
        print(
            f"[{time.time() - t0:7.1f}s] tried {tried} | kept {kept}/{args.num_episodes} | "
            f"diamond rate {result['has_diamond'].mean():.2%} | "
            f"mean length {result['length'].mean():.0f} | next seed {seed}",
            flush=True,
        )
    total = writer.num_episodes
    writer.close()
    print(f"Wrote {kept} episodes to {args.out} ({total} in file)")


if __name__ == "__main__":
    main()
