"""Craftax-Classic as a Gymnasium environment for stable-worldmodel.

Pixel frames are rendered from the same state that produces the symbolic
observation. The dynamics follow the convention of
``purejaxql/scripts/collect_craftax_episodes.py``, so an episode of the
collected datasets can be re-simulated exactly::

    reset_key, step_key, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
    # reset uses reset_key; the step taken at state.timestep == t uses
    # jax.random.fold_in(step_key, t)

Conventions:
    * ``reset(seed=s)`` generates the world of dataset episode ``seed == s``.
    * ``observation``: ``{'state': symbolic observation (1345,)}``. The frame
      is ``render()`` (uint8, 63x63x3), so use ``image_shape=(63, 63)`` in
      ``World`` to keep frames at native resolution.
    * ``info`` has ``achievements`` (22 flags), ``inventory`` (12 counts),
      ``seed``, and once a goal is set ``goal_required`` (number of
      achievements the goal has and the start lacks) and ``goal_progress``
      (the share of them unlocked).
    * ``terminated`` means the goal test succeeded (``World`` counts it as
      success). Death and the 10000-step horizon are ``truncated``; the env
      then stays frozen on its last frame.
    * ``_set_state(full_state)`` loads a ``full_state`` snapshot from the
      dataset, ``_set_goal_state(full_state)`` sets the goal from the
      snapshot of the goal frame. Both are meant for the ``callables`` of
      ``World.evaluate_from_dataset``.

JAX is forced onto the CPU unless ``JAX_PLATFORMS`` is already set, so that it
does not claim GPU memory next to torch.
"""

import functools
import os
from collections.abc import Callable

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import gymnasium as gym  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from craftax.craftax_classic.constants import (  # noqa: E402
    BLOCK_PIXEL_SIZE_AGENT,
    Achievement,
)
from craftax.craftax_classic.renderer import (  # noqa: E402
    make_craftax_pixel_renderer,
)
from craftax.craftax_env import make_craftax_env_from_name  # noqa: E402
from gymnasium import spaces  # noqa: E402


ENV_NAME = 'Craftax-Classic-Symbolic-v1'
NUM_ACTIONS = 17
NUM_ACHIEVEMENTS = len(Achievement)
OBS_DIM = 1345
MAX_SEED = 2**31 - 1


class _Sim:
    """Jitted Craftax functions, shared by all env instances of a process."""

    def __init__(self):
        env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
        params = env.default_params
        render = make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_AGENT)

        def frame(state):
            return jnp.clip(jnp.round(render(state)), 0, 255).astype(jnp.uint8)

        @jax.jit
        def reset(seed):
            keys = jax.random.split(jax.random.PRNGKey(seed), 3)
            obs, state = env.reset(keys[0], params)
            return obs, state, keys[1], frame(state)

        @jax.jit
        def step(state, step_key, action):
            key = jax.random.fold_in(step_key, state.timestep)
            obs, state, reward, done, _ = env.step(key, state, action, params)
            return obs, state, reward, done, frame(state)

        @jax.jit
        def observe(state):
            return (
                env.get_obs(state),
                env.is_terminal(state, params),
                frame(state),
            )

        self.reset, self.step, self.observe = reset, step, observe
        leaves, self.treedef = jax.tree_util.tree_flatten(reset(0)[1])
        self.leaf_specs = [(np.shape(x), np.asarray(x).dtype) for x in leaves]
        self.state_nbytes = sum(
            int(np.prod(shape)) * dtype.itemsize
            for shape, dtype in self.leaf_specs
        )


@functools.lru_cache(maxsize=1)
def _sim() -> _Sim:
    return _Sim()


def pack_state(state) -> np.ndarray:
    """Flat uint8 snapshot of a state; same layout as the dataset column."""
    parts = []
    for leaf in jax.tree_util.tree_leaves(state):
        leaf = np.asarray(leaf)
        if leaf.dtype == np.bool_:
            leaf = leaf.astype(np.uint8)
        parts.append(np.ascontiguousarray(leaf).reshape(-1).view(np.uint8))
    return np.concatenate(parts)


def unpack_state(full_state):
    """Inverse of ``pack_state``; leaves are numpy arrays."""
    sim = _sim()
    buf = np.asarray(full_state, dtype=np.uint8).reshape(-1)
    if buf.size != sim.state_nbytes:
        raise ValueError(
            f'full_state has {buf.size} bytes, expected {sim.state_nbytes}; '
            'the dataset was written with a different Craftax version.'
        )
    leaves, offset = [], 0
    for shape, dtype in sim.leaf_specs:
        size = int(np.prod(shape)) * dtype.itemsize
        chunk = buf[offset : offset + size]
        offset += size
        if dtype == np.bool_:
            leaf = chunk.astype(np.bool_)
        else:
            leaf = np.frombuffer(chunk.tobytes(), dtype=dtype)
        leaves.append(leaf.reshape(shape))
    return jax.tree_util.tree_unflatten(sim.treedef, leaves)


def _inventory(state) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(state.inventory)
    return np.array([int(x) for x in leaves], dtype=np.int32)


def _achievements(state) -> np.ndarray:
    return np.asarray(state.achievements, dtype=np.uint8)


def required_achievements(goal, start) -> np.ndarray:
    """Boolean mask of the achievements the goal has and the start lacks."""
    if goal is None or start is None:
        return np.zeros(NUM_ACHIEVEMENTS, dtype=bool)
    return goal['achievements'].astype(bool) & ~start['achievements'].astype(
        bool
    )


def new_achievements_test(current, goal, start) -> bool:
    """Success when every achievement that is new in the goal is unlocked.

    The task is defined by progress: the achievements of the goal frame that
    the start frame does not have. Achievements never lock again, so reaching
    them in any order and by any route counts. A goal without progress over the
    start has nothing to reach and never succeeds, so evaluation tasks must
    contain progress.
    """
    required = required_achievements(goal, start)
    return bool(required.any() and current['achievements'][required].all())


def achievements_inventory_test(current, goal, start=None) -> bool:
    """Success when every achievement and inventory count of the goal is met.

    A goal with no progress over the start state is met immediately.
    """
    return bool(
        np.all(current['achievements'] >= goal['achievements'])
        and np.all(current['inventory'] >= goal['inventory'])
    )


GOAL_TESTS = {
    'new_achievements': new_achievements_test,
    'achievements_inventory': achievements_inventory_test,
}


class CraftaxClassicEnv(gym.Env):
    """Craftax-Classic with pixel frames and a snapshot-based state API."""

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        goal_test: str | Callable = 'new_achievements',
        render_mode: str | None = 'rgb_array',
    ):
        self._sim = _sim()
        self.render_mode = render_mode
        self.goal_test = (
            GOAL_TESTS[goal_test] if isinstance(goal_test, str) else goal_test
        )
        self.env_name = 'CraftaxClassic'

        self.observation_space = spaces.Dict(
            {'state': spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)}
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._seed = None
        self._state = None
        self._step_key = None
        self._obs = None
        self._frame = None
        self._over = False
        self._goal = None
        self._start = None

    def _snapshot(self) -> dict:
        return {
            'achievements': _achievements(self._state),
            'inventory': _inventory(self._state),
        }

    def _info(self) -> dict:
        current = self._snapshot()
        required = required_achievements(self._goal, self._start)
        reached = current['achievements'].astype(bool)[required]
        return {
            'env_name': self.env_name,
            'seed': np.int64(self._seed),
            **current,
            # how much of the goal's new achievements is unlocked
            'goal_required': np.int32(required.sum()),
            'goal_progress': np.float32(
                reached.mean() if required.any() else 0.0
            ),
        }

    def reset(self, seed=None, options=None):
        if seed is not None:  # dataset columns give numpy integers
            seed = int(seed)
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, MAX_SEED))
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f'seed must be in [0, 2**31 - 1], got {seed}')
        self._seed = seed
        obs, self._state, self._step_key, frame = self._sim.reset(seed)
        self._obs = np.asarray(obs, dtype=np.float32)
        self._frame = np.asarray(frame)
        self._over = False
        self._goal = None
        self._start = self._snapshot()
        return {'state': self._obs}, self._info()

    def step(self, action):
        if self._over:  # frozen after death or the horizon
            return {'state': self._obs}, 0.0, False, True, self._info()
        obs, self._state, reward, done, frame = self._sim.step(
            self._state, self._step_key, int(action)
        )
        self._obs = np.asarray(obs, dtype=np.float32)
        self._frame = np.asarray(frame)
        self._over = bool(done)
        info = self._info()
        terminated = self._goal is not None and self.goal_test(
            info, self._goal, self._start
        )
        truncated = self._over and not terminated
        return {'state': self._obs}, float(reward), terminated, truncated, info

    def render(self):
        return self._frame

    def get_state(self) -> np.ndarray:
        """Snapshot of the full state, as stored in the dataset column."""
        return pack_state(self._state)

    def _set_state(self, full_state):
        state = unpack_state(full_state)
        obs, terminal, frame = self._sim.observe(state)
        self._state = state
        self._obs = np.asarray(obs, dtype=np.float32)
        self._frame = np.asarray(frame)
        self._over = bool(terminal)
        self._start = self._snapshot()  # tasks are defined relative to this

    def _set_goal_state(self, full_state):
        state = unpack_state(full_state)
        self._goal = {
            'achievements': _achievements(state),
            'inventory': _inventory(state),
        }
        if self._start is None:
            self._start = self._snapshot()

    def close(self):
        """Nothing to release: the jitted functions are shared."""
