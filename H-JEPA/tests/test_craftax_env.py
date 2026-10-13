"""CraftaxClassicEnv against an episode written by the purejaxql collector.

``data/craftax_classic_episode.npz`` comes from
``purejaxql/scripts/collect_craftax_episodes.py`` (uniform-random policy,
seed 0, 60 steps). It ends in a death. If these tests fail, the env and the
collector no longer agree on the dynamics or on the snapshot layout. To
regenerate the fixture after an intended change, rerun the collector with
those settings and store its columns (see the README).
"""

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip('craftax')

from stable_worldmodel.envs.craftax import env as craftax_env  # noqa: E402


DATA = Path(__file__).parent / 'data' / 'craftax_classic_episode.npz'


@pytest.fixture(scope='module')
def episode():
    return dict(np.load(DATA))


@pytest.fixture()
def env(episode):
    env = craftax_env.CraftaxClassicEnv()
    env.reset(seed=int(episode['seed']))
    return env


def assert_frame(env, obs, info, episode, t):
    np.testing.assert_array_equal(env.render(), episode['pixels'][t])
    np.testing.assert_array_equal(obs['state'], episode['state'][t])
    np.testing.assert_array_equal(env.get_state(), episode['full_state'][t])
    np.testing.assert_array_equal(
        info['achievements'], episode['achievements'][t]
    )


def replay(env, episode, start, check=True):
    """Step the recorded actions from frame `start`; yield (t, step result)."""
    for t in range(start, len(episode['actions'])):
        result = env.step(int(episode['actions'][t]))
        if check:
            obs, reward, _, _, info = result
            assert_frame(env, obs, info, episode, t + 1)
            assert reward == pytest.approx(episode['reward'][t + 1])
        yield t + 1, result


def test_reset_matches_dataset(env, episode):
    obs, info = env.reset(seed=int(episode['seed']))
    assert info['seed'] == episode['seed']
    assert_frame(env, obs, info, episode, 0)


def test_replay_matches_dataset(env, episode):
    *_, (_, last) = replay(env, episode, 0)
    assert last[3] is True  # the episode ends in a death: truncated
    assert last[2] is False


def test_set_state_continues_dataset(env, episode):
    start = 20
    env._set_state(episode['full_state'][start])
    assert np.array_equal(env.render(), episode['pixels'][start])
    list(replay(env, episode, start))


def test_pack_unpack_roundtrip(episode):
    snapshot = episode['full_state'][30]
    np.testing.assert_array_equal(
        craftax_env.pack_state(craftax_env.unpack_state(snapshot)), snapshot
    )


def test_frozen_after_episode_end(env, episode):
    *_, (_, last) = replay(env, episode, 0)
    obs, reward, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated and reward == 0.0
    np.testing.assert_array_equal(obs['state'], last[0]['state'])


def test_goal_success_is_terminated(env, episode):
    goal = craftax_env.unpack_state(episode['full_state'][-1])
    env._set_goal_state(episode['full_state'][-1])
    expected = None
    for t in range(len(episode['actions'])):
        state = craftax_env.unpack_state(episode['full_state'][t + 1])
        met = craftax_env.achievements_inventory_test(
            {
                'achievements': craftax_env._achievements(state),
                'inventory': craftax_env._inventory(state),
            },
            {
                'achievements': craftax_env._achievements(goal),
                'inventory': craftax_env._inventory(goal),
            },
        )
        _, _, terminated, _, _ = env.step(int(episode['actions'][t]))
        assert terminated == met
        expected = expected or met
    assert expected, 'the goal is the last frame, so it must be met there'


def test_seed_out_of_range_raises():
    with pytest.raises(ValueError):
        craftax_env.CraftaxClassicEnv().reset(seed=2**31)


def test_state_size_mismatch_raises():
    with pytest.raises(ValueError, match='Craftax version'):
        craftax_env.unpack_state(np.zeros(10, dtype=np.uint8))
