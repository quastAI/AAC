"""CraftaxClassicEnv through World.evaluate_from_dataset.

A dataset in the layout written by the purejaxql collector is rebuilt from the
fixture episode. A policy that replays the recorded actions must see exactly
the recorded frames, which covers the `seed` column, the `_set_state` and
`_set_goal_state` callables and the frame handling of World.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest


pytest.importorskip('craftax')

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.policy import BasePolicy  # noqa: E402


FIXTURE = Path(__file__).parent / 'data' / 'craftax_classic_episode.npz'
START, OFFSET = 10, 30


@pytest.fixture(scope='module')
def episode():
    return dict(np.load(FIXTURE))


@pytest.fixture()
def dataset(episode, tmp_path):
    n = len(episode['pixels'])
    action = np.full((n, 1), np.nan, dtype=np.float32)
    action[:-1, 0] = episode['actions']
    columns = dict(
        pixels=episode['pixels'],
        state=episode['state'],
        full_state=episode['full_state'],
        achievements=episode['achievements'],
        reward=episode['reward'],
        action=action,
        step_idx=np.arange(n, dtype=np.int32),
        ep_idx=np.zeros(n, dtype=np.int32),
        seed=np.full(n, episode['seed'], dtype=np.int64),
        source=np.full(n, 2, dtype=np.int8),
    )
    with h5py.File(tmp_path / 'craftax.h5', 'w') as f:
        for key, value in columns.items():
            f.create_dataset(key, data=value)
        f.create_dataset('ep_offset', data=np.array([0], dtype=np.int64))
        f.create_dataset('ep_len', data=np.array([n], dtype=np.int32))
    return swm.data.HDF5Dataset(
        'craftax',
        cache_dir=tmp_path,
        level1=dict(frameskip=1, num_steps=1),
    )


class ReplayPolicy(BasePolicy):
    def __init__(self, actions):
        super().__init__()
        self.type = 'replay'
        self.seed = None
        self.actions = actions
        self.t = 0
        self.frames = []

    def reset(self):
        self.t, self.frames = 0, []

    def get_action(self, info, **kwargs):
        self.frames.append(np.array(info['pixels'])[:, 0])
        action = self.actions[self.t]
        self.t += 1
        return np.array([action], dtype=np.int64)


def test_evaluate_from_dataset_replays_recorded_frames(episode, dataset):
    world = swm.World(
        'swm/CraftaxClassic-v1',
        num_envs=1,
        image_shape=(63, 63),
        max_episode_steps=500,
        verbose=0,
    )
    policy = ReplayPolicy(episode['actions'][START:])
    world.set_policy(policy)
    results = world.evaluate_from_dataset(
        dataset,
        episodes_idx=[0],
        start_steps=[START],
        goal_offset_steps=OFFSET,
        eval_budget=OFFSET,
        callables=[
            dict(
                method='_set_state',
                args=dict(full_state=dict(value='full_state')),
            ),
            dict(
                method='_set_goal_state',
                args=dict(full_state=dict(value='goal_full_state')),
            ),
        ],
    )
    assert results['seeds'][0] == episode['seed']
    assert len(policy.frames) == OFFSET
    for t, frame in enumerate(policy.frames):
        np.testing.assert_array_equal(frame[0], episode['pixels'][START + t])
    # The goal frame is reached by replaying the dataset: goal test passes.
    assert results['success_rate'] == 100.0


def test_initial_action_marker_is_discrete():
    world = swm.World(
        'swm/CraftaxClassic-v1',
        num_envs=1,
        image_shape=(63, 63),
        max_episode_steps=10,
        verbose=0,
    )
    world.reset(seed=3)
    assert world.infos['action'].dtype.kind == 'i'
    assert (world.infos['action'] == -1).all()
