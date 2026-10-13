"""CraftaxClassicEnv through World.evaluate_from_dataset.

A dataset in the layout written by the purejaxql collector is rebuilt from the
fixture episode. A policy that replays the recorded actions must see exactly
the recorded frames, which covers the `seed` column, the `_set_state` and
`_set_goal_state` callables and the frame handling of World.
"""

import numpy as np
import pytest


pytest.importorskip('craftax')

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.policy import BasePolicy  # noqa: E402


START, OFFSET = 2, 30  # wood is unlocked at frame 5, inside the task


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
    # Replaying the dataset unlocks the goal's new achievement: success.
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
