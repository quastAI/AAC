"""The oracle chain (P2): the goal frame the policy sees switches by step."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

from stable_worldmodel.world import _switch_goals  # noqa: E402


def test_switch_goals_sets_the_subgoal_at_its_step():
    goal = np.zeros((2, 3, 4, 4, 3), np.uint8)
    schedule = [
        [
            {'step': 0, 'pixels': np.full((4, 4, 3), 1)},
            {'step': 5, 'pixels': np.full((4, 4, 3), 2)},
        ],
        [{'step': 0, 'pixels': np.full((4, 4, 3), 7)}],
    ]
    _switch_goals(goal, schedule, 0)
    assert (goal[0] == 1).all() and (goal[1] == 7).all()
    _switch_goals(goal, schedule, 3)  # nothing starts at step 3
    assert (goal[0] == 1).all()
    _switch_goals(goal, schedule, 5)
    assert (goal[0] == 2).all() and (goal[1] == 7).all()


def test_the_policy_sees_the_subgoals_while_the_goal_test_stays(dataset):
    pytest.importorskip('craftax')
    import eval_tasks
    import stable_worldmodel as swm

    class Watch(swm.policy.RandomPolicy):
        def get_action(self, info_dict, **kwargs):
            seen.append(
                (
                    np.asarray(info_dict['goal'])[0, -1].copy(),
                    int(np.asarray(info_dict['goal_required'])[0, -1]),
                )
            )
            return super().get_action(info_dict, **kwargs)

    tasks = eval_tasks.make_tasks(
        dataset, mode='achievements', num_tasks=1, depths=(1,), sources=None
    )
    payload = eval_tasks.build_payload(dataset, tasks, eval_budget=6)
    final = np.asarray(payload['data'][0]['pixels'][-1].permute(1, 2, 0))
    first = np.asarray(payload['data'][0]['pixels'][0].permute(1, 2, 0))
    payload['goal_schedule'] = [
        [{'step': 0, 'pixels': first}, {'step': 3, 'pixels': final}]
    ]
    callables = [
        {
            'method': '_set_state',
            'args': {'full_state': {'value': 'full_state'}},
        },
        {
            'method': '_set_goal_state',
            'args': {'full_state': {'value': 'goal_full_state'}},
        },
    ]
    world = swm.World(
        'swm/CraftaxClassic-v1',
        num_envs=1,
        image_shape=(63, 63),
        max_episode_steps=12,
        verbose=0,
    )
    seen = []
    world.set_policy(Watch())
    try:
        world.evaluate_from_dataset(
            None,
            None,
            None,
            goal_offset_steps=payload['goal_offset_steps'],
            eval_budget=6,
            callables=callables,
            load_eval_trajs_path=payload,
        )
    finally:
        world.close()
    assert len(seen) >= 3
    assert all((g == first).all() for g, _ in seen[:3])  # the first subgoal
    assert all((g == final).all() for g, _ in seen[3:])  # then the final goal
    assert not (first == final).all()
    # the env's goal test is the final goal throughout (from the first step:
    # the infos of the reset carry no goal test yet, with or without subgoals)
    assert {r for _, r in seen[1:]} == {len(tasks[0]['required'])}
