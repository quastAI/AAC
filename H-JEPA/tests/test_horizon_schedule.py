"""Horizon schedules and the coupling of horizon, stride and subgoals."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

from hierarchical_solver import (  # noqa: E402
    HierarchicalSolver,
    check_subgoal_horizon,
)
from stable_worldmodel import HierarchicalPlanConfig, PlanConfig  # noqa: E402
from stable_worldmodel.solver.solver import (  # noqa: E402
    get_planning_horizon,
    horizon_schedule,
)


def test_fixed_schedule_keeps_the_configured_horizon():
    assert horizon_schedule('fixed', num_calls=5, configured_horizon=4) == [
        4
    ] * 5
    decreasing = horizon_schedule(
        'decreasing', num_calls=8, configured_horizon=4
    )
    assert decreasing[0] == 4 and decreasing[-1] == 1
    with pytest.raises(ValueError, match='horizon_schedule'):
        horizon_schedule('sometimes', num_calls=5, configured_horizon=4)


def test_planning_horizon_follows_the_schedule_kind():
    kwargs = dict(
        configured_horizon=10, replanning_interval=5, eval_budget=100
    )
    late = get_planning_horizon(steps_taken=95, **kwargs)
    assert late < 10  # the default shrinks towards the end of the budget
    assert get_planning_horizon(steps_taken=95, schedule='fixed', **kwargs) == 10


@pytest.mark.parametrize(
    'horizon, ok', [(3, False), (4, True), (7, True), (8, False), (12, False)]
)
def test_one_subgoal_needs_a_horizon_within_one_to_two_strides(horizon, ok):
    # a longer horizon compares the last frame with a subgoal of an earlier
    # time without an error, a shorter one leaves no frame to compare
    def check():
        check_subgoal_horizon(
            level=1, num_subgoals=1, planning_horizon=horizon, stride=4
        )

    if ok:
        check()
    else:
        with pytest.raises(ValueError, match='subgoals'):
            check()


def test_several_subgoals_match_the_horizon():
    check_subgoal_horizon(
        level=1, num_subgoals=3, planning_horizon=13, stride=4
    )
    with pytest.raises(ValueError):
        check_subgoal_horizon(
            level=1, num_subgoals=2, planning_horizon=13, stride=4
        )


def solver(schedule, horizons=(4, 4, 4), strides=(4, 4)):
    levels = tuple(
        PlanConfig(
            horizon=h,
            receding_horizon=2,
            horizon_schedule=schedule,
            pre_decay_horizon=None,
        )
        for h in horizons
    )
    config = HierarchicalPlanConfig(receding_horizon=2, levels=levels)
    models = [SimpleNamespace(temporal_stride=s) for s in (1, *strides)]
    out = HierarchicalSolver({}, models)
    out._config = config
    return out


def test_fixed_hierarchical_schedule_never_shrinks_or_skips_a_level():
    schedules = solver('fixed')._horizon_schedules(eval_budget=100)
    assert set(schedules) == {1, 2, 3}
    for level in schedules:
        assert set(schedules[level]) == {4}
        assert len(schedules[level]) == 50


def test_decreasing_hierarchical_schedule_skips_upper_levels_late():
    schedules = solver('decreasing')._horizon_schedules(eval_budget=100)
    assert schedules[3][-1] == 1  # the top level ends skipped
