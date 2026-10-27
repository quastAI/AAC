"""Evaluation task generation on a synthetic dataset."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

import eval_tasks  # noqa: E402


LENGTH = 90


def unlock_schedule(e):
    schedule = {0: 10 + e, 1: 30 + e}
    if e % 2 == 0:
        schedule[9] = 60
    if e % 3 == 0:
        schedule[19] = 80  # diamond
    return schedule


@pytest.fixture()
def dataset(tmp_path):
    import stable_worldmodel as swm

    columns = {
        k: []
        for k in (
            'pixels',
            'achievements',
            'full_state',
            'seed',
            'step_idx',
            'action',
            'source',
        )
    }
    for e in range(12):
        flags = np.zeros((LENGTH, 22), dtype=np.uint8)
        for index, frame in unlock_schedule(e).items():
            flags[frame:, index] = 1
        columns['achievements'].append(flags)
        columns['pixels'].append(np.full((LENGTH, 4, 4, 3), e, dtype=np.uint8))
        columns['full_state'].append(np.full((LENGTH, 5), e, dtype=np.uint8))
        columns['seed'].append(np.full(LENGTH, 1000 + e, dtype=np.int64))
        columns['step_idx'].append(np.arange(LENGTH, dtype=np.int32))
        columns['source'].append(np.zeros(LENGTH, dtype=np.int8))
        action = np.zeros((LENGTH, 1), dtype=np.float32)
        action[-1] = np.nan
        columns['action'].append(action)
    with h5py.File(tmp_path / 'tasks.h5', 'w') as f:
        for name, parts in columns.items():
            f.create_dataset(name, data=np.concatenate(parts))
        f.create_dataset('ep_len', data=np.full(12, LENGTH, dtype=np.int32))
        f.create_dataset(
            'ep_offset', data=np.arange(12, dtype=np.int64) * LENGTH
        )
    return swm.data.HDF5Dataset(
        'tasks', cache_dir=tmp_path, level1={'frameskip': 1, 'num_steps': 1}
    )


def test_unlock_frames_one_entry_per_achievement():
    flags = np.zeros((10, 22), dtype=bool)
    flags[3:, 0] = True
    flags[7:, 1] = True
    flags[7:, 5] = True  # two at once
    assert eval_tasks.unlock_frames(flags).tolist() == [3, 7, 7]
    assert eval_tasks.unlock_frames(flags[:3]).tolist() == []


def test_achievement_names_match_craftax():
    constants = pytest.importorskip('craftax.craftax_classic.constants')
    names = [a.name.lower() for a in constants.Achievement]
    assert names == eval_tasks.ACHIEVEMENT_NAMES
    assert eval_tasks.DIAMOND == constants.Achievement.COLLECT_DIAMOND.value


def check_tasks(dataset, tasks):
    flags = eval_tasks.episode_achievements(dataset)
    assert len({t['episode'] for t in tasks}) == len(tasks)  # one per world
    for t in tasks:
        f = flags[t['episode']]
        required = np.nonzero(f[t['goal']] & ~f[t['start']])[0].tolist()
        assert t['required'] == required and required  # always progress
        assert t['distance'] == t['goal'] - t['start'] > 0
        assert t['required_names'] == [
            eval_tasks.ACHIEVEMENT_NAMES[i] for i in required
        ]


def test_offset_tasks_have_progress_and_spread_over_offsets(dataset):
    tasks = eval_tasks.make_tasks(
        dataset, mode='offset', num_tasks=12, seed=1, offsets=(15, 40)
    )
    assert len(tasks) == 12
    check_tasks(dataset, tasks)
    assert {t['offset'] for t in tasks} == {15, 40}
    assert all(t['distance'] == t['offset'] for t in tasks)
    again = eval_tasks.make_tasks(
        dataset, mode='offset', num_tasks=12, seed=1, offsets=(15, 40)
    )
    assert tasks == again  # deterministic given the seed


def test_achievement_depth_tasks_end_on_the_dth_unlock(dataset):
    tasks = eval_tasks.make_tasks(
        dataset, mode='achievements', num_tasks=12, seed=2, depths=(1, 2, 3)
    )
    check_tasks(dataset, tasks)
    flags = eval_tasks.episode_achievements(dataset)
    for t in tasks:
        events = eval_tasks.unlock_frames(flags[t['episode']])
        ahead = events[(events > t['start']) & (events <= t['goal'])]
        assert len(ahead) == t['depth'] and ahead[-1] == t['goal']
        assert len(t['required']) == t['depth']  # no simultaneous unlocks here


def test_max_distance_limits_the_goal(dataset):
    tasks = eval_tasks.make_tasks(
        dataset,
        mode='achievements',
        num_tasks=12,
        depths=(1, 2),
        max_distance=25,
    )
    assert tasks and all(t['distance'] <= 25 for t in tasks)


def test_episode_tasks_run_from_reset_to_diamond(dataset):
    tasks = eval_tasks.make_tasks(dataset, mode='episode', num_tasks=12)
    check_tasks(dataset, tasks)
    assert len(tasks) == 4  # episodes 0, 3, 6, 9 reach the diamond
    for t in tasks:
        assert t['start'] == 0 and t['goal'] == 80
        assert eval_tasks.DIAMOND in t['required']
    with pytest.raises(ValueError):
        eval_tasks.make_tasks(dataset, mode='nope', num_tasks=1)


def test_chain_tasks_switch_through_expert_frames_to_the_final_goal(dataset):
    tasks = eval_tasks.make_tasks(
        dataset, mode='chain', num_tasks=12, offsets=(20,)
    )
    episode = eval_tasks.make_tasks(dataset, mode='episode', num_tasks=12)
    # scored as protocol B: the same tasks, start and final goal
    assert [(t['episode'], t['start'], t['goal']) for t in tasks] == [
        (t['episode'], t['start'], t['goal']) for t in episode
    ]
    for t in tasks:
        assert t['subgoals'] == [20, 40, 60, 80]  # the last is the goal
        assert t['switch_steps'] == [0, 40, 80, 120]  # 2x the offset each
    held = eval_tasks.make_tasks(
        dataset, mode='chain', num_tasks=1, offsets=(30,), subgoal_budget=50
    )[0]
    assert held['subgoals'] == [30, 60, 80]
    assert held['switch_steps'] == [0, 50, 100]
    payload = eval_tasks.build_payload(dataset, tasks, eval_budget=200)
    for t, schedule in zip(tasks, payload['goal_schedule']):
        assert [g['frame'] for g in schedule] == t['subgoals']
        assert [g['step'] for g in schedule] == t['switch_steps']
        pixels = schedule[0]['pixels']
        assert pixels.shape == (4, 4, 3) and pixels.dtype == np.uint8
        assert int(pixels[0, 0, 0]) == t['episode']  # the right world
    assert 'subgoals per task' in eval_tasks.describe(tasks)
    plain = eval_tasks.build_payload(dataset, episode, eval_budget=200)
    assert 'goal_schedule' not in plain


def test_payload_is_loadable_with_variable_lengths(dataset):
    from stable_worldmodel.world import _pad_loaded_eval_trajectories

    tasks = eval_tasks.make_tasks(
        dataset, mode='achievements', num_tasks=8, seed=3, depths=(1, 3)
    )
    payload = eval_tasks.build_payload(dataset, tasks, eval_budget=200)
    assert payload['eval_budget'] == 200
    assert payload['goal_offset_steps'] == max(t['distance'] for t in tasks)
    assert payload['task_info']['tasks'] == tasks
    for t, ep in zip(tasks, payload['data']):
        # only the start and the goal frame are stored, not the frames between
        assert ep['pixels'].shape[0] == 2
        assert int(ep['pixels'][0, 0, 0, 0]) == t['episode']  # the right world
        assert ep['step_idx'].tolist() == [t['start'], t['goal']]
        assert ep['full_state'].shape[0] == 2

    padded = _pad_loaded_eval_trajectories(payload['data'], payload['columns'])
    lengths = {int(ep['pixels'].shape[0]) for ep in padded}
    assert (
        len(lengths) == 1
    )  # padded to one length by repeating the last frame
    for t, ep in zip(tasks, padded):
        assert int(ep['pixels'][-1, 0, 0, 0]) == t['episode']
    assert 'goal' in eval_tasks.describe(
        tasks
    ) or 'tasks' in eval_tasks.describe(tasks)


def reload_with_sources(dataset, sources):
    import stable_worldmodel as swm

    with h5py.File(dataset.h5_path, 'r+') as f:
        del f['source']
        f.create_dataset(
            'source', data=np.repeat(np.asarray(sources, dtype=np.int8), LENGTH)
        )
    return swm.data.HDF5Dataset(
        'tasks',
        cache_dir=dataset.h5_path.parent,
        level1={'frameskip': 1, 'num_steps': 1},
    )


@pytest.mark.parametrize(
    'mode', ['offset', 'achievements', 'episode', 'chain']
)
def test_tasks_come_from_expert_episodes_only(dataset, mode):
    sources = [e % 2 for e in range(12)]  # diamond episodes: e % 3 == 0
    dataset = reload_with_sources(dataset, sources)
    kwargs = dict(mode=mode, num_tasks=12, seed=1)
    tasks = eval_tasks.make_tasks(dataset, **kwargs)
    assert tasks
    assert {sources[t['episode']] for t in tasks} == {0}
    everything = eval_tasks.make_tasks(dataset, sources=None, **kwargs)
    assert {sources[t['episode']] for t in everything} - {0}


def test_a_dataset_without_sources_is_refused_unless_asked(dataset):
    import stable_worldmodel as swm

    with h5py.File(dataset.h5_path, 'r+') as f:
        del f['source']
    plain = swm.data.HDF5Dataset(
        'tasks',
        cache_dir=dataset.h5_path.parent,
        level1={'frameskip': 1, 'num_steps': 1},
    )
    with pytest.raises(ValueError, match='source'):
        eval_tasks.make_tasks(plain, mode='offset', num_tasks=3)
    assert eval_tasks.make_tasks(
        plain, mode='offset', num_tasks=3, sources=None
    )


def test_too_few_tasks_stop_the_generator():
    # the evaluation asks for eval.num_eval tasks and fails late otherwise
    eval_tasks.require_tasks([{}] * 5, 5)
    eval_tasks.require_tasks([{}] * 6, 5)
    with pytest.raises(ValueError, match='only 5 of 300 tasks found'):
        eval_tasks.require_tasks([{}] * 5, 300)
