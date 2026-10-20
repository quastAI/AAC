"""Planning videos: which frames are kept, and a GIF from a real evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

H_JEPA = Path(__file__).parents[1] / 'h_jepa'
sys.path.insert(0, str(H_JEPA))

import planning_video as pv  # noqa: E402


def infos(step, n=2, unlock_at=None, done_at=None):
    flags = np.zeros((n, 1, 22), bool)
    if unlock_at is not None and step >= unlock_at:
        flags[:, :, 0] = True
    return {
        'pixels': np.full((n, 1, 63, 63, 3), step % 255, np.uint8),
        'goal': np.zeros((n, 1, 63, 63, 3), np.uint8),
        'achievements': flags,
        'goal_progress': flags[..., 0].astype(float),
        'goal_required': np.ones((n, 1)),
        'terminated': np.full((n, 1), done_at is not None and step >= done_at),
    }


def test_long_budgets_are_subsampled_but_unlocks_and_the_end_kept():
    rec = pv.VideoRecorder([3, None], budget=100, scale=1, max_frames=10)
    assert rec.every == 10
    for step in range(100):
        rec.capture(infos(step, unlock_at=37), step)
    rec.capture(infos(100, unlock_at=37), 100, final=True)
    assert set(rec.frames) == {0}  # the second env is not recorded
    assert len(rec.frames[0]) == 10 + 1 + 1  # every 10th, the unlock, the end


def test_recording_stops_at_success(tmp_path):
    rec = pv.VideoRecorder([0, 1], budget=20, scale=2)
    for step in range(20):
        rec.capture(infos(step, unlock_at=5, done_at=5), step)
    assert all(len(f) == 6 for f in rec.frames.values())  # steps 0..5
    paths = rec.write(tmp_path)
    assert [p.name for p in paths] == ['task_000.gif', 'task_001.gif']
    gif = Image.open(paths[0])
    assert gif.n_frames == 6
    gif.seek(5)
    assert gif.info['duration'] == pv.HOLD_MS  # the last frame is held
    assert gif.size == (2 * 126 + 3 * pv.GAP, pv.HEADER + 126 + pv.FOOTER)


def test_recorder_takes_the_first_tasks_of_a_run():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({'eval': {'videos': 8, 'eval_budget': 400}})
    assert pv.recorder_for(cfg, 8, 18) is None
    assert pv.recorder_for(cfg, 0, 10).tasks == list(range(8)) + [None] * 2
    assert pv.recorder_for(cfg, 5, 10).tasks == [5, 6, 7, None, None]
    cfg.eval.videos = 0
    assert pv.recorder_for(cfg, 0, 10) is None


def test_a_real_evaluation_writes_a_video(dataset, tmp_path):
    pytest.importorskip('craftax')
    import eval_tasks
    import stable_worldmodel as swm

    tasks = eval_tasks.make_tasks(
        dataset, mode='achievements', num_tasks=1, depths=(1,), sources=None
    )
    payload = eval_tasks.build_payload(dataset, tasks, eval_budget=6)
    world = swm.World(
        'swm/CraftaxClassic-v1',
        num_envs=1,
        image_shape=(63, 63),
        max_episode_steps=12,
        verbose=0,
    )
    world.set_policy(swm.policy.RandomPolicy())
    rec = pv.VideoRecorder([0], budget=6, scale=2)
    try:
        world.evaluate_from_dataset(
            None,
            None,
            None,
            goal_offset_steps=payload['goal_offset_steps'],
            eval_budget=6,
            callables=[
                {
                    'method': '_set_state',
                    'args': {'full_state': {'value': 'full_state'}},
                },
                {
                    'method': '_set_goal_state',
                    'args': {'full_state': {'value': 'goal_full_state'}},
                },
            ],
            load_eval_trajs_path=payload,
            recorder=rec,
        )
    finally:
        world.close()
    frames = rec.frames[0]
    assert 2 <= len(frames) <= 7  # steps 0..5 and the end, or less on success
    (path,) = rec.write(tmp_path)
    assert 1 <= Image.open(path).n_frames <= len(frames)  # equal frames merge
    # the goal panel shows the goal frame, not a blank
    goal_panel = np.asarray(frames[0])[pv.HEADER + 10, 2 * pv.GAP + 126 + 10]
    assert goal_panel.sum() > 0
