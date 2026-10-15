"""time_run: the frame geometry matches the dataset, and both timings run."""

import sys
from pathlib import Path

import pytest
import torch


pytest.importorskip('transformers')
pytest.importorskip('stable_pretraining')
pytest.importorskip('hydra')
pytest.importorskip('sklearn')

H_JEPA = Path(__file__).parents[1] / 'h_jepa'
sys.path.insert(0, str(H_JEPA))

import time_run  # noqa: E402


@pytest.mark.parametrize(
    'name, levels',
    [
        ('craftax_hjepa', 1),
        ('craftax_hjepa', 2),
        ('craftax_hjepa', 3),
        ('craftax_hjepa', 4),
        ('craftax_hwm', 3),
    ],
)
def test_required_frames_match_the_dataset_span(name, levels, dataset):
    import stable_worldmodel as swm

    cfg = time_run.compose('train', name, [f'num_levels={levels}'])
    levels = {
        f'level{n}': {
            'frameskip': int(cfg[f'level{n}'].stride),
            'window_size': int(cfg[f'level{n}'].window_size),
            'num_steps': 4,
        }
        for n in range(2, int(cfg.num_levels) + 1)
    }
    ds = swm.data.HDF5Dataset(
        'craftax',
        cache_dir=dataset.h5_path.parent,
        level1={'frameskip': 1, 'num_steps': 4},
        **levels,
    )
    assert time_run.required_frames(cfg) == ds.last_level['span']


def test_timings_run_on_cpu():
    cfg = time_run.compose('train', 'craftax_hjepa')
    device = torch.device('cpu')
    t = time_run.time_training(
        cfg, batch_size=2, steps=1, warmup=0, device=device
    )
    assert t['step_seconds'] > 0 and t['tflops_per_step'] > 0
    assert t['frames_per_clip'] == 13 and t['parameters_m'] > 1

    eval_cfg = time_run.compose('eval', 'craftax_plan')
    for level in (1, 2):
        eval_cfg.solver.solvers[f'level{level}'].update(
            num_samples=6, n_iters=2, topk=2, batch_size=2
        )
    p = time_run.time_planning(
        cfg, eval_cfg, num_envs=2, solves=1, device=device
    )
    assert p['solve_seconds'] > 0 and p['tflops_per_solve'] > 0
    assert p['solves_per_task'] >= 1
    assert p['tflops_per_task'] == pytest.approx(
        p['tflops_per_solve'] / 2 * p['solves_per_task']
    )


def test_training_timing_with_frameskip_above_one():
    # level-1 actions are one one-hot per frameskip step, concatenated
    cfg = time_run.compose('train', 'craftax_hjepa', ['num_levels=1'])
    cfg.data.dataset.level1.frameskip = 2
    t = time_run.time_training(
        cfg, batch_size=2, steps=1, warmup=0, device=torch.device('cpu')
    )
    assert t['step_seconds'] > 0
