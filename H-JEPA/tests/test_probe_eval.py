"""Probing: ridge probes, scores, findings, and the whole pipeline."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


pytest.importorskip('transformers')
pytest.importorskip('stable_pretraining')
pytest.importorskip('lightning')
hydra = pytest.importorskip('hydra')
pytest.importorskip('sklearn')

H_JEPA = Path(__file__).parents[1] / 'h_jepa'
sys.path.insert(0, str(H_JEPA))

import data as hjepa_data  # noqa: E402
import health  # noqa: E402
import probe_eval as pe  # noqa: E402
from hjepa_utils import create_world_model  # noqa: E402

torch.manual_seed(0)


def test_ridge_recovers_a_linear_map_and_scores_noise_at_zero():
    z = torch.randn(400, 8)
    y = z @ torch.randn(8, 3) + 2.0
    predict, _ = pe.fit_with_validation(z[:300], y[:300])
    assert min(pe.r2_columns(y[300:], predict(z[300:]))) > 0.98
    noise = torch.randn(400, 3)
    predict, _ = pe.fit_with_validation(z[:300], noise[:300])
    assert max(pe.r2_columns(noise[300:], predict(z[300:]))) < 0.15


def test_r2_skips_constant_targets_and_auroc_needs_both_classes():
    y = torch.stack([torch.arange(10.0), torch.ones(10)], dim=1)
    scores = pe.r2_columns(y, y.clone())
    assert scores[0] == pytest.approx(1) and scores[1] is None
    label = np.array([0] * 20 + [1] * 20)
    assert pe.auroc(label, label + 0.0) == pytest.approx(1)
    assert pe.auroc(label, np.random.rand(40)) == pytest.approx(0.5, abs=0.2)
    assert pe.auroc(np.zeros(40), np.random.rand(40)) is None
    assert pe.auroc(np.array([1] * 3 + [0] * 37), np.random.rand(40)) is None


def make_latents(n, informative):
    z = torch.randn(n, 12)
    k = torch.randint(0, 4, (n, 1)).float()
    targets = {
        'position': z[:, :2] * 2 if informative else torch.randn(n, 2),
        'facing': k if informative else torch.randint(0, 4, (n, 1)).float(),
        'mobs_in_view': torch.randn(n, 4),
        'vitals': torch.randn(n, 4),
        'inventory': torch.randn(n, 12),
        'daylight': torch.randn(n, 1),
        'achievements': (torch.randn(n, 22) > 0).float(),
    }
    if informative:
        z[:, 2:6] += torch.nn.functional.one_hot(k[:, 0].long(), 4).float() * 3
    return z, targets


def test_probe_levels_find_planted_information():
    train, val = {1: make_latents(600, True)}, {1: make_latents(300, True)}
    level = pe.probe_levels(train, val)[1]
    assert level['targets']['position']['score'] > 0.9
    assert level['targets']['facing']['score'] > 0.8
    assert level['groups']['fast'] > 0.5
    assert (
        abs(level['groups']['slow']) < 0.2
    )  # nothing planted for the slow group
    assert set(level['targets']['inventory']['columns']) == set(
        pe.TARGETS['inventory']['names']
    )


def test_frame_alignment_follows_strides_and_windows():
    cfg = OmegaConf.create(
        {
            'level2': {'stride': 4, 'window_size': 1},
            'level3': {'stride': 3, 'window_size': 2},
        }
    )
    steps = torch.arange(3)
    assert pe.frame_of_step(1, steps, cfg).tolist() == [0, 1, 2]
    assert pe.frame_of_step(2, steps, cfg).tolist() == [0, 4, 8]
    # level 3 windows of two level-2 steps end one level-2 step later
    assert pe.frame_of_step(3, steps, cfg).tolist() == [4, 16, 28]


def level(fast, medium, slow, control=None):
    entry = {
        'groups': {'fast': fast, 'medium': medium, 'slow': slow},
        'targets': {},
    }
    if control is not None:
        entry['control_groups'] = dict(
            zip(('fast', 'medium', 'slow'), control)
        )
    return entry


def codes(findings):
    return {f.code for f in findings}


def test_findings_for_empty_and_untrained_looking_latents():
    nothing = {'levels': {1: level(0.0, 0.01, -0.02, (0.0, 0.0, 0.0))}}
    assert 'PROBE_NOTHING' in codes(health.diagnose_probing(nothing))
    no_gain = {'levels': {1: level(0.4, 0.3, 0.3, (0.38, 0.3, 0.29))}}
    assert 'PROBE_NO_GAIN' in codes(health.diagnose_probing(no_gain))
    missing = {'levels': {1: level(0.8, 0.7, 0.02, (0.1, 0.1, 0.0))}}
    assert 'PROBE_GROUP_MISSING' in codes(health.diagnose_probing(missing))


def test_findings_flag_too_few_probe_samples():
    summary = {
        'levels': {1: level(0.5, 0.5, 0.5)},
        'n_train': {1: 400},
        'dim': {1: 192},
    }
    assert 'PROBE_FEW_SAMPLES' in codes(health.diagnose_probing(summary))
    summary['n_train'][1] = 4000
    assert 'PROBE_FEW_SAMPLES' not in codes(health.diagnose_probing(summary))


def test_findings_describe_what_upper_levels_keep():
    summary = {'levels': {1: level(0.8, 0.6, 0.7), 2: level(0.1, 0.4, 0.65)}}
    found = health.diagnose_probing(summary)
    retention = [f for f in found if f.code == 'PROBE_RETENTION']
    assert retention and 'slow 93%' in retention[0].message
    assert 'PROBE_UPPER_LOSES_SLOW' not in codes(found)
    lost = {'levels': {1: level(0.8, 0.6, 0.7), 2: level(0.5, 0.4, 0.05)}}
    assert 'PROBE_UPPER_LOSES_SLOW' in codes(health.diagnose_probing(lost))
    same = {'levels': {1: level(0.8, 0.6, 0.7), 2: level(0.8, 0.6, 0.7)}}
    assert 'PROBE_NO_SPECIALISATION' in codes(health.diagnose_probing(same))


# --- the whole pipeline on a synthetic dataset ------------------------------


def write_dataset(path, episodes=6, length=70):
    rng = np.random.default_rng(0)
    n = episodes * length
    action = rng.integers(0, 17, (n, 1)).astype(np.float32)
    action[length - 1 :: length] = np.nan
    columns = {
        'pixels': rng.integers(0, 255, (n, 63, 63, 3), dtype=np.uint8),
        'action': action,
        'achievements': (rng.random((n, 22)) > 0.9).astype(np.uint8),
        'inventory': rng.integers(0, 5, (n, 12)).astype(np.int16),
        'vitals': rng.integers(0, 10, (n, 4)).astype(np.int8),
        'facing': rng.integers(1, 5, (n, 1)).astype(np.int8),
        'daylight': rng.random((n, 1)).astype(np.float32),
        'position': rng.integers(0, 64, (n, 2)).astype(np.int16),
        'mobs_in_view': rng.integers(0, 3, (n, 4)).astype(np.int8),
        'step_idx': np.tile(np.arange(length), episodes).astype(np.int32),
    }
    with h5py.File(path, 'w') as f:
        for name, value in columns.items():
            f.create_dataset(name, data=value)
        f.create_dataset('ep_len', data=np.full(episodes, length, np.int32))
        f.create_dataset(
            'ep_offset', data=np.arange(episodes, dtype=np.int64) * length
        )


def test_run_probing_end_to_end(tmp_path):
    write_dataset(tmp_path / 'probe_train.h5')
    write_dataset(tmp_path / 'probe_val.h5')
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    with hydra.initialize_config_dir(
        config_dir=str(H_JEPA / 'config' / 'train'), version_base=None
    ):
        cfg = hydra.compose(
            config_name='craftax_hjepa',
            overrides=[
                'data.dataset.name=probe_train',
                'data.dataset.val_name=probe_val',
            ],
        )
    model = create_world_model(cfg)[0]
    model.train()

    def float_pixels(batch):
        for key, value in batch.items():
            if key.startswith('pixels'):
                batch[key] = value.float() / 255

    summary, findings = pe.run_probing(
        model,
        cfg,
        tmp_path,
        transform=hjepa_data.Compose(
            hjepa_data.OneHotColumn(17, 'action', 'action')
        ),
        normalize=float_pixels,
        device='cpu',
        frames=33,
        max_clips=8,
        thin=2,
        batch_size=4,
    )
    assert set(summary['levels']) == {1, 2}
    for entry in summary['levels'].values():
        assert set(entry['groups']) == {'fast', 'medium', 'slow'}
        assert set(entry['targets']) == set(pe.TARGETS)
        assert (
            entry['control_groups'] and 'control' in entry['targets']['vitals']
        )
    # every level is probed on the same states: the same number of samples
    assert summary['n_train'][1] == summary['n_train'][2]
    assert summary['n_val'][1] == summary['n_val'][2]
    assert model.training  # the model is left in the mode it was in
    report = health.render_probing_report(summary, findings)
    assert '# Probing report' in report and 'untrained' in report


def test_levels_are_probed_on_the_frames_they_share():
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            'level2': {'stride': 4, 'window_size': 1},
            'level3': {'stride': 4, 'window_size': 1},
        }
    )
    model = SimpleNamespace(num_levels=3)
    frames = pe.common_frames(model, cfg, 129, thin=1)
    assert frames.tolist() == list(range(0, 129, 16))  # level 3 steps
    assert pe.common_frames(model, cfg, 129, thin=2).tolist() == list(
        range(0, 129, 32)
    )
    assert pe.common_frames(
        SimpleNamespace(num_levels=1), cfg, 8, thin=1
    ).tolist() == list(range(8))
