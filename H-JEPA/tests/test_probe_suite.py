"""The probe suite end to end on a synthetic dataset, with the real (untrained)
model on the CPU: every section runs, the report renders, the json writes."""

import json
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

H_JEPA = Path(__file__).parents[1] / 'h_jepa'
sys.path.insert(0, str(H_JEPA))

import data as hjepa_data  # noqa: E402
import health  # noqa: E402
import probe_suite  # noqa: E402
from hjepa_utils import create_world_model  # noqa: E402
from probing import targets as tg  # noqa: E402
from probing.report import render  # noqa: E402


def write_dataset(path, episodes=6, length=70, seed=0):
    rng = np.random.default_rng(seed)
    n = episodes * length
    action = rng.integers(0, 17, (n, 1)).astype(np.float32)
    action[length - 1 :: length] = np.nan
    state = np.zeros((n, tg.STATE_DIM), np.float32)
    tiles = state[:, : tg.VIEW_TILES * tg.VIEW_CHANNELS].reshape(n, 63, 21)
    blocks = rng.choice([2, 3, 4, 5, 7, 8, 9, 13], size=(n, 63))
    tiles[np.arange(n)[:, None], np.arange(63), blocks] = 1
    mob_tiles = rng.integers(0, 63, (n, 2))
    mob_kind = rng.integers(0, 4, (n, 2))
    for j in range(2):
        tiles[np.arange(n), mob_tiles[:, j], 17 + mob_kind[:, j]] = 1
    state[:, -1] = rng.random(n) > 0.9  # asleep
    columns = {
        'pixels': rng.integers(0, 255, (n, 63, 63, 3), dtype=np.uint8),
        'action': action,
        'achievements': (rng.random((n, 22)) > 0.9).astype(np.uint8),
        'inventory': rng.integers(0, 5, (n, 12)).astype(np.int16),
        'vitals': rng.integers(0, 10, (n, 4)).astype(np.int8),
        'facing': rng.integers(1, 5, (n, 1)).astype(np.int8),
        'daylight': np.where(rng.random((n, 1)) > 0.5, 1.0, 0.3).astype(
            np.float32
        ),
        'position': rng.integers(0, 64, (n, 2)).astype(np.int16),
        'mobs_in_view': rng.integers(0, 3, (n, 4)).astype(np.int8),
        'state': state,
        'source': np.repeat(np.arange(episodes) % 2, length)[:, None].astype(
            np.int8
        ),
        'step_idx': np.tile(np.arange(length), episodes).astype(np.int32),
    }
    with h5py.File(path, 'w') as f:
        for name, value in columns.items():
            f.create_dataset(name, data=value)
        f.create_dataset('ep_len', data=np.full(episodes, length, np.int32))
        f.create_dataset(
            'ep_offset', data=np.arange(episodes, dtype=np.int64) * length
        )


def float_pixels(batch):
    for key, value in batch.items():
        if key.startswith('pixels'):
            batch[key] = value.float() / 255


@pytest.fixture(scope='module')
def suite(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('suite')
    write_dataset(tmp / 'probe_train.h5')
    write_dataset(tmp / 'probe_val.h5', seed=1)
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    with hydra.initialize_config_dir(
        config_dir=str(H_JEPA / 'config' / 'train'), version_base=None
    ):
        cfg = hydra.compose(
            config_name='craftax_hjepa',
            overrides=[
                'num_levels=1',
                'data.dataset.name=probe_train',
                'data.dataset.val_name=probe_val',
            ],
        )
    torch.manual_seed(1)  # the untrained control is seeded with 0
    model = create_world_model(cfg)[0]
    model.train()
    summary, findings = probe_suite.run_suite(
        model,
        cfg,
        tmp,
        transform=hjepa_data.Compose(
            hjepa_data.OneHotColumn(17, 'action', 'action')
        ),
        normalize=float_pixels,
        device='cpu',
        frames=33,
        max_clips=8,
        batch_size=4,
        steps=20,
        layers=(3,),
        token_cap=150,
        swap_frames=6,
        boot=4,
        horizons=(1, 2, 4, 8, 16),
        offsets=(4, 8),
        spectrum_ks=(4, 8),
        min_n=5,
        log=lambda msg: None,
    )
    return model, summary, findings, tmp, cfg


def test_the_suite_runs_every_section(suite):
    model, summary, findings, _, _ = suite
    assert model.training  # left in the mode it was in
    for key in (
        'facts',
        'state',
        'levels',
        'maps',
        'tokens',
        'strip',
        'causes',
        'spectrum',
        'swaps',
        'distance',
        'loss',
        'change',
        'events',
        'mobs',
        'monotonicity',
        'idm',
        'verdicts',
    ):
        assert summary.get(key), key
    # disjoint clips: 2 per episode of 70 frames, 8 kept
    assert summary['n_val'][1] == 8 * 33
    assert set(summary['levels'][1]['groups']) == set(tg.GROUPS)
    assert summary['levels'][1]['control_groups']
    assert set(summary['maps']) == {'mean_patches', 'cls', 'latent', 'control'}
    assert set(summary['tokens']) == {'raw', 'layer3', 'final', 'control'}
    assert set(summary['change']) == {1, 3}  # planning and training history
    assert summary['change_plan'] is summary['change'][1]
    assert summary['monotonicity']['expert_only']  # the source column
    assert 'null' in summary['swaps']['targets']
    assert any(k.startswith('block:') for k in summary['swaps']['targets'])
    assert summary['verdicts']['blocks'] and summary['verdicts']['items']
    assert isinstance(findings, list)


def test_the_report_and_the_json(suite):
    _, summary, findings, tmp, _ = suite
    text = render(summary, findings)
    for heading in (
        '# Probe suite',
        '## Verdicts',
        '### Block types',
        '## A. Encoder',
        '### Block type of a view tile, stage by stage',
        '### Sprite swaps',
        '### Mobs on a view tile, stage by stage',
        '## B. Predictor',
        '### Change test, history 1',
        '### Change test, history 3',
        '### Which transitions carry the prediction loss',
    ):
        assert heading in text, heading
    probe_suite.write(tmp / 'out', summary, findings)
    doc = json.loads((tmp / 'out' / 'probe_summary.json').read_text())
    assert doc['summary']['levels']['1']['groups']
    assert (tmp / 'out' / 'probe_report.md').read_text() == text
    # compare_runs reads the same keys as before
    assert {'groups', 'control_groups'} <= set(doc['summary']['levels']['1'])


def test_findings_come_from_the_suite(suite):
    _, summary, _, _, _ = suite
    codes = {f.code for f in health.diagnose_probing(summary)}
    assert 'PROBE_ROLLOUT_HORIZON' in codes


def test_the_held_out_clips_share_no_frame(suite):
    from training_monitor import build_health_loader

    *_, tmp, cfg = suite
    loader, frames = build_health_loader(
        cfg,
        tmp,
        None,
        frames=33,
        batch_size=4,
        max_clips=50,
        name='probe_val',
        columns=('pixels', 'action'),
        disjoint=True,
    )
    dataset = loader.dataset.dataset
    clips = [dataset.clip_indices[i] for i in loader.dataset.indices]
    assert frames == 33 and len(clips) == 12  # 2 per episode of 70 frames
    assert all(start % 33 == 0 for _, start in clips)
    assert len(set(map(tuple, clips))) == len(clips)
