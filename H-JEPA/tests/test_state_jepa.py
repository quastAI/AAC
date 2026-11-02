"""The MLP-JEPA on the state (A3, A4): the clips, the predictor's contexts,
the stop-gradient, a short training with its trace, and the report."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

from probing import targets as tg  # noqa: E402
from state_jepa import data, report, train  # noqa: E402
from state_jepa.model import Config, StateJEPA  # noqa: E402

VIEW = tg.VIEW_TILES * tg.VIEW_CHANNELS


def write_file(path, lengths, seed=0):
    """Episodes with random views, inventory, vitals and daylight; random
    actions, NaN in the last frame of an episode."""
    rng = np.random.default_rng(seed)
    n = sum(lengths)
    state = np.zeros((n, tg.STATE_DIM), np.float32)
    tiles = state[:, :VIEW].reshape(n, tg.VIEW_TILES, tg.VIEW_CHANNELS)
    blocks = rng.integers(1, tg.NUM_BLOCKS, (n, tg.VIEW_TILES, 1))
    np.put_along_axis(tiles[:, :, : tg.NUM_BLOCKS], blocks, 1, axis=2)
    tiles[:, :, tg.NUM_BLOCKS :] = rng.random(
        (n, tg.VIEW_TILES, tg.NUM_MOBS)
    ) < 0.05
    action = rng.integers(0, 17, (n, 1)).astype(np.float32)
    action[np.cumsum(lengths) - 1] = np.nan
    with h5py.File(path, 'w') as f:
        f['ep_len'] = np.array(lengths, np.int32)
        f['ep_offset'] = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        f['source'] = np.zeros(n, np.int8)
        f['state'] = state
        f['action'] = action
        f['inventory'] = rng.integers(0, 9, (n, 12)).astype(np.int16)
        f['vitals'] = rng.integers(0, 9, (n, 4)).astype(np.int8)
        f['daylight'] = rng.random((n, 1)).astype(np.float32)
    return state, action


def tiny(**kwargs):
    return Config(latent_dim=8, hidden=32, idm_hidden=16, **kwargs)


def test_clips_stay_in_one_episode_and_carry_their_actions(tmp_path):
    state, action = write_file(tmp_path / 'f.h5', (6, 9))
    table = data.load(tmp_path / 'f.h5')
    starts = data.clip_starts(table, 4)
    # episode 0 has frames 0-5, episode 1 frames 6-14
    assert starts.tolist() == [0, 1, 2, 6, 7, 8, 9, 10, 11]
    x, a = data.batch(table, starts, 4)
    assert x.shape == (len(starts), 4, tg.STATE_DIM)
    rows = starts[:, None] + torch.arange(4)
    assert np.array_equal(x.numpy(), state[rows.numpy()])
    assert (a[:, :3] >= 0).all()  # the first three frames have an action
    assert a[:, :3].tolist() == action[rows.numpy()[:, :3], 0].tolist()


def test_the_predictor_sees_only_the_frames_before_the_one_it_predicts():
    model = StateJEPA(tiny(history=3))
    z = torch.randn(2, 5, 8)
    action = torch.randint(0, 17, (2, 5))
    base = model.predict(z, action)
    assert base.shape == (2, 4, 8)  # frames 1 .. 4
    moved = z.clone()
    moved[:, 2] += 1.0
    after = model.predict(moved, action)
    # frame 2 is in the context of frames 3 and 4, not of frames 1 and 2
    assert torch.allclose(after[:, :2], base[:, :2])
    assert not torch.allclose(after[:, 2:], base[:, 2:])
    # history 3: frame 1 sees frame 0 only, the empty slots carry no flag
    ctx = model.contexts(z, action).reshape(2, 4, 3, -1)
    assert (ctx[:, 0, :2] == 0).all() and (ctx[:, 0, 2, -1] == 1).all()
    assert (ctx[:, 3, :, -1] == 1).all()  # frame 4 has three frames before


@pytest.mark.parametrize('stop_grad', [False, True])
def test_stop_grad_cuts_only_the_target_path(stop_grad):
    torch.manual_seed(0)
    model = StateJEPA(tiny(stop_grad=stop_grad))
    model.eval()  # BatchNorm with fixed statistics: frames do not mix
    x = torch.randn(16, 4, tg.STATE_DIM, requires_grad=True)
    action = torch.randint(0, 17, (16, 4))
    model.losses(x, action)['pred'].backward()
    last = x.grad[:, -1].abs().sum()  # the last frame is only a target
    first = x.grad[:, 0].abs().sum()  # the first frame is only a context
    assert first > 0
    assert (last == 0) if stop_grad else (last > 0)


def test_checks_are_dense_early_and_the_schedule_ends_at_zero():
    steps = train.check_steps(2500)
    assert steps[:5] == [0, 50, 100, 200, 300] and steps[-2:] == [2000, 2500]
    assert train.check_steps(6, every=3) == [0, 3, 6]
    total = 1000
    assert train.schedule(0, total) == pytest.approx(0.1)  # warm-up, 10 steps
    assert train.schedule(9, total) == pytest.approx(1.0)
    assert train.schedule(total, total) == pytest.approx(0.0, abs=1e-9)


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    folder = tmp_path_factory.mktemp('state_jepa')
    write_file(folder / 'train.h5', (30,) * 20, seed=0)
    write_file(folder / 'val.h5', (30,) * 12, seed=1)
    out = {}
    for name, extra in (('A3', []), ('A4', ['--stop-grad'])):
        out[name] = folder / name
        train.main([
            '--train-file', str(folder / 'train.h5'),
            '--val-file', str(folder / 'val.h5'),
            '--out', str(out[name]), '--max-steps', '6', '--check-every', '3',
            '--log-every', '2', '--batch-size', '16', '--latent-dim', '16',
            '--hidden', '32', '--val-windows', '2', '--val-length', '8',
            *extra,
        ])
    return out


def test_a_short_training_writes_its_trace_and_model(runs):
    run = runs['A3']
    trace = json.loads((run / 'trace.json').read_text())
    assert [e['step'] for e in trace] == [0, 3, 6]  # step 0: untrained
    for entry in trace:
        assert {'r2', 'near', 'rank', 'dead_axes', 'clip_share',
                'pred_over_copy'} <= set(entry)
        assert 'mob:zombie' in entry['r2'] and 'daylight' in entry['r2']
        assert 0 < entry['rank'] <= 16
    config = json.loads((run / 'config.json').read_text())
    assert config['steps'] == 6 and config['model']['stop_grad'] is False
    assert json.loads((runs['A4'] / 'config.json').read_text())['model'][
        'stop_grad'
    ]
    saved = torch.load(run / 'model.pt', weights_only=False)
    model = StateJEPA(Config(**saved['config']))
    model.load_state_dict(saved['state_dict'])
    lines = (run / 'losses.csv').read_text().splitlines()
    assert lines[0].startswith('step,lr,loss,pred,sigreg,idm')
    assert len(lines) == 1 + 3  # every 2 of 6 steps


def test_the_report_puts_the_runs_side_by_side(runs, tmp_path):
    out = tmp_path / 'A4_trace.md'
    report.main([
        '--out', str(out), f'A3_s42={runs["A3"]}', f'A4_s42={runs["A4"]}'
    ])
    text = out.read_text()
    assert '| A3_s42 | 0 |' in text and '| A4_s42 | 6 |' in text
    assert 'stop-grad True' in text and 'stop-grad False' in text
    assert (tmp_path / 'A4_trace.png').exists()
