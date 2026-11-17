"""How many latent axes a factor needs (sieve/concentration.py): a planted latent and the whole pass."""

import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))
sys.path.insert(0, str(Path(__file__).parent))

from probing import targets as tg  # noqa: E402
from sieve import concentration as cn  # noqa: E402


def planted(n=4000, d=32, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, d)).astype(np.float32)
    y = np.stack([
        z[:, 3] + 0.05 * rng.standard_normal(n),  # one axis
        z[:, :8].sum(1) / np.sqrt(8) + 0.05 * rng.standard_normal(n),  # eight
        rng.standard_normal(n),  # nothing the latent holds
    ], 1).astype(np.float32)
    train = np.arange(n) < int(n * 0.75)
    return z, y, ['one', 'eight', 'noise'], train, ~train


def test_a_factor_on_one_axis_needs_one_axis_and_a_rotation_spreads_it():
    z, y, names, train, test = planted()
    out = cn.analyse(z, y, names, train, test)
    one, eight, noise = (out['factors'][n] for n in names)
    assert one['r2'] > 0.9 and eight['r2'] > 0.9
    assert one['k90_axes'] == 1 and one['pr_axes'] < 1.5
    assert 6 <= eight['k90_axes'] <= 12  # the contributions split 8 ways
    assert one['k90_rand'] >= 12 and one['pr_rand'] > 10  # no alignment left
    assert noise['r2'] < cn.HELD  # not held: out of the summary
    s = out['summary']
    assert s['held'] == 2 and s['analysed'] == 3
    assert s['rotation_check'] < 1e-6  # the full ridge fit does not depend on the basis


def test_an_unused_factor_is_skipped_and_the_report_lists_the_rest():
    z, y, names, train, test = planted()
    y2 = np.concatenate([y, np.ones((len(y), 1), np.float32)], 1)
    out = cn.analyse(z, y2, names + ['constant'], train, test)
    assert 'constant' not in out['factors']
    text = cn.render({'planted': out})
    for part in ('one', 'eight', '(not held)', 'Rotation check', 'planted'):
        assert part in text, part


def test_the_factors_come_from_the_label_columns():
    n = 5
    state = np.zeros((n, tg.STATE_DIM), np.float32)
    view = np.zeros((n, tg.VIEW_TILES, tg.VIEW_CHANNELS), np.float32)
    view[:, :, tg.BLOCK_NAMES.index('grass')] = 1.0
    view[2, 20, tg.BLOCK_NAMES.index('grass')] = 0.0
    view[2, 20, tg.BLOCK_NAMES.index('diamond')] = 1.0
    view[3, 10, tg.NUM_BLOCKS + tg.MOB_NAMES.index('cow')] = 1.0
    state[:, : tg.VIEW_TILES * tg.VIEW_CHANNELS] = view.reshape(n, -1)
    part = types.SimpleNamespace(
        state=state, inventory=np.arange(n * 12, dtype=np.float32).reshape(n, 12),
        vitals=np.ones((n, 4), np.float32), daylight=np.linspace(0, 1, n),
        facing=np.array([1, 2, 3, 4, 1]),
    )
    y, names = cn.factor_targets(part)
    assert y.shape == (n, len(names))
    col = names.index
    assert y[2, col('in_view:diamond')] == 1 and y[0, col('in_view:diamond')] == 0
    assert y[3, col('mob:cow')] == 1 and y[0, col('mob:cow')] == 0
    assert y[1, col('facing:right')] == 1 and y[1, col('facing:left')] == 0
    assert y[4, col('inventory:wood')] == 48 and y[0, col('daylight')] == 0


def test_the_split_keeps_whole_episodes_apart():
    part = types.SimpleNamespace(episode=np.repeat(np.arange(8), 10))
    train, test = cn.split_by_episode(part, 0.25, seed=1)
    assert test.sum() == 20 and not (train & test).any()
    assert not set(part.episode[train]) & set(part.episode[test])


def test_the_whole_pass_on_a_random_model_writes_the_report(tmp_path):
    from hjepa_utils import create_world_model
    from test_sieve import compose, write_files

    write_files(tmp_path)
    model = create_world_model(compose('craftax_sieve'))[0]
    model.get_level(1).decoder.set_pixel_variance(0.05)
    ckpt = tmp_path / 'm.ckpt'
    torch.save(model, ckpt)
    cn.main(['--checkpoint', f'random={ckpt}', '--checkpoint', f'again={ckpt}',
             '--val-file', str(tmp_path / 'val.h5'), '--val-episodes', '8',
             '--windows', '2', '--length', '8',
             '--out', str(tmp_path / 'out'), '--device', 'cpu'])
    text = (tmp_path / 'out' / 'concentration.md').read_text()
    assert '## random' in text and '## again' in text
    assert (tmp_path / 'out' / 'concentration.json').exists()
