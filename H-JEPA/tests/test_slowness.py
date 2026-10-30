"""Slowness analysis: the layout of the nested sample and the variance by time
scale on data with planted scales."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

from probing import targets as tg  # noqa: E402
from slowness import factors as fa  # noqa: E402
from slowness import predict as pe  # noqa: E402
from slowness import report as rp  # noqa: E402
from slowness import sample as sm  # noqa: E402
from slowness import variance as va  # noqa: E402

E, W, L, D = 40, 4, 16, 6


def planted(episode=1.0, window=1.0, clip_means=1.0, clip=1.0, seed=0):
    """Features whose variance at each scale is the given amount in every
    dimension: independent offsets per episode, window, clip and frame."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, episode**0.5, (E, 1, 1, 1, D))
    x = x + rng.normal(0, window**0.5, (E, W, 1, 1, D))
    x = x + rng.normal(0, clip_means**0.5, (E, W, L // 4, 1, D))
    x = x + rng.normal(0, clip**0.5, (E, W, L // 4, 4, D))
    return x.reshape(E, W, L, D)


def test_sample_layout():
    lengths = np.array([10, 300, 256, 1000])
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    s = sm.nested_sample(lengths, offsets, windows=4, length=64, seed=1)
    assert s['episode'].tolist() == [1, 2, 3]  # the 10-frame episode is out
    assert s['rows'].shape == (3, 4, 64)
    for i, ep in enumerate(s['episode']):
        t = s['t'][i]
        assert (np.diff(t, axis=-1) == 1).all()  # a window is contiguous
        assert (s['rows'][i] == offsets[ep] + t).all()
        assert t.min() >= 0 and t.max() < lengths[ep]
        # windows are ordered, never overlap, one per equal segment
        assert (t[1:, 0] > t[:-1, -1]).all()
        edges = np.linspace(0, lengths[ep], 5)
        for w in range(4):
            assert edges[w] <= t[w, 0] and t[w, -1] < edges[w + 1]


def test_sample_has_no_duplicate_rows():
    lengths = np.full(30, 500)
    offsets = np.arange(30) * 500
    s = sm.nested_sample(lengths, offsets, seed=3)
    assert len(np.unique(s['rows'])) == s['rows'].size


def test_sample_is_seeded_and_limited():
    lengths = np.full(30, 500)
    offsets = np.arange(30) * 500
    a = sm.nested_sample(lengths, offsets, seed=3, max_episodes=5)
    b = sm.nested_sample(lengths, offsets, seed=3, max_episodes=5)
    assert len(a['episode']) == 5 and (a['rows'] == b['rows']).all()
    with pytest.raises(ValueError):
        sm.nested_sample([10], [0])


def test_read_columns_follows_the_rows(tmp_path):
    lengths = np.array([300, 300])
    offsets = np.array([0, 300])
    path = tmp_path / 'f.h5'
    with h5py.File(path, 'w') as f:
        f['ep_len'], f['ep_offset'] = lengths, offsets
        f['step_idx'] = np.concatenate([np.arange(300), np.arange(300)])
        f['source'] = np.repeat([0, 3], 300).astype(np.int8)
        f['dead'] = np.zeros(600, np.uint8)
    s = sm.nested_sample(lengths, offsets, windows=2, length=32)
    got = sm.read_columns(path, s, ('step_idx',))
    assert got['step_idx'].shape == (2, 2, 32)
    assert (got['step_idx'] == s['t']).all()
    assert got['source'].tolist() == [0, 3]


def test_decomposition_adds_up_and_recovers_the_scales():
    x = planted(episode=4.0, window=2.0, clip_means=1.0, clip=0.5)
    parts = va.decompose(x)
    assert np.allclose(sum(parts[k] for k in va.LEVELS), parts['total'])
    share = va.shares(parts)
    assert sum(share.values()) == pytest.approx(1.0)
    # finite samples: each estimate is within sampling noise of the planted
    for k, planted_share in zip(va.LEVELS, np.array([4, 2, 1, 0.5]) / 7.5):
        assert share[k] == pytest.approx(planted_share, abs=0.1)


def test_null_removes_the_episode_share():
    x = planted(episode=4.0, window=1.0, clip_means=0.0, clip=1.0)
    real = va.shares(va.decompose(x))['episode']
    null, spread = va.null_shares(x, n=8)['episode']
    assert real > 0.4
    # windows dealt to random episodes: only the finite-sample share is left
    assert null < real / 3
    assert spread >= 0


def test_lag_correlation_of_an_ar1_process():
    rng = np.random.default_rng(0)
    rho = 0.9
    x = np.zeros((20, 4, 256, 2))
    for t in range(1, 256):
        x[:, :, t] = rho * x[:, :, t - 1] + rng.normal(0, 1, (20, 4, 2))
    corr = va.lag_correlation(x[:, :, 32:], lags=(1, 8))
    assert corr[1].mean() == pytest.approx(rho, abs=0.03)
    assert corr[8].mean() == pytest.approx(rho**8, abs=0.06)


def test_slowness_spectrum_separates_a_slow_and_a_fast_axis():
    rng = np.random.default_rng(1)
    slow = rng.normal(0, 3.0, (E, W, 1, 1))  # big, constant per window
    fast = rng.normal(0, 1.0, (E, W, L, 1))  # small, new every frame
    x = np.concatenate([np.broadcast_to(slow, fast.shape), fast], -1)
    out = va.per_axis(x, lags=(1,))
    assert out['variance'][0] > out['variance'][1]  # sorted, largest first
    assert out['share']['episode'][0] + out['share']['window'][0] > 0.95
    assert out['share']['clip'][1] > 0.65
    assert out['lag_correlation'][1][0] > 0.99
    assert abs(out['lag_correlation'][1][1]) < 0.2


def test_within_rank_drops_the_directions_that_only_tell_episodes_apart():
    rng = np.random.default_rng(2)
    episode = rng.normal(0, 5.0, (E, 1, 1, D))  # six big episode-wide axes
    inner = np.zeros((E, W, L, D))
    inner[..., :2] = rng.normal(0, 1.0, (E, W, L, 2))  # two axes vary inside
    x = episode + inner
    whole = va.effective_rank(va.pca(x)[2])
    inside = va.within_rank(x, 'episode')
    assert whole > 4.5 and inside == pytest.approx(2.0, abs=0.2)
    assert va.within_rank(x, 'window') <= inside + 0.1
    with pytest.raises(ValueError):
        va.within_rank(x, 'clip')


def fake_columns(e=30, w=2, length=8, seed=0):
    """Columns of a sample with known content: a random block type on every
    view tile, one zombie, the player facing a random direction."""
    rng = np.random.default_rng(seed)
    shape = (e, w, length)
    state = np.zeros(shape + (tg.STATE_DIM,), np.float32)
    tiles = state[..., : tg.VIEW_TILES * tg.VIEW_CHANNELS].reshape(
        shape + (tg.VIEW_TILES, tg.VIEW_CHANNELS)
    )
    blocks = rng.integers(0, tg.NUM_BLOCKS, shape + (tg.VIEW_TILES,))
    np.put_along_axis(tiles, blocks[..., None], 1.0, axis=-1)
    tiles[..., 0, tg.NUM_BLOCKS] = 1.0  # a zombie on tile 0
    return {
        'state': state,
        'inventory': rng.integers(0, 5, shape + (12,)).astype(np.int16),
        'vitals': rng.integers(0, 9, shape + (4,)).astype(np.int8),
        'daylight': rng.random(shape + (1,)).astype(np.float32),
        'facing': rng.integers(1, 5, shape + (1,)).astype(np.int8),
        'achievements': rng.integers(0, 2, shape + (22,)).astype(np.uint8),
        'action': rng.integers(0, 17, shape + (1,)).astype(np.float32),
    }, blocks


def test_groups_follow_the_state():
    cols, blocks = fake_columns()
    groups = fa.build_groups(cols)
    assert set(groups) == set(fa.GROUPS)
    assert groups['slow'].shape[-1] == 12 + 22 + 1
    assert groups['near'].shape[-1] == 4 * tg.NUM_BLOCKS
    # the composition counts add up to the 63 tiles, the zombie is counted
    counts = groups['view'][..., : tg.NUM_BLOCKS]
    assert (counts.sum(-1) == tg.VIEW_TILES).all()
    assert (groups['view'][..., tg.NUM_BLOCKS] == 1).all()
    # the left neighbour of the player is the third neighbour tile
    left = groups['near'][..., 2 * tg.NUM_BLOCKS : 3 * tg.NUM_BLOCKS]
    assert (left.argmax(-1) == blocks[..., tg.PLAYER_INDEX - 1]).all()
    assert groups['facing'].sum(-1).min() == 1


def test_episode_split_is_disjoint():
    tr, te = fa.episode_split(40, test=0.25, seed=1)
    assert len(tr) == 30 and len(te) == 10
    assert not set(tr) & set(te)


def test_attribution_finds_the_group_that_makes_z():
    cols, _ = fake_columns(e=60, length=16)
    groups = fa.build_groups(cols)
    rng = np.random.default_rng(1)
    slow = groups['slow'][..., :4]
    z = slow @ rng.normal(size=(4, 8)) + 0.1 * rng.normal(size=slow.shape[:3] + (8,))
    out = fa.attribution(z, groups, fa.episode_split(60), 'ridge')
    assert out['all'] > 0.9
    assert out['only']['slow'] > 0.9
    assert out['unique']['slow'] > 0.8
    assert max(out['unique'][g] for g in ('medium', 'facing', 'near')) < 0.05
    assert out['residual']['variance_share'] < 0.1


def test_decodability_reads_what_z_holds():
    cols, _ = fake_columns(e=60, length=16)
    targets = fa.scalar_targets(cols)
    rng = np.random.default_rng(2)
    z = np.stack(
        [targets['vital:food'], rng.normal(size=targets['vital:food'].shape)],
        -1,
    )
    out = fa.decodability(z, targets, fa.episode_split(60))
    assert out['vital:food'] > 0.95
    assert out['vital:drink'] < 0.1


def test_cost_is_what_the_action_does_not_explain():
    rng = np.random.default_rng(3)
    shape = (20, 2, 64)
    action = rng.integers(0, 4, shape + (1,)).astype(np.float32)
    steps = np.array([0.0, 1.0, -1.0, 2.0])
    # a factor that moves by a fixed amount per action: nothing unexplained
    # (action t belongs to frame t and moves the factor to frame t + 1)
    moves = steps[action[..., 0].astype(int)]
    f = np.concatenate([np.zeros(shape[:2] + (1,)), np.cumsum(moves, -1)], -1)
    f = f[..., :-1]
    # a random walk: the action tells nothing
    walk = np.cumsum(rng.choice([-1.0, 1.0], shape), -1)
    out = fa.costs({'driven': f, 'walk': walk}, action)
    assert out['driven']['cost'] < 1e-6
    assert out['walk']['cost'] > 0.9 * out['walk']['change_variance'] / walk.var()
    assert out['driven']['change_rate'] > 0.5


class CopyJepa:
    """A stand-in whose predictor copies the last context latent."""

    def __init__(self):
        self.action_encoder = torch.nn.Identity()
        self.action_embed = torch.nn.Identity()
        self.predictor = torch.nn.Identity()

    def predict(self, emb, act_emb):
        return emb


def test_one_step_lines_up_the_transitions_and_drops_missing_actions():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(3, 2, 10, 5)).astype(np.float32)
    action = rng.integers(0, 3, (3, 2, 10, 1)).astype(np.float32)
    action[0, 0, 6] = np.nan  # no action at frame 6: that transition is out
    pred, target, last = pe.one_step(
        CopyJepa(), latent, action, 3, history=3
    )
    n = 3 * 2 * (10 - 3) - 1
    assert pred.shape == target.shape == last.shape == (n, 5)
    assert np.allclose(pred, last)  # the copy predictor returns the last
    # the first transition of window (0, 0) predicts frame 3 from frames 0-2
    assert np.allclose(last[0], latent[0, 0, 2])
    assert np.allclose(target[0], latent[0, 0, 3])


def test_per_axis_error_of_a_copy_equals_the_copy_error():
    rng = np.random.default_rng(1)
    latent = np.cumsum(rng.normal(size=(6, 2, 40, 4)), 2).astype(np.float32)
    action = np.zeros((6, 2, 40, 1), np.float32)
    pred, target, last = pe.one_step(
        CopyJepa(), latent, action, 3, history=3
    )
    out = pe.per_axis_error(latent, pred, target, last)
    assert np.allclose(out['error'], out['copy'])
    assert out['pred_over_copy'] == pytest.approx(1.0)
    assert (np.diff(out['variance']) <= 1e-9).all()  # largest axis first
    # the error splits exactly over the axes: the total is the plain mean
    # squared error of the latent
    plain = ((pred - target) ** 2).sum(-1).mean()
    assert out['error'].sum() == pytest.approx(plain, rel=1e-6)


def scale(episode, window, rank=100.0, null=0.12):
    rest = 1 - episode - window
    return {
        'shares': {
            'episode': episode,
            'window': window,
            'clip_means': rest,
            'clip': 0.0,
        },
        'null_episode': null,
        'erank': rank,
        'dim': 192,
        'within_episode_rank': rank,
        'within_window_rank': rank,
        'lag': {8: 0.9},
    }


def fake_numbers(episode=0.2):
    rows = lambda w: [  # noqa: E731
        {'kind': k, 'episodes': 8, 'raw': (v, 0.0), 'centered': (v, 0.0)}
        for k, v in (('iid', 1.0), ('window', w))
    ]
    dec = {'ridge': {'mob:cow': 0.6, 'daylight': 0.9}}
    last = {'ridge': {'mob:cow': 0.02, 'daylight': 0.95}}
    return {
        'names': ['untrained', 'epoch1', 'epoch2'],
        'scales': {
            'latent/untrained': scale(0.12, 0.4, rank=5),
            'latent/epoch1': scale(episode, 0.4, rank=60),
            'latent/epoch2': scale(episode, 0.4, rank=100),
            'cls/epoch2': scale(episode, 0.4),
        },
        'batches': {'epoch2': rows(8.0), 'state': rows(2.0)},
        'decodability': {
            'latent/untrained': dec,
            'latent/epoch2': last,
            'cls/epoch2': last,
        },
        'costs': {
            'mob:cow': {'cost': 0.3, 'change_rate': 0.09},
            'daylight': {'cost': 0.0, 'change_rate': 1.0},
        },
        'attribution': {},
    }


def test_reading_follows_the_numbers():
    text = ' '.join(rp.reading(fake_numbers(episode=0.2)))
    assert 'not what the variance is made of' in text  # 0.2 - 0.12 < 0.1
    assert 'slower than the whole state' in text  # 8x against 2x
    assert 'mob:cow' in text and 'expensive' in text
    assert 'daylight' not in text.split('lost')[1]
    text = ' '.join(rp.reading(fake_numbers(episode=0.3)))
    assert 'Episode identity carries variance' in text


def test_timescale_summary_on_planted_data():
    x = planted(episode=4.0, window=2.0, clip_means=1.0, clip=0.5)
    out = rp.timescale(x)
    assert out['dim'] == D
    assert out['shares']['episode'] > out['null_episode']
    assert 0 < out['within_episode_rank'] <= D
    assert 0.0 < out['lag'][8] < 1.0
