"""timescales on signals with known answers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import timescales as ts  # noqa: E402

W = 128


def sine(period, n=2048, phase=0.0):
    return np.sin(2 * np.pi * np.arange(n) / period + phase)


def pooled(signals, window=W):
    """Pooled spectrum of several episodes (one 1-D signal each)."""
    power = np.zeros(window // 2 + 1)
    segments = 0
    for x in signals:
        p, s = ts.episode_power(x, window)
        power += p
        segments += s
    return power, segments


def test_the_centroid_period_of_a_sine_is_its_period():
    for period in (4, 16, 32):  # periods that fit the window exactly
        psd, segments = pooled([sine(period)])
        assert segments > 0
        s = ts.spectrum_summary(psd, W)
        assert s['period'] == pytest.approx(period, rel=0.05)
        assert not s['limited']


def test_two_entities_with_periods_4_and_32_have_a_gap_of_8():
    fast = ts.spectrum_summary(pooled([sine(4)])[0], W)
    slow = ts.spectrum_summary(pooled([sine(32)])[0], W)
    gap = ts.frequency_gap({'fast': fast, 'slow': slow})
    assert gap['ratio'] == pytest.approx(8, rel=0.1)
    assert gap['slowest'] == 'slow' and gap['fastest'] == 'fast'
    assert not gap['lower_bound']


def test_amplitude_and_offset_do_not_change_the_centroid():
    a = ts.spectrum_summary(pooled([sine(16)])[0], W)
    b = ts.spectrum_summary(pooled([1000 * sine(16) + 5])[0], W)
    assert a['period'] == pytest.approx(b['period'], rel=1e-3)


def test_a_staircase_is_limited_by_the_window_and_the_gap_is_a_lower_bound():
    n = 2048
    stairs = np.cumsum(np.random.default_rng(0).random(n) < 0.004)  # ~8 steps
    s = ts.spectrum_summary(pooled([stairs.astype(float)])[0], W)
    assert s['limited'] and s['low_share'] > 0.5
    fast = ts.spectrum_summary(pooled([sine(4)])[0], W)
    gap = ts.frequency_gap({'stairs': s, 'fast': fast})
    assert gap['lower_bound'] and gap['slowest'] == 'stairs'


def test_bands_hold_the_requested_power_and_are_ordered():
    s = ts.spectrum_summary(pooled([sine(8) + sine(32)])[0], W)
    band = s['band']
    assert band['fast_edge'] <= band['median'] <= band['slow_edge']
    # 90% of the power is reached at the period-8 peak
    assert 7 < band['fast_edge'] < 10
    assert band['slow_edge'] == pytest.approx(32, rel=0.1)


def test_short_and_constant_episodes_have_no_segment():
    assert ts.episode_power(sine(8, n=W - 1), W)[1] == 0
    assert ts.episode_power(np.ones(500), W)[1] == 0
    p, s = ts.episode_power(np.stack([sine(8, 500), np.ones(500)], 1), W)
    # the constant dimension is dropped, not counted as zero power
    assert s > 0 and p.sum() > 0


def test_dimensions_are_averaged_not_summed():
    one, _ = ts.episode_power(sine(8, 500), W)
    two, _ = ts.episode_power(np.stack([sine(8, 500)] * 2, 1), W)
    assert two.sum() == pytest.approx(one.sum(), rel=1e-4)


def test_the_bootstrap_interval_contains_the_true_gap():
    rng = np.random.default_rng(1)
    acc = {'fast': [], 'slow': []}
    for _ in range(30):
        phase = rng.uniform(0, 6.28)
        acc['fast'].append(ts.episode_power(sine(4, 600, phase), W)[0])
        acc['slow'].append(ts.episode_power(sine(32, 600, phase), W)[0])
    boot = ts.bootstrap_gap({k: np.stack(v) for k, v in acc.items()}, W)
    assert boot['p5'] <= boot['p50'] <= boot['p95']
    assert boot['p5'] < 8 < boot['p95'] * 1.2
    assert boot['p95'] - boot['p5'] < 2  # identical episodes: a tight interval


def test_periodic_events_have_no_spread_and_the_mode_is_the_period():
    masks = []
    for _ in range(3):
        m = np.zeros(200, bool)
        m[::21] = True
        masks.append(m)
    st = ts.inter_event_stats(masks)
    assert st['mode'] == 21 and st['mode_share'] == 1.0
    assert st['cv'] == pytest.approx(0)
    assert st['p10'] == st['p90'] == 21
    assert st['events'] == 3 * 10 and st['rate'] == pytest.approx(30 / 600)
    # one cut-off gap per episode: 3 of (3 * 9 complete + 3)
    assert st['censored_share'] == pytest.approx(3 / 30)


def test_irregular_events_have_a_large_cv():
    rng = np.random.default_rng(0)
    m = [rng.random(2000) < 0.02]
    regular = [np.arange(2000) % 50 == 0]
    assert ts.inter_event_stats(m)['cv'] > 0.7
    assert ts.inter_event_stats(regular)['cv'] < 0.01


def test_masks_of_changes_rises_and_drops():
    x = np.array([[0, 5], [1, 5], [1, 4], [1, 4], [2, 6]])
    assert ts.change_mask(x).tolist() == [True, True, False, True]
    assert ts.rise_mask(x).tolist() == [True, False, False, True]
    assert ts.drop_mask(x).tolist() == [False, True, False, False]
    union = ts.union_masks([[ts.rise_mask(x)], [ts.drop_mask(x)]])
    assert union[0].tolist() == [True, True, False, True]


def test_clusters_split_where_neighbours_are_two_times_apart():
    values = {'a': 2.0, 'b': 3.0, 'c': 20.0, 'd': 21.0, 'e': 100.0}
    assert ts.cluster_scales(values) == [['a', 'b'], ['c', 'd'], ['e']]
    same = {'a': 5.0, 'b': 6.0, 'c': 7.0}
    assert ts.cluster_scales(same) == [['a', 'b', 'c']]
    assert ts.cluster_scales({'a': 1, 'b': 1, 'c': 8}) == [['a', 'b'], ['c']]
    assert ts.cluster_scales({}) == []


def test_boundaries_give_the_interval_between_clusters():
    values = {'a': 2.0, 'b': 3.0, 'c': 20.0, 'd': 90.0}
    out = ts.boundaries(ts.cluster_scales(values), values)
    assert len(out) == 2
    assert (out[0]['from'], out[0]['to']) == (3.0, 20.0)
    assert out[0]['ratio'] == pytest.approx(20 / 3)
    assert out[1]['fast'] == ['c'] and out[1]['slow'] == ['d']
