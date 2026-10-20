"""The spectral estimators of dataset_section.spectral on signals with known answers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from dataset_section import spectral as ts  # noqa: E402

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
