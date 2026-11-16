"""The weight solver of analysis/mass_weights.py: a rare class is lifted to the threshold, within the cap and the effective sample size."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

import mass_weights as mw  # noqa: E402


def synthetic(frames=4000, seed=0):
    rng = np.random.default_rng(seed)
    S = np.ones(frames)
    C = np.zeros((frames, 3))
    C[:, 0] = 0.30 + 0.05 * rng.random(frames)        # a common class: always there, 30% of the variance
    rare = rng.random(frames) < 0.10                    # a rare class: in 10% of the frames, 0.2% of the variance overall
    C[rare, 1] = 0.02
    C[:, 2] = 1e-9                                      # a class that never matters (under min_mass): left alone
    return C, S, rare


def test_a_rare_class_reaches_the_threshold_within_the_limits():
    C, S, rare = synthetic()
    m0 = mw.mass(np.ones(len(S)), C, S)
    tau = 0.005
    assert m0[1] < tau / 2
    w, targets, _ = mw.solve_weights(C, S, tau, cap=8.0, ess=0.5)
    m1 = mw.mass(w, C, S)
    assert list(targets) == [1]
    assert 0.9 * tau <= m1[1] <= 1.6 * tau                # reaches the threshold, does not overshoot far
    assert abs(w.mean() - 1) < 1e-9 and w.min() >= 1 / 8 - 1e-9 and w.max() <= 8 + 1e-9
    assert mw.ess_of(w) >= 0.5
    assert w[rare].mean() > w[~rare].mean()               # the frames that carry the class are the ones upweighted


def test_an_unreachable_target_is_approached_within_the_effective_sample_size():
    C, S, _ = synthetic()
    m0 = mw.mass(np.ones(len(S)), C, S)[1]
    w, _, _ = mw.solve_weights(C, S, 0.01, cap=8.0, ess=0.5)  # 1% from a class in 10% of the frames needs an effective sample size under 0.5
    m1 = mw.mass(w, C, S)[1]
    assert m0 < m1 < 0.01 and mw.ess_of(w) >= 0.5 and w.max() <= 8 + 1e-9


def test_nothing_to_do_when_every_class_is_above_the_threshold():
    C, S, _ = synthetic()
    w, targets, _ = mw.solve_weights(C, S, 1e-4)
    assert len(targets) == 0 and np.allclose(w, 1)
