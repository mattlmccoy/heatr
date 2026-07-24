#!/usr/bin/env python3
"""Unit tests for the per-node adaptive-gain absolute-target tuning law.

Implements Allison's Tune_Conductivity.m law as a pure function:

    sigma_i(k+1) = clip( sigma_i(k) + K1_i * (Tt - T_i) / maxDiff,  0, sigma_max )

with K1_i halved whenever node i's error (Tt - T_i) changes sign between
iterations, and convergence when maxDiff = max_i |Tt - T_i| <= eps.

TWO-SIDED actuation: sigma may rise ABOVE the uniform baseline (0.04) up to
sigma_max, or fall to 0.  Run with:

    ./.venv312/bin/python -m pytest test_pernode_tuning.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from pernode_tuning import pernode_sigma_update


# ---------------------------------------------------------------------------
# 1. Proportional move toward target
# ---------------------------------------------------------------------------

def test_proportional_move_extreme_nodes():
    """The worst-error nodes (|err| = maxDiff) move by exactly +/- K1."""
    sigma = np.array([0.04, 0.04])
    T = np.array([100.0, 300.0])          # one cold, one hot
    Tt = 200.0
    K1 = np.array([0.01, 0.01])
    sign_prev = np.array([0.0, 0.0])      # no history -> no halving
    res = pernode_sigma_update(sigma, T, Tt, K1, sign_prev, sigma_max=0.06)
    # err = [+100, -100], maxDiff = 100 -> step = +/-K1
    assert res.max_diff == pytest.approx(100.0)
    assert res.sigma[0] == pytest.approx(0.05)   # cold node gains sigma
    assert res.sigma[1] == pytest.approx(0.03)   # hot node loses sigma


def test_proportional_move_scales_with_error():
    """A node with half the max error moves by half of K1."""
    sigma = np.array([0.04, 0.04])
    T = np.array([150.0, 300.0])          # err = [+50, -100], maxDiff = 100
    Tt = 200.0
    K1 = np.array([0.01, 0.01])
    res = pernode_sigma_update(sigma, T, Tt, K1, np.zeros(2), sigma_max=0.06)
    assert res.sigma[0] == pytest.approx(0.045)  # 0.04 + 0.01*(50/100)
    assert res.sigma[1] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# 2. Gain halving on sign flip
# ---------------------------------------------------------------------------

def test_gain_halved_on_sign_flip():
    """A node whose error sign flips vs the previous iter halves its K1."""
    sigma = np.array([0.04])
    T = np.array([300.0])                 # err = -100 (sign -1)
    Tt = 200.0
    K1 = np.array([0.01])
    sign_prev = np.array([1.0])           # was positive -> now negative: flip
    res = pernode_sigma_update(sigma, T, Tt, K1, sign_prev, sigma_max=0.06)
    assert res.K1[0] == pytest.approx(0.005)          # halved
    assert res.sigma[0] == pytest.approx(0.035)       # 0.04 + 0.005*(-1)
    assert res.sign[0] == pytest.approx(-1.0)


def test_gain_unchanged_without_flip():
    """Same-sign error keeps K1 constant."""
    sigma = np.array([0.04])
    T = np.array([300.0])                 # err = -100 (sign -1)
    Tt = 200.0
    K1 = np.array([0.01])
    sign_prev = np.array([-1.0])          # already negative: no flip
    res = pernode_sigma_update(sigma, T, Tt, K1, sign_prev, sigma_max=0.06)
    assert res.K1[0] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 3. Clamping both ends
# ---------------------------------------------------------------------------

def test_clamp_high_end():
    """A node driven above sigma_max is clamped to sigma_max."""
    sigma = np.array([0.059])
    T = np.array([0.0])                   # err = +200, huge upward push
    res = pernode_sigma_update(sigma, T, 200.0, np.array([0.01]),
                               np.zeros(1), sigma_max=0.06)
    assert res.sigma[0] == pytest.approx(0.06)


def test_clamp_low_end():
    """A node driven below zero is clamped to sigma_min (0)."""
    sigma = np.array([0.005])
    T = np.array([400.0])                 # err = -200, huge downward push
    res = pernode_sigma_update(sigma, T, 200.0, np.array([0.01]),
                               np.zeros(1), sigma_max=0.06)
    assert res.sigma[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Convergence flag
# ---------------------------------------------------------------------------

def test_converged_flag_true_within_eps():
    T = np.array([197.0, 203.0, 200.0])   # maxDiff = 3 <= 5
    res = pernode_sigma_update(np.full(3, 0.04), T, 200.0,
                               np.full(3, 0.01), np.zeros(3),
                               sigma_max=0.06, eps=5.0)
    assert res.max_diff == pytest.approx(3.0)
    assert res.converged is True


def test_converged_flag_false_outside_eps():
    T = np.array([190.0, 203.0, 200.0])   # maxDiff = 10 > 5
    res = pernode_sigma_update(np.full(3, 0.04), T, 200.0,
                               np.full(3, 0.01), np.zeros(3),
                               sigma_max=0.06, eps=5.0)
    assert res.max_diff == pytest.approx(10.0)
    assert res.converged is False


# ---------------------------------------------------------------------------
# 5. Scalar K1 broadcast convenience
# ---------------------------------------------------------------------------

def test_scalar_K1_broadcasts():
    res = pernode_sigma_update(np.full(3, 0.04), np.array([100.0, 200.0, 300.0]),
                               200.0, 0.01, np.zeros(3), sigma_max=0.06)
    assert res.K1.shape == (3,)
    assert np.all(res.K1 == 0.01)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
