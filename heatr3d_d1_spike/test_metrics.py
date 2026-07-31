"""Unit gates for the pure comparison math used by Tasks 2 and 3.

Runs in EITHER environment (pure numpy, no dolfinx):
    ./.venv312/bin/python -m pytest heatr3d_d1_spike/test_metrics.py
"""
from __future__ import annotations

import numpy as np
import pytest

import metrics as m


# --------------------------------------------------------------------------- #
# normalized pattern + relative L2
# --------------------------------------------------------------------------- #
def test_unit_mean_removes_absolute_scale():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    out = m.unit_mean(q)
    assert out.mean() == pytest.approx(1.0)
    # scale invariance: the pattern of 7*q is identical
    assert np.allclose(out, m.unit_mean(7.0 * q))


def test_unit_mean_rejects_zero_mean():
    with pytest.raises(ValueError):
        m.unit_mean(np.zeros(5))


def test_rel_l2_is_zero_for_identical_and_scaled_patterns():
    a = np.array([1.0, 3.0, 5.0, 11.0])
    assert m.rel_l2_pattern(a, a) == pytest.approx(0.0)
    assert m.rel_l2_pattern(3.0 * a, a) == pytest.approx(0.0, abs=1e-14)


def test_rel_l2_known_value():
    # b pattern = [0.5, 1.5]; a pattern = [1.5, 0.5]; diff = [1, -1]
    a = np.array([3.0, 1.0])
    b = np.array([1.0, 3.0])
    expect = np.sqrt(2.0) / np.sqrt(0.5 ** 2 + 1.5 ** 2)
    assert m.rel_l2_pattern(a, b) == pytest.approx(expect)


# --------------------------------------------------------------------------- #
# corner-edge distance (Task 3)
# --------------------------------------------------------------------------- #
def test_corner_edge_distance_zero_on_the_edge():
    half = 0.010
    x = np.array([half, -half, half, -half])
    y = np.array([half, half, -half, -half])
    assert np.allclose(m.corner_edge_distance(x, y, half), 0.0)


def test_corner_edge_distance_center_and_face():
    half = 0.010
    # centre of the square: distance sqrt(2)*half to the nearest corner
    d = m.corner_edge_distance(np.array([0.0]), np.array([0.0]), half)
    assert d[0] == pytest.approx(np.sqrt(2.0) * half)
    # mid-face point (0, half): distance = half (along x to the corner)
    d = m.corner_edge_distance(np.array([0.0]), np.array([half]), half)
    assert d[0] == pytest.approx(half)


# --------------------------------------------------------------------------- #
# power-law fit (Task 3 gate)
# --------------------------------------------------------------------------- #
def test_power_law_fit_recovers_exact_law():
    h = np.array([1.0, 0.5, 0.25, 0.125])
    q = 3.0 * h ** (-0.4)
    fit = m.power_law_fit(h, q)
    assert fit["exponent"] == pytest.approx(-0.4)
    assert fit["prefactor"] == pytest.approx(3.0)
    assert fit["r2"] == pytest.approx(1.0)


def test_power_law_fit_r2_drops_for_noisy_data():
    h = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    q = np.array([1.0, 4.0, 1.5, 9.0, 2.0])
    fit = m.power_law_fit(h, q)
    assert fit["r2"] < 0.9


def test_power_law_fit_needs_three_points():
    with pytest.raises(ValueError):
        m.power_law_fit(np.array([1.0, 0.5]), np.array([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# mask-aware gradient (Task 2 diagnostic: is the heatr3d surface Q spike an
# artifact of differencing ACROSS the material interface?)
# --------------------------------------------------------------------------- #
def _grid(nx, ny, h):
    x = (np.arange(nx) + 0.5) * h
    y = (np.arange(ny) + 0.5) * h
    return np.meshgrid(x, y, indexing="ij")


def test_masked_grad_is_exact_for_a_linear_field_including_boundary_cells():
    h = 0.5
    X, Y = _grid(9, 7, h)
    V = 3.0 * X - 2.0 * Y
    mask = np.zeros(V.shape, bool)
    mask[2:7, 1:6] = True
    # poison everything outside the mask: a mask-aware stencil must ignore it
    V = np.where(mask, V, 1e9)
    Ex, Ey = m.masked_grad_2d(V, mask, h)
    assert np.allclose(Ex[mask], -3.0)
    assert np.allclose(Ey[mask], +2.0)


def test_masked_grad_naive_gradient_would_fail_the_same_case():
    """Guards the diagnostic's premise: np.gradient DOES get poisoned."""
    h = 0.5
    X, Y = _grid(9, 7, h)
    mask = np.zeros(X.shape, bool)
    mask[2:7, 1:6] = True
    V = np.where(mask, 3.0 * X - 2.0 * Y, 1e9)
    gx, gy = np.gradient(V, h, edge_order=1)
    assert np.abs(gx[mask] + 3.0).max() > 1.0


def test_masked_grad_isolated_cell_has_zero_gradient():
    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True
    V = np.zeros((5, 5))
    V[2, 2] = 7.0
    Ex, Ey = m.masked_grad_2d(V, mask, 1.0)
    assert Ex[2, 2] == 0.0 and Ey[2, 2] == 0.0


def test_weighted_percentile_equal_weights_is_the_order_statistic():
    v = np.arange(1.0, 101.0)
    w = np.ones_like(v)
    assert m.weighted_percentile(v, w, 99.0) == pytest.approx(99.0)
    assert m.weighted_percentile(v, w, 100.0) == pytest.approx(100.0)
    assert m.weighted_percentile(v, w, 50.0) == pytest.approx(50.0)


def test_weighted_percentile_respects_weights():
    v = np.array([1.0, 2.0])
    w = np.array([99.0, 1.0])
    assert m.weighted_percentile(v, w, 99.0) == pytest.approx(1.0)
    assert m.weighted_percentile(v, w, 99.5) == pytest.approx(2.0)


def test_weighted_percentile_is_order_independent():
    rng = np.random.default_rng(0)
    v = rng.random(50)
    w = rng.random(50) + 0.1
    k = rng.permutation(50)
    assert m.weighted_percentile(v, w, 90.0) == pytest.approx(
        m.weighted_percentile(v[k], w[k], 90.0))


def test_masked_grad_handles_complex_fields():
    h = 1.0
    X, Y = _grid(6, 6, h)
    mask = np.ones(X.shape, bool)
    V = (1.0 + 2.0j) * X
    Ex, Ey = m.masked_grad_2d(V, mask, h)
    assert np.allclose(Ex, -(1.0 + 2.0j))
    assert np.allclose(Ey, 0.0)
