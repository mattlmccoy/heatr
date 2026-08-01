"""Red-first tests for the chi-parameterized melt-region objective.

The only difference from `shape_objective` is that the target indicator is an
argument instead of the solve-grid binary raster. The tests that matter are the
flag-off identity (passing the raster back in must reproduce the old numbers to
the last bit) and the seed, because the seed is what the adjoint integrates.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import shape_objective as so
from adjoint2d import topopt_objective as to


class _Pins:
    t_pc_c = 180.0
    dt_pc_c = 10.0
    ambient_c = 25.0


class _Case:
    def __init__(self, pm):
        self.part_mask = pm
        self.pins = _Pins()


def _case(n=12, seed=2):
    pm = np.zeros((n, n), dtype=bool)
    pm[3:9, 4:10] = True
    rng = np.random.default_rng(seed)
    T = 150.0 + 60.0 * rng.random((n, n))
    return _Case(pm), T


def test_raster_chi_reproduces_shape_objective_bitwise():
    case, T = _case()
    J_new, g_new = to.J_and_seed(T, case, case.part_mask.astype(float))
    J_old, g_old = so.shape_J_and_seed(T, case)
    assert J_new == J_old
    assert np.array_equal(g_new, g_old)


def test_objective_is_zero_when_the_melt_fraction_equals_the_target():
    case, _ = _case()
    chi = np.clip(case.part_mask.astype(float) * 0.7 + 0.1, 0.0, 1.0)
    T = case.pins.t_pc_c + case.pins.dt_pc_c * (chi - 0.5)
    J, _g = to.J_and_seed(T, case, chi)
    assert J == pytest.approx(0.0, abs=1e-20)


def test_seed_is_the_derivative_of_J_with_respect_to_temperature():
    case, T = _case(seed=9)
    chi = np.clip(0.3 + 0.5 * np.linspace(0, 1, T.size).reshape(T.shape), 0.0, 1.0)
    _J, g = to.J_and_seed(T, case, chi)
    h = 1e-4
    for idx in ((5, 6), (2, 2), (8, 9)):
        Tp = T.copy(); Tp[idx] += h
        Tm = T.copy(); Tm[idx] -= h
        fd = (to.J_and_seed(Tp, case, chi)[0] - to.J_and_seed(Tm, case, chi)[0]) / (2 * h)
        assert abs(fd - g[idx]) <= 1e-6 * max(abs(g[idx]), 1.0)


def test_area_weighted_iou_reduces_to_the_raster_iou_for_a_binary_target():
    case, T = _case(seed=4)
    chi = case.part_mask.astype(float)
    melted = to.phi_of(T, case) >= 0.5
    inter = float(np.sum(melted & case.part_mask))
    union = float(np.sum(melted | case.part_mask))
    assert to.area_iou(melted.astype(float), chi) == pytest.approx(inter / union)


def test_area_weighted_iou_is_between_zero_and_one():
    rng = np.random.default_rng(1)
    a = rng.random((9, 9))
    b = rng.random((9, 9))
    v = to.area_iou(a, b)
    assert 0.0 <= v <= 1.0


def test_chi_shape_mismatch_raises():
    case, T = _case()
    with pytest.raises(ValueError):
        to.J_and_seed(T, case, np.ones((3, 3)))
