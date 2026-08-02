"""Phase C Task 1: the shape objective, both weightings.

Pure numpy closed-form checks on synthetic phi/chi fields -- runs in either
environment. The asymmetry is the point, so it is tested directly rather than
inferred from a solve.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import objective as obj


def test_symmetric_control_is_the_phase_b_functional():
    vol = np.array([1.0, 1.0, 2.0])
    phi = np.array([1.0, 0.0, 0.5])
    chi = np.array([1.0, 1.0, 0.0])
    # (0)^2*1 + (-1)^2*1 + (0.5)^2*2 = 0 + 1 + 0.5
    assert obj.j_symmetric(phi, chi, vol) == pytest.approx(1.5)
    assert obj.j_symmetric(chi, chi, vol) == pytest.approx(0.0)


def test_asymmetric_hinge_gives_zero_for_in_bounds_density_above_the_floor():
    """THE point of the refinement: in-bounds density above the floor is FREE.
    The symmetric control penalizes the same state, which is what makes it the
    wrong objective for Matt's stated goal."""
    vol = np.array([1.0])
    chi = np.array([1.0])
    phi = np.array([0.90])                       # 90 % dense, inside the band
    assert obj.j_asymmetric(phi, chi, vol) == pytest.approx(0.0)
    assert obj.j_symmetric(phi, chi, vol) == pytest.approx(0.01)


def test_asymmetric_closed_form_on_both_error_kinds():
    vol = np.array([1.0, 1.0])
    phi = np.array([1.0, 0.0])
    chi = np.array([0.0, 1.0])
    # cell 0: fully melted where there is no part -> e_out = 1
    # cell 1: no melt where the part is        -> e_in = 0.85 - 0 = 0.85
    w = obj.W_OUT_OVER_W_IN
    assert obj.j_asymmetric(phi, chi, vol) == pytest.approx(w * 1.0 + 0.85 ** 2)


def test_out_of_bounds_is_penalized_by_exactly_the_registered_ratio():
    """An equal-magnitude error must cost W_out/W_in times more outside the
    part than inside it. This is the whole content of 'hard versus soft'."""
    d = 0.1
    vol = np.array([1.0])
    chi_out, phi_out = np.array([0.0]), np.array([d])
    chi_in, phi_in = np.array([1.0]), np.array([obj.PHI_FLOOR - d])
    j_out = obj.j_asymmetric(phi_out, chi_out, vol)
    j_in = obj.j_asymmetric(phi_in, chi_in, vol)
    assert j_out / j_in == pytest.approx(obj.W_OUT_OVER_W_IN)


def test_registered_constants_come_from_the_preregistration():
    pr = obj.preregistration()["objective"]["asymmetric"]
    assert obj.PHI_FLOOR == pr["phi_floor"]
    assert obj.W_OUT_OVER_W_IN == pr["w_out_over_w_in"]
    assert obj.W_SENSITIVITY == pr["sensitivity_arm_w_out_over_w_in"]


def test_sensitivity_arm_changes_only_the_ratio():
    vol, phi, chi = np.array([1.0, 1.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    j10 = obj.j_asymmetric(phi, chi, vol, w_ratio=10.0)
    j3 = obj.j_asymmetric(phi, chi, vol, w_ratio=3.0)
    assert j10 - j3 == pytest.approx(7.0 * 1.0)


@pytest.mark.parametrize("fn,dfn", [(obj.j_symmetric, obj.dj_symmetric_dphi),
                                    (obj.j_asymmetric, obj.dj_asymmetric_dphi)])
def test_dj_dphi_matches_central_differences(fn, dfn):
    rng = np.random.default_rng(3)
    n = 40
    vol = rng.random(n) + 0.5
    chi = rng.random(n)
    phi = np.clip(chi + 0.3 * rng.standard_normal(n), 0.0, 1.0)
    g = dfn(phi, chi, vol)
    d = rng.standard_normal(n)
    d /= np.linalg.norm(d)
    h = 1e-6
    fd = (fn(phi + h * d, chi, vol) - fn(phi - h * d, chi, vol)) / (2 * h)
    assert abs(fd - float(np.dot(g, d))) <= 1e-6 * max(abs(fd), 1.0)


def test_asymmetric_gradient_is_zero_in_the_free_region():
    """Where the hinge is inactive and there is no spill, the objective is flat
    -- the optimizer must see no reason to push density beyond the floor."""
    vol = np.ones(3)
    chi = np.array([1.0, 1.0, 1.0])
    phi = np.array([0.90, 0.95, 1.0])
    assert np.allclose(obj.dj_asymmetric_dphi(phi, chi, vol), 0.0)
