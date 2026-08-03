"""RED-first tests for the v2.1.0 1/|g0| objective rescale at solve start.

WHAT IT IS. Divide the objective and its gradient by the Euclidean norm of the
gradient at the START point, once, before the first optimizer step. That is a
PURE REPARAMETERIZATION by a positive constant: it cannot move a minimizer and
it cannot change a descent direction. What it does change is every
absolute-scale-sensitive quantity inside the optimizer, and that is the whole
point.

WHY IT WAS ADOPTED. The upper-rail stall class: when the objective's magnitude
sits far below 1, `scipy.optimize.minimize(method="L-BFGS-B")` compares the
objective decrease against `max(|f_k|, |f_k+1|, 1)`, so its relative tolerance
becomes an ABSOLUTE one and the solve terminates at or near the start point,
which for a full-depth start is the upper rail of the box. Confirmed
independently in the two-dimensional gear8 solve and in the three-dimensional
port lane's Phase C.

The tests below pin, in order: the factor, the guards, the invariance
(reparameterization cannot move the minimizer), the smooth-case regression at a
stated tolerance, and the rail-stall case actually un-stalling.

Acronyms: L-BFGS-B = limited-memory Broyden-Fletcher-Goldfarb-Shanno with box
constraints. g0 = the gradient at the solve's start point.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from adjoint2d import objective_scale as osc


# ---------------------------------------------------------------------------
# 1. the factor and its guards
# ---------------------------------------------------------------------------

def test_the_factor_is_exactly_one_over_the_start_gradient_norm():
    g0 = np.array([3.0, 4.0])       # norm exactly 5
    rs = osc.ObjectiveRescale.from_start_gradient(g0)
    assert rs.g0_norm == pytest.approx(5.0)
    assert rs.factor == pytest.approx(1.0 / 5.0)
    assert rs.applied is True


def test_disabled_is_the_identity_and_says_so():
    rs = osc.ObjectiveRescale.from_start_gradient(np.array([3.0, 4.0]),
                                                  enabled=False)
    assert rs.factor == 1.0
    assert rs.applied is False
    assert "disabled" in rs.reason


def test_a_zero_start_gradient_falls_back_to_the_identity_loudly():
    rs = osc.ObjectiveRescale.from_start_gradient(np.zeros(5))
    assert rs.factor == 1.0
    assert rs.applied is False
    assert "zero" in rs.reason or "non-finite" in rs.reason


def test_a_non_finite_start_gradient_falls_back_to_the_identity():
    rs = osc.ObjectiveRescale.from_start_gradient(np.array([1.0, np.nan]))
    assert rs.factor == 1.0
    assert rs.applied is False
    assert "non-finite" in rs.reason


def test_the_record_is_immutable_and_json_ready():
    rs = osc.ObjectiveRescale.from_start_gradient(np.array([2.0]))
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        rs.factor = 2.0  # type: ignore[misc]
    d = rs.as_dict()
    assert set(d) == {"applied", "factor", "g0_norm", "reason"}
    assert isinstance(d["factor"], float)


# ---------------------------------------------------------------------------
# 2. it is a reparameterization, not a change of problem
# ---------------------------------------------------------------------------

def test_apply_scales_the_objective_and_the_gradient_by_the_same_constant():
    rs = osc.ObjectiveRescale.from_start_gradient(np.array([0.0, 10.0]))
    J, g = rs.apply(4.0, np.array([2.0, -6.0]))
    assert J == pytest.approx(4.0 * 0.1)
    assert g == pytest.approx(np.array([0.2, -0.6]))


def test_the_descent_direction_is_preserved_to_roundoff():
    """A positive scalar cannot rotate the gradient.

    Stated honestly: the multiply and the renormalization are floating point,
    so the claim is preserved TO ROUNDOFF (1e-15 per component), not bitwise.
    The signs are exact, and those are what a descent method acts on.
    """
    g = np.array([1.5, -2.5, 0.25])
    rs = osc.ObjectiveRescale.from_start_gradient(g)
    _J, gs = rs.apply(1.0, g)
    assert np.allclose(gs / np.linalg.norm(gs), g / np.linalg.norm(g),
                       rtol=0, atol=1e-15)
    assert np.array_equal(np.sign(gs), np.sign(g))


def test_the_ordering_of_two_candidate_designs_is_preserved_exactly():
    """The solve picks the best evaluated design by raw J; scaling cannot flip it."""
    rs = osc.ObjectiveRescale.from_start_gradient(np.array([4.0]))
    a, _ = rs.apply(3.0, np.array([1.0]))
    b, _ = rs.apply(3.0000001, np.array([1.0]))
    assert a < b


# ---------------------------------------------------------------------------
# 3. regression: same converged map on a smooth case
# ---------------------------------------------------------------------------

def _quadratic(c: float, x_star: np.ndarray):
    """A smooth, box-constrained, strictly convex test objective of scale c."""
    def f(x):
        d = np.asarray(x, dtype=float) - x_star
        return c * 0.5 * float(d @ d), c * d
    return f


def _solve(f, x0, rescale: bool):
    rs = osc.ObjectiveRescale.from_start_gradient(f(x0)[1], enabled=rescale)

    def fun(x):
        return rs.apply(*f(x))

    r = minimize(fun, x0, jac=True, method="L-BFGS-B",
                 bounds=[(0.0, 1.0)] * len(x0))
    return r, rs


def test_rescaled_and_unrescaled_reach_the_same_converged_map_smooth_case():
    """STATED TOLERANCE: 1e-9 max absolute difference per design variable.

    A well-scaled smooth case is the control: the rescale must be a no-op on
    the ANSWER even though it changes the iterate path.
    """
    x_star = np.full(8, 0.3)
    f = _quadratic(1.0, x_star)
    x0 = np.ones(8)
    r_raw, _ = _solve(f, x0, rescale=False)
    r_res, rs = _solve(f, x0, rescale=True)
    assert rs.applied is True
    assert float(np.max(np.abs(r_res.x - r_raw.x))) <= 1e-9
    assert float(np.max(np.abs(r_res.x - x_star))) <= 1e-9


# ---------------------------------------------------------------------------
# 4. the rail-stall case demonstrably un-stalls
# ---------------------------------------------------------------------------

def test_the_upper_rail_stall_is_real_without_the_rescale():
    """The failure this change exists for, reproduced before it is fixed.

    Objective scale 1e-10, start at the upper rail of the box. scipy's
    convergence test degenerates to an absolute one, so the solve stops with
    the design still pinned at the rail.
    """
    x_star = np.full(8, 0.3)
    f = _quadratic(1e-10, x_star)
    x0 = np.ones(8)
    r_raw, _ = _solve(f, x0, rescale=False)
    assert r_raw.nit == 0
    assert float(np.max(np.abs(r_raw.x - x0))) == 0.0        # still on the rail
    assert float(np.max(np.abs(r_raw.x - x_star))) > 0.5


def test_the_rescale_un_stalls_that_same_case_to_the_true_minimizer():
    x_star = np.full(8, 0.3)
    f = _quadratic(1e-10, x_star)
    x0 = np.ones(8)
    r_res, rs = _solve(f, x0, rescale=True)
    assert rs.applied is True
    assert r_res.nit >= 1
    assert float(np.max(np.abs(r_res.x - x_star))) <= 1e-9


def test_the_un_stalled_answer_equals_the_well_scaled_answer():
    """Same minimizer, two objective magnitudes ten decades apart."""
    x_star = np.full(8, 0.3)
    x0 = np.ones(8)
    r_a, _ = _solve(_quadratic(1.0, x_star), x0, rescale=True)
    r_b, _ = _solve(_quadratic(1e-10, x_star), x0, rescale=True)
    assert float(np.max(np.abs(r_a.x - r_b.x))) <= 1e-9
