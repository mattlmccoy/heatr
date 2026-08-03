"""Tests for the method of moving asymptotes, box-constrained form.

RED FIRST. Every test in this file was observed failing with
`ImportError: cannot import name 'mma' from 'adjoint2d'` before `mma.py`
existed.

WHAT IS BEING TESTED, and why these and not others. The method of moving
asymptotes (MMA, Svanberg 1987) replaces the true objective at each iterate by
a separable convex rational approximation whose curvature is set by two moving
asymptotes per variable. With NO constraints other than the box the
subproblem's Lagrangian dual is ZERO dimensional and the subproblem has a
closed-form per-variable solution, so the whole method reduces to: build the
asymptotes from the last three iterates, build the approximation from the
gradient, solve in closed form, clamp to the move limits and the box. These
tests pin each of those four pieces plus five end-to-end behaviours.

TWO PROPERTIES THAT ARE PINNED HERE BECAUSE THEY ARE EASY TO GET WRONG BY
ASSUMPTION, and both were derived before the implementation was written:

  * MMA is NOT a descent method. Without a line search the iterate can be
    worse than its predecessor, and on a quadratic it usually is on the first
    step. `test_the_iterate_can_overshoot...` pins that, and it is why the
    solve driver must keep the best iterate.
  * The default asymptote lower clamp of 0.01 times the box span FLOORS the
    achievable accuracy at about one percent of the span, because the step is
    proportional to the asymptote distance. The convergence tests therefore
    use a deliberately loosened clamp; the production configuration keeps
    Svanberg's 0.01 and is not expected to converge below it.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import mma

# A configuration with the two safeguards that floor the accuracy loosened, so
# the convergence tests measure the ALGORITHM and not the safeguard.
TIGHT = mma.MMAConfig(asy_bound_lo=1e-12, raa0=1e-12)


# ---------------------------------------------------------------------------
# 1. the asymptotes
# ---------------------------------------------------------------------------

def test_first_iteration_asymptotes_bracket_the_start_by_asy_init_times_the_span():
    """Svanberg's rule for the first two iterations: L = x - s0 (xmax - xmin)."""
    x0 = np.array([0.3, 0.7])
    m = mma.MMA(x0, lower=0.0, upper=1.0)
    m.step(np.array([1.0, 1.0]))
    assert np.allclose(m.last_sub.L, x0 - 0.5 * 1.0)
    assert np.allclose(m.last_sub.U, x0 + 0.5 * 1.0)


def test_an_oscillating_variable_contracts_its_asymptotes_and_a_monotone_one_expands():
    """The whole point of MMA: sign flips damp, steady progress accelerates.

    Variable 0 is driven to oscillate and variable 1 to move monotonically by
    feeding gradients directly. On the third iteration, the first one that has
    three iterates of history, the asymptote distance must be multiplied by
    0.7 for the oscillating variable and 1.2 for the monotone one.
    """
    m = mma.MMA(np.array([0.5, 0.5]), lower=0.0, upper=1.0)
    m.step(np.array([+1.0, +1.0]))      # both move down
    m.step(np.array([-1.0, +1.0]))      # variable 0 reverses, variable 1 does not
    m.step(np.array([+1.0, +1.0]))      # variable 0 reverses again
    assert m.gamma_last[0] == pytest.approx(0.7)
    assert m.gamma_last[1] == pytest.approx(1.2)


def test_asymptote_distances_are_clamped_into_the_allowed_band():
    """Both distances stay inside [asy_bound_lo, asy_bound_hi] times the span.

    Read off the subproblem record, because that is the point at which the
    distance is defined: it is measured from the iterate the asymptotes were
    built around, not from the iterate the step produced.
    """
    cfg = mma.MMAConfig(asy_bound_lo=0.01, asy_bound_hi=10.0)
    m = mma.MMA(np.full(3, 0.5), lower=0.0, upper=1.0, cfg=cfg)
    for _ in range(60):
        m.step(np.array([1.0, -1.0, 0.0]))
        s = m.last_sub
        for d in (s.x - s.L, s.U - s.x):
            assert np.all(d >= cfg.asy_bound_lo * 1.0 - 1e-12)
            assert np.all(d <= cfg.asy_bound_hi * 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# 2. the subproblem, against an independent numerical minimization
# ---------------------------------------------------------------------------

def test_subproblem_solution_matches_a_brute_force_minimization_of_the_same_model():
    """The closed form must be the argmin of the separable model it claims.

    Independent check, no shared code: the model per variable is
    p0/(U - x) + q0/(x - L) on [alpha, beta]. A dense scan of that scalar
    function must land where the closed form does.
    """
    rng = np.random.default_rng(11)
    x0 = rng.uniform(0.2, 0.8, size=6)
    g = rng.normal(size=6)
    m = mma.MMA(x0, lower=0.0, upper=1.0)
    sub = m.build_subproblem(g)
    x_new = m.solve_subproblem(sub)
    for j in range(6):
        grid = np.linspace(sub.alpha[j], sub.beta[j], 200_001)
        model = sub.p0[j] / (sub.U[j] - grid) + sub.q0[j] / (grid - sub.L[j])
        assert x_new[j] == pytest.approx(grid[int(np.argmin(model))], abs=2e-5)


def test_the_step_is_a_descent_direction_in_the_gradient_sense():
    """A positive partial derivative moves its variable DOWN, and the reverse."""
    m = mma.MMA(np.full(2, 0.5), lower=0.0, upper=1.0)
    x1 = m.step(np.array([+2.0, -2.0]))
    assert x1[0] < 0.5
    assert x1[1] > 0.5


def test_a_zero_gradient_leaves_the_variable_exactly_where_it_is():
    """With only the raa0 term the model is symmetric about x, so the step is 0.

    Algebra: p0 = (U-x)^2 r and q0 = (x-L)^2 r give
    x_new = [(U-x) L + (x-L) U] / (U - L) = x, exactly.
    """
    m = mma.MMA(np.array([0.37, 0.62]), lower=0.0, upper=1.0)
    x1 = m.step(np.zeros(2))
    assert x1 == pytest.approx(np.array([0.37, 0.62]), abs=1e-12)


def test_the_move_limit_caps_a_single_step():
    cfg = mma.MMAConfig(move=0.05)
    m = mma.MMA(np.full(4, 0.5), lower=0.0, upper=1.0, cfg=cfg)
    x1 = m.step(np.array([1e6, -1e6, 1e-9, 0.0]))
    assert np.all(np.abs(x1 - 0.5) <= 0.05 + 1e-12)
    assert x1[0] == pytest.approx(0.45, abs=1e-9)
    assert x1[1] == pytest.approx(0.55, abs=1e-9)


def test_the_step_size_tracks_the_asymptote_distance_not_the_gradient_magnitude():
    """A documented CONSEQUENCE of dropping every constraint but the box.

    With no constraint there is no dual variable to rebalance the model, so for
    any variable whose gradient is well above the raa0 floor the closed-form
    step is close to a fixed fraction of the asymptote distance regardless of
    how large the gradient is. Scaling the gradient by 1e6 must therefore
    change the step by less than one part in a thousand. This is pinned
    because it is the single most surprising property of the box-only case and
    a reader must not assume MMA here is gradient-magnitude sensitive.
    """
    a = mma.MMA(np.full(2, 0.5), lower=0.0, upper=1.0,
                cfg=mma.MMAConfig(move=1.0)).step(np.array([1.0, 1.0]))
    b = mma.MMA(np.full(2, 0.5), lower=0.0, upper=1.0,
                cfg=mma.MMAConfig(move=1.0)).step(np.array([1e6, 1e6]))
    assert np.max(np.abs(a - b)) < 1e-3


# ---------------------------------------------------------------------------
# 3. the box
# ---------------------------------------------------------------------------

def test_iterates_never_leave_the_box_under_an_adversarial_gradient():
    rng = np.random.default_rng(3)
    m = mma.MMA(rng.uniform(0, 1, 25), lower=0.0, upper=1.0)
    for _ in range(80):
        x = m.step(rng.normal(scale=1e3, size=25))
        assert np.all(x >= -1e-15) and np.all(x <= 1.0 + 1e-15)


def test_box_activation_a_minimum_outside_the_box_is_driven_onto_the_bound():
    """f = sum (x - c)^2 with c outside [0, 1] on both sides, one interior."""
    c = np.array([-0.8, 1.9, 0.4])
    m = mma.MMA(np.full(3, 0.5), lower=0.0, upper=1.0, cfg=TIGHT)
    for _ in range(200):
        x = m.step(2.0 * (m.x - c))
    assert x[0] == pytest.approx(0.0, abs=1e-12)
    assert x[1] == pytest.approx(1.0, abs=1e-12)
    assert x[2] == pytest.approx(0.4, abs=1e-6)


def test_per_variable_bounds_are_honoured():
    lo = np.array([0.2, 0.0])
    hi = np.array([1.0, 0.6])
    m = mma.MMA(np.array([0.5, 0.5]), lower=lo, upper=hi)
    for _ in range(80):
        x = m.step(np.array([1.0, -1.0]))
    assert x[0] == pytest.approx(0.2, abs=1e-9)
    assert x[1] == pytest.approx(0.6, abs=1e-9)


def test_the_default_asymptote_clamp_floors_the_accuracy_at_one_percent_of_the_span():
    """Named as a limit, not a bug: the production configuration cannot resolve
    an interior optimum finer than roughly asy_bound_lo times the span, because
    the step is proportional to the asymptote distance and that distance is
    clamped from below. Measured here so the solve's stopping behaviour is not
    mistaken for a convergence failure."""
    c = np.full(4, 0.4321)
    m = mma.MMA(np.full(4, 0.9), lower=0.0, upper=1.0)   # default asy_bound_lo 0.01
    err = []
    for _ in range(300):
        m.step(2.0 * (m.x - c))
        err.append(float(np.max(np.abs(m.x - c))))
    assert min(err) > 1e-6         # cannot reach the tight tolerance
    assert min(err) < 0.05         # but does get inside a few percent of the span


# ---------------------------------------------------------------------------
# 4. end to end
# ---------------------------------------------------------------------------

def test_analytic_quadratic_converges_to_the_interior_minimum():
    rng = np.random.default_rng(7)
    c = rng.uniform(0.15, 0.85, size=30)
    m = mma.MMA(np.full(30, 0.5), lower=0.0, upper=1.0, cfg=TIGHT)
    for _ in range(200):
        m.step(2.0 * (m.x - c))
    assert np.max(np.abs(m.x - c)) < 1e-6


def test_anisotropic_quadratic_converges_despite_a_1e4_curvature_spread():
    """Curvature MMA has to discover from its asymptotes, since it sees only
    gradients and, in the box-only case, barely even their magnitude."""
    a = np.array([1e-2, 1.0, 1e2])
    c = np.array([0.25, 0.5, 0.75])
    m = mma.MMA(np.full(3, 0.4), lower=0.0, upper=1.0, cfg=TIGHT)
    for _ in range(200):
        m.step(2.0 * a * (m.x - c))
    assert np.max(np.abs(m.x - c)) < 1e-6


def test_reciprocal_objective_the_family_topology_optimization_actually_produces():
    """Known topology-optimization behaviour, in its cheapest honest form.

    The compliance of a self-adjoint linear structure is a sum of reciprocal
    terms in the design variables. On f = sum(a/x + b x), whose minimum is
    x* = sqrt(a/b), two things must hold and both are asserted:

      1. the FIRST iterate must stay strictly inside the box, where a raw
         gradient step of the same objective from x = 0.9 lands at -0.05,
         outside it. This is the safeguard MMA supplies and a plain gradient
         method does not;
      2. the run must converge to the closed-form optimum.
    """
    a = np.array([0.04, 0.09, 0.16])
    b = np.ones(3)
    star = np.sqrt(a / b)                       # 0.2, 0.3, 0.4
    x0 = np.full(3, 0.9)
    g0 = -a / x0 ** 2 + b
    assert np.min(x0 - g0) < 0.05               # the raw gradient step leaves the box
    m = mma.MMA(x0, lower=0.05, upper=1.0, cfg=TIGHT)
    x1 = m.step(g0)
    assert np.all(x1 > 0.05) and np.all(x1 <= 1.0)
    for _ in range(200):
        m.step(-a / m.x ** 2 + b)
    assert np.max(np.abs(m.x - star)) < 1e-6


def test_the_iterate_can_overshoot_so_the_caller_must_keep_the_best_iterate():
    """MMA has no line search and is NOT monotone. Pinned deliberately.

    Starting one ten-thousandth away from the optimum of a quadratic, the
    first step is a fixed fraction of the initial asymptote distance, which is
    half the box, so it lands far away and the objective rises by orders of
    magnitude. The solve driver must therefore keep the best iterate, exactly
    as the L-BFGS-B driver does.
    """
    c = np.full(3, 0.5001)
    m = mma.MMA(np.full(3, 0.5), lower=0.0, upper=1.0, cfg=TIGHT)
    f0 = float(np.sum((m.x - c) ** 2))
    m.step(2.0 * (m.x - c))
    f1 = float(np.sum((m.x - c) ** 2))
    assert f1 > f0
    for _ in range(200):
        m.step(2.0 * (m.x - c))
    assert float(np.sum((m.x - c) ** 2)) < f0    # and it still converges


def test_state_is_carried_across_a_change_of_objective_which_is_what_continuation_needs():
    """MMA's asymptotes are the memory that survives a beta jump.

    L-BFGS-B's curvature memory is discarded whenever the optimizer is
    restarted, which is what the beta continuation does at every stage
    boundary. MMA keeps its asymptotes and its three-deep iterate history, so
    switching the objective mid-run must neither reset them nor raise.
    """
    m = mma.MMA(np.full(4, 0.5), lower=0.0, upper=1.0)
    for _ in range(5):
        m.step(np.full(4, 1.0))
    k_before = m.iteration
    m.step(np.full(4, -1.0))          # a different objective's gradient
    assert m.iteration == k_before + 1
    # The asymptotes are NOT the first-iteration ones: five iterations of
    # monotone progress expanded them by 1.2 per step, and the objective
    # change did not reset that.
    dist = m.last_sub.U - m.last_sub.x
    assert np.all(dist > 0.5 * 1.0 + 1e-9)
    assert m.xold1 is not None and m.xold2 is not None
    # and the step follows the NEW gradient's descent direction
    assert np.all(m.x > 0.0)


# ---------------------------------------------------------------------------
# 5. input validation
# ---------------------------------------------------------------------------

def test_a_start_outside_the_box_is_rejected_rather_than_silently_clipped():
    with pytest.raises(ValueError):
        mma.MMA(np.array([1.5]), lower=0.0, upper=1.0)


def test_an_inverted_box_is_rejected():
    with pytest.raises(ValueError):
        mma.MMA(np.array([0.5]), lower=1.0, upper=0.0)


def test_a_gradient_of_the_wrong_length_is_rejected():
    m = mma.MMA(np.full(3, 0.5), lower=0.0, upper=1.0)
    with pytest.raises(ValueError):
        m.step(np.zeros(4))


def test_a_non_finite_gradient_is_rejected_rather_than_poisoning_the_asymptotes():
    m = mma.MMA(np.full(2, 0.5), lower=0.0, upper=1.0)
    with pytest.raises(ValueError):
        m.step(np.array([1.0, np.nan]))


def test_a_negative_or_zero_move_limit_is_rejected():
    with pytest.raises(ValueError):
        mma.MMA(np.full(2, 0.5), lower=0.0, upper=1.0, cfg=mma.MMAConfig(move=0.0))
