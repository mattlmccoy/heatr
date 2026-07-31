"""Tests for the pure gain-calibration logic (line search + hold-out selection).

Written BEFORE the implementation (red-green). No forward solves here: the objective
is a callable, so every test is a pure-logic test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gain_calibration as gc  # noqa: E402


# --------------------------------------------------------------------------
# 1. warm-start bracketing
# --------------------------------------------------------------------------

def test_bracket_interior_minimum_uses_neighbours() -> None:
    """Grid argmin in the interior -> bracket is the two neighbouring gains."""
    grid = {0.3: 10.0, 0.5: 8.0, 0.7: 9.0, 0.85: 12.0}
    br = gc.bracket_from_grid(grid, domain=(0.05, 2.5))
    assert br.lo == pytest.approx(0.3)
    assert br.hi == pytest.approx(0.7)
    assert br.needs_expansion is False


def test_bracket_upper_endpoint_minimum_flags_expansion() -> None:
    """Grid argmin at the top endpoint -> the minimum is not bracketed."""
    grid = {0.3: 10.761, 0.5: 9.586, 0.7: 7.913, 0.85: 6.993}   # the real square curve
    br = gc.bracket_from_grid(grid, domain=(0.05, 2.5))
    assert br.needs_expansion is True
    assert br.lo == pytest.approx(0.7)
    assert br.hi == pytest.approx(0.85)


def test_bracket_lower_endpoint_minimum_flags_expansion_downward() -> None:
    grid = {0.3: 5.0, 0.5: 6.0, 0.7: 7.0, 0.85: 8.0}
    br = gc.bracket_from_grid(grid, domain=(0.05, 2.5))
    assert br.needs_expansion is True
    assert br.lo == pytest.approx(0.3)
    assert br.hi == pytest.approx(0.5)
    assert br.direction == -1


def test_bracket_requires_at_least_three_points() -> None:
    with pytest.raises(ValueError):
        gc.bracket_from_grid({0.3: 1.0, 0.5: 2.0}, domain=(0.05, 2.5))


# --------------------------------------------------------------------------
# 2. expansion step
# --------------------------------------------------------------------------

def test_expansion_step_grows_geometrically_and_respects_the_cap() -> None:
    """Expanding past the top endpoint steps out by the golden ratio, capped."""
    assert gc.next_expansion_point(0.7, 0.85, domain=(0.05, 2.5)) == pytest.approx(
        0.85 + gc.GOLDEN * (0.85 - 0.7)
    )
    # near the cap the step is clipped to the cap
    assert gc.next_expansion_point(1.8, 2.4, domain=(0.05, 2.5)) == pytest.approx(2.5)
    # at the cap there is nowhere left to go
    assert gc.next_expansion_point(2.0, 2.5, domain=(0.05, 2.5)) is None


# --------------------------------------------------------------------------
# 3. the full budgeted search
# --------------------------------------------------------------------------

def test_search_finds_interior_minimum_of_a_quadratic() -> None:
    """A budgeted search on a smooth unimodal objective lands near the true argmin."""
    calls: list[float] = []

    def f(x: float) -> float:
        calls.append(x)
        return (x - 1.30) ** 2 + 1.0

    res = gc.calibrate_gain(f, warm_start={0.3: f(0.3), 0.5: f(0.5),
                                           0.7: f(0.7), 0.85: f(0.85)},
                            domain=(0.05, 2.5), max_new_evals=4)
    calls.clear()
    assert res.best_gain == pytest.approx(1.30, abs=0.20)
    assert res.n_new_evals <= 4
    assert res.n_total_evals == 4 + res.n_new_evals
    # every evaluated gain is inside the domain
    assert all(0.05 <= g <= 2.5 for g in res.evaluated)


def test_search_never_returns_a_gain_worse_than_the_warm_start_best() -> None:
    """A line search cannot choose a step that makes the fit objective worse."""
    def f(x: float) -> float:
        return (x - 1.30) ** 2 + 1.0

    warm = {0.3: f(0.3), 0.5: f(0.5), 0.7: f(0.7), 0.85: f(0.85)}
    res = gc.calibrate_gain(f, warm_start=warm, domain=(0.05, 2.5), max_new_evals=4)
    assert res.best_fit <= min(warm.values()) + 1e-12


def test_search_respects_a_zero_new_eval_budget() -> None:
    def f(x: float) -> float:
        return (x - 1.30) ** 2

    warm = {0.3: f(0.3), 0.5: f(0.5), 0.7: f(0.7), 0.85: f(0.85)}
    res = gc.calibrate_gain(f, warm_start=warm, domain=(0.05, 2.5), max_new_evals=0)
    assert res.n_new_evals == 0
    assert res.best_gain == pytest.approx(0.85)


def test_search_handles_a_monotone_objective_by_running_to_the_cap() -> None:
    """Monotone-decreasing fit metric -> the search pushes toward the domain cap."""
    def f(x: float) -> float:
        return -x

    warm = {0.3: f(0.3), 0.5: f(0.5), 0.7: f(0.7), 0.85: f(0.85)}
    res = gc.calibrate_gain(f, warm_start=warm, domain=(0.05, 2.5), max_new_evals=4)
    assert res.best_gain > 1.5


def test_search_skips_infeasible_evaluations_without_crashing() -> None:
    """The objective may return None (solve produced no usable fit state)."""
    def f(x: float) -> float | None:
        return None if x > 1.5 else (x - 1.2) ** 2

    warm = {0.3: f(0.3), 0.5: f(0.5), 0.7: f(0.7), 0.85: f(0.85)}
    res = gc.calibrate_gain(f, warm_start=warm, domain=(0.05, 2.5), max_new_evals=4)
    assert res.best_gain is not None
    assert res.best_fit is not None


# --------------------------------------------------------------------------
# 4. hold-out selection rule
# --------------------------------------------------------------------------

def _cand(g: float, fit: float | None, hold: float | None, melt: bool) -> gc.GainCandidate:
    return gc.GainCandidate(gain=g, fit_score=fit, holdout_score=hold, melt_reached=melt)


def test_selection_uses_the_fit_metric_not_the_holdout() -> None:
    """The hold-out value must never influence which gain is chosen."""
    cands = [
        _cand(0.5, fit=9.0, hold=2.0, melt=True),
        _cand(0.85, fit=7.0, hold=8.0, melt=True),   # better fit, worse hold-out
    ]
    sel = gc.select_on_holdout(cands)
    assert sel.selected.gain == pytest.approx(0.85)
    assert sel.selected.holdout_score == pytest.approx(8.0)
    assert sel.reported_holdout == pytest.approx(8.0)


def test_selection_excludes_candidates_that_never_reach_melt() -> None:
    """Feasibility reads melt_reached only, never the hold-out value."""
    cands = [
        _cand(0.5, fit=9.0, hold=12.0, melt=True),
        _cand(0.85, fit=4.0, hold=None, melt=False),   # best fit but never melts
    ]
    sel = gc.select_on_holdout(cands)
    assert sel.selected.gain == pytest.approx(0.5)
    assert sel.unconstrained.gain == pytest.approx(0.85)
    assert sel.n_infeasible == 1


def test_selection_reports_not_reached_when_nothing_melts() -> None:
    cands = [
        _cand(0.5, fit=9.0, hold=None, melt=False),
        _cand(0.85, fit=4.0, hold=None, melt=False),
    ]
    sel = gc.select_on_holdout(cands)
    assert sel.selected is None
    assert sel.reported_holdout is None
    assert sel.status == "NOT_REACHED"


def test_selection_ignores_candidates_with_no_fit_score() -> None:
    cands = [
        _cand(0.5, fit=None, hold=1.0, melt=True),
        _cand(0.85, fit=9.0, hold=5.0, melt=True),
    ]
    sel = gc.select_on_holdout(cands)
    assert sel.selected.gain == pytest.approx(0.85)


def test_selection_rejects_an_empty_candidate_list() -> None:
    with pytest.raises(ValueError):
        gc.select_on_holdout([])


# --------------------------------------------------------------------------
# 5. argmin pinned to a domain edge (the harmful-shape case)
# --------------------------------------------------------------------------

def test_boundary_argmin_at_domain_floor_still_spends_the_budget_inside() -> None:
    """Grid argmin pinned at the domain floor must still refine inside (floor, next).

    Regression: the first campaign run stopped after one new solve on triangle,
    H_shape and trapezoid because a two-point bracket produced no probe.
    """
    def f(x: float) -> float:
        return x                      # monotone increasing -> argmin at the floor

    warm = {0.05: f(0.05), 0.3: f(0.3), 0.5: f(0.5), 0.85: f(0.85)}
    res = gc.calibrate_gain(f, warm_start=warm, domain=(0.05, 2.5), max_new_evals=3)
    assert res.n_new_evals == 3
    assert all(0.05 < g < 0.3 for g in res.new_gains)


# --------------------------------------------------------------------------
# 6. unseeded, multimodality-robust search (v2)
# --------------------------------------------------------------------------

# Fixture CAPTURED from the real 26-point triangle gain scan reported in
# ADJOINT_PROTOTYPE_REPORT.md Section 7.1 (heating-peak sigma_T, deg C). Local
# maximum 48.82 at m = 0.638, global minimum 23.01 at m = 2.402. The three points
# above m = 2.5 are an ASSUMED monotone extension past the scanned range (the map
# becomes progressively more two-level), flagged here so the fixture is not
# mistaken for measurement.
_TRIANGLE_SCAN_X = [0.050, 0.344, 0.540, 0.638, 0.834, 1.030, 1.520, 2.108,
                    2.402, 2.500, 3.000, 4.000, 6.000]
_TRIANGLE_SCAN_Y = [40.45, 45.77, 48.34, 48.82, 46.14, 33.97, 25.04, 23.08,
                    23.01, 23.04, 23.50, 25.00, 28.00]


def _triangle_fit(x: float) -> float:
    """Piecewise-linear interpolation of the captured triangle gain curve."""
    xs, ys = _TRIANGLE_SCAN_X, _TRIANGLE_SCAN_Y
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    raise AssertionError("unreachable")


def test_coarse_scan_spans_the_domain_with_constant_log_ratio() -> None:
    pts = gc.coarse_scan_gains(domain=(0.05, 6.0), n_points=7)
    assert len(pts) == 7
    assert pts[0] == pytest.approx(0.05)
    assert pts[-1] == pytest.approx(6.0)
    assert all(pts[i] < pts[i + 1] for i in range(len(pts) - 1))
    ratios = [pts[i + 1] / pts[i] for i in range(len(pts) - 1)]
    assert max(ratios) == pytest.approx(min(ratios), rel=1e-9)


def test_coarse_scan_rejects_a_nonpositive_lower_bound() -> None:
    with pytest.raises(ValueError):
        gc.coarse_scan_gains(domain=(0.0, 6.0), n_points=7)


def test_seeded_search_is_trapped_by_the_triangle_local_maximum() -> None:
    """Reproduces the v1 artifact: the stored grid sits left of the local maximum."""
    warm = {m: _triangle_fit(m) for m in (0.30, 0.50, 0.70, 0.85)}
    res = gc.calibrate_gain(_triangle_fit, warm_start=warm,
                            domain=(0.05, 6.0), max_new_evals=4)
    assert res.best_gain < 0.35          # walked toward the domain floor


def test_unseeded_search_escapes_the_local_maximum_and_finds_the_global_minimum() -> None:
    res = gc.calibrate_gain_unseeded(_triangle_fit, domain=(0.05, 6.0),
                                     n_coarse=7, max_refine=3)
    assert res.best_gain > 1.8
    assert res.best_fit < 24.0
    assert res.n_new_evals == 7 + 3


def test_unseeded_search_counts_every_evaluation_it_spends() -> None:
    calls: list[float] = []

    def f(x: float) -> float:
        calls.append(x)
        return (x - 2.0) ** 2

    res = gc.calibrate_gain_unseeded(f, domain=(0.05, 6.0), n_coarse=6, max_refine=2)
    assert len(calls) == res.n_new_evals
    assert res.n_total_evals == res.n_new_evals


def test_unseeded_search_tolerates_infeasible_coarse_points() -> None:
    def f(x: float) -> float | None:
        return None if x > 3.0 else (x - 1.0) ** 2

    res = gc.calibrate_gain_unseeded(f, domain=(0.05, 6.0), n_coarse=7, max_refine=3)
    assert res.best_gain is not None
    assert res.best_gain == pytest.approx(1.0, abs=0.35)
