"""Tests for the optimizer-agnostic continuation stage runner.

RED FIRST. Observed before `topopt_stage.py` existed:
`ImportError: cannot import name 'topopt_stage' from 'adjoint2d'`.

WHAT THIS PROTECTS. The retest compares two optimizers on the SAME objective at
the SAME budget. The only way that comparison means anything is if both
optimizers consume a single, shared evaluation callable and count their budget
the same way. These tests pin exactly that, without touching the physics: the
evaluation callable here is a quadratic.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import mma, topopt_stage as ts


def quadratic(c: np.ndarray):
    """A shared evaluate callable, plus the call log both optimizers write to."""
    calls: list[np.ndarray] = []

    def evaluate(vec: np.ndarray):
        v = np.asarray(vec, dtype=float)
        calls.append(v.copy())
        return float(np.sum((v - c) ** 2)), 2.0 * (v - c)

    return evaluate, calls


BOX = (0.0, 1.0)


# ---------------------------------------------------------------------------
# budget accounting, which is the whole basis of the matched-budget comparison
# ---------------------------------------------------------------------------

def test_mma_uses_exactly_the_evaluation_budget_one_per_design_update():
    ev, calls = quadratic(np.full(5, 0.3))
    r = ts.StageRunner(ev, BOX, optimizer="mma", mma_state=mma.MMA(
        np.full(5, 0.9), lower=BOX[0], upper=BOX[1]))
    r.run(np.full(5, 0.9), n_new=4)
    assert len(calls) == 4
    assert r.n_used == 4


def test_lbfgsb_never_exceeds_the_evaluation_budget():
    ev, calls = quadratic(np.full(5, 0.3))
    r = ts.StageRunner(ev, BOX, optimizer="lbfgsb")
    r.run(np.full(5, 0.9), n_new=4)
    assert len(calls) <= 4
    assert r.n_used == len(calls)


def test_both_optimizers_consume_the_identical_evaluate_callable():
    ev, _calls = quadratic(np.full(3, 0.3))
    a = ts.StageRunner(ev, BOX, optimizer="lbfgsb")
    b = ts.StageRunner(ev, BOX, optimizer="mma",
                       mma_state=mma.MMA(np.full(3, 0.9), lower=0.0, upper=1.0))
    assert a.evaluate is ev and b.evaluate is ev


def test_a_zero_budget_stage_makes_no_evaluation_and_returns_the_start():
    ev, calls = quadratic(np.full(3, 0.3))
    r = ts.StageRunner(ev, BOX, optimizer="mma",
                       mma_state=mma.MMA(np.full(3, 0.9), lower=0.0, upper=1.0))
    best, info = r.run(np.full(3, 0.9), n_new=0)
    assert calls == []
    assert best == pytest.approx(np.full(3, 0.9))
    assert info["n_new_used"] == 0


# ---------------------------------------------------------------------------
# the best-iterate rule, which MMA needs because it is not a descent method
# ---------------------------------------------------------------------------

def test_the_stage_returns_the_best_iterate_and_not_the_last():
    """Deliberately adversarial evaluate: the last call is the worst."""
    seen = {"n": 0}
    good = np.full(2, 0.42)

    def evaluate(vec):
        seen["n"] += 1
        if seen["n"] == 2:
            return -100.0, np.zeros(2)          # the best J of the stage
        return 10.0 * seen["n"], np.ones(2) * 1e-9

    r = ts.StageRunner(evaluate, BOX, optimizer="mma",
                       mma_state=mma.MMA(np.full(2, 0.5), lower=0.0, upper=1.0))
    best, info = r.run(np.full(2, 0.5), n_new=4)
    assert info["best_J"] == -100.0
    assert info["best_eval_index"] == 2
    assert best == pytest.approx(r.points[2])
    del good


def test_every_evaluation_is_recorded_with_its_design_point():
    ev, calls = quadratic(np.full(4, 0.3))
    r = ts.StageRunner(ev, BOX, optimizer="mma",
                       mma_state=mma.MMA(np.full(4, 0.9), lower=0.0, upper=1.0))
    r.run(np.full(4, 0.9), n_new=3)
    assert sorted(r.points) == [1, 2, 3]
    for k, v in r.points.items():
        assert v == pytest.approx(calls[k - 1])


# ---------------------------------------------------------------------------
# continuation: the state that carries, and the state that does not
# ---------------------------------------------------------------------------

def test_mma_state_carries_across_two_stages_so_the_asymptotes_are_not_relearnt():
    ev, _ = quadratic(np.full(5, 0.3))
    state = mma.MMA(np.full(5, 0.9), lower=0.0, upper=1.0)
    x = np.full(5, 0.9)
    x, _ = ts.StageRunner(ev, BOX, optimizer="mma", mma_state=state).run(x, 3)
    assert state.iteration == 3
    x, _ = ts.StageRunner(ev, BOX, optimizer="mma", mma_state=state).run(x, 3)
    assert state.iteration == 6              # not reset to 3


def test_mma_requires_its_state_object_so_a_silent_reset_is_impossible():
    ev, _ = quadratic(np.full(2, 0.3))
    with pytest.raises(ValueError):
        ts.StageRunner(ev, BOX, optimizer="mma").run(np.full(2, 0.5), 2)


def test_an_unknown_optimizer_name_is_rejected():
    ev, _ = quadratic(np.full(2, 0.3))
    with pytest.raises(ValueError):
        ts.StageRunner(ev, BOX, optimizer="adam")


# ---------------------------------------------------------------------------
# the mechanics claim the retest exists to test, on a clean quadratic
# ---------------------------------------------------------------------------

def test_on_a_short_stage_mma_produces_one_design_update_per_evaluation():
    """The mechanics fix, stated as a testable property.

    MMA has no line search, so every evaluation moves the design. The
    assertion is that all n evaluated points are distinct, which a line-search
    method cannot guarantee because a rejected trial point is re-evaluated
    near the incumbent.
    """
    ev, calls = quadratic(np.linspace(0.2, 0.8, 6))
    state = mma.MMA(np.full(6, 0.95), lower=0.0, upper=1.0)
    ts.StageRunner(ev, BOX, optimizer="mma", mma_state=state).run(np.full(6, 0.95), 5)
    for i in range(1, len(calls)):
        assert np.max(np.abs(calls[i] - calls[i - 1])) > 1e-12


def test_the_box_is_respected_by_both_optimizers():
    ev, calls = quadratic(np.full(4, 5.0))       # optimum far outside the box
    for name, state in (("lbfgsb", None),
                        ("mma", mma.MMA(np.full(4, 0.5), lower=0.0, upper=1.0))):
        calls.clear()
        ts.StageRunner(ev, BOX, optimizer=name, mma_state=state).run(np.full(4, 0.5), 6)
        for v in calls:
            assert np.all(v >= -1e-12) and np.all(v <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# the result-file stem, so four arms cannot overwrite one another
# ---------------------------------------------------------------------------

def test_the_output_stem_is_unchanged_for_the_production_recipe():
    from adjoint2d import topopt_solve as tos
    assert tos.output_tag("square") == "square"
    assert tos.output_tag("square", "filteronly") == "square_control_filteronly"


def test_the_output_stem_separates_optimizer_and_budget():
    from adjoint2d import topopt_solve as tos
    stems = {
        tos.output_tag("square", "filteronly", "lbfgsb", 40.0),
        tos.output_tag("square", "", "mma", 40.0),
        tos.output_tag("square", "", "mma", 80.0),
        tos.output_tag("square", "", "lbfgsb", 80.0),
    }
    assert len(stems) == 4


# ---------------------------------------------------------------------------
# the acceptance-gate runner must be able to read any arm's result files
# ---------------------------------------------------------------------------

def test_robust_source_resolution_defaults_to_the_original_pass_unchanged():
    """RED first: `resolve_source` did not exist.

    The two acceptance gates were written against one fixed directory and one
    fixed stem convention. The retest has four arms in a different directory,
    so the resolution is made explicit and pure. The default must reproduce the
    previous behaviour exactly, or the earlier pass's numbers stop being
    reproducible.
    """
    from adjoint2d import topopt_robust as trb
    assert trb.resolve_stem("square", "") == "square"
    assert trb.resolve_stem("square", "control_filteronly") == "square_control_filteronly"
    assert trb.resolve_source(None, "square", "").parent == trb.OUT_TOPOPT
    assert trb.resolve_source(None, "square", "").name == "square.json"


def test_robust_source_resolution_accepts_an_explicit_directory_and_stem():
    from adjoint2d import topopt_robust as trb
    p = trb.resolve_source("/tmp/out_mma", "square", "", stem="square_mma_b80")
    assert str(p) == "/tmp/out_mma/square_mma_b80.json"


# ---------------------------------------------------------------------------
# the carried-history arm: scipy giving up early must not silently cut budget
# ---------------------------------------------------------------------------

def test_lbfgsb_carry_re_enters_when_scipy_stops_early_so_the_budget_is_spent():
    """RED first. OBSERVED FAILURE that motivated this test, on real physics:
    `logs_mma/triangle_lbfgsb_carry_b80.log` shows L-BFGS-B declaring
    convergence after 12 of 32 allowed evaluations, at beta 2, so the beta 4,
    8 and 16 stages of the continuation received NO evaluation at all and the
    arm never reached the projection sharpness it was supposed to be compared
    at. A single `minimize` call cannot be relied on to consume the budget,
    especially in this arm where the objective changes underneath the line
    search at each stage boundary.

    The fix is to re-enter L-BFGS-B from the current point until the budget is
    spent, and to COUNT the re-entries, because each one discards the curvature
    memory that this arm exists to carry.
    """
    def evaluate(vec):                       # flat: scipy converges at once
        return 1.0, np.zeros(np.size(vec))

    r = ts.StageRunner(evaluate, BOX, optimizer="lbfgsb_carry")
    _best, info = r.run(np.full(3, 0.5), n_new=5)
    assert r.n_used == 5
    assert info["n_optimizer_restarts"] >= 1


def test_plain_lbfgsb_does_not_re_enter_because_a_stage_is_one_call_by_definition():
    def evaluate(vec):
        return 1.0, np.zeros(np.size(vec))

    r = ts.StageRunner(evaluate, BOX, optimizer="lbfgsb")
    _best, info = r.run(np.full(3, 0.5), n_new=5)
    assert r.n_used == 1
    assert info["n_optimizer_restarts"] == 0
