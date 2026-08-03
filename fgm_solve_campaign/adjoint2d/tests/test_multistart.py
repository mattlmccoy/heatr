"""Pure logic of the multi-start / warm-start melt-objective solve.

Everything here is arithmetic or array algebra with no forward run in it: the
budget split across starts, the early-kill rule, the deterministic perturbed
start, and the box projection of a stored map into a design variable.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import multistart as ms


# ---------------------------------------------------------------------------
# budget arithmetic
# ---------------------------------------------------------------------------

def test_probe_evals_splits_the_pool_evenly():
    # 14 total gradient evaluations, 4 starts, 2 wanted each -> 2 each
    assert ms.probe_evals(14, 4, 2) == 2


def test_probe_evals_never_exceeds_an_even_share():
    # 6 total, 4 starts: an even share is 1, so a request of 2 is cut to 1
    assert ms.probe_evals(6, 4, 2) == 1


def test_probe_evals_is_at_least_one_when_the_pool_is_tiny():
    assert ms.probe_evals(2, 4, 2) == 1


def test_probe_evals_rejects_a_non_positive_start_count():
    with pytest.raises(ValueError):
        ms.probe_evals(10, 0, 2)


def test_continuation_evals_divides_the_remainder_among_survivors():
    # 14 total, 8 already spent, 2 survivors -> 3 each (floor)
    assert ms.continuation_evals(14, 8, 2) == 3


def test_continuation_evals_is_zero_when_the_budget_is_exhausted():
    assert ms.continuation_evals(8, 8, 2) == 0
    assert ms.continuation_evals(8, 12, 2) == 0


def test_continuation_evals_is_zero_with_no_survivors():
    assert ms.continuation_evals(14, 8, 0) == 0


# ---------------------------------------------------------------------------
# the early-kill rule
# ---------------------------------------------------------------------------

def test_survivors_keeps_the_leader_and_a_close_second():
    best = {"cold": 100.0, "warm": 104.0, "prev": 500.0, "pert": 600.0}
    assert ms.survivors(best, margin=0.10, max_keep=2) == ["cold", "warm"]


def test_survivors_kills_a_start_that_lags_beyond_the_margin():
    best = {"cold": 100.0, "warm": 130.0}
    assert ms.survivors(best, margin=0.10, max_keep=2) == ["cold"]


def test_survivors_respects_max_keep_even_when_all_are_close():
    best = {"a": 100.0, "b": 101.0, "c": 102.0, "d": 103.0}
    assert ms.survivors(best, margin=0.10, max_keep=2) == ["a", "b"]


def test_survivors_always_keeps_at_least_the_leader():
    best = {"a": 100.0, "b": 101.0}
    assert ms.survivors(best, margin=0.0, max_keep=1) == ["a"]


def test_survivors_orders_by_J_not_by_insertion():
    best = {"warm": 3.0, "cold": 1.0, "prev": 2.0}
    assert ms.survivors(best, margin=10.0, max_keep=3) == ["cold", "prev", "warm"]


def test_survivors_breaks_exact_ties_by_insertion_order():
    best = {"warm": 5.0, "cold": 5.0}
    assert ms.survivors(best, margin=0.0, max_keep=2) == ["warm", "cold"]


def test_survivors_skips_starts_with_no_result():
    best = {"cold": 100.0, "warm": None, "prev": 101.0}
    assert ms.survivors(best, margin=0.10, max_keep=2) == ["cold", "prev"]


def test_survivors_of_an_empty_dict_is_empty():
    assert ms.survivors({}, margin=0.10, max_keep=2) == []


def test_survivors_handles_a_zero_leader_without_dividing_by_zero():
    best = {"a": 0.0, "b": 1e-9}
    assert ms.survivors(best, margin=0.10, max_keep=2) == ["a"]


# ---------------------------------------------------------------------------
# the deterministic perturbed start
# ---------------------------------------------------------------------------

def test_perturbed_start_is_the_midpoint_of_uniform_and_the_proportional_map():
    pm = np.zeros((4, 4), dtype=bool)
    pm[1:3, 1:3] = True
    pi = np.full((4, 4), 0.25)
    v = ms.perturbed_start(pi, pm)
    assert np.allclose(v[pm], 0.625)          # (1.0 + 0.25) / 2


def test_perturbed_start_holds_the_nominal_value_outside_the_part():
    pm = np.zeros((3, 3), dtype=bool)
    pm[1, 1] = True
    v = ms.perturbed_start(np.zeros((3, 3)), pm)
    assert np.allclose(v[~pm], 1.0)


def test_perturbed_start_stays_in_the_unit_box():
    rng = np.random.default_rng(0)
    pm = np.ones((5, 5), dtype=bool)
    v = ms.perturbed_start(rng.uniform(-1.0, 2.0, (5, 5)), pm)
    assert v.min() >= 0.0 and v.max() <= 1.0


def test_perturbed_start_differs_from_uniform_when_the_map_is_not_uniform():
    pm = np.ones((4, 4), dtype=bool)
    pi = np.linspace(0.0, 1.0, 16).reshape(4, 4)
    v = ms.perturbed_start(pi, pm)
    assert not np.allclose(v, 1.0)


def test_perturbed_start_is_seedless_and_reproducible():
    pm = np.ones((4, 4), dtype=bool)
    pi = np.linspace(0.0, 1.0, 16).reshape(4, 4)
    assert np.array_equal(ms.perturbed_start(pi, pm), ms.perturbed_start(pi, pm))


# ---------------------------------------------------------------------------
# projecting a stored map into a design variable
# ---------------------------------------------------------------------------

def test_start_from_map_clips_into_the_box_and_holds_the_outside():
    pm = np.zeros((3, 3), dtype=bool)
    pm[1, 1] = True
    s = np.full((3, 3), 1.7)
    v = ms.start_from_map(s, pm, (0.0, 1.0))
    assert v[1, 1] == pytest.approx(1.0)
    assert np.allclose(v[~pm], 1.0)


def test_start_from_map_leaves_an_in_box_value_alone():
    pm = np.ones((2, 2), dtype=bool)
    s = np.array([[0.1, 0.9], [0.3, 0.5]])
    assert np.allclose(ms.start_from_map(s, pm, (0.0, 1.0)), s)


# ---------------------------------------------------------------------------
# picking the winner
# ---------------------------------------------------------------------------

def test_best_final_picks_the_lowest_J():
    r = {"cold": {"J": 10.0}, "warm": {"J": 3.0}, "prev": {"J": 7.0}}
    assert ms.best_final(r) == "warm"


def test_best_final_ignores_starts_with_no_result():
    r = {"cold": {"J": 10.0}, "warm": None}
    assert ms.best_final(r) == "cold"


def test_best_final_of_nothing_is_none():
    assert ms.best_final({}) is None
    assert ms.best_final({"a": None}) is None


# ---------------------------------------------------------------------------
# the evaluation cache that makes the probe-to-continuation handover free
# ---------------------------------------------------------------------------

def test_eval_cache_returns_the_stored_value_for_an_identical_vector():
    c = ms.EvalCache()
    v = np.array([1.0, 2.0, 3.0])
    c.put(v, (5.0, np.array([0.1, 0.2, 0.3])))
    hit = c.get(v.copy())
    assert hit is not None and hit[0] == 5.0


def test_eval_cache_misses_on_a_different_vector():
    c = ms.EvalCache()
    c.put(np.array([1.0, 2.0]), (5.0, np.array([0.0, 0.0])))
    assert c.get(np.array([1.0, 2.0000001])) is None


def test_eval_cache_counts_hits_and_misses():
    c = ms.EvalCache()
    v = np.array([1.0])
    c.get(v)
    c.put(v, (1.0, v))
    c.get(v)
    assert c.n_miss == 1 and c.n_hit == 1
