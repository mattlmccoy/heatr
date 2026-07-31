"""Pure-logic tests for the temporal power schedule p(t).

Written RED first. The module `adjoint2d.schedule` did not exist when these
were run the first time and the ImportError was observed.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import schedule as sch


# ---------------------------------------------------------------------------
# segment mapping
# ---------------------------------------------------------------------------

def test_segment_index_is_uniform_over_the_nominal_horizon():
    # 1500 steps, 16 segments: 93.75 steps per segment, so the boundaries are
    # not integers and the mapping must still cover every step exactly once.
    n_steps, n_seg = 1500, 16
    idx = [sch.segment_index(it, n_steps, n_seg) for it in range(n_steps)]
    assert idx[0] == 0
    assert idx[-1] == n_seg - 1
    assert sorted(set(idx)) == list(range(n_seg))
    # monotone non-decreasing
    assert all(b >= a for a, b in zip(idx, idx[1:]))


def test_segment_index_clamps_beyond_the_nominal_horizon():
    # A march that runs past its nominal horizon must not index out of range.
    assert sch.segment_index(2000, 1500, 16) == 15


def test_segment_bounds_partition_the_horizon():
    b = sch.segment_bounds(1500, 16)
    assert len(b) == 16
    assert b[0][0] == 0
    assert b[-1][1] == 1500
    assert all(b[i][1] == b[i + 1][0] for i in range(15))
    assert sum(hi - lo for lo, hi in b) == 1500


def test_expand_is_piecewise_constant_and_matches_segment_index():
    p = np.linspace(0.1, 1.4, 8)
    full = sch.expand(p, n_steps=100, n_seg=8)
    assert full.shape == (100,)
    for it in range(100):
        assert full[it] == p[sch.segment_index(it, 100, 8)]


def test_accumulate_to_segments_is_the_transpose_of_expand():
    # The gradient of a scalar built as sum_it f(it) * expand(p)[it] with
    # respect to p is the segment-wise sum of f. That is exactly the adjoint of
    # `expand`, and the test states it as an inner-product identity.
    rng = np.random.default_rng(0)
    n_steps, n_seg = 37, 5
    per_step = rng.standard_normal(n_steps)
    p = rng.standard_normal(n_seg)
    lhs = float(np.dot(per_step, sch.expand(p, n_steps, n_seg)))
    rhs = float(np.dot(sch.accumulate_to_segments(per_step, n_steps, n_seg), p))
    assert lhs == pytest.approx(rhs, rel=1e-14, abs=1e-14)


def test_accumulate_handles_a_truncated_march():
    # The march can stop early. Steps that never ran contribute nothing, and
    # the trailing segments must come back as exact zeros, not be dropped.
    per_step = np.ones(10)
    out = sch.accumulate_to_segments(per_step, n_steps=100, n_seg=4)
    assert out.shape == (4,)
    assert out[0] == 10.0
    assert out[1] == 0.0 and out[2] == 0.0 and out[3] == 0.0


# ---------------------------------------------------------------------------
# rounding to the binary on/off class
# ---------------------------------------------------------------------------

def test_round_binary_uses_a_half_threshold_and_stays_in_the_set():
    p = np.array([0.0, 0.4999, 0.5, 0.5001, 1.0])
    b = sch.round_binary(p)
    assert set(np.unique(b)).issubset({0.0, 1.0})
    assert list(b) == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_duty_cycle_is_the_time_weighted_mean_not_the_segment_mean():
    # Unequal segment lengths must be weighted by their length in steps.
    p = np.array([1.0, 0.0, 0.0])
    # 10 steps, 3 segments -> lengths 4, 3, 3 under floor partitioning
    d = sch.duty_cycle(p, n_steps=10, n_seg=3)
    assert d == pytest.approx(4.0 / 10.0)


# ---------------------------------------------------------------------------
# budget bookkeeping for the alternating block solve
# ---------------------------------------------------------------------------

def test_block_budget_split_spends_every_evaluation():
    parts = sch.block_budget(13, 4)
    assert sum(parts) == 13
    assert len(parts) == 4
    assert max(parts) - min(parts) <= 1


def test_block_budget_never_returns_an_empty_block_when_it_can_avoid_it():
    parts = sch.block_budget(4, 4)
    assert parts == [1, 1, 1, 1]
    parts = sch.block_budget(3, 4)
    assert sum(parts) == 3
    assert parts.count(0) == 1


# ---------------------------------------------------------------------------
# generator-instruction rendering
# ---------------------------------------------------------------------------

def test_schedule_instructions_give_segment_times_and_levels():
    rows = sch.instructions(np.array([1.0, 0.0, 1.0]), n_steps=30, n_seg=3, dt_s=0.5)
    assert len(rows) == 3
    assert rows[0]["t_start_s"] == 0.0
    assert rows[0]["t_end_s"] == pytest.approx(5.0)
    assert rows[0]["level"] == 1.0
    assert rows[-1]["t_end_s"] == pytest.approx(15.0)


def test_instructions_merge_adjacent_equal_levels_when_asked():
    rows = sch.instructions(np.array([1.0, 1.0, 0.0]), n_steps=30, n_seg=3,
                            dt_s=0.5, merge=True)
    assert len(rows) == 2
    assert rows[0]["t_end_s"] == pytest.approx(10.0)
    assert rows[1]["level"] == 0.0
