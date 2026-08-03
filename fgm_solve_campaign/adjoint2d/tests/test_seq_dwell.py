"""Red-first tests for SEQUENTIAL dwell scheduling, the pure logic.

The actuator under test is an ORDERED list of (position, duration) segments:
hold one orientation long enough to melt one limb, then move and hold another.
That is a different object from the CYCLED dwell of `dwell.py`, whose repeated
short cycles make the part see only the time average of the heating.

Everything in this file is arithmetic on the segment-to-outer-step overlap
matrix. No physics, no solver.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adjoint2d import seq_dwell as sq  # noqa: E402


# ---------------------------------------------------------------------------
# the overlap matrix
# ---------------------------------------------------------------------------

def test_step_mix_rows_sum_to_one():
    f = sq.step_mix([3.3, 2.2, 1.1], dt_s=0.5, n_steps=20)
    assert f.shape == (20, 3)
    assert np.allclose(f.sum(axis=1), 1.0, atol=1e-14)
    assert np.all(f >= 0.0)


def test_step_mix_on_exact_step_multiples_is_one_hot():
    """Boundaries on the control grid: every step belongs to ONE segment."""
    f = sq.step_mix([2.0, 1.5, 1.0], dt_s=0.5, n_steps=9)
    assert set(np.unique(f)) <= {0.0, 1.0}
    idx = np.argmax(f, axis=1)
    assert list(idx) == [0, 0, 0, 0, 1, 1, 1, 2, 2]


def test_step_mix_splits_the_straddled_step_in_proportion():
    """A boundary 30 percent into a step splits that step 0.3 / 0.7."""
    f = sq.step_mix([1.15, 5.0], dt_s=0.5, n_steps=6)
    assert f[0, 0] == pytest.approx(1.0)
    assert f[1, 0] == pytest.approx(1.0)
    assert f[2, 0] == pytest.approx(0.3)
    assert f[2, 1] == pytest.approx(0.7)
    assert f[3, 1] == pytest.approx(1.0)


def test_step_mix_holds_the_last_segment_past_the_program_end():
    """A machine that has finished its program holds the last position."""
    f = sq.step_mix([1.0, 1.0], dt_s=0.5, n_steps=10)
    assert np.allclose(f[4:, 1], 1.0)
    assert np.allclose(f[4:, 0], 0.0)


def test_step_mix_rejects_a_negative_duration():
    with pytest.raises(ValueError):
        sq.step_mix([1.0, -0.5], dt_s=0.5, n_steps=10)


def test_step_mix_allows_a_segment_shorter_than_one_control_step():
    """Two boundaries inside one step: the step carries three segments."""
    f = sq.step_mix([0.1, 0.2, 5.0], dt_s=0.5, n_steps=4)
    assert f[0, 0] == pytest.approx(0.2)
    assert f[0, 1] == pytest.approx(0.4)
    assert f[0, 2] == pytest.approx(0.4)
    assert f.sum(axis=1) == pytest.approx(np.ones(4))


# ---------------------------------------------------------------------------
# the vector-Jacobian product, against a central difference
# ---------------------------------------------------------------------------

def test_mix_vjp_against_central_difference():
    """dJ/d(durations) for a random linear functional of the overlap matrix.

    Every switch time sits strictly INSIDE a control step (3.3, 6.05, 10.15),
    which is the almost-everywhere case. The measure-zero case where a switch
    lands exactly on a step edge is a genuine kink and is tested separately.
    """
    rng = np.random.default_rng(4)
    d = np.array([3.3, 2.75, 4.1, 6.0])
    dt, n = 0.5, 30
    g = rng.standard_normal((n, d.size))

    def J(dd):
        return float(np.sum(g * sq.step_mix(dd, dt, n)))

    ana = sq.mix_vjp(g, d, dt, n)
    for k in range(d.size):
        e = 1e-6
        dp, dm = d.copy(), d.copy()
        dp[k] += e
        dm[k] -= e
        fd = (J(dp) - J(dm)) / (2 * e)
        assert fd == pytest.approx(ana[k], abs=1e-7, rel=1e-7)


def test_mix_vjp_at_a_switch_on_a_step_edge_is_the_right_derivative():
    """A switch exactly on a control-step edge is a KINK, named not hidden.

    `mix_vjp` returns the RIGHT derivative there. The central difference sits
    halfway between the two one-sided derivatives, so the two disagree by
    construction and this is the one case the gate must avoid.
    """
    rng = np.random.default_rng(5)
    d = np.array([3.0, 4.0])                  # c_1 = 3.0, exactly a step edge
    dt, n = 0.5, 20
    g = rng.standard_normal((n, 2))

    def J(dd):
        return float(np.sum(g * sq.step_mix(dd, dt, n)))

    right = (J(d + np.array([1e-6, 0.0])) - J(d)) / 1e-6
    left = (J(d) - J(d - np.array([1e-6, 0.0]))) / 1e-6
    assert sq.mix_vjp(g, d, dt, n)[0] == pytest.approx(right, rel=1e-6)
    assert abs(right - left) > 1e-3


def test_mix_vjp_is_zero_for_the_last_segment():
    """The final boundary is where the machine holds; moving it changes nothing."""
    rng = np.random.default_rng(11)
    d = np.array([3.0, 4.0, 2.0])
    g = rng.standard_normal((40, 3))
    assert sq.mix_vjp(g, d, 0.5, 40)[-1] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# the machine-readable program
# ---------------------------------------------------------------------------

def test_sequential_program_orders_moves_and_conserves_exposure():
    p = sq.sequential_program([0.0, 45.0, 90.0], [0, 2, 1], [100.0, 200.0, 50.0],
                              dt_s=0.5, horizon_s=750.0)
    assert [m["position_deg"] for m in p.moves] == [0.0, 90.0, 45.0]
    assert [m["move_at_s"] for m in p.moves] == [0.0, 100.0, 300.0]
    assert sum(m["dwell_s"] for m in p.moves) == pytest.approx(750.0)
    assert p.moves[-1]["dwell_s"] == pytest.approx(450.0)   # holds to the horizon
    assert p.as_json()["n_moves"] == 3
    assert p.as_json()["schedule_kind"] == "sequential"


def test_sequential_program_snaps_to_the_control_step():
    p = sq.sequential_program([0.0, 90.0], [0, 1], [100.3, 100.0],
                              dt_s=0.5, horizon_s=400.0, snap=True)
    assert p.moves[0]["dwell_s"] == pytest.approx(100.5)
    assert all(abs(m["dwell_s"] / 0.5 - round(m["dwell_s"] / 0.5)) < 1e-9
               for m in p.moves)


def test_sequential_program_merges_a_repeated_adjacent_position():
    p = sq.sequential_program([0.0, 90.0], [0, 0, 1], [10.0, 20.0, 5.0],
                              dt_s=0.5, horizon_s=100.0)
    assert [m["position_deg"] for m in p.moves] == [0.0, 90.0]
    assert p.moves[0]["dwell_s"] == pytest.approx(30.0)


def test_segment_orders_enumerates_every_ordering_without_repeats():
    got = sq.segment_orders([0, 3, 5], n_segments=2)
    assert sorted(got) == sorted([(0, 3), (0, 5), (3, 0), (3, 5), (5, 0), (5, 3)])
    assert len(sq.segment_orders([0, 1, 2, 3], n_segments=3)) == 24
