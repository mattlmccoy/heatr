"""Red-first tests for the asymmetric dwell schedule.

Three groups, in the order they were written:

  1. the simplex projection and the softmax parameterization of the dwell
     fractions (pure logic, no physics);
  2. the cycle scheduler that turns a dwell fraction vector into a machine
     readable turntable program (pure logic, no physics);
  3. the weighted kernel itself, whose DEGENERACY at uniform weights must
     reproduce the already-gated `rot_kernel.AveragedKernel` bit for bit, and
     whose weight gradient must match a finite difference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
CAMPAIGN = HERE.parents[2]
REPO = HERE.parents[3]
for p in (str(CAMPAIGN), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adjoint2d import dwell                                    # noqa: E402
from adjoint2d import gradops, library_solve as lib            # noqa: E402
from adjoint2d import shape_objective as so                    # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                 # noqa: E402
from adjoint2d.pins import load_cfg                            # noqa: E402
from adjoint2d.rot_frame import rotation_operator              # noqa: E402
from adjoint2d.rot_kernel import AveragedKernel                # noqa: E402


# ---------------------------------------------------------------------------
# 1. the simplex projection and the softmax
# ---------------------------------------------------------------------------

def test_project_to_simplex_leaves_a_point_already_on_the_simplex_unchanged():
    w = np.array([0.1, 0.4, 0.25, 0.25])
    assert np.allclose(dwell.project_to_simplex(w), w, atol=1e-14)


def test_project_to_simplex_clips_a_negative_and_renormalizes():
    # The Euclidean projection of [0.6, 0.6, -0.2] onto the unit simplex.
    # Two active coordinates, tau = (0.6 + 0.6 - 1) / 2 = 0.1.
    got = dwell.project_to_simplex(np.array([0.6, 0.6, -0.2]))
    assert np.allclose(got, [0.5, 0.5, 0.0], atol=1e-14)
    assert got.sum() == pytest.approx(1.0, abs=1e-14)


def test_project_to_simplex_is_the_nearest_point_of_the_simplex():
    """Brute force: no random simplex point may be nearer than the projection."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        v = rng.normal(size=6) * 1.5
        p = dwell.project_to_simplex(v)
        d0 = float(np.sum((v - p) ** 2))
        cand = rng.dirichlet(np.ones(6), size=4000)
        d = np.sum((cand - v) ** 2, axis=1)
        assert d0 <= float(d.min()) + 1e-12


def test_project_to_simplex_scales_to_a_total_other_than_one():
    got = dwell.project_to_simplex(np.array([1.0, 2.0, -5.0]), total=750.0)
    assert got.sum() == pytest.approx(750.0, abs=1e-9)
    assert got.min() >= 0.0


def test_softmax_weights_are_positive_and_sum_to_one():
    w = dwell.softmax_weights(np.array([-3.0, 0.0, 2.5, 1.0]))
    assert w.min() > 0.0
    assert w.sum() == pytest.approx(1.0, abs=1e-15)


def test_softmax_weights_are_shift_invariant_and_overflow_safe():
    z = np.array([800.0, 802.0, 799.0])
    w = dwell.softmax_weights(z)
    assert np.all(np.isfinite(w))
    assert np.allclose(w, dwell.softmax_weights(z - 800.0), atol=1e-15)


def test_softmax_vjp_matches_a_central_finite_difference():
    rng = np.random.default_rng(11)
    z = rng.normal(size=5)
    gw = rng.normal(size=5)

    def f(zz):
        return float(np.dot(gw, dwell.softmax_weights(zz)))

    ana = dwell.softmax_vjp(dwell.softmax_weights(z), gw)
    eps = 1e-6
    num = np.array([(f(z + eps * e) - f(z - eps * e)) / (2 * eps)
                    for e in np.eye(5)])
    assert np.allclose(ana, num, rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. the cycle scheduler
# ---------------------------------------------------------------------------

ANG4 = (0.0, 90.0, 180.0, 270.0)


def test_cycle_program_dwells_sum_to_the_exposure():
    prog = dwell.cycle_program([0.4, 0.3, 0.2, 0.1], ANG4,
                               cycle_time_s=20.0, total_s=750.0, dt_s=0.5)
    assert sum(m["dwell_s"] for m in prog.moves) == pytest.approx(750.0, abs=1e-9)


def test_cycle_program_move_times_are_monotone_and_on_the_step_grid():
    prog = dwell.cycle_program([0.4, 0.3, 0.2, 0.1], ANG4,
                               cycle_time_s=20.0, total_s=750.0, dt_s=0.5)
    t = [m["move_at_s"] for m in prog.moves]
    assert t[0] == 0.0
    assert all(b > a for a, b in zip(t, t[1:]))
    assert all(abs(round(x / 0.5) - x / 0.5) < 1e-9 for x in t)


def test_cycle_program_allocates_time_in_proportion_to_the_weights():
    w = [0.4, 0.3, 0.2, 0.1]
    prog = dwell.cycle_program(w, ANG4, cycle_time_s=20.0, total_s=800.0, dt_s=0.5)
    got = {a: 0.0 for a in ANG4}
    for m in prog.moves:
        got[m["position_deg"]] += m["dwell_s"]
    for a, wi in zip(ANG4, w):
        assert got[a] == pytest.approx(wi * 800.0, rel=0.02)


def test_cycle_program_omits_a_zero_weight_position_entirely():
    prog = dwell.cycle_program([0.5, 0.0, 0.5, 0.0], ANG4,
                               cycle_time_s=20.0, total_s=100.0, dt_s=0.5)
    assert {m["position_deg"] for m in prog.moves} == {0.0, 180.0}
    assert prog.kept_positions_deg == (0.0, 180.0)


def test_cycle_program_merges_the_wrap_when_one_position_takes_everything():
    """A single kept position is one dwell for the whole exposure, not N of them."""
    prog = dwell.cycle_program([1.0, 0.0], (0.0, 45.0), cycle_time_s=20.0,
                               total_s=100.0, dt_s=0.5)
    assert len(prog.moves) == 1
    assert prog.moves[0] == {"position_deg": 0.0, "dwell_s": 100.0, "move_at_s": 0.0}


def test_cycle_program_allotment_is_integer_steps_summing_to_the_cycle():
    prog = dwell.cycle_program([1 / 3, 1 / 3, 1 / 3], (0.0, 120.0, 240.0),
                               cycle_time_s=20.0, total_s=100.0, dt_s=0.5)
    # 40 steps per cycle across three positions: 14, 13, 13 by largest remainder.
    assert sum(prog.steps_per_position) == 40
    assert sorted(prog.steps_per_position) == [13, 13, 14]


def test_cycle_program_rejects_a_cycle_too_short_to_give_each_position_a_step():
    with pytest.raises(ValueError):
        dwell.cycle_program([0.25] * 4, ANG4, cycle_time_s=1.0,
                            total_s=100.0, dt_s=0.5)


def test_cycle_program_realized_weights_are_the_quantized_ones():
    """What the machine executes, not what the optimizer asked for."""
    prog = dwell.cycle_program([0.37, 0.33, 0.30], (0.0, 120.0, 240.0),
                               cycle_time_s=20.0, total_s=800.0, dt_s=0.5)
    assert prog.realized_weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(prog.realized_weights,
                       np.asarray(prog.steps_per_position) / 40.0)


# ---------------------------------------------------------------------------
# 3. the weighted kernel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cross_cfg():
    return load_cfg(lib.shape_config("cross"))


def test_dwell_kernel_at_uniform_weights_is_bit_identical_to_the_averaged_kernel(cross_cfg):
    """DEGENERACY GATE. Equal dwells must reproduce the already-gated kernel."""
    ang = np.array([0.0, 90.0, 180.0, 270.0])
    ka = AveragedKernel.build(cross_cfg, angles=ang)
    kd = DwellKernel.build(cross_cfg, angles=ang)
    kd.set_weights(np.full(4, 0.25))
    case = ka.case0
    ops = gradops.gradient_matrices(case.x, case.y)
    s = np.where(case.part_mask, 0.7, 1.0)

    ta = ka.forward(s, keep_checkpoints=True, n_steps=30)
    td = kd.forward(s, keep_checkpoints=True, n_steps=30)
    assert np.array_equal(ta.T_final, td.T_final)
    assert np.array_equal(ta.Q_avg_a, td.Q_avg_a)
    assert np.array_equal(ta.Q_avg_b, td.Q_avg_b)

    Ja, seed_a = so.shape_J_and_seed(ta.T_at_end(29), case)
    Jd, seed_d = so.shape_J_and_seed(td.T_at_end(29), case)
    assert Ja == Jd
    ga = ka.gradient(s, ta, {29: seed_a}, grad_ops=ops)
    gd = kd.gradient(s, td, {29: seed_d}, grad_ops=ops)
    rel = np.max(np.abs(ga - gd)) / max(np.max(np.abs(ga)), 1e-30)
    assert rel < 1e-13


def test_dwell_kernel_weights_must_be_non_negative_and_normalized(cross_cfg):
    kd = DwellKernel.build(cross_cfg, angles=np.array([0.0, 90.0]))
    with pytest.raises(ValueError):
        kd.set_weights(np.array([-0.1, 1.1]))
    with pytest.raises(ValueError):
        kd.set_weights(np.array([0.3, 0.3]))
    with pytest.raises(ValueError):
        kd.set_weights(np.array([0.3, 0.3, 0.4]))


def test_weight_gradient_matches_a_central_finite_difference(cross_cfg):
    ang = np.array([0.0, 45.0, 90.0, 135.0])
    kd = DwellKernel.build(cross_cfg, angles=ang)
    case = kd.case0
    pm = case.part_mask
    s = np.where(pm, 0.8, 1.0)
    w0 = np.array([0.45, 0.25, 0.20, 0.10])
    n = 40
    read = n - 1

    def J_of(w):
        kd.set_weights(w)
        tr = kd.forward(s, keep_checkpoints=False, n_steps=n)
        return so.shape_J_and_seed(tr.T_at_end(read), case)[0]

    kd.set_weights(w0)
    tr0 = kd.forward(s, keep_checkpoints=True, n_steps=n)
    _J0, seed = so.shape_J_and_seed(tr0.T_at_end(read), case)
    gw = kd.weight_gradient(tr0, {read: seed})

    # A direction that stays on the simplex, so the finite difference is a
    # legal move in the design space and no renormalization is folded in.
    d = np.array([1.0, -1.0, 0.5, -0.5])
    ana = float(np.dot(gw, d))
    best = min(abs((J_of(w0 + e * d) - J_of(w0 - e * d)) / (2 * e) - ana)
               / max(abs(ana), 1e-30) for e in (1e-4, 1e-5, 1e-6, 1e-7))
    assert best < 1e-6


def test_weighted_kernel_is_equivariant_under_a_quarter_turn_of_map_and_weights(cross_cfg):
    """The regression-control identity, at the operator level.

    For a part mask invariant under 90 degrees and an angle set that is the
    multiples of 90, rotating the design map by a quarter turn and cyclically
    shifting the dwell vector rotates the averaged heating by the same quarter
    turn and changes nothing else. This is what makes equal dwells a meaningful
    prediction on a four-fold symmetric shape.
    """
    ang = np.array([0.0, 90.0, 180.0, 270.0])
    kd = DwellKernel.build(cross_cfg, angles=ang)
    case = kd.case0
    R90 = rotation_operator(case.part_mask.shape, 90.0)
    rng = np.random.default_rng(5)
    s = np.where(case.part_mask, rng.uniform(0.3, 1.0, case.part_mask.shape), 1.0)
    w = np.array([0.4, 0.3, 0.2, 0.1])

    kd.set_weights(w)
    Qa, _Qb = kd.averaged_Q(s)
    kd.set_weights(np.roll(w, -1))
    Qa2, _ = kd.averaged_Q(np.where(case.part_mask, R90.apply(s, outside=1.0), 1.0))

    inner = case.part_mask & np.roll(np.roll(case.part_mask, 1, 0), 1, 1)
    ref = R90.apply(Qa, outside=0.0)
    assert np.max(np.abs(Qa2 - ref)[inner]) < 1e-9 * max(np.max(np.abs(ref)), 1e-30)
