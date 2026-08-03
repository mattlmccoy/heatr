"""Red-first tests for the SECONDARY arm: dwell schedule plus power schedule.

Power scheduling p(t) is deprioritized as an actuator (dwell time is the
dose-steering knob). It is kept here as ONE labelled counterexample check on
the cross, the single shape whose optimized power schedule contained a real
generator OFF period with a measured non-dose benefit
(`TEMPORAL_SCHEDULING_REPORT.md` Section 8). The question this module exists to
answer is narrow: does p(t) add anything ON TOP OF an optimized dwell schedule?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
for p in (str(HERE.parents[2]), str(HERE.parents[3])):
    if p not in sys.path:
        sys.path.insert(0, p)

from adjoint2d import dwell_power as dp                     # noqa: E402
from adjoint2d import gradops, library_solve as lib         # noqa: E402
from adjoint2d import schedule as sch                       # noqa: E402
from adjoint2d import shape_objective as so                 # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel              # noqa: E402
from adjoint2d.pins import load_cfg                         # noqa: E402


@pytest.fixture(scope="module")
def cross_cfg():
    return load_cfg(lib.shape_config("cross"))


def test_a_flat_unit_schedule_reproduces_the_unscheduled_kernel_bit_for_bit(cross_cfg):
    """FLAG-OFF IDENTITY. Adding a switched-off channel must change nothing."""
    kern = DwellKernel.build(cross_cfg, angles=np.array([0.0, 45.0, 90.0, 135.0]))
    case = kern.case0
    ops = gradops.gradient_matrices(case.x, case.y)
    s = np.where(case.part_mask, 0.8, 1.0)
    w = np.array([0.4, 0.2, 0.3, 0.1])
    kern.set_weights(w)

    ref = kern.forward(s, keep_checkpoints=True, n_steps=30)
    got = dp.scheduled_forward(kern, s, np.ones(30), n_steps=30, keep_checkpoints=True)
    assert np.array_equal(ref.T_final, got.T_final)
    assert np.array_equal(ref.rho_final, got.rho_final)

    _J, seed = so.shape_J_and_seed(ref.T_at_end(29), case)
    g_s0, g_w0 = kern.both_gradients(s, ref, {29: seed}, grad_ops=ops)
    g_s1, g_w1, g_p = dp.scheduled_gradients(kern, s, got, {29: seed}, np.ones(30),
                                             n_seg=4, grad_ops=ops)
    assert np.max(np.abs(g_s0 - g_s1)) < 1e-13 * max(np.max(np.abs(g_s0)), 1e-30)
    assert np.max(np.abs(g_w0 - g_w1)) < 1e-13 * max(np.max(np.abs(g_w0)), 1e-30)
    assert g_p.shape == (4,)


def test_the_power_gradient_matches_a_central_finite_difference(cross_cfg):
    kern = DwellKernel.build(cross_cfg, angles=np.array([0.0, 90.0]))
    case = kern.case0
    s = np.where(case.part_mask, 0.9, 1.0)
    kern.set_weights(np.array([0.6, 0.4]))
    n, n_seg, read = 60, 4, 59
    p0 = np.array([1.2, 0.7, 1.0, 0.4])

    def J_of(pv):
        tr = dp.scheduled_forward(kern, s, sch.expand_full(pv, n, n, n_seg), n_steps=n)
        return so.shape_J_and_seed(tr.T_at_end(read), case)[0]

    tr0 = dp.scheduled_forward(kern, s, sch.expand_full(p0, n, n, n_seg),
                               n_steps=n, keep_checkpoints=True)
    _J, seed = so.shape_J_and_seed(tr0.T_at_end(read), case)
    _gs, _gw, gp = dp.scheduled_gradients(kern, s, tr0, {read: seed},
                                          sch.expand_full(p0, n, n, n_seg), n_seg=n_seg)
    rng = np.random.default_rng(2)
    d = rng.standard_normal(n_seg)
    d /= np.linalg.norm(d)
    ana = float(np.dot(gp, d))
    best = min(abs((J_of(p0 + e * d) - J_of(p0 - e * d)) / (2 * e) - ana)
               / max(abs(ana), 1e-30) for e in (1e-4, 1e-5, 1e-6, 1e-7))
    assert best < 1e-6


def test_scheduled_forward_refuses_when_the_heating_cap_binds(cross_cfg):
    """The scaling convention holds only while the cap is slack; say so loudly.

    The dwell kernel applies the `max_qrf` cap PER POSITION in the lab frame,
    before the angle average, so a power scale applied to the averaged field
    cannot re-apply it. Rather than silently violate the convention the call
    refuses when any position is at the cap.
    """
    kern = DwellKernel.build(cross_cfg, angles=np.array([0.0, 90.0]))
    case = kern.case0
    s = np.where(case.part_mask, 1.0, 1.0)
    kern.set_weights(np.array([0.5, 0.5]))
    kern.case0.pins.__dict__["max_qrf"] = 1.0     # force the cap to bind
    try:
        with pytest.raises(ValueError):
            dp.scheduled_forward(kern, s, np.ones(10), n_steps=10)
    finally:
        del kern.case0.pins.__dict__["max_qrf"]
