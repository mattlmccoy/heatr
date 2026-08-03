"""dJ/dp_k from the same reverse sweep that produces dJ/ds.

Written RED first. Small-horizon central-difference checks; the full epsilon
sweep on the production horizon lives in `adjoint2d.gate_sched`.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import adjoint, forward as fwd, gradops, shape_objective as so
from adjoint2d.library_solve import shape_config
from adjoint2d.pins import build_case, load_cfg

N_STEPS = 240
N_SEG = 4


def _case(shape: str = "star"):
    return build_case(load_cfg(shape_config(shape)))


def _run(case, s, p):
    return fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                       n_steps=N_STEPS, p_seg=p, n_seg=N_SEG, p_horizon=N_STEPS)


def _J_fixed(case, s, p, index):
    tr = _run(case, s, p)
    return so.shape_J_and_seed(tr.T_at_end(min(index, tr.n_outer - 1)), case)[0]


def test_unit_schedule_gradient_reproduces_the_no_schedule_gradient_bit_for_bit():
    case = _case()
    ops = gradops.gradient_matrices(case.x, case.y)
    s = np.full(case.part_mask.shape, 0.9)
    a = fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None, n_steps=N_STEPS)
    b = _run(case, s, np.ones(N_SEG))
    idx = N_STEPS - 1
    ga = adjoint.gradient(case, s, a, {idx: so.shape_J_and_seed(a.T_at_end(idx), case)[1]},
                          grad_ops=ops)
    gb, gp = adjoint.gradient(case, s, b, {idx: so.shape_J_and_seed(b.T_at_end(idx), case)[1]},
                              grad_ops=ops, with_schedule=True)
    assert np.max(np.abs(ga)) > 0.0        # the check must not be vacuous
    assert np.array_equal(ga, gb)
    assert gp.shape == (N_SEG,)
    assert np.all(gp != 0.0)


def test_dJ_dp_matches_central_difference_at_the_nominal_schedule():
    case = _case()
    ops = gradops.gradient_matrices(case.x, case.y)
    s = np.full(case.part_mask.shape, 0.9)
    p0 = np.ones(N_SEG)
    idx = N_STEPS - 1
    tr = _run(case, s, p0)
    _J, seed = so.shape_J_and_seed(tr.T_at_end(idx), case)
    _gs, gp = adjoint.gradient(case, s, tr, {idx: seed}, grad_ops=ops, with_schedule=True)
    for k in range(N_SEG):
        # epsilon 1e-6: above about 1e-5 a global power perturbation drags a
        # large population of part cells across the bottom of the phase ramp at
        # once and the central difference measures the one-sided breakpoint
        # instead of the derivative. The full sweep is in `gate_sched`.
        eps = 1e-6
        pp, pm = p0.copy(), p0.copy()
        pp[k] += eps
        pm[k] -= eps
        fd = (_J_fixed(case, s, pp, idx) - _J_fixed(case, s, pm, idx)) / (2 * eps)
        assert fd == pytest.approx(gp[k], rel=2e-5, abs=1e-9), f"segment {k}"


def test_dJ_dp_is_zero_for_a_segment_the_march_never_reached():
    # The march is truncated at three quarters of the horizon; the last segment
    # cannot influence the objective and its gradient must be exactly zero.
    case = _case()
    ops = gradops.gradient_matrices(case.x, case.y)
    s = np.full(case.part_mask.shape, 0.9)
    half = 3 * N_STEPS // 4
    tr = fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                     n_steps=half, p_seg=np.ones(N_SEG), n_seg=N_SEG, p_horizon=N_STEPS)
    idx = half - 1
    _J, seed = so.shape_J_and_seed(tr.T_at_end(idx), case)
    _gs, gp = adjoint.gradient(case, s, tr, {idx: seed}, grad_ops=ops, with_schedule=True)
    assert gp[3] == 0.0
    assert gp[0] != 0.0
