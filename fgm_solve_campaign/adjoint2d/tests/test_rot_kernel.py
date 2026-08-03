"""Red-first tests for the rotationally-averaged heating kernel.

The strongest available gate on the whole construction is a DEGENERACY gate:
with a single averaging angle at zero degrees the rotation operators are the
identity and the averaged-kernel forward must reduce, bit for bit, to the
ordinary forward `adjoint2d.forward.forward`, and its adjoint to
`adjoint2d.adjoint.gradient`. Anything wired backwards (a transpose, a scale, a
frame) shows up there before a single expensive solve is run.

These run on a small synthetic case rather than a production config so they
stay fast; `gate_rot.py` runs the finite-difference gate on the real geometry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "fgm_solve_campaign") not in sys.path:
    sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

CFG = REPO / "outputs_eqs/fgm_calibrated_control/configs/cross_m0p0500.yaml"
N_STEPS = 12


def _case():
    from adjoint2d.pins import build_case, load_cfg
    cfg = load_cfg(CFG)
    return build_case(cfg)


def _s0(case, seed: int = 3):
    rng = np.random.default_rng(seed)
    s = np.ones(case.part_mask.shape)
    s[case.part_mask] = rng.uniform(0.4, 1.0, size=int(case.part_mask.sum()))
    return s


def test_single_angle_zero_reproduces_the_plain_forward():
    from adjoint2d import forward as fwd
    from adjoint2d.rot_kernel import AveragedKernel
    case = _case()
    s = _s0(case)
    kern = AveragedKernel.build(case.cfg, angles=np.array([0.0]))
    tr_ref = fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                         n_steps=N_STEPS)
    tr = kern.forward(s, keep_checkpoints=True, n_steps=N_STEPS)
    assert tr.n_outer == tr_ref.n_outer
    assert np.array_equal(tr.T_final, tr_ref.T_final)
    assert np.array_equal(tr.rho_final, tr_ref.rho_final)
    assert tr.P_abs_B == pytest.approx(tr_ref.P_abs_B, rel=0, abs=0)


def test_single_angle_zero_reproduces_the_plain_gradient():
    from adjoint2d import adjoint, forward as fwd, gradops
    from adjoint2d import shape_objective as so
    from adjoint2d.rot_kernel import AveragedKernel
    case = _case()
    s = _s0(case)
    ops = gradops.gradient_matrices(case.x, case.y)
    kern = AveragedKernel.build(case.cfg, angles=np.array([0.0]))

    tr_ref = fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                         n_steps=N_STEPS)
    i = N_STEPS - 1
    _J, seed = so.shape_J_and_seed(tr_ref.T_at_end(i), case)
    g_ref = adjoint.gradient(case, s, tr_ref, {i: seed}, grad_ops=ops)

    tr = kern.forward(s, keep_checkpoints=True, n_steps=N_STEPS)
    g = kern.gradient(s, tr, {i: seed}, grad_ops=ops)
    num = float(np.max(np.abs(g - g_ref)))
    den = max(float(np.max(np.abs(g_ref))), 1e-30)
    assert num / den < 1e-12, (num, den)


def test_averaged_kernel_is_the_mean_of_the_per_angle_kernels():
    """Q_avg must literally be the arithmetic mean of the back-rotated
    per-angle heating patterns, not a re-solve at a mean conductivity."""
    from adjoint2d import forward as fwd
    from adjoint2d.pins import build_case
    from adjoint2d.rot_frame import rotation_operator
    from adjoint2d.rot_kernel import AveragedKernel
    import copy
    case = _case()
    s = _s0(case)
    angles = np.array([0.0, 90.0])
    kern = AveragedKernel.build(case.cfg, angles=angles)
    Qa, Qb = kern.averaged_Q(s)

    acc_a = np.zeros(case.part_mask.shape)
    acc_b = np.zeros(case.part_mask.shape)
    for th in angles:
        cfg = copy.deepcopy(case.cfg)
        cfg["geometry"]["part"]["rotation_deg"] = float(th)
        cj = build_case(cfg)
        Rl = rotation_operator(case.part_mask.shape, float(th))
        Rb = rotation_operator(case.part_mask.shape, -float(th))
        s_lab = np.where(cj.part_mask, Rl.apply(s, outside=1.0), 1.0)
        sa = fwd.solve_electric(cj, fwd.sigma_state_a(cj, s_lab, False),
                                fwd.eps_field(cj))
        sb = fwd.solve_electric(cj, fwd.sigma_state_b(cj, s_lab)[0],
                                fwd.eps_field(cj))
        acc_a += Rb.apply(sa.Qrf, outside=0.0) / len(angles)
        acc_b += Rb.apply(sb.Qrf, outside=0.0) / len(angles)
    assert np.allclose(Qa, acc_a, rtol=0, atol=1e-9 * max(acc_a.max(), 1.0))
    assert np.allclose(Qb, acc_b, rtol=0, atol=1e-9 * max(acc_b.max(), 1.0))


def test_uniform_map_kernel_of_a_symmetric_shape_is_nearly_symmetric():
    """Sanity on the physics, not just the plumbing: averaging a four-fold
    symmetric part over a full turn must wash out the electrode axis, so the
    averaged kernel of the cross under a uniform map is close to its own
    90-degree rotation."""
    from adjoint2d.rot_frame import rotation_operator
    from adjoint2d.rot_kernel import AveragedKernel
    case = _case()
    kern = AveragedKernel.build(case.cfg, angles=np.arange(0.0, 360.0, 30.0))
    _Qa, Qb = kern.averaged_Q(np.ones(case.part_mask.shape))
    R90 = rotation_operator(case.part_mask.shape, 90.0)
    pm = case.part_mask
    d = np.abs(R90.apply(Qb, outside=0.0) - Qb)[pm]
    assert float(np.mean(d)) / max(float(np.mean(Qb[pm])), 1e-30) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
