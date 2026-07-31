"""Piecewise gates on the adjoint, so a whole-march finite-difference failure
can be bisected instead of guessed at.

1. the diffusion vector-Jacobian product is the exact transpose of the
   bilinear diffusion operator;
2. the electro-quasi-static (EQS) vector-Jacobian product reproduces a
   brute-force finite difference of the absorbed power field with respect to
   conductivity, on a grid small enough to do it exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import adjoint, eqs, forward as fwd, gradops
from adjoint2d.pins import EPS0
from adjoint2d.prod import rfam


def test_diffusion_vjp_is_the_exact_transpose():
    rng = np.random.default_rng(0)
    ny, nx = 7, 9
    dx = dy = 5.0e-4
    T = 20.0 + 200.0 * rng.random((ny, nx))
    k = 0.1 + 0.2 * rng.random((ny, nx))
    u = rng.standard_normal((ny, nx))
    dT = rng.standard_normal((ny, nx))
    dk = rng.standard_normal((ny, nx))
    lin = (rfam.diffusion_divergence(dT, k, dx, dy)
           + rfam.diffusion_divergence(T, dk, dx, dy))
    lhs = float(np.sum(u * lin))
    g_T, g_k = adjoint.diffusion_vjp(u, T, k, dx, dy)
    rhs = float(np.sum(g_T * dT) + np.sum(g_k * dk))
    assert abs(lhs - rhs) <= 1e-9 * max(1.0, abs(lhs))


class _MiniCase:
    """Enough of a Case for the EQS vector-Jacobian product."""

    def __init__(self, ny=11, nx=13):
        self.dx = self.dy = 5.0e-4
        self.x = np.arange(nx) * self.dx
        self.y = np.arange(ny) * self.dy
        self.part_mask = np.zeros((ny, nx), dtype=bool)
        self.part_mask[3:8, 4:9] = True
        self.doped_mask = self.part_mask.copy()
        self.elec_hi = np.zeros((ny, nx), dtype=bool)
        self.elec_lo = np.zeros((ny, nx), dtype=bool)
        self.elec_hi[-1, :] = True
        self.elec_lo[0, :] = True

        class P:
            omega = 2.0 * np.pi * 27.12e6
            power_factor = 1.0
            max_qrf = 1.0e11
            v_hi = 1500.0
            v_lo = 0.0
            zero_qrf_outside_doped = True
        self.pins = P()


def _state(case, sigma, eps_r):
    return fwd.solve_electric(case, sigma, eps_r)


def test_eqs_vjp_matches_brute_force_finite_difference():
    rng = np.random.default_rng(3)
    case = _MiniCase()
    ny, nx = case.part_mask.shape
    eps_r = np.where(case.part_mask, 20.0, 2.0)
    sigma0 = np.where(case.part_mask, 0.04, 1e-8)
    sigma0 = sigma0 * (1.0 + 0.20 * rng.standard_normal((ny, nx)) * case.part_mask)
    gQ = np.zeros((ny, nx))
    gQ[case.doped_mask] = rng.standard_normal(int(case.doped_mask.sum()))

    Gx, Gy = gradops.gradient_matrices(case.x, case.y)
    st = _state(case, sigma0, eps_r)
    dJ = adjoint.eqs_vjp(case, st, gQ, Gx, Gy)

    d = np.zeros((ny, nx))
    d[case.part_mask] = rng.standard_normal(int(case.part_mask.sum()))
    d *= 0.04
    ana = float(np.sum(dJ * d))

    best = np.inf
    for h in (1e-4, 1e-5, 1e-6, 1e-7):
        jp = float(np.sum(gQ * _state(case, sigma0 + h * d, eps_r).Qrf))
        jm = float(np.sum(gQ * _state(case, sigma0 - h * d, eps_r).Qrf))
        fd = (jp - jm) / (2 * h)
        best = min(best, abs(fd - ana) / max(abs(ana), 1e-30))
    assert best < 1e-7, f"best relative error {best:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:warnings"]))
