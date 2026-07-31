"""RED-first gate: the vectorized EQS assembly must reproduce the production
`rfam_eqs_coupled.solve_eqs_complex` potential BIT-IDENTICALLY.

EQS = electro-quasi-static. The adjoint reuses the assembled matrix, so the
assembly must be the same discrete operator the forward solved, not a look-alike.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import eqs
from adjoint2d.prod import rfam


def _small_case(seed: int = 0):
    rng = np.random.default_rng(seed)
    ny, nx = 9, 11
    sigma = 1e-8 + 0.05 * rng.random((ny, nx))
    eps_r = 2.0 + 18.0 * rng.random((ny, nx))
    omega = 2.0 * np.pi * 27.12e6
    gamma = sigma + 1j * omega * 8.8541878128e-12 * eps_r
    elec_hi = np.zeros((ny, nx), dtype=bool)
    elec_lo = np.zeros((ny, nx), dtype=bool)
    elec_hi[-1, :] = True
    elec_lo[0, :] = True
    return gamma, elec_hi, elec_lo, 1234.5, 0.0, 5.0e-4, 5.0e-4


def test_assembled_solve_is_bit_identical_to_production():
    gamma, hi, lo, v_hi, v_lo, dx, dy = _small_case()
    V_ref = rfam.solve_eqs_complex(
        gamma=gamma, elec_hi=hi, elec_lo=lo, v_hi=v_hi, v_lo=v_lo,
        dx=dx, dy=dy, steps=3000, tol=1e-9,
    )
    op = eqs.assemble(gamma, hi, lo, v_hi, v_lo, dx, dy)
    V_new = op.solve()
    assert np.max(np.abs(V_new - V_ref)) == 0.0


def test_residual_of_assembled_operator_is_tiny():
    gamma, hi, lo, v_hi, v_lo, dx, dy = _small_case(3)
    op = eqs.assemble(gamma, hi, lo, v_hi, v_lo, dx, dy)
    V = op.solve()
    r = op.A @ V.ravel() - op.b
    assert np.max(np.abs(r)) < 1e-9 * max(1.0, float(np.max(np.abs(op.b))))


def test_transpose_solve_matches_dense_transpose():
    gamma, hi, lo, v_hi, v_lo, dx, dy = _small_case(7)
    op = eqs.assemble(gamma, hi, lo, v_hi, v_lo, dx, dy)
    rng = np.random.default_rng(11)
    p = rng.random(op.A.shape[0]) + 1j * rng.random(op.A.shape[0])
    lam = op.solve_transpose(p)
    resid = op.A.T @ lam - p
    assert np.max(np.abs(resid)) < 1e-8 * max(1.0, float(np.max(np.abs(p))))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
