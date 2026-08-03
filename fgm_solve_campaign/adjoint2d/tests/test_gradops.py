"""RED-first: the sparse gradient stencils must reproduce np.gradient and be
exact transposes of themselves."""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import gradops


def test_matches_numpy_gradient():
    ny, nx = 7, 9
    x = np.linspace(-0.03, 0.03, nx)
    y = np.linspace(-0.03, 0.03, ny)
    rng = np.random.default_rng(0)
    V = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
    dVdy, dVdx = np.gradient(V, y, x, edge_order=1)
    Gx, Gy = gradops.gradient_matrices(x, y)
    ax = (Gx @ V.ravel()).reshape(ny, nx)
    ay = (Gy @ V.ravel()).reshape(ny, nx)
    scale = float(np.max(np.abs(dVdx)))
    assert np.max(np.abs(ax - dVdx)) < 1e-12 * scale
    assert np.max(np.abs(ay - dVdy)) < 1e-12 * scale


def test_dot_product_transpose_identity():
    ny, nx = 6, 5
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 2.0, ny)
    Gx, Gy = gradops.gradient_matrices(x, y)
    rng = np.random.default_rng(2)
    v = rng.standard_normal(ny * nx)
    u = rng.standard_normal(ny * nx)
    for G in (Gx, Gy):
        lhs = float(np.dot(G @ v, u))
        rhs = float(np.dot(v, G.T @ u))
        assert abs(lhs - rhs) <= 1e-12 * max(1.0, abs(lhs))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:warnings"]))
