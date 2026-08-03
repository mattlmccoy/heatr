"""Sparse matrices for the `np.gradient(..., edge_order=1)` stencils.

The electric field is built by the production engine as
`dVdy, dVdx = np.gradient(V, y, x, edge_order=1)`. The adjoint needs the
transpose of that linear map. Building it explicitly (rather than hand-rolling
a transpose) means the same coefficients are used in both directions, and the
`test_gradops.py` dot-product test proves the transpose is the transpose.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sparse


def _gradient_1d_matrix(coord: np.ndarray) -> sparse.csr_matrix:
    """Matrix G with (G f)_i = np.gradient(f, coord, edge_order=1)_i."""
    n = len(coord)
    if n < 2:  # pragma: no cover
        raise ValueError("need at least 2 points")
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    h = np.diff(coord)
    # interior, second-order non-uniform stencil (numpy's formula)
    for i in range(1, n - 1):
        hs = h[i - 1]
        hd = h[i]
        rows += [i, i, i]
        cols += [i - 1, i, i + 1]
        vals += [-hd / (hs * (hs + hd)),
                 (hd - hs) / (hs * hd),
                 hs / (hd * (hs + hd))]
    # edge_order = 1 one-sided edges
    rows += [0, 0, n - 1, n - 1]
    cols += [0, 1, n - 2, n - 1]
    vals += [-1.0 / h[0], 1.0 / h[0], -1.0 / h[-1], 1.0 / h[-1]]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def gradient_matrices(x: np.ndarray, y: np.ndarray) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """(Gx, Gy) acting on a row-major flattened (ny, nx) field."""
    nx, ny = len(x), len(y)
    gx1 = _gradient_1d_matrix(np.asarray(x, dtype=float))
    gy1 = _gradient_1d_matrix(np.asarray(y, dtype=float))
    Gx = sparse.kron(sparse.eye(ny, format="csr"), gx1, format="csr")
    Gy = sparse.kron(gy1, sparse.eye(nx, format="csr"), format="csr")
    return Gx, Gy
