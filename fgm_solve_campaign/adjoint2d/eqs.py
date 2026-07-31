"""Vectorized assembly of the 2-D electro-quasi-static (EQS) operator.

The production engine assembles `div(gamma grad V) = 0` with harmonic face
averaging, Dirichlet electrodes and zero-normal-flux outer walls in a Python
double loop (`rfam_eqs_coupled.solve_eqs_complex`). This module builds the
identical sparse matrix in vectorized form, keeps it, and exposes both the
forward solve and the transposed solve so the adjoint reuses the SAME discrete
operator and the SAME factorization.

Sign and transpose convention
-----------------------------
For every free (non-electrode) cell k the assembled residual is

    R_k = sum_{n in nb(k)} w_{k,n} * (V_k - V_n),      w_{k,n} = gf_{k,n} / h^2

with gf the harmonic face conductance 2 g_k g_n / (g_k + g_n). Electrode rows
are the identity with the prescribed potential on the right-hand side, so the
solved V equals the prescribed value there exactly and the residual above may
be written with V_n rather than a separately prescribed boundary value.

The adjoint of a real functional J of the complex potential needs
`A^T lambda = p` (plain transpose, not conjugate transpose) because the
sensitivity is written as dJ = 2 Re(p^T dV). See `adjoint.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

_DIAG_REG = 1e-18


@dataclass
class EqsOperator:
    """Assembled EQS system plus the pieces the adjoint needs."""

    A: sparse.csr_matrix
    b: np.ndarray
    shape: tuple[int, int]
    gamma: np.ndarray
    free: np.ndarray            # (ny, nx) bool, True where the row is a PDE row
    inv_h2: tuple[float, float]  # (inv_dx2, inv_dy2)
    _lu: object = None

    def solve(self) -> np.ndarray:
        """Forward solve, using the same call the production engine uses."""
        v_flat = spla.spsolve(self.A, self.b)
        if np.any(~np.isfinite(v_flat)):
            raise RuntimeError("EQS solve produced non-finite values.")
        return v_flat.reshape(self.shape)

    def _lu_factor(self):
        if self._lu is None:
            self._lu = spla.splu(self.A.tocsc())
        return self._lu

    def solve_transpose(self, p: np.ndarray) -> np.ndarray:
        """Solve A^T lambda = p, reusing the forward factorization."""
        lu = self._lu_factor()
        return lu.solve(np.asarray(p, dtype=np.complex128).ravel(), trans="T")


def cmul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Naive complex multiply (a*c - b*d, a*d + b*c), each product rounded once.

    NumPy's complex128 ARRAY multiply loop and its SCALAR multiply path do not
    agree bit-for-bit (measured: 2363 of 5000 random pairs differ by one unit in
    the last place, `scratch_probe3.py`). The production assembly loop uses the
    scalar path. Reproducing it exactly is what makes the L0 bit-identity gate
    reachable, so the naive formula is used deliberately, not for speed.
    """
    xr, xi = x.real, x.imag
    yr, yi = y.real, y.imag
    return (xr * yr - xi * yi) + 1j * (xr * yi + xi * yr)


def _face_conductance(g0: np.ndarray, g1: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Harmonic face conductance, replicating the production branch logic."""
    den = g0 + g1
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gf = cmul(2.0 * g0, g1) / den
    small = np.abs(den) <= 1e-30
    if np.any(small):
        gf = np.where(small, 0.5 * (g0 + g1), gf)
    bad = ~(np.isfinite(gf.real) & np.isfinite(gf.imag))
    if np.any(bad):
        gf = np.where(bad, 1e-9 + 0.0j, gf)
    return np.where(present, gf, 0.0 + 0.0j)


def _shifted(arr: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Neighbor value in `direction` and a presence mask (False on the wall)."""
    out = np.zeros_like(arr)
    present = np.zeros(arr.shape, dtype=bool)
    if direction == "up":       # neighbor (i-1, j)
        out[1:, :] = arr[:-1, :]
        present[1:, :] = True
    elif direction == "down":   # neighbor (i+1, j)
        out[:-1, :] = arr[1:, :]
        present[:-1, :] = True
    elif direction == "left":   # neighbor (i, j-1)
        out[:, 1:] = arr[:, :-1]
        present[:, 1:] = True
    elif direction == "right":  # neighbor (i, j+1)
        out[:, :-1] = arr[:, 1:]
        present[:, :-1] = True
    else:  # pragma: no cover
        raise ValueError(direction)
    return out, present


# Neighbor order MUST match the production loop: up, down, left, right.
DIRECTIONS = ("up", "down", "left", "right")
_IS_VERTICAL = {"up": True, "down": True, "left": False, "right": False}
_OFFSET = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def face_weights(
    gamma: np.ndarray,
    dx: float,
    dy: float,
) -> dict[str, np.ndarray]:
    """Per-cell, per-direction face weight w = gf / h^2 (zero on the walls)."""
    inv_dx2 = 1.0 / max(dx * dx, 1e-24)
    inv_dy2 = 1.0 / max(dy * dy, 1e-24)
    g = np.asarray(gamma, dtype=np.complex128)
    out: dict[str, np.ndarray] = {}
    for d in DIRECTIONS:
        g1, present = _shifted(g, d)
        gf = _face_conductance(g, g1, present)
        out[d] = gf * (inv_dy2 if _IS_VERTICAL[d] else inv_dx2)
    return out


def assemble(
    gamma: np.ndarray,
    elec_hi: np.ndarray,
    elec_lo: np.ndarray,
    v_hi: float,
    v_lo: float,
    dx: float,
    dy: float,
) -> EqsOperator:
    ny, nx = gamma.shape
    n = ny * nx
    hi = np.asarray(elec_hi, dtype=bool)
    lo = np.asarray(elec_lo, dtype=bool)
    dirich = hi | lo
    free = ~dirich

    w = face_weights(gamma, dx, dy)

    # Diagonal: accumulate in the production order (up, down, left, right).
    diag = np.zeros((ny, nx), dtype=np.complex128)
    rhs = np.zeros((ny, nx), dtype=np.complex128)
    v_prescribed = np.where(hi, complex(v_hi), np.where(lo, complex(v_lo), 0.0 + 0.0j))
    for d in DIRECTIONS:
        diag = diag + w[d]
        nb_dirich, _ = _shifted(dirich.astype(np.float64), d)
        nb_v, _ = _shifted(v_prescribed, d)
        rhs = rhs + np.where(nb_dirich > 0.0, w[d] * nb_v, 0.0 + 0.0j)

    idx = np.arange(n).reshape(ny, nx)
    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    vals_l: list[np.ndarray] = []

    # Off-diagonals: only for free rows with a free neighbor.
    for d in DIRECTIONS:
        di, dj = _OFFSET[d]
        src = np.zeros((ny, nx), dtype=bool)
        i0, i1 = (max(0, -di), ny - max(0, di))
        j0, j1 = (max(0, -dj), nx - max(0, dj))
        src[i0:i1, j0:j1] = True
        nb_idx = np.full((ny, nx), -1, dtype=np.int64)
        nb_idx[i0:i1, j0:j1] = idx[i0 + di:i1 + di, j0 + dj:j1 + dj]
        nb_free = np.zeros((ny, nx), dtype=bool)
        nb_free[i0:i1, j0:j1] = free[i0 + di:i1 + di, j0 + dj:j1 + dj]
        sel = src & free & nb_free
        rows_l.append(idx[sel])
        cols_l.append(nb_idx[sel])
        vals_l.append(-w[d][sel])

    rows_l.append(idx[free])
    cols_l.append(idx[free])
    vals_l.append(diag[free])
    rows_l.append(idx[dirich])
    cols_l.append(idx[dirich])
    vals_l.append(np.full(int(dirich.sum()), 1.0 + 0.0j))

    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A = A + sparse.eye(n, format="csr", dtype=np.complex128) * (_DIAG_REG + 0.0j)

    b = np.where(free, rhs, v_prescribed).ravel().astype(np.complex128)

    return EqsOperator(
        A=A,
        b=b,
        shape=(ny, nx),
        gamma=np.asarray(gamma, dtype=np.complex128),
        free=free,
        inv_h2=(1.0 / max(dx * dx, 1e-24), 1.0 / max(dy * dy, 1e-24)),
    )
