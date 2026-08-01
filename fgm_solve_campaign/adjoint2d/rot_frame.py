"""PART-frame <-> LAB-frame rotation as an assembled linear operator.

The continuous-rotation averaged-kernel solve rotates the design map into the
lab frame at every sampled turntable angle, solves the electro-quasi-static
(EQS) problem there, and rotates the resulting radio-frequency heating back
into the part frame before averaging. Both directions sit inside the chain
rule, so the rotation is assembled ONCE as a sparse matrix and forward and
adjoint share it. That is the same discipline the EQS operator follows:
assemble once, transpose exactly, never re-derive the adjoint by hand.

Convention, inherited and not re-derived
----------------------------------------
`test_orientation_map_rotation.py` proved on the real engine that the
production `geometry.part.rotation_deg = +90` equals `np.rot90(k=-1)` in array
coordinates with ZERO mismatched cells (T_shape, grid 120), which is
`scipy.ndimage.rotate(angle = -rotation_deg, reshape=False, order=1)`. This
module reproduces `scipy.ndimage.affine_transform` with that same matrix and
offset explicitly, so `rotation_operator(shape, deg).apply(m)` is
`rotate_sat_map(m, deg)` with an exact transpose attached.

Source-coordinate map, with `a = -deg` in radians and `c = (n - 1) / 2`:

    src_row = c0 + cos(a) * (row - c0) + sin(a) * (col - c1)
    src_col = c1 - sin(a) * (row - c0) + cos(a) * (col - c1)

Out-of-extent source points take the fill value `outside`, which is
`scipy.ndimage`'s `mode="constant"` rule. That fill enters as an AFFINE OFFSET
carrying no design sensitivity, so `apply_T` is the transpose of the linear
part alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse


@dataclass(frozen=True)
class RotationOperator:
    """Bilinear rotation `R` on a (ny, nx) grid, plus its exact transpose.

    Attributes
    ----------
    R        sparse (N, N) matrix, at most four non-zeros per row.
    deficit  (ny, nx) array of `1 - rowsum(R)`: the weight that fell outside
             the array extent. `apply(x, outside=c) = R x + c * deficit`.
    """

    shape: tuple[int, int]
    degrees: float
    R: sparse.csr_matrix
    deficit: np.ndarray

    def apply(self, x: np.ndarray, outside: float = 0.0) -> np.ndarray:
        v = self.R @ np.asarray(x, dtype=float).ravel()
        out = v.reshape(self.shape)
        if outside:
            out = out + float(outside) * self.deficit
        return out

    def apply_T(self, g: np.ndarray) -> np.ndarray:
        """Transpose of the LINEAR part; the `outside` offset has no design
        sensitivity, so it does not appear here."""
        return (self.R.T @ np.asarray(g, dtype=float).ravel()).reshape(self.shape)


def rotation_operator(shape: tuple[int, int], degrees: float) -> RotationOperator:
    """Assemble the rotation by the production part-rotation convention."""
    ny, nx = int(shape[0]), int(shape[1])
    a = math.radians(-float(degrees))
    ca, sa = math.cos(a), math.sin(a)
    # Exact trig at multiples of 90 degrees. Without this `sin(pi)` is -1.2e-16
    # rather than 0, which pushes the source point of an edge cell a hair
    # outside the extent and silently zeroes a whole row: the 180-degree case
    # then stops being a pixel permutation. Caught by the red test, not assumed.
    q = (-float(degrees)) / 90.0
    if abs(q - round(q)) < 1e-12:
        ca, sa = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][int(round(q)) % 4]
    c0, c1 = 0.5 * (ny - 1), 0.5 * (nx - 1)

    rr, cc = np.meshgrid(np.arange(ny, dtype=float),
                         np.arange(nx, dtype=float), indexing="ij")
    sr = c0 + ca * (rr - c0) + sa * (cc - c1)
    sc = c1 - sa * (rr - c0) + ca * (cc - c1)

    # scipy's mode="constant": a source point outside the input extent
    # [0, n-1] contributes the fill value in full, not a partial blend.
    inside = (sr >= 0.0) & (sr <= ny - 1) & (sc >= 0.0) & (sc <= nx - 1)
    # Snap coordinates that sit on a grid point to within round-off so the
    # 90/180/270-degree cases collapse to exact pixel permutations.
    sr = np.where(np.abs(sr - np.round(sr)) < 1e-12, np.round(sr), sr)
    sc = np.where(np.abs(sc - np.round(sc)) < 1e-12, np.round(sc), sc)

    r0 = np.clip(np.floor(sr), 0, ny - 1).astype(np.int64)
    c0i = np.clip(np.floor(sc), 0, nx - 1).astype(np.int64)
    r1 = np.minimum(r0 + 1, ny - 1)
    c1i = np.minimum(c0i + 1, nx - 1)
    fr = np.clip(sr - r0, 0.0, 1.0)
    fc = np.clip(sc - c0i, 0.0, 1.0)

    out_idx = (np.arange(ny * nx, dtype=np.int64))
    rows, cols, vals = [], [], []
    for ri, ci, w in ((r0, c0i, (1 - fr) * (1 - fc)),
                      (r0, c1i, (1 - fr) * fc),
                      (r1, c0i, fr * (1 - fc)),
                      (r1, c1i, fr * fc)):
        wgt = np.where(inside, w, 0.0).ravel()
        keep = wgt != 0.0
        rows.append(out_idx[keep])
        cols.append((ri * nx + ci).ravel()[keep])
        vals.append(wgt[keep])

    R = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ny * nx, ny * nx)).tocsr()
    R.sum_duplicates()
    rowsum = np.asarray(R.sum(axis=1)).ravel().reshape(ny, nx)
    return RotationOperator(shape=(ny, nx), degrees=float(degrees), R=R,
                            deficit=(1.0 - rowsum))


def averaging_angles(step_deg: float) -> np.ndarray:
    """Uniform sampling of a FULL turn, endpoint excluded.

    A full turn and not the shape's symmetry period: the solved dopant map is
    not constrained to carry the shape's symmetry, so `R_theta s` and
    `R_(theta + period) s` are different lab-frame maps even when the part mask
    is identical. Averaging over the period would only be exact for a
    symmetric map, and the whole point of the solve is that the map is free.
    """
    step = float(step_deg)
    if step <= 0.0:
        raise ValueError(f"step_deg must be positive, got {step_deg!r}")
    n = 360.0 / step
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"step_deg must divide 360 exactly, got {step_deg!r}")
    return np.arange(int(round(n)), dtype=float) * step
