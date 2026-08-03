"""Sub-cell VOLUME fill for the Phase C target indicator chi.

Pure numpy (no scipy, no matplotlib), so it runs in the dolfinx spike env as
well as the geo-prewarp venv, and so the 2-D lane's shared contract can be run
against it from either side.

THE CONVENTION IS NOT DEFINED HERE. It is the shared fill contract published by
the 2-D lane at `fgm_solve_campaign/adjoint2d/tests/fill_contract.py` (their
commit d881455, GEOMETRY_GENERALIZATION_REPORT.md), which this module is
written to satisfy and which `solve3d/tests/test_fill_contract.py` runs
verbatim against these functions. Restating it here rather than re-deriving it:

  * `fill(polygon, x, y)[j, i]` is the fraction of the cell CENTRED on
    `(x[i], y[j])` that lies inside the polygon. Cells are centred on the grid
    points, not corner-anchored.
  * inside-ness is the EVEN-ODD (crossing-number) rule, so winding is
    irrelevant and a reversed ring must give a bit-identical answer;
  * sub-cell sampling is a regular `n_sub` grid at cell offsets
    `linspace(-0.5 + 0.5/n, 0.5 - 0.5/n, n)` -- the production offsets of
    `rfam_eqs_coupled._subpixel_fill_fraction`. Deterministic, not Monte Carlo.
  * a sample exactly on the boundary is decided by the inside test, not
    special-cased; no cell is clipped analytically.

WHY A VOLUME FILL AT ALL, given that the Phase C solve mesh CONFORMS to the
part boundary and therefore has no partial cells (every tetrahedron is wholly
in or wholly out, asserted in the Phase B design test): because "chi is exact
here" is a property of THIS geometry on THIS mesh, not a property of the
method. The moment the target comes from an STL that the mesh does not conform
to, or the map is scored on a mesh built for a different shape, the fill is
what keeps the target grid-independent. It is built now, contract-checked now,
and its exactness on the conforming mesh is measured rather than assumed.

DEVIATION, recorded rather than silently forked (see `chi_on_cells`): the
contract's sampling convention is a regular n_sub-by-n_sub grid inside a
RECTANGULAR cell. A tetrahedron has no such grid. For per-tetrahedron fill this
module uses a symmetric barycentric lattice instead. On a conforming mesh the
choice is inert -- every sample of a cell lands on the same side, so the fill is
exactly 0 or 1 for ANY quadrature -- and `chi_on_cells` reports whether that
was in fact the case for the mesh it was handed.
"""
from __future__ import annotations

import numpy as np

N_SUB = 32          # matches the 2-D lane's CHI_N_SUB (their section 2)


# --------------------------------------------------------------------------- #
# Point-in-polygon: even-odd crossing number, vectorised, pure numpy
# --------------------------------------------------------------------------- #
def points_in_polygon(poly: np.ndarray, px: np.ndarray,
                      py: np.ndarray) -> np.ndarray:
    """Even-odd (crossing-number) test for many points against a closed ring.

    Winding-independent by construction, which is what the contract requires:
    a crossing count's parity does not depend on the direction of travel."""
    poly = np.asarray(poly, dtype=float)
    px = np.asarray(px, dtype=float).ravel()
    py = np.asarray(py, dtype=float).ravel()
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    inside = np.zeros(px.size, dtype=bool)
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        # the horizontal ray from (px, py) toward +x crosses this edge iff the
        # edge straddles py in the half-open sense and the crossing is to the
        # right. The half-open comparison is what makes a vertex hit count once.
        cond = (ay > py) != (by > py)
        if not np.any(cond):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = ax + (py - ay) * (bx - ax) / np.where(by - ay == 0.0,
                                                         np.nan, by - ay)
        inside ^= cond & (px < xint)
    return inside


def _dist_to_boundary(poly: np.ndarray, px: np.ndarray,
                      py: np.ndarray) -> np.ndarray:
    """Minimum distance from each point to the polygon's edge set."""
    poly = np.asarray(poly, dtype=float)
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    L2 = np.where(L2 == 0.0, 1.0, L2)
    P = np.column_stack([np.asarray(px, float).ravel(),
                         np.asarray(py, float).ravel()])
    best = np.full(P.shape[0], np.inf)
    for k in range(a.shape[0]):
        d = P - a[k]
        t = np.clip((d @ ab[k]) / L2[k], 0.0, 1.0)
        proj = a[k] + t[:, None] * ab[k]
        best = np.minimum(best, np.hypot(*(P - proj).T))
    return best


def _offsets(n: int) -> np.ndarray:
    """The production sub-cell offsets, in units of the cell size."""
    return np.linspace(-0.5 + 0.5 / n, 0.5 - 0.5 / n, int(n))


# --------------------------------------------------------------------------- #
# 2-D area fill (the contract's fill_fn signature)
# --------------------------------------------------------------------------- #
def area_fill_2d(poly: np.ndarray, x: np.ndarray, y: np.ndarray,
                 n_sub: int = N_SUB) -> np.ndarray:
    """Sub-cell AREA fill; returns shape (len(y), len(x)) per the contract.

    BAND RESTRICTED for cost, which is the 2-D lane's own documented
    optimization (their section 2): only cells whose centre lies within half a
    cell diagonal of the polygon boundary can be partial, so only those are
    supersampled and the rest take their centre value. Interior and exterior
    cells are exactly 1 and 0 either way, so the restriction is not an
    approximation -- it is the same number computed without wasting samples on
    cells whose answer is already decided.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y, indexing="xy")            # [j, i] per the contract
    centre_in = points_in_polygon(poly, X.ravel(), Y.ravel()).reshape(X.shape)
    out = centre_in.astype(float)

    half_diag = 0.5 * float(np.hypot(dx, dy))
    dist = _dist_to_boundary(poly, X.ravel(), Y.ravel()).reshape(X.shape)
    band = dist <= half_diag
    if not np.any(band):
        return out

    off = _offsets(n_sub)
    ox = (off * dx)[None, :, None] * np.ones((1, 1, n_sub))
    oy = (off * dy)[None, None, :] * np.ones((1, n_sub, 1))
    bx = X[band][:, None, None] + ox
    by = Y[band][:, None, None] + oy
    ins = points_in_polygon(poly, bx.ravel(), by.ravel())
    out[band] = ins.reshape(-1, n_sub * n_sub).mean(axis=1)
    return out


# --------------------------------------------------------------------------- #
# 3-D volume fill
# --------------------------------------------------------------------------- #
def volume_fill_3d(poly: np.ndarray, x: np.ndarray, y: np.ndarray,
                   z_centers: np.ndarray, dz: float,
                   z_lo: float, z_hi: float,
                   n_sub: int = N_SUB, n_sub_z: int | None = None) -> np.ndarray:
    """Sub-cell VOLUME fill of an extruded prism, shape (len(z), len(y), len(x)).

    The prism is `poly` extruded over [z_lo, z_hi]. The in-plane sampling is the
    contract's; the z sampling uses the same offset rule, so a cell straddling
    an end plane fills the true fraction of its z-extent rather than snapping."""
    z_centers = np.atleast_1d(np.asarray(z_centers, dtype=float))
    nz = int(n_sub_z or n_sub)
    area = area_fill_2d(poly, x, y, n_sub=n_sub)             # (ny, nx)
    offz = _offsets(nz)
    out = np.empty((z_centers.size, area.shape[0], area.shape[1]), dtype=float)
    for k, zc in enumerate(z_centers):
        zs = zc + offz * float(dz)
        frac_z = float(np.mean((zs >= z_lo) & (zs <= z_hi)))
        out[k] = area * frac_z
    return out


def volume_fill_slice(poly: np.ndarray, x: np.ndarray, y: np.ndarray,
                      n_sub: int = N_SUB) -> np.ndarray:
    """THE CONTRACT'S 3-D ENTRY POINT.

    The volume fill of one z-slice lying strictly inside the extrusion. Every z
    sample is inside, so the z factor is exactly 1 and the result must equal the
    area fill cell for cell -- which is what
    `fill_contract.assert_extrusion_slice_reduction` asserts, at atol 1e-12."""
    return volume_fill_3d(poly, x, y, np.array([0.0]), dz=1e-4,
                          z_lo=-1.0, z_hi=1.0, n_sub=n_sub)[0]


# --------------------------------------------------------------------------- #
# Per-cell fill on an unstructured mesh
# --------------------------------------------------------------------------- #
def _tet_lattice(order: int) -> np.ndarray:
    """Symmetric barycentric lattice inside the reference tetrahedron.

    The contract's regular n_sub-by-n_sub CELL grid has no tetrahedral
    analogue; this is the documented substitute. See the module docstring."""
    pts = []
    n = int(order)
    for i in range(n):
        for j in range(n - i):
            for k in range(n - i - j):
                l = n - 1 - i - j - k
                pts.append(((i + 0.25) / n, (j + 0.25) / n, (k + 0.25) / n))
    return np.asarray(pts, dtype=float)


def chi_on_cells(cell_vertices: np.ndarray, poly: np.ndarray,
                 z_lo: float, z_hi: float, order: int = 6) -> dict:
    """Volume fill per tetrahedron for an extruded-prism target.

    `cell_vertices` is (ncell, 4, 3). Returns the fill plus the MEASURED
    statement of whether the mesh conformed (every cell exactly 0 or 1), so the
    quadrature deviation recorded in the module docstring can be shown to be
    inert instead of assumed to be."""
    V = np.asarray(cell_vertices, dtype=float)
    bary = _tet_lattice(order)
    w = np.concatenate([1.0 - bary.sum(axis=1, keepdims=True), bary], axis=1)
    pts = np.einsum("qk,ckd->cqd", w, V)                     # (ncell, nq, 3)
    flat = pts.reshape(-1, 3)
    ins = points_in_polygon(poly, flat[:, 0], flat[:, 1])
    ins &= (flat[:, 2] >= z_lo) & (flat[:, 2] <= z_hi)
    frac = ins.reshape(V.shape[0], -1).mean(axis=1)
    exact = bool(np.all((frac == 0.0) | (frac == 1.0)))
    return {"fill": frac, "n_quadrature_points": int(bary.shape[0]),
            "mesh_conformed": exact,
            "n_partial_cells": int(np.count_nonzero((frac > 0.0) & (frac < 1.0))),
            "quadrature": "symmetric barycentric lattice (documented deviation "
                          "from the contract's rectangular n_sub grid, which "
                          "has no tetrahedral analogue)"}
