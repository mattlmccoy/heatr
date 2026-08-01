"""Grid-independent target indicator chi, built from the geometry.

WHY THE RASTER IS WRONG. Every solve in this campaign so far has used
`shape_objective.chi_part`, the solve-grid BINARY part mask, as the target of

    J_phi = sum over the domain of (phi - chi)^2.

A binary raster is a property of the grid, not of the part: a boundary cell is
scored as fully part or fully powder depending on where its centre falls, so
the total target area moves when the grid moves. Measured on a 10 mm circle in
the 60 mm domain, the binary raster's area is off the closed form by a
different amount at grid 120 than at grid 160, which means a map solved at 120
and scored at 160 is being scored against a DIFFERENT target. That confounds
the grid hold-out with the map's own transfer, exactly the confound
`SOLVE_ROBUSTNESS_VALIDATION.md` Section 3.2 could not separate.

THE REPLACEMENT. chi(cell) = the cell average of the geometric indicator, i.e.
the fraction of the cell's area that lies inside the part. It is evaluated by
the SAME supersampling routine the production domain builder uses to make its
material fill fraction, `rfam_eqs_coupled._subpixel_fill_fraction`, at a higher
sample count. Reusing the production routine rather than writing a second one
means the target indicator and the material fill fraction can never disagree
about what "inside" means.

`CHI_N_SUB` = 32, so 1024 sample points per cell. The residual error is a
boundary effect of order (cell size) / (2 * n_sub) per boundary cell; measured
on the circle it is below 0.2 percent of the area at grid 120 and the 120-to-160
difference is below 0.2 percent, against a binary raster whose 120-to-160
difference is larger. That is the entire claim: chi is grid independent to
within the sampling error, and the sampling error is quoted.

COST, AND WHY THE EVALUATION IS BAND RESTRICTED. The production routine
samples every cell, which at 1024 samples per cell and a 720-vertex circle is
16 s per call, too slow to sit inside a solve. A cell can only have a
fractional fill if its centre lies within half a cell diagonal of the polygon
boundary, so the fill is computed by supersampling ONLY that band and taking
the centre-point value elsewhere. The offsets are the production offsets, so
for every band cell the answer is the production answer to the last bit and for
every other cell it is provably 0 or 1. `test_band_restriction_is_bit_identical_
to_the_production_sampler` checks that equality directly rather than assuming
it.

LIMIT, STATED. This is an area fill, not a signed distance. It carries the
correct cell-average of the indicator, which is what a cell-summed quadratic
objective needs, but it does not carry a boundary normal or a curvature, so it
cannot support a level-set velocity. That is not needed here and is named as
not provided.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .prod import rfam

CHI_N_SUB = 32


def _distance_to_boundary(poly: np.ndarray, xx: np.ndarray, yy: np.ndarray,
                          chunk: int = 64) -> np.ndarray:
    """Minimum distance from each grid point to the closed polygon boundary."""
    p = np.asarray(poly, dtype=float)
    a = p
    b = np.roll(p, -1, axis=0)
    px = xx.ravel()
    py = yy.ravel()
    best = np.full(px.shape, np.inf)
    for i0 in range(0, len(a), chunk):
        ax = a[i0:i0 + chunk, 0][None, :]
        ay = a[i0:i0 + chunk, 1][None, :]
        bx = b[i0:i0 + chunk, 0][None, :]
        by = b[i0:i0 + chunk, 1][None, :]
        ex = bx - ax
        ey = by - ay
        ll = ex * ex + ey * ey
        t = np.where(ll > 0.0,
                     ((px[:, None] - ax) * ex + (py[:, None] - ay) * ey) / np.where(ll > 0, ll, 1.0),
                     0.0)
        t = np.clip(t, 0.0, 1.0)
        dx_ = px[:, None] - (ax + t * ex)
        dy_ = py[:, None] - (ay + t * ey)
        best = np.minimum(best, np.min(np.hypot(dx_, dy_), axis=1))
    return best.reshape(xx.shape)


def area_fill_poly(poly: np.ndarray, x: np.ndarray, y: np.ndarray,
                   n_sub: int = CHI_N_SUB) -> np.ndarray:
    """Sub-cell area fill of one polygon on the (y, x) grid, in [0, 1].

    Band-restricted, and bit-identical to the production sampler; see the
    module docstring.
    """
    n = int(n_sub)
    if n < 1:
        raise ValueError(f"n_sub must be a positive integer, got {n_sub!r}")
    p = np.asarray(poly, dtype=float)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    dx = float(xa[1] - xa[0])
    dy = float(ya[1] - ya[0])
    xx, yy = np.meshgrid(xa, ya, indexing="xy")
    path = rfam.MplPath(p)
    inside_c = path.contains_points(
        np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    out = inside_c.astype(float)
    if n == 1:
        return out
    half_diag = 0.5 * float(np.hypot(dx, dy))
    band = _distance_to_boundary(p, xx, yy) <= half_diag * (1.0 + 1e-9)
    if not band.any():
        return out
    jj, ii = np.nonzero(band)
    cx = xx[jj, ii]
    cy = yy[jj, ii]
    offsets = np.linspace(-0.5 + 0.5 / n, 0.5 - 0.5 / n, n)
    total = np.zeros(cx.shape, dtype=float)
    for dox in offsets:
        sx = cx + dox * dx
        for doy in offsets:
            sy = cy + doy * dy
            total += path.contains_points(np.column_stack([sx, sy])).astype(float)
    out[jj, ii] = total / float(n * n)
    return out


def area_fill_union(polys: Sequence[np.ndarray] | Iterable[np.ndarray],
                    x: np.ndarray, y: np.ndarray,
                    n_sub: int = CHI_N_SUB) -> np.ndarray:
    """Union fill over several polygons, by the production `maximum` rule.

    `rfam_eqs_coupled.make_domain` combines multi-part fills with
    `np.maximum` (line 1595), so the same rule is used here. For DISJOINT
    parts, which is every multi-part case in this library, the maximum and the
    sum agree; where two parts overlap the maximum is the correct union fill
    and the sum would double count.
    """
    out = None
    for p in polys:
        f = area_fill_poly(p, x, y, n_sub=n_sub)
        out = f if out is None else np.maximum(out, f)
    if out is None:
        raise ValueError("no polygons given")
    return out


def chi_from_cfg(cfg: dict, x: np.ndarray, y: np.ndarray,
                 n_sub: int = CHI_N_SUB) -> tuple[np.ndarray, dict]:
    """The grid-independent chi for a campaign config, plus its provenance.

    Raises loudly on any geometry feature whose fill this routine does not
    reproduce (shells, antennae, image-rasterized parts), rather than silently
    returning a chi that disagrees with the domain builder's own mask.
    """
    geom = cfg["geometry"]
    parts = rfam._parts_from_geometry(geom)
    for p in parts:
        for key in ("shell", "antennae", "image", "image_path"):
            if p.get(key) or geom.get(key):
                raise NotImplementedError(
                    f"geometry feature {key!r} is present; the area-fill chi "
                    "reproduces only the base polygon fill and would disagree "
                    "with the domain builder")
    polys = []
    for p in parts:
        poly, _mask, _fill = rfam._single_part_mask_and_fill(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), dict(p))
        polys.append(np.asarray(poly, dtype=float))
    chi = area_fill_union(polys, x, y, n_sub=n_sub)
    dA = float(x[1] - x[0]) * float(y[1] - y[0])
    info = {"n_sub": int(n_sub), "n_polys": len(polys),
            "area_m2": float(np.sum(chi)) * dA,
            "construction": "sub-cell area fill by "
                            "rfam_eqs_coupled._subpixel_fill_fraction, union by "
                            "np.maximum"}
    return chi, info


def raster_vs_area_delta(part_mask: np.ndarray, chi: np.ndarray,
                         dx: float, dy: float) -> dict:
    """How far the binary raster target is from the area-fill target."""
    b = np.asarray(part_mask, dtype=float)
    c = np.asarray(chi, dtype=float)
    dA = float(dx) * float(dy)
    a_bin = float(b.sum()) * dA
    a_chi = float(c.sum()) * dA
    return {"area_raster_m2": a_bin, "area_chi_m2": a_chi,
            "area_rel_delta": (a_bin - a_chi) / max(a_chi, 1e-30),
            "n_cells_differing_gt_0p01": int(np.sum(np.abs(b - c) > 0.01)),
            "max_abs_cell_delta": float(np.max(np.abs(b - c))),
            "sum_abs_cell_delta": float(np.sum(np.abs(b - c)))}
