#!/usr/bin/env python3
"""The two variants that decide the cross's rotating-arm grid scatter.

`ROTATING_GRID_LADDER_REPORT.md` Section 4.5 offers ONE candidate mechanism for
the cross's non-converging rotating intersection-over-union sequence, labelled
ASSUMED: the whole-cell rounding of the cross's two rectilinear boundaries,
which correlates with the scatter at Pearson r = +0.773 over the seven ladder
grids. Section 10 names the two experiments that decide it. This module holds
the pieces both of them need, so the ladder driver is EXTENDED rather than
forked.

VARIANT A, SNAPPED GEOMETRY. At each grid the cross's limb half-length and arm
half-width are moved to the nearest whole multiple of the cell size. Two
consequences, and the second is the one that makes the variant a clean test:

  1. the part mask realizes both boundaries exactly, so the rounding error the
     correlation is computed against is identically zero at every grid;
  2. the sub-cell area-fill target chi becomes BINARY, because every cell is
     then wholly inside or wholly outside the part, so the raster target and
     the area-fill target coincide and cannot disagree.

WHAT VARIANT A DOES NOT DO, stated because it is the honest limit. It does not
hold the part fixed. It moves each boundary by up to half a cell, which is up
to 2.4 percent of the limb half-length at grid 96 and 0.3 percent at grid 360.
The original ladder holds the PART fixed and lets the RASTER wobble by half a
cell; this variant holds the RASTER exact and lets the PART wobble by the same
half cell. So a collapse of the scatter says the mismatch between raster and
target was the cause; a persistence says a half-cell change of the physical
part is enough to move the answer, which is a statement about the physics and
not about the rasterizer.

VARIANT B, SUB-CELL MELT AREA FILL. The melted region is `phi >= 0.5`, a
Heaviside on a field whose phase-change window is 10 C wide, so a whole ring of
cells can enter or leave the melted set for a small change of field. The
replacement is the sub-cell AREA FRACTION of each cell that lies inside the
melt front, evaluated by bilinear reconstruction of phi.

WHY THAT INDICATOR AND NOT ANOTHER. The campaign's target indicator chi is
already the sub-cell area fraction of each cell that lies inside the part
boundary (`adjoint2d/chi_area.py`). Scoring the melt front by the SAME
convention makes both sides of the intersection over union the same kind of
object, and it inherits chi's own degeneracy: when the front sits on cell
edges, the area fill is exactly the binary set again
(`test_melt_area_fill_reproduces_a_binary_field_exactly`). The alternative
offered in the task, a Gaussian-smoothed front, does not have that property
against chi and introduces a width that has no counterpart in the target; it is
computed here anyway, over a shrinking width, as the regularizer-width
diagnostic that the discipline asks for, with sigma = 0 required to reproduce
the original binary metric bit for bit.

Neither variant touches any solver module. Variant B is a pure RE-SCORING of
melt fields the original ladder already stored, so it re-runs no physics at all
and reproduces the original numbers by construction.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

# The cross polygon, from the production shape builder `shapes.make_cross` at
# the pinned 0.022 m width with the default thickness = width / 3: unique
# coordinates +- 11.000 mm (the limb half-length) and +- 11/3 mm (the arm half
# width).
CROSS_LIMB_HALF_M = 0.011
CROSS_ARM_HALF_M = 0.011 / 3.0
CHAMBER_M = 0.06
MELT_LEVEL = 0.5


# ---------------------------------------------------------------------------
# variant A: snap the cross to the grid
# ---------------------------------------------------------------------------

def dx_of_grid(n_grid: int, chamber_m: float = CHAMBER_M) -> float:
    """The cell size the production domain builder produces at `n_grid`.

    `rfam_eqs_coupled.make_domain` builds `x = linspace(-w/2, w/2, nx)`, so the
    spacing is the chamber width over `nx - 1`.
    """
    return float(chamber_m) / (int(n_grid) - 1)


def snap_cross_geometry(limb_half_m: float, arm_half_m: float,
                        dx_m: float) -> dict:
    """Both cross boundaries moved to the nearest whole multiple of the cell.

    Returns the snapped half-dimensions, the whole-cell counts they realize,
    and the geometry deltas in metres, in cells and as a fraction of the
    dimension itself, because those deltas are the price of the variant and the
    report has to quote them.
    """
    dx = float(dx_m)
    m = max(1, int(round(float(limb_half_m) / dx)))
    k = max(1, int(round(float(arm_half_m) / dx)))
    if k >= m:
        raise ValueError(f"snapped arm half {k} cells is not inside the limb "
                         f"half {m} cells; the cross would degenerate")
    limb = m * dx
    arm = k * dx
    return {
        "dx_m": dx,
        "limb_half_m": limb, "arm_half_m": arm,
        "limb_cells": m, "arm_cells": k,
        "limb_half_m_original": float(limb_half_m),
        "arm_half_m_original": float(arm_half_m),
        "d_limb_m": limb - float(limb_half_m),
        "d_arm_m": arm - float(arm_half_m),
        "d_limb_frac_of_cell": (limb - float(limb_half_m)) / dx,
        "d_arm_frac_of_cell": (arm - float(arm_half_m)) / dx,
        "d_limb_pct_of_dimension":
            100.0 * (limb - float(limb_half_m)) / float(limb_half_m),
        "d_arm_pct_of_dimension":
            100.0 * (arm - float(arm_half_m)) / float(arm_half_m),
    }


def snap_cross_cfg(cfg: dict, n_grid: int) -> tuple[dict, dict]:
    """The configuration with the cross snapped to the grid at `n_grid`.

    The width and the height carry the snapped limb SPAN and the explicit
    `thickness` key carries the snapped arm width, which is the argument
    `shapes.make_shape` passes straight to `shapes.make_cross`; without it the
    builder would fall back to width / 3 and un-snap the arm.
    """
    part = cfg["geometry"]["part"]
    if str(part.get("shape", "")).strip().lower() not in {"cross", "plus"}:
        raise ValueError(f"the snap is defined for the cross only, got "
                         f"{part.get('shape')!r}")
    dx = dx_of_grid(n_grid, float(cfg["geometry"]["chamber_x"]))
    limb0 = 0.5 * float(part["width"])
    arm0 = 0.5 * float(part.get("thickness", float(part["width"]) / 3.0))
    s = snap_cross_geometry(limb0, arm0, dx)
    out = copy.deepcopy(cfg)
    p = out["geometry"]["part"]
    p["width"] = 2.0 * s["limb_half_m"]
    p["height"] = 2.0 * s["limb_half_m"]
    p["thickness"] = 2.0 * s["arm_half_m"]
    s["n_grid"] = int(n_grid)
    s["is_no_op"] = bool(abs(s["d_limb_m"]) < 1e-15 and abs(s["d_arm_m"]) < 1e-15)
    return out, s


# ---------------------------------------------------------------------------
# variant B: the smoothed melt indicators
# ---------------------------------------------------------------------------

def melt_area_fill(phi: np.ndarray, n_sub: int = 8,
                   level: float = MELT_LEVEL) -> np.ndarray:
    """Sub-cell area fraction of each cell that lies inside the melt front.

    The field is reconstructed bilinearly between grid points and sampled on
    the SAME offset pattern `chi_area.area_fill_poly` uses, `n_sub` by `n_sub`
    points strictly inside the cell, so the melt indicator and the target chi
    are built by one convention. Outside the array the reconstruction is
    clamped to the edge value, which is the only choice that cannot invent a
    front at the domain boundary.
    """
    a = np.asarray(phi, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"phi must be two dimensional, got shape {a.shape}")
    n = int(n_sub)
    if n < 1:
        raise ValueError(f"n_sub must be a positive integer, got {n_sub!r}")
    ny, nx = a.shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    jr = jj.ravel().astype(float)
    ir = ii.ravel().astype(float)
    off = np.linspace(-0.5 + 0.5 / n, 0.5 - 0.5 / n, n)
    total = np.zeros(jr.shape, dtype=float)
    for dj in off:
        for di in off:
            v = map_coordinates(a, np.vstack([jr + dj, ir + di]), order=1,
                                mode="nearest")
            total += (v >= level)
    return (total / float(n * n)).reshape(ny, nx)


def gaussian_melt_indicator(phi: np.ndarray, sigma_cells: float,
                            level: float = MELT_LEVEL) -> np.ndarray:
    """The melted SET after smoothing the melt fraction by `sigma_cells`.

    At sigma = 0 this is the original `phi >= 0.5` set exactly, which is the
    width-shrink limit the discipline requires of any regularized threshold.
    """
    s = float(sigma_cells)
    if s < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma_cells!r}")
    if s == 0.0:
        return np.asarray(phi, dtype=float) >= level
    return gaussian_filter(np.asarray(phi, dtype=float), s, mode="nearest") >= level


def area_iou(a: np.ndarray, b: np.ndarray) -> float:
    """sum(min) / sum(max), the campaign's area-weighted overlap.

    Imported convention, re-exported here so the variant scripts cannot drift
    from `adjoint2d.topopt_objective.area_iou`; the import is checked by
    `test_area_iou_is_the_campaign_function`.
    """
    from adjoint2d import topopt_objective as tobj
    return float(tobj.area_iou(a, b))


def binary_iou(melted: np.ndarray, part_mask: np.ndarray) -> float:
    m = np.asarray(melted, dtype=bool)
    p = np.asarray(part_mask, dtype=bool)
    u = int(np.sum(m | p))
    return float(int(np.sum(m & p)) / u) if u else float("nan")


def smoothed_scores(phi: np.ndarray, chi: np.ndarray, part_mask: np.ndarray,
                    n_sub: int = 8,
                    sigmas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0)) -> dict:
    """Every variant-B reading of one stored melt field, all labelled.

    `IoU_binary` is recomputed here from the stored field and must equal the
    number the original ladder recorded; it is the re-scoring reproduction
    check, not a new result.
    """
    fill = melt_area_fill(phi, n_sub=n_sub)
    out = {
        "IoU_binary": binary_iou(np.asarray(phi) >= MELT_LEVEL, part_mask),
        "IoU_phi_vs_chi_area": area_iou(phi, chi),
        "IoU_subcell_melt_vs_chi": area_iou(fill, chi),
        "IoU_subcell_melt_vs_raster": area_iou(fill, np.asarray(part_mask,
                                                               dtype=float)),
        "melted_area_cells_binary": float(np.sum(np.asarray(phi) >= MELT_LEVEL)),
        "melted_area_cells_subcell": float(np.sum(fill)),
        "n_sub": int(n_sub),
    }
    for s in sigmas:
        out[f"IoU_gauss_sigma{s:g}"] = binary_iou(
            gaussian_melt_indicator(phi, s), part_mask)
    return out


def spread(values, floor_n=None, grids=None) -> float:
    """max minus min, optionally over the grids at or above `floor_n`."""
    v = np.asarray(list(values), dtype=float)
    if floor_n is not None and grids is not None:
        g = np.asarray(list(grids), dtype=float)
        v = v[g >= float(floor_n)]
    return float(np.max(v) - np.min(v)) if v.size else float("nan")


def pearson_r(a, b) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
