"""Shared FILL CONTRACT, importable as a test utility by any lane.

WHY THIS FILE IS NOT A TEST CLASS. The three-dimensional port lane needs its
sub-cell VOLUME fill to reduce to the two-dimensional sub-cell AREA fill on an
extrusion slice. A contract that lives inside a two-dimensional test class
cannot be re-run against a three-dimensional implementation. So the analytic
checks are written here as plain functions that take the fill implementation as
an ARGUMENT, and the two-dimensional test module simply calls them with its own
implementation. The three-dimensional lane imports the same functions and
passes its slice evaluator.

THE CONVENTION THE CONTRACT ASSUMES, stated so a second implementation cannot
drift from it silently:

  * `fill(polygon, x, y)` returns an array of shape `(len(y), len(x))` whose
    entry `[j, i]` is the fraction of the CELL CENTRED on `(x[i], y[j])` that
    lies inside the polygon. Cells are `dx` by `dy` rectangles centred on the
    grid points, not corner-anchored; the campaign grid is uniform.
  * The polygon is a closed ring of `(x, y)` vertices in metres. WINDING IS
    IRRELEVANT: inside-ness is the even-odd (crossing-number) rule, which is
    the rule `matplotlib.path.Path.contains_points` applies with the default
    zero radius and the rule `rfam_eqs_coupled.polygon_mask` therefore applies.
    A clockwise and a counter-clockwise ring must give the same fill.
  * Sub-cell sampling is a REGULAR `n_sub` by `n_sub` grid of points at cell
    offsets `linspace(-0.5 + 0.5/n, 0.5 - 0.5/n, n)`, the production offsets of
    `rfam_eqs_coupled._subpixel_fill_fraction`. The fill is the fraction of
    those sample points that are inside. It is therefore a MONTE-CARLO-FREE
    deterministic quadrature with a boundary error of order
    `cell_size / (2 n_sub)` per boundary cell.
  * Boundary handling: a sample point exactly on the boundary is decided by the
    inside test, not special-cased. No cell is clipped analytically.
  * The value is in [0, 1] everywhere, exactly 1 for a cell fully interior and
    exactly 0 for a cell fully exterior.

For a three-dimensional volume fill on an extruded prism, the same contract
holds slice by slice: the volume fill of a cell whose z-extent lies strictly
inside the extrusion must EQUAL the area fill of the corresponding
two-dimensional cell. `assert_extrusion_slice_reduction` states that as a
callable check the three-dimensional lane can run directly.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

FillFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

# Tolerances, quoted as fractions of the closed-form area. The area fill's
# residual is a boundary effect; at n_sub = 32 on the campaign grid the
# measured circle error is below 0.05 percent, so 0.2 percent is a loose but
# meaningful bar that a genuinely wrong implementation cannot pass.
AREA_REL_TOL = 2.0e-3
GRID_MOVE_REL_TOL = 2.0e-3


def circle_polygon(radius: float, n: int = 720, cx: float = 0.0,
                   cy: float = 0.0) -> np.ndarray:
    """A regular `n`-gon INSCRIBED in the circle of the given radius."""
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return np.column_stack([cx + radius * np.cos(t), cy + radius * np.sin(t)])


def inscribed_ngon_area(radius: float, n: int) -> float:
    """Closed-form area of the inscribed regular n-gon the sampler actually sees.

    The polygon, not the circle, is the ground truth: a fill routine given an
    n-gon must reproduce the n-gon. Quoting pi r^2 instead would charge the
    implementation for the caller's own discretization of the circle.
    """
    return 0.5 * int(n) * float(radius) ** 2 * math.sin(2.0 * math.pi / int(n))


def rotated_rect_polygon(w: float, h: float, deg: float,
                         cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    p = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return np.column_stack([cx + p[:, 0] * c - p[:, 1] * s,
                            cy + p[:, 0] * s + p[:, 1] * c])


def _area_of(fill: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    dA = float(x[1] - x[0]) * float(y[1] - y[0])
    return float(np.sum(fill)) * dA


def assert_circle_area(fill_fn: FillFn, x: np.ndarray, y: np.ndarray,
                       radius: float = 0.010, n_vertices: int = 720,
                       rel_tol: float = AREA_REL_TOL) -> dict:
    """The analytic circle check. Returns the measured numbers for logging."""
    poly = circle_polygon(radius, n_vertices)
    exact = inscribed_ngon_area(radius, n_vertices)
    measured = _area_of(fill_fn(poly, x, y), x, y)
    rel = (measured - exact) / exact
    assert abs(rel) <= rel_tol, (
        f"circle area fill is off by {rel*100:.4f} percent "
        f"(measured {measured:.6e} m^2 against closed form {exact:.6e} m^2), "
        f"tolerance {rel_tol*100:.4f} percent")
    return {"area_measured_m2": measured, "area_closed_form_m2": exact,
            "rel_error": rel}


def assert_rotated_rect_area(fill_fn: FillFn, x: np.ndarray, y: np.ndarray,
                             w: float = 0.020, h: float = 0.008,
                             deg: float = 31.0,
                             rel_tol: float = AREA_REL_TOL) -> dict:
    """The analytic rotated-rectangle check: area is rotation invariant."""
    exact = float(w) * float(h)
    measured = _area_of(fill_fn(rotated_rect_polygon(w, h, deg), x, y), x, y)
    rel = (measured - exact) / exact
    assert abs(rel) <= rel_tol, (
        f"rotated-rectangle area fill is off by {rel*100:.4f} percent "
        f"(measured {measured:.6e} m^2 against w*h {exact:.6e} m^2)")
    return {"area_measured_m2": measured, "area_closed_form_m2": exact,
            "rel_error": rel, "deg": float(deg)}


def assert_winding_invariance(fill_fn: FillFn, x: np.ndarray, y: np.ndarray,
                              poly: np.ndarray | None = None) -> None:
    """Reversing the vertex order must not change one cell of the fill."""
    p = rotated_rect_polygon(0.020, 0.008, 17.0) if poly is None else np.asarray(poly)
    a = fill_fn(p, x, y)
    b = fill_fn(p[::-1].copy(), x, y)
    d = float(np.max(np.abs(a - b)))
    assert d == 0.0, f"fill is winding dependent; max cell difference {d:.3e}"


def assert_grid_independence(fill_fn: FillFn, axes_a, axes_b,
                             poly: np.ndarray | None = None,
                             rel_tol: float = GRID_MOVE_REL_TOL) -> dict:
    """The total filled area must barely move when the grid changes."""
    p = circle_polygon(0.010, 720) if poly is None else np.asarray(poly)
    xa, ya = axes_a
    xb, yb = axes_b
    aa = _area_of(fill_fn(p, xa, ya), xa, ya)
    ab = _area_of(fill_fn(p, xb, yb), xb, yb)
    rel = (ab - aa) / aa
    assert abs(rel) <= rel_tol, (
        f"filled area moved {rel*100:.4f} percent between grids "
        f"{len(xa)} and {len(xb)}")
    return {"area_grid_a_m2": aa, "area_grid_b_m2": ab, "rel_move": rel}


def assert_extrusion_slice_reduction(volume_fill_slice_fn, area_fill_fn: FillFn,
                                     x: np.ndarray, y: np.ndarray,
                                     poly: np.ndarray | None = None,
                                     atol: float = 1e-12) -> None:
    """THREE-DIMENSIONAL LANE ENTRY POINT.

    `volume_fill_slice_fn(poly, x, y)` must return the sub-cell VOLUME fill of
    one z-slice that lies strictly inside an extrusion of `poly`. On such a
    slice the volume fill is the area fill, cell for cell. This is the only
    statement that ties the two lanes' target indicators together, and it is
    written here so the three-dimensional lane can run it without importing a
    two-dimensional test class.
    """
    p = circle_polygon(0.010, 720) if poly is None else np.asarray(poly)
    a = area_fill_fn(p, x, y)
    v = volume_fill_slice_fn(p, x, y)
    d = float(np.max(np.abs(np.asarray(v) - a)))
    assert d <= atol, (
        f"the volume fill of an interior extrusion slice differs from the area "
        f"fill by {d:.3e} (tolerance {atol:.1e})")
