"""Red-first tests for the mask-to-polygon tracer.

An imported geometry arrives as pixels at least as often as it arrives as
vertices: a binary mask from a slicer, a portable network graphics (PNG) mask
from a drawing tool, a rasterized layer of a stereolithography (STL) file. The
solve stack, however, is polygon-based end to end: the production domain
builder rasterizes a polygon, and the area-fill target indicator supersamples
that same polygon. So the intake needs ONE conversion, mask to polygon, and it
has to be exact rather than approximate: the polygon must be the pixel region
itself, not a smoothed outline of it, or the target the solve chases is no
longer the geometry the user imported.

The tracer therefore emits the STAIRCASE boundary of the pixel region, built
from the pixel edges that are not shared by two inside pixels. That is exact by
construction, and these tests pin it.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import geometry_contour as gc


def _axes(n: int, half: float = 0.030):
    return np.linspace(-half, half, n), np.linspace(-half, half, n)


def _block_mask(ny=20, nx=20, r0=4, r1=12, c0=6, c1=15):
    m = np.zeros((ny, nx), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def test_rectangle_block_traces_to_one_loop_with_exact_area():
    m = _block_mask()
    loops = gc.mask_to_polygons(m, dx=1.0, dy=1.0, x0=0.0, y0=0.0)
    assert len(loops) == 1
    area = gc.signed_area(loops[0])
    assert area > 0.0, "the outer loop must come back counter-clockwise"
    assert area == pytest.approx(float(m.sum()), abs=1e-12)


def test_rectangle_block_has_four_corners_after_collinear_merge():
    loops = gc.mask_to_polygons(_block_mask(), dx=1.0, dy=1.0, x0=0.0, y0=0.0)
    assert loops[0].shape == (4, 2)


def test_staircase_of_a_step_shape_keeps_every_corner():
    m = np.zeros((10, 10), dtype=bool)
    m[2:6, 2:6] = True
    m[6:8, 2:4] = True          # an L, six corners
    loops = gc.mask_to_polygons(m, dx=1.0, dy=1.0, x0=0.0, y0=0.0)
    assert len(loops) == 1
    assert loops[0].shape == (6, 2)
    assert gc.signed_area(loops[0]) == pytest.approx(float(m.sum()), abs=1e-12)


def test_two_disjoint_blocks_trace_to_two_loops():
    m = np.zeros((20, 20), dtype=bool)
    m[2:6, 2:6] = True
    m[12:16, 12:16] = True
    loops = gc.mask_to_polygons(m, dx=1.0, dy=1.0, x0=0.0, y0=0.0)
    assert len(loops) == 2
    assert sum(gc.signed_area(p) for p in loops) == pytest.approx(32.0, abs=1e-12)


def test_a_hole_is_detected_and_raised_loudly():
    m = np.zeros((20, 20), dtype=bool)
    m[4:16, 4:16] = True
    m[8:12, 8:12] = False        # an interior hole
    with pytest.raises(gc.UnsupportedGeometry, match="hole"):
        gc.mask_to_polygons(m, dx=1.0, dy=1.0, x0=0.0, y0=0.0)


def test_physical_scaling_places_the_loop_in_metres():
    m = _block_mask()
    dx = dy = 5.0e-4
    loops = gc.mask_to_polygons(m, dx=dx, dy=dy, x0=-0.03, y0=-0.03)
    area = gc.signed_area(loops[0])
    assert area == pytest.approx(float(m.sum()) * dx * dy, rel=1e-12)
    assert loops[0][:, 0].min() >= -0.03 - dx
    assert loops[0][:, 1].max() <= -0.03 + m.shape[0] * dy + dy


def test_even_odd_point_in_polygon_agrees_with_the_production_rule():
    """The dependency-light inside test must match matplotlib's, point for point.

    `rfam_eqs_coupled.polygon_mask` uses `matplotlib.path.Path`, so any second
    implementation that the three-dimensional lane might carry has to agree
    with it or the two lanes disagree about what "inside" means.
    """
    from matplotlib.path import Path as MplPath

    t = np.linspace(0.0, 2.0 * np.pi, 11, endpoint=False)
    r = np.where(np.arange(11) % 2 == 0, 0.010, 0.004)
    poly = np.column_stack([r * np.cos(t), r * np.sin(t)])
    rng = np.random.default_rng(20260801)
    pts = rng.uniform(-0.012, 0.012, size=(4000, 2))
    mine = gc.point_in_polygon_evenodd(poly, pts[:, 0], pts[:, 1])
    theirs = MplPath(poly).contains_points(pts)
    assert int(np.sum(mine != theirs)) == 0
