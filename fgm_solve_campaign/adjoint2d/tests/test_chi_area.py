"""Red-first tests for the grid-independent target indicator chi.

The objective's target must be a property of the GEOMETRY, not of the solve
grid. The solve-grid binary raster is not: a cell is either fully in the part
or fully out of it, so the target area jumps as the grid changes and a solve at
grid 120 is chasing a different target than a score at grid 160.

The replacement is the sub-cell area fill, the cell average of the geometric
indicator function, evaluated by the SAME supersampling routine the production
domain builder uses for its material fill fraction, at a higher sample count.
The analytic case pinned here is a circle, whose area is known in closed form.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import chi_area


R_CIRCLE = 0.010          # 10 mm radius
HALF = 0.030              # 60 mm square domain, matching the campaign


def _axes(n: int):
    return (np.linspace(-HALF, HALF, n), np.linspace(-HALF, HALF, n))


def _circle_poly(radius: float = R_CIRCLE, n: int = 720) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def _area(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(field)) * float(x[1] - x[0]) * float(y[1] - y[0])


def test_area_fill_of_a_circle_matches_the_closed_form():
    x, y = _axes(120)
    chi = chi_area.area_fill_poly(_circle_poly(), x, y)
    exact = np.pi * R_CIRCLE ** 2
    assert abs(_area(chi, x, y) - exact) / exact < 2.0e-3


def test_area_fill_beats_the_binary_raster_on_the_same_grid():
    x, y = _axes(120)
    poly = _circle_poly()
    exact = np.pi * R_CIRCLE ** 2
    err_fill = abs(_area(chi_area.area_fill_poly(poly, x, y), x, y) - exact)
    err_bin = abs(_area(chi_area.area_fill_poly(poly, x, y, n_sub=1), x, y) - exact)
    assert err_fill < err_bin


def test_area_fill_is_grid_independent_where_the_binary_raster_is_not():
    poly = _circle_poly()
    exact = np.pi * R_CIRCLE ** 2
    fills, bins = [], []
    for n in (120, 160):
        x, y = _axes(n)
        fills.append(_area(chi_area.area_fill_poly(poly, x, y), x, y))
        bins.append(_area(chi_area.area_fill_poly(poly, x, y, n_sub=1), x, y))
    assert abs(fills[1] - fills[0]) / exact < abs(bins[1] - bins[0]) / exact
    assert abs(fills[1] - fills[0]) / exact < 2.0e-3


def test_area_fill_stays_in_the_unit_interval():
    x, y = _axes(64)
    chi = chi_area.area_fill_poly(_circle_poly(), x, y)
    assert chi.min() >= 0.0 and chi.max() <= 1.0
    assert chi[0, 0] == 0.0                       # a corner is far outside
    assert chi[32, 32] == pytest.approx(1.0)      # the centre is fully inside


def test_n_sub_must_be_a_positive_integer():
    x, y = _axes(32)
    with pytest.raises(ValueError):
        chi_area.area_fill_poly(_circle_poly(), x, y, n_sub=0)


def test_frozen_sample_count():
    assert chi_area.CHI_N_SUB == 32


def test_band_restriction_is_bit_identical_to_the_production_sampler():
    """The fast path must not be a second, slightly different convention."""
    from adjoint2d.prod import rfam
    x, y = _axes(48)
    poly = _circle_poly(n=96)
    fast = chi_area.area_fill_poly(poly, x, y, n_sub=8)
    slow = rfam._subpixel_fill_fraction(poly, x, y, n_sub=8)
    assert np.array_equal(fast, slow)


def test_band_restriction_is_bit_identical_on_a_grid_aligned_square():
    """A grid-aligned square is the case where a band that is one cell too
    narrow would silently drop a half-filled row."""
    from adjoint2d.prod import rfam
    x, y = _axes(41)
    poly = np.array([[-0.0101, -0.0101], [0.0101, -0.0101],
                     [0.0101, 0.0101], [-0.0101, 0.0101]])
    fast = chi_area.area_fill_poly(poly, x, y, n_sub=8)
    slow = rfam._subpixel_fill_fraction(poly, x, y, n_sub=8)
    assert np.array_equal(fast, slow)


def test_union_of_two_disjoint_polygons_adds_their_areas():
    x, y = _axes(200)
    a = _circle_poly(0.005) + np.array([-0.012, 0.0])
    b = _circle_poly(0.005) + np.array([+0.012, 0.0])
    chi = chi_area.area_fill_union([a, b], x, y)
    exact = 2.0 * np.pi * 0.005 ** 2
    assert abs(_area(chi, x, y) - exact) / exact < 5.0e-3
    assert chi.max() <= 1.0
