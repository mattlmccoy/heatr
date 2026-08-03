"""Phase C Task 2 (chi): the CROSS-LANE fill contract.

The 2-D lane published a shared, importable contract at
fgm_solve_campaign/adjoint2d/tests/fill_contract.py (their d881455) so the two
lanes' target indicators are provably the same convention rather than two
independent conventions that happen to agree today. This module imports THEIR
file unmodified and runs it against MY fill.

Pure numpy on both sides, so it runs in either environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fgm_solve_campaign" / "adjoint2d" / "tests"))

import fill_contract as fc            # noqa: E402  THEIR file, read-only

from solve3d import fill              # noqa: E402


def _axes(n: int = 120, L: float = 0.060):
    h = L / n
    c = (np.arange(n) + 0.5) * h - L / 2.0
    return c, c


# --------------------------------------------------------------------- #
# Their parameterized checks, run against MY implementation
# --------------------------------------------------------------------- #
def test_their_circle_area_check_passes_on_my_fill():
    x, y = _axes()
    out = fc.assert_circle_area(fill.area_fill_2d, x, y)
    assert abs(out["rel_error"]) <= fc.AREA_REL_TOL


def test_their_rotated_rect_check_passes_on_my_fill():
    x, y = _axes()
    fc.assert_rotated_rect_area(fill.area_fill_2d, x, y)


def test_their_winding_invariance_check_passes_on_my_fill():
    x, y = _axes()
    fc.assert_winding_invariance(fill.area_fill_2d, x, y)


def test_their_grid_independence_check_passes_on_my_fill():
    fc.assert_grid_independence(fill.area_fill_2d, _axes(120), _axes(160))


def test_their_extrusion_slice_reduction_is_exact():
    """THE cross-lane statement: my sub-cell VOLUME fill of a z-slice strictly
    inside an extrusion must equal their sub-cell AREA fill, cell for cell."""
    x, y = _axes()
    fc.assert_extrusion_slice_reduction(fill.volume_fill_slice,
                                        fill.area_fill_2d, x, y)


# --------------------------------------------------------------------- #
# My own additional checks
# --------------------------------------------------------------------- #
def test_fill_is_bounded_and_saturates():
    x, y = _axes(60)
    f = fill.area_fill_2d(fc.circle_polygon(0.010, 720), x, y)
    assert f.min() == 0.0 and f.max() == 1.0
    assert np.all((f >= 0.0) & (f <= 1.0))


def test_volume_fill_of_a_prism_recovers_the_closed_form_volume():
    """3-D analogue of their area check: an extruded circle's volume."""
    x, y = _axes(120)
    zc, dz = np.array([0.0]), 0.020
    poly = fc.circle_polygon(0.010, 720)
    v = fill.volume_fill_3d(poly, x, y, zc, dz, z_lo=-0.010, z_hi=0.010)
    dV = float(x[1] - x[0]) * float(y[1] - y[0]) * dz
    measured = float(v.sum()) * dV
    exact = fc.inscribed_ngon_area(0.010, 720) * dz
    assert abs(measured - exact) / exact <= fc.AREA_REL_TOL


def test_partial_z_slice_is_a_true_fraction():
    """A slice straddling the extrusion end must fill exactly the fraction of
    its z-extent that lies inside -- otherwise the volume fill is only an area
    fill wearing a third index."""
    x, y = _axes(60)
    poly = fc.circle_polygon(0.010, 720)
    deep = fill.volume_fill_3d(poly, x, y, np.array([0.0]), 0.004,
                               z_lo=-0.010, z_hi=0.010)
    # centre at the end plane: exactly half the cell's z-extent is inside
    half = fill.volume_fill_3d(poly, x, y, np.array([0.010]), 0.004,
                               z_lo=-0.010, z_hi=0.010)
    assert float(half.sum()) / float(deep.sum()) == pytest.approx(0.5, abs=0.02)
