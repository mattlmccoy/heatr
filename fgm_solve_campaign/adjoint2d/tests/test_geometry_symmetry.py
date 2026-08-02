"""Red-first tests for symmetry analysis of an arbitrary imported geometry.

WHY THE SOLVE STACK NEEDS THIS. Two frozen conventions depend on the part's
symmetry and were, until now, applied by hand from knowledge of which shape was
being run.

  * The DWELL candidate set. `DWELL_SCHEDULE_REPORT.md` Section 2 measured that
    the part-frame heating at theta and at theta + 180 degrees is the SAME
    FIELD to 4e-13 relative, because the grounded parallel-plate drive is
    invariant under a half turn. Eight candidate positions therefore carry only
    four distinct heating patterns, and every dwell vector is reported on the
    distinct positions. That is a GAUGE rule and it holds for every geometry,
    so the intake must apply it universally rather than per shape.
  * The indexing set. `CONTINUOUS_ROTATION_REPORT.md` Section 7 measured that
    symmetry-matched indexing beats the finest available rotation: the cross
    and the square win at 90-degree indexing by factors of 2.8 and 2.1 over
    their best continuous period. Choosing that indexing needs the part's
    rotational order.

The recovery targets below are the orders those shapes are DEFINED with in
`shapes.py`: the cross four, the five-pointed star five, star6 and the hexagon
six, the equilateral triangle three, the L none, the T none but one mirror.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import chi_area, geometry_intake as gi, geometry_symmetry as gs
from adjoint2d.library_solve import shape_config
from adjoint2d.pins import load_cfg

GRID = 120


def _chi(shape: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = gi.grid_axes(GRID)
    chi, _info = chi_area.chi_from_cfg(load_cfg(shape_config(shape)), x, y)
    return chi, x, y


# --- the gauge rule, which is universal ---------------------------------------

def test_gauge_reduce_folds_the_half_turn_redundancy():
    got = gs.gauge_reduce([0.0, 30.0, 180.0, 210.0, 200.0, 359.0])
    assert np.allclose(got, [0.0, 20.0, 30.0, 179.0])


def test_gauge_reduce_is_idempotent():
    a = gs.gauge_reduce([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    assert np.allclose(a, gs.gauge_reduce(a))
    assert np.allclose(a, [0.0, 45.0, 90.0, 135.0])


# --- rotational order ----------------------------------------------------------

def test_zero_rotation_has_zero_mismatch():
    chi, x, y = _chi("cross")
    assert gs.rotation_mismatch(chi, 0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("shape,order", [
    ("cross", 4),
    ("square", 4),
    ("star", 5),
    ("star6", 6),
    ("hexagon", 6),
    ("equilateral_triangle", 3),
    ("L_shape", 1),
    ("T_shape", 1),
])
def test_rotational_order_is_recovered_on_the_library_shapes(shape, order):
    chi, x, y = _chi(shape)
    rep = gs.analyze(chi)
    assert rep.rotational_order == order, (
        f"{shape}: expected order {order}, got {rep.rotational_order} "
        f"(mismatch curve minima at {rep.diagnostics['tested_orders']})")


def test_the_circle_is_flagged_as_continuously_symmetric():
    chi, x, y = _chi("circle")
    rep = gs.analyze(chi)
    assert rep.continuous_rotational is True
    assert rep.rotational_order >= 8


def test_a_deliberately_asymmetric_polygon_has_no_rotational_symmetry():
    poly = np.array([[-0.008, -0.006], [0.010, -0.004], [0.006, 0.009],
                     [-0.003, 0.011], [-0.009, 0.002]])
    it = gi.from_polygon(poly, grid=GRID)
    rep = gs.analyze(it.chi)
    assert rep.rotational_order == 1
    assert rep.mirror_axes_deg == ()


# --- mirrors --------------------------------------------------------------------

def test_the_T_shape_has_exactly_one_mirror_axis_and_no_rotation():
    chi, x, y = _chi("T_shape")
    rep = gs.analyze(chi)
    assert rep.rotational_order == 1
    assert len(rep.mirror_axes_deg) == 1


def test_the_L_shape_has_no_mirror_axis():
    chi, x, y = _chi("L_shape")
    assert gs.analyze(chi).mirror_axes_deg == ()


def test_the_square_has_four_mirror_axes():
    chi, x, y = _chi("square")
    assert len(gs.analyze(chi).mirror_axes_deg) == 4


# --- the candidate set the dwell and indexing layers consume ---------------------

def test_candidate_angles_span_the_symmetry_reduced_gauge_interval():
    """Orientations distinct under BOTH the part symmetry and the half-turn gauge."""
    assert gs.candidate_span_deg(1) == pytest.approx(180.0)
    assert gs.candidate_span_deg(2) == pytest.approx(180.0)
    assert gs.candidate_span_deg(4) == pytest.approx(90.0)
    assert gs.candidate_span_deg(6) == pytest.approx(60.0)
    # An ODD order does not contain the half turn, so the gauge halves the span.
    assert gs.candidate_span_deg(3) == pytest.approx(60.0)
    assert gs.candidate_span_deg(5) == pytest.approx(36.0)


def test_candidate_angles_are_inside_the_span_and_start_at_zero():
    ang = gs.candidate_angles(order=4, step_deg=15.0)
    assert np.allclose(ang, [0.0, 15.0, 30.0, 45.0, 60.0, 75.0])
    ang5 = gs.candidate_angles(order=5, step_deg=12.0)
    assert np.allclose(ang5, [0.0, 12.0, 24.0])


def test_candidate_angles_of_a_continuous_shape_collapse_to_one():
    chi, x, y = _chi("circle")
    rep = gs.analyze(chi)
    assert len(gs.candidate_angles(report=rep, step_deg=15.0)) == 1


def test_indexing_orders_are_the_divisor_compatible_ones():
    assert gs.indexing_orders(4) == (2, 4)
    assert gs.indexing_orders(6) == (2, 3, 6)
    assert gs.indexing_orders(5) == (5,)
    assert gs.indexing_orders(1) == ()
