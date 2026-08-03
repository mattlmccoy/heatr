"""Red-first tests for the joint per-angle map re-solve helper logic.

The joint campaign asks whether the best orientation MOVES once the dopant map
is re-solved at every angle instead of being rigidly rotated from the zero-degree
solution. Three pieces of pure logic decide the headline and are therefore
tested before the driver exists:

  * `rotated_warm_start`, the composition of the proven lab-frame map rotation
    (`scripts/analysis/orientation_map_rotation.rotate_sat_map`) with the
    campaign's design-variable start convention (clip into the box inside the
    part, hold saturation 1 outside);
  * `argmin_angle`, the per-shape joint optimum read off scored rows;
  * `angle_delta_deg`, the signed angle move against the fixed-map sweep's best
    angle, wrapped by that shape's symmetry period so that, for example, a cross
    result at 90 degrees is a zero move and not a 90-degree move.

Ground-truth contract for the rotation itself is already PROVEN in
`test_orientation_map_rotation.py` (production `rotation_deg = +90` equals
`np.rot90(k=-1)` with zero mismatched cells).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent
MAPS = REPO / "fgm_solve_campaign/out_lib/T_shape_maps.npz"


# ---------------------------------------------------------------------------
# rotated_warm_start
# ---------------------------------------------------------------------------

def test_rotated_warm_start_is_the_clipped_map_at_zero_degrees():
    from scripts.analysis.joint_angle_lib import rotated_warm_start
    d = np.load(MAPS)
    m, pm = d["A1_cont"], d["part_mask"].astype(bool)
    out = rotated_warm_start(m, 0.0, pm, box=(0.0, 1.0))
    assert np.allclose(out[pm], np.clip(m, 0.0, 1.0)[pm], atol=1e-12)


def test_rotated_warm_start_holds_saturation_one_outside_the_rotated_part():
    from scripts.analysis.joint_angle_lib import rotated_warm_start
    d = np.load(MAPS)
    m, pm = d["A1_cont"], d["part_mask"].astype(bool)
    pm90 = np.rot90(pm, k=-1)
    out = rotated_warm_start(m, 90.0, pm90, box=(0.0, 1.0))
    assert np.all(out[~pm90] == 1.0)


def test_rotated_warm_start_at_90_is_the_exact_pixel_permutation_inside_the_part():
    from scripts.analysis.joint_angle_lib import rotated_warm_start
    d = np.load(MAPS)
    m, pm = d["A1_cont"], d["part_mask"].astype(bool)
    pm90 = np.rot90(pm, k=-1)
    out = rotated_warm_start(m, 90.0, pm90, box=(0.0, 1.0))
    ref = np.clip(np.rot90(m, k=-1), 0.0, 1.0)
    assert np.allclose(out[pm90], ref[pm90], atol=1e-9)


def test_rotated_warm_start_respects_a_tighter_box():
    from scripts.analysis.joint_angle_lib import rotated_warm_start
    d = np.load(MAPS)
    m, pm = d["A1_cont"], d["part_mask"].astype(bool)
    out = rotated_warm_start(m, 0.0, pm, box=(0.2, 0.8))
    assert out[pm].min() >= 0.2 - 1e-12
    assert out[pm].max() <= 0.8 + 1e-12


# ---------------------------------------------------------------------------
# argmin_angle and best_row
# ---------------------------------------------------------------------------

def test_argmin_angle_picks_the_smallest_J():
    from scripts.analysis.joint_angle_lib import argmin_angle
    rows = [{"angle_deg": 0.0, "J": 500.0},
            {"angle_deg": 45.0, "J": 300.0},
            {"angle_deg": 90.0, "J": 400.0}]
    assert argmin_angle(rows) == 45.0


def test_argmin_angle_breaks_ties_toward_the_smaller_angle():
    from scripts.analysis.joint_angle_lib import argmin_angle
    rows = [{"angle_deg": 90.0, "J": 300.0}, {"angle_deg": 30.0, "J": 300.0}]
    assert argmin_angle(rows) == 30.0


def test_argmin_angle_raises_on_no_rows():
    from scripts.analysis.joint_angle_lib import argmin_angle
    with pytest.raises(ValueError):
        argmin_angle([])


def test_best_row_returns_the_lowest_J_row_object():
    from scripts.analysis.joint_angle_lib import best_row
    rows = [{"eval_index": 1, "J": 10.0}, {"eval_index": 2, "J": 3.0},
            {"eval_index": 3, "J": 3.0}]
    assert best_row(rows)["eval_index"] == 2


# ---------------------------------------------------------------------------
# angle_delta_deg
# ---------------------------------------------------------------------------

def test_angle_delta_is_zero_for_the_same_angle():
    from scripts.analysis.joint_angle_lib import angle_delta_deg
    assert angle_delta_deg(90.0, 90.0, 180.0) == 0.0


def test_angle_delta_wraps_by_the_symmetry_period():
    from scripts.analysis.joint_angle_lib import angle_delta_deg
    # A cross has a 90-degree period, so 90 degrees IS zero degrees.
    assert angle_delta_deg(90.0, 0.0, 90.0) == 0.0
    # A five-point star has a 72-degree period.
    assert angle_delta_deg(72.0, 0.0, 72.0) == 0.0


def test_angle_delta_is_signed_and_takes_the_short_way_round():
    from scripts.analysis.joint_angle_lib import angle_delta_deg
    assert angle_delta_deg(135.0, 0.0, 180.0) == pytest.approx(-45.0)
    assert angle_delta_deg(22.5, 0.0, 180.0) == pytest.approx(22.5)
    assert angle_delta_deg(30.0, 0.0, 90.0) == pytest.approx(30.0)


def test_angle_delta_rejects_a_non_positive_period():
    from scripts.analysis.joint_angle_lib import angle_delta_deg
    with pytest.raises(ValueError):
        angle_delta_deg(10.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# the operating-ceiling flag
# ---------------------------------------------------------------------------

def test_ceiling_flag_fires_above_250_c_and_not_at_it():
    from scripts.analysis.joint_angle_lib import exceeds_ceiling
    assert exceeds_ceiling(254.5) is True
    assert exceeds_ceiling(250.0) is False
    assert exceeds_ceiling(242.2) is False


# ---------------------------------------------------------------------------
# the angle sets carried over from the orientation sweep
# ---------------------------------------------------------------------------

def test_angle_sets_match_the_orientation_sweep():
    from scripts.analysis.joint_angle_lib import ANGLES, SYMMETRY_PERIOD_DEG
    from scripts.analysis.run_orientation_optimization import ANGLES as SWEEP
    for shape in ("T_shape", "L_shape", "cross", "star"):
        assert list(ANGLES[shape]) == list(SWEEP[shape])
    assert SYMMETRY_PERIOD_DEG == {"T_shape": 180.0, "L_shape": 180.0,
                                   "cross": 90.0, "star": 72.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
