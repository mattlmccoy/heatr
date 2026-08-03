"""Red-first tests for the two cross-scatter variants.

VARIANT A, geometry snapping. The rotating forward grid ladder found a
non-monotone intersection-over-union sequence on the cross and attributed it,
as an ASSUMED mechanism only, to the whole-cell rounding of the cross's two
rectilinear boundaries (`ROTATING_GRID_LADDER_REPORT.md` Section 4.5). The
snap moves each boundary to the nearest whole multiple of the cell size, so the
raster and the sub-cell area-fill target agree exactly.

VARIANT B, a smoothed melt indicator. The melted region is `phi >= 0.5`, a
Heaviside on a field whose boundary layer is a few cells wide, so a whole ring
of cells can enter or leave the melted set for a small change of field. The
replacement scores the SUB-CELL AREA FILL of the melt front, which is the same
convention the target indicator chi already uses (`adjoint2d/chi_area.py`).

Both variants must reduce to the original convention in their own zero limit,
and those two degeneracies are the first two tests written.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from rot_ladder_variants import (                                # noqa: E402
    CROSS_ARM_HALF_M, CROSS_LIMB_HALF_M, dx_of_grid,
    gaussian_melt_indicator, melt_area_fill, snap_cross_geometry,
)


# ---------------------------------------------------------------------------
# variant A: the snap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [96, 120, 160, 180, 200, 240, 360])
def test_snap_puts_both_cross_boundaries_on_whole_cell_multiples(n: int) -> None:
    dx = dx_of_grid(n)
    s = snap_cross_geometry(CROSS_LIMB_HALF_M, CROSS_ARM_HALF_M, dx)
    for key in ("limb_half_m", "arm_half_m"):
        cells = s[key] / dx
        assert abs(cells - round(cells)) < 1e-9, (key, cells)


def test_snap_is_a_no_op_at_grid_181_where_the_original_is_already_exact() -> None:
    """Grid 181 has dx = 1/3 mm, so 11.000 mm is 33 cells and 11/3 mm is 11.

    This is the identity control for the whole variant: at this grid the snap
    must change nothing at all, so a snapped run and an unsnapped run at grid
    181 are the same run.
    """
    dx = dx_of_grid(181)
    s = snap_cross_geometry(CROSS_LIMB_HALF_M, CROSS_ARM_HALF_M, dx)
    assert s["limb_half_m"] == pytest.approx(CROSS_LIMB_HALF_M, rel=0, abs=1e-15)
    assert s["arm_half_m"] == pytest.approx(CROSS_ARM_HALF_M, rel=0, abs=1e-15)
    assert s["limb_cells"] == 33 and s["arm_cells"] == 11
    assert abs(s["d_limb_frac_of_cell"]) < 1e-9
    assert abs(s["d_arm_frac_of_cell"]) < 1e-9


@pytest.mark.parametrize("n", [96, 120, 160, 180, 200, 240, 360, 181])
def test_snap_moves_each_boundary_by_at_most_half_a_cell(n: int) -> None:
    dx = dx_of_grid(n)
    s = snap_cross_geometry(CROSS_LIMB_HALF_M, CROSS_ARM_HALF_M, dx)
    assert abs(s["d_limb_m"]) <= 0.5 * dx + 1e-15
    assert abs(s["d_arm_m"]) <= 0.5 * dx + 1e-15
    assert abs(s["d_limb_frac_of_cell"]) <= 0.5 + 1e-12
    assert abs(s["d_arm_frac_of_cell"]) <= 0.5 + 1e-12


def test_snapped_and_unsnapped_cases_are_identical_where_the_snap_is_a_no_op() -> None:
    """The identity control for the whole snap code path.

    At grid 181 the snap changes no dimension, so it must produce the SAME
    case: the same polygon, the same part mask, the same sub-cell target. This
    is what rules out the snap silently perturbing something else, in
    particular the explicit `thickness` key it has to write into the part
    definition to keep the arm from falling back to width / 3.

    Grid 181 is deliberately not in the ladder itself: with an odd grid number
    the domain builder puts a grid point at the origin, so a boundary at a
    whole multiple of the cell size lands ON grid points rather than between
    them, which is the ambiguous case rather than the exact one.
    """
    sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
    from adjoint2d import chi_area, robust_rot as rr
    from adjoint2d.library_solve import shape_config
    from adjoint2d.pins import build_case, load_cfg

    from rot_ladder_variants import snap_cross_cfg

    base = rr.cfg_at_grid(load_cfg(shape_config("cross")), 181)
    snapped, info = snap_cross_cfg(base, 181)
    assert info["is_no_op"]

    c0 = build_case(base)
    c1 = build_case(snapped)
    assert np.array_equal(c0.part_mask, c1.part_mask)
    assert np.array_equal(c0.fill_frac, c1.fill_frac)
    chi0, _ = chi_area.chi_from_cfg(base, c0.x, c0.y)
    chi1, _ = chi_area.chi_from_cfg(snapped, c1.x, c1.y)
    assert np.array_equal(chi0, chi1)


# ---------------------------------------------------------------------------
# variant B: the smoothed melt indicator
# ---------------------------------------------------------------------------

def test_gaussian_melt_indicator_at_zero_width_is_the_binary_threshold() -> None:
    """The width-shrink limit. Zero smoothing must be the original metric."""
    rng = np.random.default_rng(0)
    phi = rng.random((37, 41))
    assert np.array_equal(gaussian_melt_indicator(phi, 0.0), phi >= 0.5)


def test_melt_area_fill_reproduces_a_straight_binary_front_exactly() -> None:
    """A straight melt front already on cell edges gives back the binary set.

    This is the same degeneracy `chi_area` has against the binary raster when
    the geometric boundary sits on a cell edge, and it is what makes the
    sub-cell melt fill a like-for-like replacement rather than a new metric.
    """
    b = np.zeros((21, 23))
    b[:, 7:17] = 1.0                      # a strip spanning the whole array
    f = melt_area_fill(b, n_sub=8)
    assert np.array_equal(f, b)


def test_melt_area_fill_rounds_a_convex_corner_by_a_measured_amount() -> None:
    """The limit of a bilinear reconstruction, measured rather than assumed.

    At a right-angle corner the reconstruction cannot be exact: the corner cell
    of a convex corner loses area and the corner cell of a concave corner gains
    the same amount. The number is pinned here so the report can quote it, and
    so a change of the sampling would fail this test rather than pass silently.
    """
    b = np.zeros((13, 13))
    b[4:9, 4:9] = 1.0                     # a square: four convex corners
    f = melt_area_fill(b, n_sub=8)
    corner = f[4, 4]
    assert 0.85 < corner < 0.95
    assert f[6, 6] == 1.0                 # the interior is untouched
    # the whole-field area error is four corners' worth and nothing else
    assert abs(f.sum() - b.sum() + 4.0 * (1.0 - corner)) < 1e-12


def test_melt_area_fill_is_continuous_in_the_front_position() -> None:
    """Sweep a straight melt front across one cell.

    The binary count moves in whole-row jumps; the area fill must move
    smoothly. This is the property the whole variant rests on.
    """
    ny, nx = 16, 64
    xs = np.arange(nx, dtype=float)
    binary_totals, fill_totals = [], []
    for shift in np.linspace(0.0, 1.0, 21):
        # a front of half width 1.5 cells, so phi is not clipped near it
        raw = 0.5 + (30.0 + shift - xs) / 3.0
        phi = np.clip(np.tile(raw, (ny, 1)), 0.0, 1.0)
        binary_totals.append(float(np.sum(phi >= 0.5)))
        fill_totals.append(float(np.sum(melt_area_fill(phi, n_sub=16))))
    b_jump = float(np.max(np.abs(np.diff(binary_totals))))
    f_jump = float(np.max(np.abs(np.diff(fill_totals))))
    assert b_jump >= ny                       # a whole column enters at once
    assert f_jump < 0.25 * ny                 # the fill creeps in instead
    # and over the full cell both must have swept the same one column
    assert abs((fill_totals[-1] - fill_totals[0]) - ny) < 0.05 * ny


def test_area_iou_is_the_campaign_function_and_not_a_second_copy() -> None:
    """The overlap convention must be the campaign's, not a re-derivation."""
    sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
    from adjoint2d import topopt_objective as tobj

    from rot_ladder_variants import area_iou

    rng = np.random.default_rng(11)
    a = rng.random((17, 19))
    b = rng.random((17, 19))
    assert area_iou(a, b) == tobj.area_iou(a, b)


def test_melt_area_fill_is_bounded_and_matches_the_binary_area_on_average() -> None:
    rng = np.random.default_rng(3)
    phi = np.clip(rng.normal(0.5, 0.4, size=(48, 48)), 0.0, 1.0)
    f = melt_area_fill(phi, n_sub=8)
    assert f.min() >= 0.0 and f.max() <= 1.0
    # no systematic area bias against the binary reading beyond the boundary band
    assert abs(f.sum() - float(np.sum(phi >= 0.5))) < 0.10 * float(np.sum(phi >= 0.5))
