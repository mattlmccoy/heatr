"""Pure-logic tests for the Studio direct-solve geometry front end (spec 7e).

RED-FIRST. Everything here is numpy-only so it runs in the geo-prewarp venv:

    ./.venv312/bin/python -m pytest solve3d/tests/test_studio_geom.py

The solve chain itself (dolfinx) is validated by the budget-6 integration run,
not here.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import studio_geom as sg

L = 0.060


def _centers(n: int, L_: float = L) -> np.ndarray:
    h = L_ / n
    return (np.arange(n) + 0.5) * h - L_ / 2.0


def _prism_part(n: int = 16, k: int = 6, kz: tuple[int, int] = (5, 10)):
    """A square prism: identical occupied slices over a z band."""
    part = np.zeros((n, n, n), dtype=bool)
    lo = (n - k) // 2
    part[lo:lo + k, lo:lo + k, kz[0]:kz[1]] = True
    return part


# --------------------------------------------------------------------- #
# 1. Extrusion detection -- DETECTED, never assumed from a shape name
# --------------------------------------------------------------------- #
def test_detect_extrusion_accepts_a_prism():
    part = _prism_part()
    rec = sg.detect_extrusion(part)
    assert rec["is_extruded"] is True
    assert rec["refusal"] is None
    assert rec["z_index_lo"] == 5 and rec["z_index_hi"] == 9
    assert rec["n_occupied_slices"] == 5


def test_detect_extrusion_refuses_a_taper_and_names_phase_e():
    part = _prism_part()
    # taper: one slice loses a row, so the slices are not identical
    part[5, :, 7] = False
    rec = sg.detect_extrusion(part)
    assert rec["is_extruded"] is False
    assert rec["n_mismatched_slices"] >= 1
    msg = rec["refusal"]
    assert "Phase E" in msg and "tet" in msg.lower() and "STL" in msg


def test_detect_extrusion_refuses_an_empty_part():
    rec = sg.detect_extrusion(np.zeros((8, 8, 8), bool))
    assert rec["is_extruded"] is False
    assert "empty" in rec["refusal"].lower()


# --------------------------------------------------------------------- #
# 2. Outline -- sub-cell marching squares, NOT the staircase
# --------------------------------------------------------------------- #
def test_square_outline_is_one_ccw_ring_with_cut_corners():
    n, k = 16, 6
    h = L / n
    m = np.zeros((n, n), bool)
    lo = (n - k) // 2
    m[lo:lo + k, lo:lo + k] = True
    rings = sg.outline_rings(m, h, L)
    assert len(rings) == 1
    area = sg.ring_area(rings[0])
    # the contour runs half a cell outside the outermost cell centres, so the
    # body spans k*h, with each of the four corners cut by an h/2 x h/2 triangle
    assert area == pytest.approx((k * h) ** 2 - 4 * (h * h / 8.0), rel=1e-12)
    assert area > 0.0                      # CCW: outer ring, inside on the left


def test_square_outline_is_not_the_staircase():
    n, k = 16, 6
    h = L / n
    m = np.zeros((n, n), bool)
    m[5:5 + k, 5:5 + k] = True
    rings = sg.outline_rings(m, h, L)
    assert sg.ring_area(rings[0]) < (k * h) ** 2      # corners are cut


def test_annulus_gives_an_outer_ccw_ring_and_an_inner_cw_hole():
    n = 40
    h = L / n
    c = _centers(n)
    X, Y = np.meshgrid(c, c, indexing="ij")
    r = np.hypot(X, Y)
    m = (r <= 0.012) & (r >= 0.005)
    rings = sg.outline_rings(m, h, L)
    assert len(rings) == 2
    areas = [sg.ring_area(rg) for rg in rings]
    assert sum(a > 0 for a in areas) == 1            # exactly one outer ring
    assert sum(a < 0 for a in areas) == 1            # exactly one hole
    assert sum(areas) == pytest.approx(m.sum() * h * h, rel=0.05)


def test_points_in_rings_reproduces_the_voxel_mask_at_cell_centres():
    n = 40
    h = L / n
    c = _centers(n)
    X, Y = np.meshgrid(c, c, indexing="ij")
    r = np.hypot(X, Y)
    m = (r <= 0.012) & (r >= 0.005)
    rings = sg.outline_rings(m, h, L)
    got = sg.points_in_rings(rings, X.ravel(), Y.ravel()).reshape(m.shape)
    assert np.array_equal(got, m)


def test_outline_refuses_two_disconnected_bodies():
    n = 24
    h = L / n
    m = np.zeros((n, n), bool)
    m[2:6, 2:6] = True
    m[14:18, 14:18] = True
    with pytest.raises(sg.OutlineError, match="disconnected|2 outer"):
        sg.outline_rings(m, h, L)


# --------------------------------------------------------------------- #
# 3. The validation tube part (runner voxelization conventions)
# --------------------------------------------------------------------- #
def test_tube_part_is_extruded_and_has_the_requested_extent():
    part = sg.make_tube_part(n=32, r_outer_m=0.012, r_bore_m=0.005,
                             height_m=0.020)
    assert part.shape == (32, 32, 32)
    rec = sg.detect_extrusion(part)
    assert rec["is_extruded"] is True
    h = L / 32
    zc = _centers(32)
    occ = np.where(part.any(axis=(0, 1)))[0]
    assert np.all(np.abs(zc[occ]) <= 0.010)
    assert (occ.size * h) == pytest.approx(0.020, abs=1.5 * h)
    # the bore is really there
    mid = part[:, :, occ[occ.size // 2]]
    assert not mid[16, 16]


# --------------------------------------------------------------------- #
# 4. The delivered artifact format (the Studio's consumer contract)
# --------------------------------------------------------------------- #
def test_map_npz_has_the_phase_c_keys_and_loads_through_dg0_to_voxel(tmp_path):
    from studio3d.transfer import dg0_to_voxel
    part = sg.make_tube_part(n=24, r_outer_m=0.012, r_bore_m=0.005,
                             height_m=0.020)
    h = L / 24
    c = _centers(24)
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    cen = np.column_stack([X[part], Y[part], Z[part]])
    s = np.full(cen.shape[0], 0.8)
    vols = np.full(cen.shape[0], h ** 3)
    p = tmp_path / "studio_solve_map.npz"
    sg.write_map_npz(p, cen, s, vols, v_raw=s.copy())
    with np.load(p) as d:
        assert set(("centroids", "s_map", "volumes", "v_raw")) <= set(d.files)
        rec = dg0_to_voxel(d["centroids"], d["s_map"], d["volumes"], part,
                           chamber_m=L)
    assert rec["state"] == "measured_and_passed"
    assert rec["sat"][part].mean() == pytest.approx(0.8, rel=1e-9)
