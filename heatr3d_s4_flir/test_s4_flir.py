"""Gate S4 red-green tests: registration operators, metrics, and the
segment-chaining equivalence that the time-resolved prediction relies on.

Run:  ./.venv312/bin/python -m pytest heatr3d_s4_flir/test_s4_flir.py -q
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from heatr3d import Grid, Params, run  # noqa: E402
from s4_flir_lib import (  # noqa: E402
    chamfer_mm,
    corner_edge_contrast,
    largest_component_roi,
    min_area_rect,
    pattern_correlation,
    resample_unit_square,
)


# --------------------------------------------------------------------------- #
# registration operators
# --------------------------------------------------------------------------- #
def _rotated_square_mask(n=200, side=60.0, angle_deg=17.0, cx=100.0, cy=95.0):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    a = np.deg2rad(angle_deg)
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    return (np.abs(u) <= side / 2) & (np.abs(v) <= side / 2)


def test_min_area_rect_recovers_a_rotated_square():
    mask = _rotated_square_mask(side=60.0, angle_deg=17.0, cx=100.0, cy=95.0)
    rect = min_area_rect(mask)
    assert rect.cx == pytest.approx(100.0, abs=1.0)
    assert rect.cy == pytest.approx(95.0, abs=1.0)
    assert rect.side_u == pytest.approx(60.0, abs=1.5)
    assert rect.side_v == pytest.approx(60.0, abs=1.5)
    # angle is only defined mod 90 deg for a square
    assert (rect.angle_deg % 90.0) == pytest.approx(17.0, abs=1.5)


def test_largest_component_roi_drops_a_detached_hot_blob():
    field = np.full((80, 80), 20.0)
    field[20:60, 20:60] = 200.0           # the part
    field[5:10, 70:75] = 300.0            # detached electrode glow
    roi = largest_component_roi(field, ambient=20.0, frac=0.5)
    assert roi[30, 30]
    assert not roi[7, 72]
    assert roi.sum() == pytest.approx(40 * 40, rel=0.05)


def test_resample_unit_square_is_orientation_and_scale_invariant():
    """The same physical pattern at two sizes/rotations resamples to the same
    part-relative field (that is the whole point of the registration)."""
    def field_at(n, side, angle, cx, cy):
        yy, xx = np.mgrid[0:n, 0:n].astype(float)
        a = np.deg2rad(angle)
        dx, dy = xx - cx, yy - cy
        u = (dx * np.cos(a) + dy * np.sin(a)) / side
        v = (-dx * np.sin(a) + dy * np.cos(a)) / side
        inside = (np.abs(u) <= 0.5) & (np.abs(v) <= 0.5)
        # corner-hot ("X") pattern on a part that is everywhere hotter than the
        # background, as a real part in cold powder is
        f = 150.0 + 50.0 * (u ** 2 + v ** 2) / 0.5
        return np.where(inside, f, 20.0)

    fa = field_at(200, 60.0, 0.0, 100.0, 100.0)
    fb = field_at(300, 110.0, 31.0, 150.0, 140.0)
    ra = resample_unit_square(fa, min_area_rect(largest_component_roi(fa, 20.0)), ng=48)
    rb = resample_unit_square(fb, min_area_rect(largest_component_roi(fb, 20.0)), ng=48)
    assert pattern_correlation(ra, rb) > 0.99


# --------------------------------------------------------------------------- #
# metric operators
# --------------------------------------------------------------------------- #
def _corner_hot(ng=48):
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, ng), np.linspace(-0.5, 0.5, ng), indexing="ij")
    return 1.0 + (u ** 2 + v ** 2)


def _center_hot(ng=48):
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, ng), np.linspace(-0.5, 0.5, ng), indexing="ij")
    return 2.0 - (u ** 2 + v ** 2)


def test_corner_edge_contrast_separates_corner_hot_from_center_hot():
    assert corner_edge_contrast(_corner_hot()) > 0
    assert corner_edge_contrast(_center_hot()) < 0


def test_pattern_correlation_is_negative_for_inverted_topology():
    assert pattern_correlation(_corner_hot(), _center_hot()) < -0.9
    assert pattern_correlation(_corner_hot(), _corner_hot()) == pytest.approx(1.0)


def test_chamfer_mm_measures_hot_set_displacement():
    ng = 41
    ramp = np.linspace(0.0, 0.5, ng)
    a = ramp[::-1, None] + ramp[None, ::-1]            # hot toward index (0,0)
    b = a.copy()
    assert chamfer_mm(a, b, part_mm=40.0, top_frac=0.10) == pytest.approx(0.0, abs=1e-9)
    shifted = np.roll(a, 2, axis=0)                    # 2-cell displacement
    d_small = chamfer_mm(a, shifted, part_mm=40.0, top_frac=0.10)
    c = a[:, ::-1].copy()                              # hot toward the opposite corner
    d_far = chamfer_mm(a, c, part_mm=40.0, top_frac=0.10)
    assert 0.0 < d_small < 3.0
    assert d_far > 15.0


# --------------------------------------------------------------------------- #
# heatr3d segment chaining (the time-resolved prediction depends on it)
# --------------------------------------------------------------------------- #
def test_chained_T0_override_segments_equal_one_long_march():
    """densify=False state is T alone, so restarting from T_final must reproduce
    a single long march exactly. If this fails, every intermediate field in the
    S4 scoring is wrong."""
    g = Grid(n=16, L=0.030)
    part = np.zeros((16, 8, 16), bool)
    part[4:12, 0:4, 4:12] = True
    p = dataclasses.replace(Params(), phase_update="enthalpy", dt_s=0.2, t_pc_c=185.0)
    long_run = run(g, part, p, max_time_s=8.0, phi_target=1.5)
    seg = run(g, part, p, max_time_s=4.0, phi_target=1.5)
    seg2 = run(g, part, p, max_time_s=4.0, phi_target=1.5,
               T0_override=seg.T_final, qrf_override=seg.Qrf)
    assert np.array_equal(long_run.T_final, seg2.T_final)
