"""Phase A close-out: shape-metric math (IoU, melt-front distance, bed melt)
and the cross-family combination rule.

RUNS IN THE geo-prewarp VENV (needs scipy for the EDT):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ./.venv312/bin/python -m pytest solve3d/tests/test_shape_metrics.py
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import gates, shape_metrics as sm


def _disc(n: int, h: float, radius: float) -> np.ndarray:
    c = (np.arange(n) + 0.5) * h - n * h / 2.0
    X, Y = np.meshgrid(c, c, indexing="ij")
    return np.sqrt(X ** 2 + Y ** 2) <= radius


# --------------------------------------------------------------------------- #
# IoU
# --------------------------------------------------------------------------- #
def test_iou_is_one_for_identical_regions_and_zero_for_disjoint():
    a = _disc(64, 1e-3, 0.010)
    assert sm.iou(a, a) == pytest.approx(1.0)
    b = np.zeros_like(a)
    b[:4, :4] = True
    assert sm.iou(a & ~a, b) == pytest.approx(0.0)


def test_iou_matches_the_closed_form_for_two_concentric_discs():
    """Concentric discs of radii r and R>r: intersection = smaller disc,
    union = larger, so IoU -> (r/R)^2 as the grid refines."""
    h = 5e-5
    n = 800
    a, b = _disc(n, h, 0.008), _disc(n, h, 0.010)
    assert sm.iou(a, b) == pytest.approx((0.008 / 0.010) ** 2, rel=2e-3)


def test_iou_of_empty_pair_is_nan_not_a_silent_one():
    """Two empty regions are NOT perfect agreement; they are no information.
    Returning 1.0 here would be a false-green (rule: unknown must never render
    as healthy)."""
    z = np.zeros((8, 8), bool)
    assert np.isnan(sm.iou(z, z))


# --------------------------------------------------------------------------- #
# Melt-front position
# --------------------------------------------------------------------------- #
def test_symmetric_surface_distance_is_zero_for_identical_regions():
    a = _disc(200, 2e-4, 0.010)
    assert sm.symmetric_surface_distance_mm(a, a, 2e-4) == pytest.approx(0.0)


def test_symmetric_surface_distance_recovers_a_known_radial_offset():
    """Two concentric discs whose radii differ by 1.0 mm: every point of one
    boundary is 1.0 mm from the other boundary, so the symmetric surface
    distance must be 1.0 mm to within one pixel."""
    h = 1e-4                      # 0.1 mm pixels
    a, b = _disc(400, h, 0.008), _disc(400, h, 0.009)
    d = sm.symmetric_surface_distance_mm(a, b, h)
    assert abs(d - 1.0) < 0.15, d


def test_symmetric_surface_distance_is_nan_when_a_region_is_empty():
    a = _disc(64, 1e-3, 0.010)
    assert np.isnan(sm.symmetric_surface_distance_mm(a, np.zeros_like(a), 1e-3))


# --------------------------------------------------------------------------- #
# Bed melt (the hard side of the asymmetric objective)
# --------------------------------------------------------------------------- #
def test_out_of_part_fraction_counts_only_melt_outside_the_part():
    part = _disc(200, 2e-4, 0.008)
    melt = _disc(200, 2e-4, 0.009)          # spills 1 mm past the part
    f = sm.out_of_part_fraction(melt, part)
    expected = ((0.009 ** 2 - 0.008 ** 2) / 0.008 ** 2)
    assert f == pytest.approx(expected, rel=0.02)
    assert sm.out_of_part_fraction(part, part) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# The cross-family combination rule (declared BEFORE any number is computed)
# --------------------------------------------------------------------------- #
def test_cross_family_rule_is_the_triangle_inequality_sum_times_the_safety():
    assert gates.COMBINATION_RULE == "sum"
    assert gates.combine_spreads(0.01, 0.02, safety=1.5) == pytest.approx(0.045)
    # it must be a BOUND, i.e. never smaller than either input's safety-scaled
    # value, which root-sum-square would violate
    rss = 1.5 * np.hypot(0.01, 0.02)
    assert gates.combine_spreads(0.01, 0.02, safety=1.5) > rss
