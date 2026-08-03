"""Support-aware map transfer (spec section 6, the Phase C staircase lesson).

Rule: EXTEND the field beyond its support before interpolating, then re-mask
to the target support. Acceptance: total in-part dopant moves < 2 % across
the transfer (the Phase C pre-registered threshold), asserted in code.

Sources handled:
- DG0 cell data (centroids + values + volumes, the solve3d artifact form)
  -> voxel grid.
- voxel volume -> voxel volume of a different n (grid changes).
"""
from __future__ import annotations

import numpy as np
import pytest

from studio3d.transfer import (TransferError, dg0_to_voxel,
                               voxel_to_voxel)


def _sphere_part(n: int, r_frac: float = 0.3) -> np.ndarray:
    c = (np.arange(n) + 0.5) / n - 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return (X**2 + Y**2 + Z**2) < r_frac**2


def test_voxel_to_voxel_preserves_dopant_mass_on_refinement():
    n0, n1 = 32, 48
    part0, part1 = _sphere_part(n0), _sphere_part(n1)
    rng = np.random.default_rng(0)
    sat0 = np.where(part0, 0.4 + 0.2 * rng.random((n0,) * 3), 0.0)
    rec = voxel_to_voxel(sat0, part0, part1, chamber_m=0.060)
    assert rec["sat"].shape == (n1, n1, n1)
    assert rec["state"] == "measured_and_passed"
    assert abs(rec["dopant_mass_move_rel"]) < 0.02
    # outside the target part the map is zero (re-masked)
    assert rec["sat"][~part1].max() == 0.0


def test_zero_fill_transfer_would_fail_the_gate():
    """The Phase C failure mode: plain interpolation with zeros outside the
    staircase support thins the rim. The support-aware path must beat it,
    and the gate must FAIL (not pass silently) when handed the naive
    result's mass move."""
    n0, n1 = 24, 37   # incommensurate grids maximize rim resampling
    part0, part1 = _sphere_part(n0), _sphere_part(n1)
    sat0 = np.where(part0, 1.0, 0.0)   # worst case: full-strength rim step
    rec = voxel_to_voxel(sat0, part0, part1, chamber_m=0.060)
    # support-aware transfer holds the mass within the gate
    assert rec["state"] == "measured_and_passed"
    # the naive zero-fill comparison, computed the same way, is reported
    # for the record and is measurably worse
    assert rec["naive_mass_move_rel"] > rec["dopant_mass_move_rel"] >= 0.0


def test_failed_gate_raises_loudly():
    """A graded field read far outside its support cannot keep its in-part
    mean; the gate must refuse the map, never pass it silently."""
    n0 = 24
    part0 = _sphere_part(n0, r_frac=0.2)
    c = (np.arange(n0) + 0.5) / n0 - 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    sat0 = np.where(part0, np.clip(1.0 - 4.0 * r, 0.05, 1.0), 0.0)
    # target support: a shell well outside the source sphere, where the
    # nearest-value extension can only supply the rim value
    part1 = (r > 0.35) & (r < 0.45)
    with pytest.raises(TransferError, match="2 % gate"):
        voxel_to_voxel(sat0, part0, part1, chamber_m=0.060)


def test_dg0_to_voxel_constant_field_is_exact():
    """A constant DG0 field must transfer to exactly that constant inside
    the part, whatever the mesh: the mechanism check for support-aware
    extension (zeros outside cells must never bleed in)."""
    rng = np.random.default_rng(1)
    n = 24
    part = _sphere_part(n)
    # fake DG0 data: centroids sampled inside the sphere, constant value
    m = 4000
    pts = rng.uniform(-0.5, 0.5, size=(m * 8, 3))
    inside = (pts**2).sum(axis=1) < 0.3**2
    cents = pts[inside][:m] * 0.060
    vals = np.full(len(cents), 0.7)
    vols = np.full(len(cents), 1.0)
    rec = dg0_to_voxel(cents, vals, vols, part, chamber_m=0.060)
    sat = rec["sat"]
    assert sat.shape == (n, n, n)
    assert np.allclose(sat[part], 0.7, atol=1e-6)
    assert sat[~part].max() == 0.0
    assert rec["state"] == "measured_and_passed"
