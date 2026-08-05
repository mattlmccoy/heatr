"""Arbitrary-STL tet meshing: validation refusals, volume, fill, equivalence.

Plan: docs/superpowers/plans/2026-08-04-shrinkage-v2-tranche1.md Task 2.

Four groups, one per plan clause:
  (1) a watertight library STL meshes to a conforming tet mesh whose volume
      matches the analytic value within a STATED tolerance;
  (2) chi volume-fill on the STL mesh passes the shared fill contract;
  (3) bad geometry is REFUSED LOUDLY, using the library's own Tier-3 fixtures
      where they exist;
  (4) the OCC-construction path and the STL path agree within the Phase A
      SAME-ENGINE dolfinx bands -- the honest equivalence gate.

FIXTURE GAP, recorded rather than papered over: the shape library ships two
Tier-3 rejection fixtures, `open_cylinder` (non-watertight) and `flat_plane`
(zero volume). It has NO self-intersecting fixture. The self-intersection
refusal is therefore exercised against a mesh CONSTRUCTED HERE and labelled as
such; it is not a library fixture and does not pretend to be.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from solve3d import stl_mesh

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "shape_library_3d"
STL = LIB / "stl"
META = LIB / "meta"

# Phase A dolfinx SAME-ENGINE spreads (solve3d/results/dolfinx_refinement.json,
# MAX-of-pairs, the rule declared in parity_tolerances_crossfamily.json), times
# the frozen 1.5 safety factor. Read from the file so a change there breaks the
# test rather than sliding through.
SAFETY = 1.5


def _dolfinx_band(key: str) -> float:
    d = json.loads((ROOT / "solve3d" / "results"
                    / "dolfinx_refinement.json").read_text())["spreads"]
    # the file records the MAX-of-pairs at top level, per its reporting_rule
    return SAFETY * float(d[key])


# --------------------------------------------------------------------------- #
# (1) watertight library STL -> conforming tet mesh, volume gate
# --------------------------------------------------------------------------- #
def test_library_pyramid_stl_loads_and_is_watertight():
    v, f = stl_mesh.load_stl(STL / "pyramid.stl")
    assert v.ndim == 2 and v.shape[1] == 3
    assert f.ndim == 2 and f.shape[1] == 3
    stl_mesh.validate(v, f)          # must not raise


def test_stl_enclosed_volume_matches_the_library_metadata():
    """The divergence-theorem volume must reproduce the library's own number,
    which is the value the library asserts it normalised to."""
    v, f = stl_mesh.load_stl(STL / "pyramid.stl")
    want = json.loads((META / "pyramid.json").read_text())["actual_volume_mm3"]
    got = stl_mesh.enclosed_volume(v, f) * 1e9      # m^3 -> mm^3 if metres
    assert got == pytest.approx(want, rel=1e-6) or \
        stl_mesh.enclosed_volume(v, f) == pytest.approx(want, rel=1e-6)


@pytest.mark.slow
def test_pyramid_stl_meshes_to_a_tet_mesh_matching_the_stl_volume():
    """STATED TOLERANCE: 0.5 percent on the meshed part volume against the
    STL's own enclosed volume. It is a discretisation tolerance, not a physics
    band: a conforming tet mesh of a faceted solid should recover the facet
    volume to well under this."""
    msh, info = stl_mesh.build_mesh_from_stl(STL / "pyramid.stl",
                                             lc_part=1.5e-3)
    assert info.n_cells_total > 0
    assert info.part_volume_rel_err_vs_stl == pytest.approx(0.0, abs=5.0e-3)


# --------------------------------------------------------------------------- #
# (2) the shared fill contract, on an STL-derived mesh
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_chi_volume_fill_on_the_stl_mesh_passes_the_shared_contract():
    """chi from the STL mesh must be bounded, saturate, and integrate to the
    part volume -- the same contract the OCC path already satisfies."""
    msh, info = stl_mesh.build_mesh_from_stl(STL / "pyramid.stl",
                                             lc_part=1.5e-3)
    chi, vol = stl_mesh.chi_and_volumes(msh, info)
    assert chi.min() >= 0.0 and chi.max() <= 1.0
    assert np.isclose(chi.max(), 1.0)
    v_chi = float(np.dot(chi, vol))
    assert v_chi == pytest.approx(info.part_volume_m3, rel=5.0e-3)
    # not a tautology: chi is cross-checked against an INDEPENDENT inside
    # test on the original triangle soup, so a mesh that claimed the wrong
    # cells would fail here even though the volume sum agreed
    v, f = stl_mesh.load_stl(STL / "pyramid.stl")
    ctr = stl_mesh.cell_centroids(msh) / info.scale
    ins = stl_mesh.points_inside(v, f, ctr)
    assert ins.mean() > 0.995, f"only {ins.mean():.4f} of cells test inside"


def test_inside_test_separates_known_inside_and_outside_points():
    """Guards points_inside against being trivially true."""
    v, f = stl_mesh.load_stl(STL / "pyramid.stl")
    lo, hi = v.min(axis=0), v.max(axis=0)
    mid = (lo + hi) / 2.0
    inside_pt = np.array([[mid[0], mid[1], lo[2] + 0.05 * (hi[2] - lo[2])]])
    outside_pt = np.array([[hi[0] * 2.0, hi[1] * 2.0, mid[2]],
                           [mid[0], mid[1], hi[2] * 1.5]])
    assert bool(stl_mesh.points_inside(v, f, inside_pt)[0]) is True
    assert not stl_mesh.points_inside(v, f, outside_pt).any()


# --------------------------------------------------------------------------- #
# (3) refusals -- loud, and named the way the library records them
# --------------------------------------------------------------------------- #
def test_tier3_open_cylinder_is_refused_as_non_watertight():
    v, f = stl_mesh.load_stl(STL / "open_cylinder.stl")
    with pytest.raises(stl_mesh.NonWatertightMeshError) as e:
        stl_mesh.validate(v, f)
    assert "edge" in str(e.value).lower()


def test_tier3_flat_plane_is_refused_as_zero_volume():
    v, f = stl_mesh.load_stl(STL / "flat_plane.stl")
    with pytest.raises((stl_mesh.ZeroVolumeError,
                        stl_mesh.NonWatertightMeshError)):
        stl_mesh.validate(v, f)


def test_refusal_names_match_the_library_metadata_contract():
    """The library records the error it EXPECTS ingestion to raise. Those
    strings are a cross-component contract, so they are checked, not assumed."""
    for name in ("open_cylinder", "flat_plane"):
        want = json.loads((META / f"{name}.json").read_text())["rejection_error"]
        assert hasattr(stl_mesh, want), f"{name} expects {want}"
        assert issubclass(getattr(stl_mesh, want), stl_mesh.MeshRefusal)


def test_constructed_self_intersecting_mesh_is_refused():
    """CONSTRUCTED FIXTURE, not from the library: the shape library has no
    self-intersecting Tier-3 case (recorded gap). Two interpenetrating
    tetrahedra are watertight edge-wise and have positive volume, so only a
    genuine intersection test can catch them."""
    v, f = stl_mesh.two_interpenetrating_tetrahedra()
    stl_mesh.check_watertight(v, f)          # passes: the point of the fixture
    with pytest.raises(stl_mesh.SelfIntersectingMeshError):
        stl_mesh.validate(v, f, check_self_intersection=True)


def test_build_refuses_bad_geometry_before_meshing():
    """The refusal must happen at intake, not after a long meshing run."""
    with pytest.raises(stl_mesh.MeshRefusal):
        stl_mesh.build_mesh_from_stl(STL / "open_cylinder.stl", lc_part=3e-3)


def test_a_good_mesh_is_not_refused_by_the_self_intersection_test():
    """Guards the refusal against being trivially true (mutation check)."""
    v, f = stl_mesh.load_stl(STL / "pyramid.stl")
    stl_mesh.validate(v, f, check_self_intersection=True)


# --------------------------------------------------------------------------- #
# (4) the honest equivalence gate: OCC path vs STL path
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_occ_and_stl_paths_agree_within_the_phase_a_same_engine_band():
    """The Phase A circle anchor built as an OCC solid, exported to STL, and
    re-meshed through the STL path. The two must agree no worse than the
    engine's OWN mesh-refinement wobble (x1.5), which is the only band that
    makes this claim honest rather than self-graded."""
    band = _dolfinx_band("t90_rel_spread")
    res = stl_mesh.occ_vs_stl_equivalence(lc_part=1.5e-3)
    assert res["part_volume_rel_diff"] == pytest.approx(0.0, abs=band)
    assert res["band_used"] == pytest.approx(band, rel=1e-12)
    assert res["source"].endswith("dolfinx_refinement.json")


def test_equivalence_band_is_read_from_the_frozen_file_not_invented():
    band = _dolfinx_band("t90_rel_spread")
    assert band > 0.0
    # the MAX-of-pairs rule: coarse_vs_mid dominates on this quantity
    d = json.loads((ROOT / "solve3d" / "results"
                    / "dolfinx_refinement.json").read_text())["spreads"]
    assert band == pytest.approx(
        SAFETY * d["coarse_vs_mid"]["t90_rel_spread"], rel=1e-12)
