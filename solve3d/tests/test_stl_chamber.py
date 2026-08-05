"""Chamber (bed) embedding for arbitrary-STL parts: the TRANCHE 1 blocker.

TRANCHE1_REPORT.md recorded `with_chamber=True` as failing in tetgen with
"PLC Error: a segment and a facet intersect", and attributed it to the inner
surface being a discrete entity built from parsed STL triangles. THAT
ATTRIBUTION WAS WRONG, and the first test here is what shows it: the chamber
box fails on its own, with no STL part present at all.

`_add_box_surface_loop` built each of the box's 12 edges TWICE -- once per
adjoining face -- because `gmsh.model.geo.addLine` does not deduplicate. Two
coincident curves are meshed independently, so the two faces meeting at that
edge carry different 1-D node sets and the shell is not conforming with
ITSELF. tetgen's complaint is literally true and has nothing to do with the
part. A box has 8 points and 12 curves; the helper produced 8 and 24.

The gates below are the ones TRANCHE1_REPORT.md pre-registered for whatever
route fixed this: part volume against the STL's own enclosed volume, the
shared fill contract, and the same OCC-vs-STL Phase A same-engine band.
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
TAMPER = (ROOT.parents[1] / "software" / "meteor" / "tools" / "uploads"
          / "feb850ec" / "Part Studio 1 - Tamper.stl")

SAFETY = 1.5
L_CHAMBER_M = 0.060                    # forward.L_DOMAIN, the frozen chamber


def _dolfinx_band(key: str = "t90_rel_spread") -> float:
    d = json.loads((ROOT / "solve3d" / "results"
                    / "dolfinx_refinement.json").read_text())["spreads"]
    return SAFETY * float(d[key])



def _raycast_agreement(msh, info, verts, faces, chi, n_sample: int = 4000,
                       seed: int = 0) -> float:
    """Fraction of cells whose chi agrees with an INDEPENDENT ray cast on the
    original triangle soup.

    SUBSAMPLED, and that is a stated limitation rather than a hidden one:
    `points_inside` loops in Python over points, so all ~30k cells of a chamber
    mesh costs minutes. A fixed-seed sample of 4000 cells is drawn from the
    WHOLE mesh (part and bed), so a route that tagged the wrong cells still
    fails: getting 0.995 agreement on 4000 uniformly drawn cells by chance is
    not a thing that happens.
    """
    ctr_stl = stl_mesh.mesh_points_to_stl_native(
        stl_mesh.cell_centroids(msh), info)
    n = ctr_stl.shape[0]
    idx = (np.arange(n) if n <= n_sample else
           np.random.default_rng(seed).choice(n, n_sample, replace=False))
    ins = stl_mesh.points_inside(verts, faces, ctr_stl[idx])
    return float((ins == (chi[idx] > 0.5)).mean())


# --------------------------------------------------------------------------- #
# (0) the actual root cause, isolated and fast
# --------------------------------------------------------------------------- #
def test_chamber_box_shell_has_exactly_one_curve_per_edge():
    """A box has 8 corners and 12 edges. Building 24 curves gives every face
    pair its own copy of the shared edge, so the shell cannot mesh
    conformingly -- which is the whole PLC failure."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("box_shell")
        stl_mesh._add_box_surface_loop(gmsh, 0.060)
        gmsh.model.geo.synchronize()
        n_pts = len(gmsh.model.getEntities(0))
        n_curves = len(gmsh.model.getEntities(1))
        n_surfs = len(gmsh.model.getEntities(2))
    finally:
        gmsh.finalize()
    assert n_pts == 8
    assert n_curves == 12, f"a box has 12 edges, helper made {n_curves}"
    assert n_surfs == 6


@pytest.mark.slow
def test_empty_chamber_box_meshes_on_its_own():
    """No part at all: if this fails, no STL fix can help. It is the control
    that TRANCHE 1 never ran, and it is what mislocated the blocker."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("box_only")
        loop = stl_mesh._add_box_surface_loop(gmsh, 0.060)
        vol = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(3, [vol], 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.008)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.012)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)
        n_tets = len(gmsh.model.mesh.getElements(3)[1][0])
    finally:
        gmsh.finalize()
    assert n_tets > 0


# --------------------------------------------------------------------------- #
# (1) the library pyramid, embedded in the bed
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_pyramid_stl_meshes_inside_the_chamber():
    """STATED TOLERANCE 0.5 percent, the same discretisation tolerance the
    part-only gate uses. The part must also be a strict minority of the mesh,
    or the "bed" is not there."""
    msh, info = stl_mesh.build_mesh_from_stl(
        STL / "pyramid.stl", lc_part=1.5e-3, with_chamber=True,
        L=L_CHAMBER_M, precomp_coeffs=None)
    assert info.with_chamber is True
    assert info.part_volume_rel_err_vs_stl == pytest.approx(0.0, abs=5.0e-3)
    assert info.n_bed_cells > 0
    assert info.n_part_cells > 0


@pytest.mark.slow
def test_chamber_chi_is_nontrivial_and_passes_the_shared_fill_contract():
    """chi must be 1 on the part and 0 on the bed, integrate to the part
    volume, and be cross-checked against an INDEPENDENT ray-cast inside test
    on the original triangle soup -- so a mesh that tagged the wrong cells
    fails here even though the volume sum agreed."""
    msh, info = stl_mesh.build_mesh_from_stl(
        STL / "pyramid.stl", lc_part=1.5e-3, with_chamber=True,
        L=L_CHAMBER_M, precomp_coeffs=None)
    chi, vol = stl_mesh.chi_and_volumes(msh, info)
    assert chi.min() == 0.0 and chi.max() == 1.0      # non-trivial: bed exists
    assert float(np.dot(chi, vol)) == pytest.approx(info.part_volume_m3,
                                                    rel=5.0e-3)
    v, f = stl_mesh.load_stl(STL / "pyramid.stl")
    agree = _raycast_agreement(msh, info, v, f, chi)
    assert agree > 0.995, f"only {agree:.4f} of cells agree with the ray cast"


@pytest.mark.slow
def test_stl_chamber_matches_the_occ_chamber_within_the_phase_a_band():
    """THE equivalence gate, run on the CHAMBER mesh rather than the part-only
    mesh. The OCC route (phase_e/geometry.build_mesh) and the STL route must
    agree on the embedded part volume no worse than the engine's own
    mesh-refinement wobble (x1.5). Same shape, same chamber, same lc."""
    from solve3d.phase_e import geometry as geo
    band = _dolfinx_band()
    lc = 1.5e-3
    _m_occ, i_occ = geo.build_mesh("pyramid", lc_part=lc, L=L_CHAMBER_M,
                                   precomp_coeffs=None)
    _m_stl, i_stl = stl_mesh.build_mesh_from_stl(
        STL / "pyramid.stl", lc_part=lc, with_chamber=True, L=L_CHAMBER_M,
        precomp_coeffs=None)
    rel = i_stl.part_volume_m3 / i_occ.part_volume_m3 - 1.0
    assert rel == pytest.approx(0.0, abs=band), (
        f"STL chamber part volume {i_stl.part_volume_m3:.6e} vs OCC "
        f"{i_occ.part_volume_m3:.6e}, rel {rel:.3e}, band {band:.5f}")


@pytest.mark.slow
def test_level0_precompensation_reaches_the_chamber_mesh():
    """L0 is default ON, and the chamber must be built around the
    PRE-COMPENSATED part -- otherwise the solve domain and the solve target
    describe different objects. The scaled part volume must grow by exactly
    xy_scale**2 * z_scale."""
    from solve3d import precomp
    c = precomp.load_defaults()
    lc = 2.0e-3
    _m0, i0 = stl_mesh.build_mesh_from_stl(
        STL / "pyramid.stl", lc_part=lc, with_chamber=True, L=L_CHAMBER_M,
        precomp_coeffs=precomp.ShrinkageL0(0.0, 0.0))
    _m1, i1 = stl_mesh.build_mesh_from_stl(
        STL / "pyramid.stl", lc_part=lc, with_chamber=True, L=L_CHAMBER_M,
        precomp_coeffs=c)
    want = c.xy_scale ** 2 * c.z_scale
    assert i1.stl_volume_m3 / i0.stl_volume_m3 == pytest.approx(want, rel=1e-12)
    assert i1.part_volume_m3 / i0.part_volume_m3 == pytest.approx(want,
                                                                  rel=1e-2)
    assert i1.precomp["enabled"] is True
    assert i1.precomp["material_only"] is True


# --------------------------------------------------------------------------- #
# (2) a genuinely arbitrary, non-library part: the real user Tamper
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_tamper_is_a_valid_arbitrary_part():
    """Not from the shape library, not a primitive: the actual STL the Studio
    job feb850ec ran on (TAMPER_DIAGNOSIS.md)."""
    assert TAMPER.exists(), TAMPER
    v, f = stl_mesh.load_stl(TAMPER)
    stl_mesh.validate(v, f, check_self_intersection=True)
    assert f.shape[0] > 2000                 # a real tessellation, not a box


@pytest.mark.slow
def test_tamper_meshes_with_chamber_and_passes_the_fill_contract():
    msh, info = stl_mesh.build_mesh_from_stl(
        TAMPER, lc_part=1.5e-3, with_chamber=True, L=L_CHAMBER_M,
        precomp_coeffs=None)
    assert info.part_volume_rel_err_vs_stl == pytest.approx(0.0, abs=5.0e-3)
    chi, vol = stl_mesh.chi_and_volumes(msh, info)
    assert chi.min() == 0.0 and chi.max() == 1.0
    assert float(np.dot(chi, vol)) == pytest.approx(info.part_volume_m3,
                                                    rel=5.0e-3)
    v, f = stl_mesh.load_stl(TAMPER)
    agree = _raycast_agreement(msh, info, v, f, chi)
    assert agree > 0.995, f"only {agree:.4f} of cells agree with the ray cast"


@pytest.mark.slow
def test_tamper_runs_one_short_forward_march_with_standing_gates_green():
    """ONE short march on the chamber-embedded Tamper. The standing gates are
    the ones every solve3d forward already reports: energy residual, the
    clamp-bound latch, and the CFL flag. A march that trips any of them is not
    a usable forward, however pretty the mesh is."""
    from solve3d import forward as fwd
    msh, info = stl_mesh.build_mesh_from_stl(
        TAMPER, lc_part=2.0e-3, with_chamber=True, L=L_CHAMBER_M,
        precomp_coeffs=None)
    p = fwd.ForwardParams()
    in_part = stl_mesh.part_mask_predicate(info)
    mats = fwd.build_materials(msh, in_part, p)
    out = fwd.march_enthalpy(msh, p, mats=mats, q_uniform=1.0e6,
                             max_time_s=20.0, L=L_CHAMBER_M)
    assert out["clamp_bound"] is False
    assert out["cfl_violated"] is False
    assert abs(float(out["energy_residual_frac"])) < 1e-6


# --------------------------------------------------------------------------- #
# (3) the surface-reconstruction refusal
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_a_too_coarse_feature_angle_is_REFUSED_not_reported():
    """Tranche 1's 40 degrees was tuned on an all-planar library primitive. On
    the Tamper's curved wall it loses 0.79 percent of the volume -- a quietly
    wrong-sized part, which is the confident-wrong-number failure mode. It must
    raise, not return."""
    with pytest.raises(stl_mesh.SurfaceReconstructionError) as e:
        stl_mesh.build_mesh_from_stl(
            TAMPER, lc_part=1.5e-3, with_chamber=True, L=L_CHAMBER_M,
            precomp_coeffs=None, feature_angle_deg=40.0)
    assert "smoothed across real geometry" in str(e.value)


@pytest.mark.slow
def test_the_default_feature_angle_recovers_the_tamper_volume():
    """Mutation guard on the refusal above: the shipped default must PASS the
    same part the coarse angle fails, or the refusal is just a blanket ban."""
    _m, info = stl_mesh.build_mesh_from_stl(
        TAMPER, lc_part=1.5e-3, with_chamber=True, L=L_CHAMBER_M,
        precomp_coeffs=None)
    assert info.feature_angle_deg == stl_mesh.FEATURE_ANGLE_DEG
    assert abs(info.part_volume_rel_err_vs_stl) < 1.0e-4


def test_the_pyramid_is_invariant_to_the_feature_angle_change():
    """The library primitive's facets are coplanar, so lowering the default
    splits nothing and the part-only Tranche 1 result is untouched. Checked on
    the SURFACE classification, which is what the angle acts on."""
    import gmsh
    v, f = stl_mesh.load_stl(STL / "pyramid.stl")
    counts = {}
    for angle in (40.0, stl_mesh.FEATURE_ANGLE_DEG):
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.clear()
            gmsh.model.add("cls")
            ent = gmsh.model.addDiscreteEntity(2)
            gmsh.model.mesh.addNodes(
                2, ent, np.arange(1, v.shape[0] + 1, dtype=np.int64),
                (v * stl_mesh.MM_TO_M).ravel())
            gmsh.model.mesh.addElementsByType(
                ent, 2, [], (f + 1).ravel().astype(np.int64))
            gmsh.model.mesh.classifySurfaces(angle * np.pi / 180.0, True, True,
                                             np.pi)
            gmsh.model.mesh.createGeometry()
            counts[angle] = len(gmsh.model.getEntities(2))
        finally:
            gmsh.finalize()
    assert counts[40.0] == counts[stl_mesh.FEATURE_ANGLE_DEG]


# --------------------------------------------------------------------------- #
# (4) adaptive chamber sizing reaches the mesher
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_the_mesher_sizes_the_chamber_adaptively_by_default():
    """L=None with a chamber must use the pre-registered rule, not a guess.
    The Tamper's 44.4 mm span plus 2x20 mm is 84.4 mm -> 85 mm."""
    from solve3d import chamber as ch
    _m, info = stl_mesh.build_mesh_from_stl(
        TAMPER, lc_part=2.5e-3, with_chamber=True, precomp_coeffs=None)
    assert info.L_chamber_m == pytest.approx(0.085)
    assert info.chamber["mode"] == "adaptive"
    assert info.chamber["tag"] == "ch085"
    assert info.chamber["margin_actual_m"] >= ch.GAP_M - 1e-9
    assert info.chamber["margin_below_preregistered"] is False


@pytest.mark.slow
def test_the_frozen_chamber_is_still_reachable_and_says_its_margin_is_short():
    """Reproduction path. It must not pretend the short margin is fine."""
    _m, info = stl_mesh.build_mesh_from_stl(
        TAMPER, lc_part=2.5e-3, with_chamber=True, L=0.060,
        precomp_coeffs=None)
    assert info.L_chamber_m == pytest.approx(0.060)
    assert info.chamber["mode"] == "frozen_override"
    assert info.chamber["tag"] == "ch060"
    assert info.chamber["margin_below_preregistered"] is True


@pytest.mark.slow
def test_the_adaptive_chamber_is_sized_on_the_PRE_COMPENSATED_part():
    """L0 grows the solid ~3 percent. Sizing the chamber on the nominal bbox
    would silently hand the grown part a smaller margin than pre-registered --
    the same class of bug as the origin-centred refinement box."""
    from solve3d import chamber as ch, precomp
    c = precomp.load_defaults()
    _m, info = stl_mesh.build_mesh_from_stl(
        TAMPER, lc_part=2.5e-3, with_chamber=True, precomp_coeffs=c)
    assert info.chamber["margin_actual_m"] >= ch.GAP_M - 1e-9
    assert info.chamber["governing_span_m"] == pytest.approx(
        0.0444 * c.xy_scale, rel=1e-3)
