"""Tests for the symmetry-consistency primitive (solve3d/symmetry_gate.py).

The PRIMITIVE only: given a solved map, node coords, weights, and an explicit
symmetry group (list of coord->coord ops), what fraction of the map's
(vol-weighted) variance survives projection onto the group-symmetric subspace?
A budget-limited solve that fit discretization-frame noise scores LOW here even
when the peak/density gates pass (the Phase C cylinder was 0.25).

GROUP DETECTION (part INTERSECT field INTERSECT objective INTERSECT
convection-BC symmetry) and the is_sendable WIRING are deliberately NOT here --
they conform to the campaign lane's SYMMETRY_GATE_REPORT.md when it lands. This
file pins only the spec-independent linear-algebra core.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import symmetry_gate as sg

IDENT = lambda p: p
XMIRROR = lambda p: p * np.array([-1.0, 1.0, 1.0])

# 4 nodes on the x-axis, symmetric about 0: -2 <-> 2, -1 <-> 1 under x-mirror.
COORDS = np.array([[-2.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
W = np.ones(4)


def _frac(s, group, weights=W):
    return sg.symmetric_variance_fraction(
        np.asarray(s, float), COORDS, weights, group)["fraction"]


def test_fully_symmetric_map_scores_one():
    # s(-x) == s(x) -> entirely in the symmetric subspace
    assert _frac([10.0, 4.0, 4.0, 10.0], [IDENT, XMIRROR]) == pytest.approx(1.0)


def test_pure_antisymmetric_map_scores_zero():
    # mean-zero, s(-x) == -s(x) -> the symmetric component is identically 0
    assert _frac([3.0, 1.0, -1.0, -3.0], [IDENT, XMIRROR]) == pytest.approx(0.0, abs=1e-12)


def test_known_symmetric_plus_antisymmetric_split():
    # s = sym[10,4,4,10] + anti[3,1,-1,-3]; sym var = 9, total var = 14 -> 9/14
    assert _frac([13.0, 5.0, 3.0, 7.0], [IDENT, XMIRROR]) == pytest.approx(9.0 / 14.0)


def test_identity_only_group_is_trivially_symmetric():
    # the trivial group projects to identity -> every map is "symmetric"
    assert _frac([13.0, 5.0, 3.0, 7.0], [IDENT]) == pytest.approx(1.0)


def test_weights_are_respected():
    # 6 nodes x=[-3,-2,-1,1,2,3]; symmetric part [9,4,1,1,4,9] carries variance on
    # BOTH pairs, the antisymmetry [5,0,0,0,0,-5] lives ONLY on the |x|=3 pair.
    # s = sym + anti = [14,4,1,1,4,4]. Down-weighting the |x|=3 nodes removes the
    # only antisymmetric content while the inner pairs still carry symmetric
    # variance -> the fraction rises toward 1. (Uniform weights ~0.566.)
    coords6 = np.array([[-3.0, 0, 0], [-2.0, 0, 0], [-1.0, 0, 0],
                        [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    s = np.array([14.0, 4.0, 1.0, 1.0, 4.0, 4.0])
    grp = [IDENT, XMIRROR]
    lo = sg.symmetric_variance_fraction(s, coords6, np.ones(6), grp)["fraction"]
    w_down = np.array([1e-3, 1.0, 1.0, 1.0, 1.0, 1e-3])
    hi = sg.symmetric_variance_fraction(s, coords6, w_down, grp)["fraction"]
    assert lo == pytest.approx(0.566, abs=1e-2)
    assert hi > lo
    assert hi == pytest.approx(1.0, abs=1e-2)


def test_reports_max_match_distance_zero_on_exact_symmetric_mesh():
    out = sg.symmetric_variance_fraction(
        np.array([10.0, 4.0, 4.0, 10.0]), COORDS, W, [IDENT, XMIRROR])
    assert out["max_match_dist"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= out["fraction"] <= 1.0


def test_verdict_pass_fail_at_threshold():
    # convenience: PASS iff fraction >= threshold (default 0.8)
    assert sg.symmetry_verdict(0.912) == "PASS"
    assert sg.symmetry_verdict(0.354) == "FAIL"
    assert sg.symmetry_verdict(0.80) == "PASS"


# ---- the conformant gate RECORD (SYMMETRY_GATE_REPORT.md 3.4/3.5/4) --------- #
# Group detection on the mesh: mirrors PERPENDICULAR to the build axis, each
# ACCEPTED only if it maps the in-part node set onto itself (else -> reductions);
# the build-axis mirror is excluded a priori (one-sided top convection). Verdict
# PASS iff fraction>=0.8 OR projection-price<=1%; VACUOUS_PASS for a trivial group.
def _grid_part():
    # a small centered symmetric point cloud in x-z (build axis y), all in-part
    xs = np.array([-2.0, -1.0, 1.0, 2.0])
    zs = np.array([-2.0, -1.0, 1.0, 2.0])
    pts = np.array([[x, 0.0, z] for x in xs for z in zs])
    return pts


def test_record_centered_symmetric_map_passes_and_is_sendable():
    coords = _grid_part()
    # map symmetric under x and z mirrors: value depends on |x|,|z| only
    s = np.array([abs(x) + abs(z) for x, _, z in coords])
    rec = sg.symmetry_gate_record(s, coords, np.ones(len(s)),
                                  in_part=np.ones(len(s), bool), build_axis="y")
    assert rec["fraction"] == pytest.approx(1.0, abs=1e-9)
    assert set(rec["group"]) == {"identity", "mirror_x", "mirror_z", "mirror_xz"}
    assert rec["build_axis"] == "y"
    assert rec["convective_faces"] == ["y=+L/2"]
    assert rec["vacuous"] is False
    assert rec["verdict"] == "PASS"
    assert rec["sendable"] is True
    # y-mirror is excluded a priori by one-sided convection -> in reductions
    assert any("mirror_y" in r["element"] and "convection" in r["reason"].lower()
               for r in rec["reductions"])


def _wedge_part():
    """Symmetric in z per column, but a WEDGE in x: the z cross-section GROWS with
    x, so the x-mirror sends the wide +x columns onto the narrow -x columns
    (macroscopically off the part) while the z-mirror is exact. This exercises a
    GENUINE accept (z) and a GENUINE reject (x) in one fixture -- unlike a dropped
    grid column, whose mirror image lands ~1 NN into the gap and is NOT a clean
    reject under a containment criterion (calibration note, 2026-08-10)."""
    colz = {-3.0: [0.0], -2.0: [0.0], -1.0: [-1.0, 0.0, 1.0],
            1.0: [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            2.0: [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            3.0: [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,
                  1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    return np.array([[x, 0.0, z] for x, zs in colz.items() for z in zs])


def test_record_accepts_symmetric_mirror_and_rejects_the_wedge_mirror():
    coords = _wedge_part()
    s = np.ones(len(coords))
    rec = sg.symmetry_gate_record(s, coords, np.ones(len(coords)),
                                  in_part=np.ones(len(coords), bool),
                                  build_axis="y")
    # z-mirror is a real part symmetry -> accepted; x-mirror is the wedge -> rejected
    assert "mirror_z" in rec["group"]
    assert "mirror_x" not in rec["group"]
    assert any("mirror_x" in r["element"] for r in rec["reductions"])
    # the accepted (z) mirror is exact on this integer grid -> zero slop
    assert rec["max_match_dist"] == pytest.approx(0.0, abs=1e-12)


def test_record_asymmetric_part_is_vacuous_pass():
    # a right-triangle wedge in BOTH x and z: neither mirror maps the part onto
    # itself (its images land macroscopically off the part) -> trivial group ->
    # vacuous pass (stated). A tiny L-corner does NOT work: with only a few nodes
    # every mirror image lands within a couple NN and is spuriously accepted.
    coords = np.array([[float(x), 0.0, float(z)]
                       for x in range(8) for z in range(8 - x)])
    s = np.linspace(0.2, 0.8, len(coords))
    rec = sg.symmetry_gate_record(s, coords, np.ones(len(coords)),
                                  in_part=np.ones(len(coords), bool),
                                  build_axis="y")
    assert rec["group"] == ["identity"]
    assert rec["vacuous"] is True
    assert rec["verdict"] == "VACUOUS_PASS"
    assert rec["sendable"] is True


def test_record_low_fraction_passes_on_price_when_residue_is_inert():
    coords = _grid_part()
    # a map with real asymmetric content -> fraction < 0.8
    rng = np.array([0.1, 0.9, -0.4, 0.6, 0.2, -0.7, 0.8, -0.3,
                    0.5, -0.6, 0.35, -0.2, 0.15, 0.44, -0.55, 0.25])
    s = rng
    # injected scorer: projected map scores ~identical to solved -> price ~0
    def scorer(_map):
        return 5.0        # constant J -> |J(proj)-J(solved)| = 0
    rec = sg.symmetry_gate_record(s, coords, np.ones(len(s)),
                                  in_part=np.ones(len(s), bool), build_axis="y",
                                  scorer=scorer, j_uniform=10.0, j_solved=5.0)
    assert rec["fraction"] < 0.8
    assert rec["projection_price"] == pytest.approx(0.0, abs=1e-9)
    assert rec["verdict"] == "PASS"           # passed on price, not fraction
    assert rec["sendable"] is True


def test_record_low_fraction_and_costly_price_fails():
    coords = _grid_part()
    s = np.array([0.1, 0.9, -0.4, 0.6, 0.2, -0.7, 0.8, -0.3,
                  0.5, -0.6, 0.35, -0.2, 0.15, 0.44, -0.55, 0.25])
    def scorer(_map):
        return 9.5        # |J(proj)-J(solved)|=4.5 vs margin |10-5|=5 -> price 0.9 >> 0.01
    rec = sg.symmetry_gate_record(s, coords, np.ones(len(s)),
                                  in_part=np.ones(len(s), bool), build_axis="y",
                                  scorer=scorer, j_uniform=10.0, j_solved=5.0)
    assert rec["fraction"] < 0.8
    assert rec["projection_price"] > 0.01
    assert rec["verdict"] == "FAIL"
    assert rec["sendable"] is False


# ---- REAL-MESH cross-check: the load-bearing acceptance test ---------------- #
# The record's group DETECTION (containment criterion, k=2.0, threshold=0.90,
# calibrated 2026-08-10 on these very meshes) must reproduce the retro fractions
# in solve3d/results/symmetry_retro_3d.json for the shapes whose group is really
# {x,z}, and must arrive at the physically CORRECT group for the pyramid, whose
# apex is along z so it is NOT z-symmetric (the retro applied a blanket {x,z} and
# false-failed it -- the same over-projection failure the 2-D report warns about
# in its Section 3.2). All maps store IN-PART cells only; weights are the cell
# volumes; build axis is y (one-sided top convection); no scorer is injected, so
# a low-fraction shape FAILs on fraction alone.
from pathlib import Path

_RESULTS = Path(__file__).resolve().parents[1] / "results"


def _load_map(name):
    d = np.load(_RESULTS / name)
    return (np.asarray(d["s_map"], float), np.asarray(d["centroids"], float),
            np.asarray(d["volumes"], float))


def _real_record(name):
    s, c, v = _load_map(name)
    return sg.symmetry_gate_record(s, c, v, in_part=np.ones(len(s), bool),
                                   build_axis="y")


@pytest.mark.parametrize("name,fname,frac,group_has,verdict", [
    # cube: {id,x,z,xz}, 0.9121 PASS  (reproduces retro exactly)
    ("cube", "map_stage_b4_cube.npz", 0.9121,
     {"mirror_x", "mirror_z", "mirror_xz"}, "PASS"),
    # square: {id,x,z,xz}, 0.8601 PASS (reproduces retro exactly)
    ("square", "map_stage_b4_square.npz", 0.8601,
     {"mirror_x", "mirror_z", "mirror_xz"}, "PASS"),
    # cylinder (axis z, full height): {id,x,z,xz}, 0.4258 FAIL residue
    ("cylinder", "phase_c_map_solve_filter_only_asymmetric_scaled.npz", 0.4258,
     {"mirror_x", "mirror_z", "mirror_xz"}, "FAIL"),
])
def test_real_mesh_cross_check_reproduces_retro(name, fname, frac, group_has,
                                                verdict):
    rec = _real_record(fname)
    assert rec["fraction"] == pytest.approx(frac, abs=1e-3), name
    assert group_has.issubset(set(rec["group"])), (name, rec["group"])
    assert rec["build_axis"] == "y"
    assert rec["convective_faces"] == ["y=+L/2"]
    assert rec["verdict"] == verdict, (name, rec["verdict"])
    # the reported max_match_dist is the exactness lower-bound; mesh-frame slop
    # is a couple NN, never zero on a real tet mesh, and must be surfaced.
    assert rec["max_match_dist"] > 0.0


def test_real_mesh_pyramid_detects_x_only_group_and_passes():
    # THE HONEST CORRECTION. The pyramid apex is along z (z cross-section shrinks
    # to a point), so the z-mirror is NOT a part symmetry and the containment
    # criterion REJECTS it. Under the correct detected group {id, mirror_x} the
    # map is 0.9917 symmetric -> PASS. The retro's 0.354 FAIL was an artifact of
    # projecting onto a z-mirror the pyramid does not have.
    rec = _real_record("map_stage_b4_pyramid.npz")
    assert set(rec["group"]) == {"identity", "mirror_x"}
    assert any("mirror_z" in r["element"] for r in rec["reductions"])
    assert rec["fraction"] == pytest.approx(0.9917, abs=1e-3)
    assert rec["verdict"] == "PASS"
    assert rec["sendable"] is True


# ---- mesh-robust (interpolated) matching: kills the coarse-mesh false-fail --- #
# Nearest-node matching snaps a mirrored point to the nearest node, so on a coarse
# / unstructured mesh a symmetric field reads as asymmetric (fraction is a LOWER
# bound; the gate warns it can false-fail). Interpolating the field AT the mirror
# point removes that slop without enabling a false-pass (a genuinely asymmetric
# field still evaluates differently at the mirror). Opt-in (interp=True); the
# default nearest-node path is unchanged so the calibrated retro cross-checks hold.
def _jittered_sym_mesh(seed=1, n=11):
    """Symmetric x-z point cloud about the origin with ASYMMETRIC positional
    jitter -> the mesh nodes are NOT mirror-partnered, so nearest-node matching
    has real slop, exactly like a coarse tet mesh."""
    rng = np.random.default_rng(seed)
    ax = np.linspace(-3.0, 3.0, n)
    base = np.array([[x, 0.0, z] for x in ax for z in ax])
    jit = rng.uniform(-0.12, 0.12, base.shape)
    jit[:, 1] = 0.0
    return base + jit


def test_interp_is_a_bounded_projection_where_nearest_overshoots():
    """interp evaluates the field AT the mirror image, so it is a true orthogonal
    projection: fraction in [0,1] and ~1 for a symmetric field. Nearest-node
    matching is NOT a projection -- on a non-partnered cloud it can report a
    fraction ABOVE 1 (matching slop adds spurious variance). This is interp's
    correctness guarantee, independent of mesh resolution."""
    rng = np.random.default_rng(2)
    p = rng.uniform(-3.0, 3.0, (80, 2))            # non-mirror-partnered cloud
    coords = np.column_stack([p[:, 0], np.zeros(80), p[:, 1]])
    s = coords[:, 0] ** 2 + coords[:, 2] ** 2       # symmetric under x and z
    grp = [IDENT,
           sg._mirror_op(np.zeros(3), [0]),
           sg._mirror_op(np.zeros(3), [2]),
           sg._mirror_op(np.zeros(3), [0, 2])]
    near = sg.symmetric_variance_fraction(s, coords, np.ones(len(s)), grp)["fraction"]
    interp = sg.symmetric_variance_fraction(
        s, coords, np.ones(len(s)), grp, interp=True)["fraction"]
    assert near > 1.0                       # nearest overshoots -> not a projection
    assert interp <= 1.0 + 1e-9             # interp is a bounded projection
    assert interp == pytest.approx(1.0, abs=0.1)   # and recovers the symmetry


def test_interp_does_not_false_pass_a_genuinely_asymmetric_map():
    coords = _jittered_sym_mesh()
    # odd in x -> the x-symmetric component is ~0; interpolation must NOT inflate it
    s = coords[:, 0].copy()
    grp = [IDENT, sg._mirror_op(np.zeros(3), [0])]
    interp = sg.symmetric_variance_fraction(
        s, coords, np.ones(len(s)), grp, interp=True)["fraction"]
    assert interp < 0.2                     # still fails -> never a false-pass


def test_interp_identity_group_is_still_trivially_symmetric():
    coords = _jittered_sym_mesh()
    s = np.linspace(0.2, 0.8, len(coords))
    out = sg.symmetric_variance_fraction(
        s, coords, np.ones(len(s)), [IDENT], interp=True)
    assert out["fraction"] == pytest.approx(1.0)


def test_record_accepts_interp_and_flags_it():
    """symmetry_gate_record threads interp=True and records which matcher was used
    so a re-scored verdict is self-documenting."""
    coords = _grid_part()
    s = np.array([abs(x) + abs(z) for x, _, z in coords])
    rec = sg.symmetry_gate_record(s, coords, np.ones(len(s)),
                                  in_part=np.ones(len(s), bool), build_axis="y",
                                  interp=True)
    assert rec["matcher"] == "interp"
    assert rec["fraction"] == pytest.approx(1.0, abs=1e-6)
    assert rec["verdict"] == "PASS"


def test_record_default_matcher_is_nearest_node():
    coords = _grid_part()
    s = np.array([abs(x) + abs(z) for x, _, z in coords])
    rec = sg.symmetry_gate_record(s, coords, np.ones(len(s)),
                                  in_part=np.ones(len(s), bool), build_axis="y")
    assert rec["matcher"] == "nearest"


# ---- studio_solve producer wiring ------------------------------------------ #
def test_studio_solve_symmetry_gate_helper_emits_conformant_record():
    # studio_solve pulls in the dolfinx forward; runs only in the spike env.
    ss = pytest.importorskip("solve3d.studio_solve")
    # a small centered x/z-symmetric part cloud; map symmetric under x,z mirrors
    xs = np.array([-2.0, -1.0, 1.0, 2.0])
    coords = np.array([[x, 0.0, z] for x in xs for z in xs])
    s = np.array([abs(x) + abs(z) for x, _, z in coords])
    rec = ss.symmetry_gate_record_for_map(
        coords, np.ones(len(s)), s,
        scorer=lambda _m: 1.0, j_uniform=2.0, j_solved=1.0)
    assert set(rec["group"]) == {"identity", "mirror_x", "mirror_z", "mirror_xz"}
    assert rec["build_axis"] == "y"
    assert rec["convective_faces"] == ["y=+L/2"]
    assert rec["vacuous"] is False
    assert rec["fraction"] == pytest.approx(1.0, abs=1e-9)
    assert rec["verdict"] == "PASS"
    assert rec["sendable"] is True
