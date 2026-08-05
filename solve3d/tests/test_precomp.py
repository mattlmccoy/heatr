"""Level 0 affine pre-compensation: coefficients, the map, and the guard.

Plan: docs/superpowers/plans/2026-08-04-shrinkage-v2-tranche1.md Task 1.
Spec: docs/superpowers/specs/2026-08-04-shrinkage-prewarp-v2-design.md sec 2.
Coefficients: SHRINKAGE_COEFFICIENTS_MEMO.md sec 3.

The three test groups map one-to-one onto the plan's three red-test clauses:
(a) the affine map, exact on analytic vertices; (b) the DOUBLE-COUNTING GUARD,
which is Matt's explicit distinction between material shrinkage (the
pre-scale's job) and densification consolidation (the model's job); (c)
config-driven coefficients with memo defaults, carried into provenance with
their uncertainty band.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from solve3d import precomp

ROOT = Path(__file__).resolve().parents[2]
MEMO = ROOT / "SHRINKAGE_COEFFICIENTS_MEMO.md"
SHARED = ROOT / "shrinkage_precomp.json"      # cross-lane, Studio mirrors it
SCHEMA_VERSION = "1.0"

# The memo's section 3 numbers, transcribed here so a silent edit to either
# the config or the memo breaks a test rather than sliding through.
MEMO_S_XY, MEMO_S_XY_BAND = 0.030, (0.020, 0.040)
MEMO_S_Z, MEMO_S_Z_BAND = 0.020, (0.010, 0.030)


# --------------------------------------------------------------------------- #
# (a) the affine map
# --------------------------------------------------------------------------- #
def test_scales_are_the_inverse_shrinkage_factors():
    c = precomp.ShrinkageL0(s_xy=0.030, s_z_mat=0.020)
    assert c.xy_scale == pytest.approx(1.0 / (1.0 - 0.030), rel=1e-15)
    assert c.z_scale == pytest.approx(1.0 / (1.0 - 0.020), rel=1e-15)


def test_map_is_exact_on_analytic_vertices():
    """A cube corner scales by the axis factors, to round-off."""
    c = precomp.ShrinkageL0(s_xy=0.030, s_z_mat=0.020)
    a = 16.119919540164695e-3 / 2.0
    v = np.array([[a, -a, a], [-a, a, -a]], dtype=float)
    out = precomp.scale_points(v, c)
    want = v * np.array([c.xy_scale, c.xy_scale, c.z_scale])
    assert np.allclose(out, want, rtol=0.0, atol=0.0)
    # and the intended physical statement: shrinking the result by s_* returns
    # the nominal vertex
    back = out * np.array([1 - 0.030, 1 - 0.030, 1 - 0.020])
    assert np.allclose(back, v, rtol=1e-15, atol=0.0)


def test_map_is_anisotropic_z_differs_from_xy():
    c = precomp.ShrinkageL0(s_xy=0.030, s_z_mat=0.020)
    assert c.xy_scale != c.z_scale


def test_scale_points_rejects_wrong_shape():
    c = precomp.ShrinkageL0(s_xy=0.03, s_z_mat=0.02)
    with pytest.raises(ValueError):
        precomp.scale_points(np.zeros((4, 2)), c)


# --------------------------------------------------------------------------- #
# (b) THE DOUBLE-COUNTING GUARD (named test, Matt's distinction)
# --------------------------------------------------------------------------- #
def test_double_counting_guard_schema_has_no_densification_parameter():
    """The L0 schema carries MATERIAL shrinkage only.

    Consolidation (powder to solid, rho 0.55 -> ~1.0) is marched by the model.
    If a densification term could enter here it would be counted twice.
    """
    fields = set(precomp.ShrinkageL0.field_names())
    assert fields == {"s_xy", "s_z_mat"}
    for banned in ("densif", "consolidat", "rho", "density", "phi", "collapse"):
        assert not any(banned in f for f in fields)


def test_double_counting_guard_refuses_a_densification_key_loudly():
    bad = {"s_xy": 0.030, "s_z_mat": 0.020, "s_z_densification": 0.45}
    with pytest.raises(precomp.DoubleCountingError) as e:
        precomp.ShrinkageL0.from_mapping(bad)
    assert "s_z_densification" in str(e.value)


def test_double_counting_guard_refuses_a_total_shrinkage_key():
    """A 'total' includes consolidation by construction, so it is refused."""
    with pytest.raises(precomp.DoubleCountingError):
        precomp.ShrinkageL0.from_mapping({"s_xy": 0.03, "s_total_z": 0.06})


def test_zero_coefficients_are_bit_identical_to_no_precompensation():
    """A run with s_* = 0 must reproduce today's geometry to the last bit."""
    off = precomp.ShrinkageL0(s_xy=0.0, s_z_mat=0.0)
    assert off.xy_scale == 1.0 and off.z_scale == 1.0
    assert off.is_identity
    rng = np.random.default_rng(0)
    p = rng.normal(size=(64, 3))
    out = precomp.scale_points(p, off)
    assert np.array_equal(out, p)
    assert out.tobytes() == p.tobytes()


def test_disabled_config_is_identity_regardless_of_coefficients():
    c = precomp.ShrinkageL0(s_xy=0.030, s_z_mat=0.020, enabled=False)
    assert c.is_identity
    rng = np.random.default_rng(1)
    p = rng.normal(size=(32, 3))
    assert np.array_equal(precomp.scale_points(p, c), p)


# --------------------------------------------------------------------------- #
# (c) config-driven coefficients, memo defaults, provenance with the band
# --------------------------------------------------------------------------- #
def test_defaults_come_from_the_shared_root_file_not_a_literal():
    """One coefficients file at repo root, shared with the Studio lane."""
    assert precomp.config_path() == SHARED
    assert SHARED.exists(), f"shared L0 coefficients missing at {SHARED}"
    d = json.loads(SHARED.read_text())
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["s_xy"] == MEMO_S_XY
    assert d["s_z_mat"] == MEMO_S_Z
    assert tuple(d["s_xy_band"]) == MEMO_S_XY_BAND
    assert tuple(d["s_z_mat_band"]) == MEMO_S_Z_BAND
    assert d["source"] == "SHRINKAGE_COEFFICIENTS_MEMO.md"
    assert d["material_only"] is True
    assert MEMO.exists()


def test_unknown_schema_version_is_refused():
    """A schema bump must stop this lane rather than be read optimistically."""
    with pytest.raises(precomp.SchemaVersionError):
        precomp.from_mapping_checked({"schema_version": "2.0", "s_xy": 0.03,
                                      "s_z_mat": 0.02, "material_only": True})


def test_double_counting_guard_requires_material_only_true():
    """material_only is the shared file's assertion of Matt's distinction; a
    file that does not claim it is refused, not silently trusted."""
    with pytest.raises(precomp.DoubleCountingError):
        precomp.from_mapping_checked({"schema_version": SCHEMA_VERSION,
                                      "s_xy": 0.03, "s_z_mat": 0.02,
                                      "material_only": False})
    with pytest.raises(precomp.DoubleCountingError):
        precomp.from_mapping_checked({"schema_version": SCHEMA_VERSION,
                                      "s_xy": 0.03, "s_z_mat": 0.02})


def test_load_defaults_matches_the_memo():
    c = precomp.load_defaults()
    assert c.s_xy == MEMO_S_XY
    assert c.s_z_mat == MEMO_S_Z
    assert c.s_xy_band == MEMO_S_XY_BAND
    assert c.s_z_mat_band == MEMO_S_Z_BAND


def test_default_is_on_per_the_approved_spec():
    """AUTHORITY for this default: the APPROVED spec
    docs/superpowers/specs/2026-08-04-shrinkage-prewarp-v2-design.md section 2
    (Level 0), verbatim: "Off by default until the memo lands; then default ON
    with the coefficients displayed."

    SHRINKAGE_COEFFICIENTS_MEMO.md is the memo, and it has landed, so the
    stated condition is met and ON is the approved state rather than a new
    default decision by this lane.

    `enabled` is deliberately NOT in the shared coefficients file: the numbers
    are cross-lane, but whether a given lane applies them is that lane's own
    state.
    """
    assert precomp.load_defaults().enabled is True
    assert "enabled" not in json.loads(SHARED.read_text())


def test_pre_l0_campaigns_are_reproduced_by_pinning_the_coefficients_to_zero():
    """The flip governs NEW runs. Every existing campaign artifact was made
    with no pre-compensation, so reproducing one means pinning s_* = 0
    explicitly; this is the supported way to do that."""
    off = precomp.load_defaults(enabled=False)
    assert off.is_identity
    zeroed = precomp.ShrinkageL0(s_xy=0.0, s_z_mat=0.0)
    assert zeroed.is_identity
    rng = np.random.default_rng(7)
    q = rng.normal(size=(16, 3))
    assert np.array_equal(precomp.scale_points(q, off), q)
    assert np.array_equal(precomp.scale_points(q, zeroed), q)


def test_provenance_records_values_scales_band_and_source():
    c = precomp.load_defaults()
    p = c.provenance()
    assert p["s_xy"] == MEMO_S_XY
    assert p["s_z_mat"] == MEMO_S_Z
    assert p["s_xy_band"] == list(MEMO_S_XY_BAND)
    assert p["s_z_mat_band"] == list(MEMO_S_Z_BAND)
    assert p["xy_scale"] == pytest.approx(1.0 / (1.0 - MEMO_S_XY), rel=1e-15)
    assert p["z_scale"] == pytest.approx(1.0 / (1.0 - MEMO_S_Z), rel=1e-15)
    assert "SHRINKAGE_COEFFICIENTS_MEMO.md" in p["source"]
    assert p["schema_version"] == SCHEMA_VERSION
    assert p["material_only"] is True
    assert p["enabled"] is True
    assert json.dumps(p)          # provenance must be JSON-serialisable


def test_provenance_carries_the_unmeasured_rfam_caveat():
    """These are SLS literature values; RFAM material shrinkage is unmeasured.
    That has to travel with the number, not sit only in the memo."""
    p = precomp.load_defaults().provenance()
    assert "unmeasured" in p["applicability"].lower()
    assert "P1" in p["applicability"]


def test_provenance_band_gives_a_dimensional_uncertainty_on_a_length():
    """Every dimensional claim carries the band as a labeled uncertainty."""
    c = precomp.load_defaults()
    lo, hi = c.xy_scale_band()
    assert lo == pytest.approx(1.0 / (1.0 - 0.020), rel=1e-15)
    assert hi == pytest.approx(1.0 / (1.0 - 0.040), rel=1e-15)
    assert lo < c.xy_scale < hi


# --------------------------------------------------------------------------- #
# intake wiring: the pre-scale reaches BOTH chi and the mesh, consistently
# --------------------------------------------------------------------------- #
def test_predicate_and_solid_scale_together():
    """chi comes from the predicate and the mesh from the OCC solid; if only
    one were pre-scaled the target and the domain would disagree."""
    from solve3d.phase_e import geometry as geo
    c = precomp.ShrinkageL0(s_xy=0.030, s_z_mat=0.020)
    a = geo.CUBE_A_M / 2.0
    pred = geo.in_part_predicate("cube", precomp_coeffs=c)
    just_in = np.array([[a * c.xy_scale * 0.999], [0.0], [0.0]])
    just_out = np.array([[a * c.xy_scale * 1.001], [0.0], [0.0]])
    assert bool(pred(just_in)[0]) is True
    assert bool(pred(just_out)[0]) is False
    # z uses the other factor
    z_in = np.array([[0.0], [0.0], [a * c.z_scale * 0.999]])
    z_out = np.array([[0.0], [0.0], [a * c.z_scale * 1.001]])
    assert bool(pred(z_in)[0]) is True
    assert bool(pred(z_out)[0]) is False


def test_predicate_without_precomp_is_unchanged():
    from solve3d.phase_e import geometry as geo
    rng = np.random.default_rng(3)
    p = rng.normal(scale=8e-3, size=(3, 200))
    assert np.array_equal(geo.in_part_predicate("cube")(p),
                          geo.in_part_predicate("cube", precomp_coeffs=None)(p))


def test_build_case_applies_l0_by_default_and_records_provenance():
    """With the approved flip, a NEW run is pre-compensated without being
    asked, and says so in its provenance."""
    from solve3d.phase_e import run as R
    tc = R.build_case("cube")
    prov = tc.info.precomp
    assert prov["enabled"] is True
    assert prov["s_xy"] == MEMO_S_XY and prov["s_z_mat"] == MEMO_S_Z
    assert prov["s_xy_band"] == list(MEMO_S_XY_BAND)
    assert "unmeasured" in prov["applicability"].lower()
    # the mesh really is the scaled solid, not the nominal one
    c = precomp.load_defaults()
    want = geo_nominal_cube_volume() * c.xy_scale ** 2 * c.z_scale
    assert tc.info.part_volume_m3 == pytest.approx(want, rel=2e-3)


def test_build_case_with_zero_coefficients_reproduces_the_recorded_mesh():
    """The documented pre-L0 reproduction path, checked against the committed
    Phase E artifact rather than against a remembered number."""
    from solve3d.phase_e import run as R
    rec = json.loads((ROOT / "solve3d" / "phase_e" / "results"
                      / "phase_e_cube.json").read_text())["arms"]["_mesh"]
    tc = R.build_case("cube", precomp_coeffs=precomp.ShrinkageL0(0.0, 0.0))
    assert int(tc.ncells) == rec["n_cells"]
    assert int(tc.vol_nodal.size) == rec["n_nodes"]


def geo_nominal_cube_volume() -> float:
    from solve3d.phase_e import geometry as geo
    return geo.CUBE_A_M ** 3


def test_precompensated_volume_grows_by_the_expected_factor():
    from solve3d.phase_e import geometry as geo
    c = precomp.ShrinkageL0(s_xy=0.030, s_z_mat=0.020)
    v0 = geo.CUBE_A_M ** 3
    v1 = geo.nominal_volume_m3("cube", precomp_coeffs=c)
    assert v1 / v0 == pytest.approx(c.xy_scale ** 2 * c.z_scale, rel=1e-12)
