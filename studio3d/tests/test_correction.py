"""Correction source selection (spec section 6): solved registry first,
2.5-D per-slice fallback, engine labeled everywhere, never blended."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from studio3d.correction import build_correction
from studio3d.registry import part_hash
from studio3d.runner import voxelize_stl


@pytest.fixture()
def box_stl(tmp_path) -> Path:
    p = tmp_path / "box20.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(p)
    return p


def _fake_dg0_npz(path: Path, part: np.ndarray, value: float) -> None:
    n = part.shape[0]
    h = 0.060 / n
    idx = np.argwhere(part)
    cents = (idx + 0.5) * h - 0.030
    np.savez(path, centroids=cents,
             s_map=np.full(len(cents), value),
             volumes=np.full(len(cents), h ** 3))


def test_solved_registry_match_uses_solve3d_map(box_stl, tmp_path):
    n = 16
    part = voxelize_stl(str(box_stl), n)
    art = tmp_path / "solved.npz"
    _fake_dg0_npz(art, part, 0.8)
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"entries": [{
        "name": "test", "part_sha256": part_hash(part), "grid_n": n,
        "artifact": str(art), "form": "dg0", "engine": "solve3d_solved",
        "trust_badge": "solve3d solved | sim-only"}]}))
    gd = tmp_path / "grade"
    prov = build_correction(gd, str(box_stl), n, registry_path=reg)
    assert prov["engine"] == "solve3d_solved"
    assert prov["transfer"]["state"] == "measured_and_passed"
    with np.load(gd / "heatr3d" / "correction_sat.npz") as d:
        assert d["sat"].shape == (n, n, n)
        assert np.allclose(d["sat"][part], 0.8, atol=1e-6)


def test_fallback_to_25d_perslice(box_stl, tmp_path):
    n = 16
    gd = tmp_path / "grade"
    (gd / "heatr").mkdir(parents=True)
    ng = 40
    yy, xx = np.meshgrid(np.arange(ng), np.arange(ng), indexing="ij")
    mask2 = (np.abs(xx - ng / 2) < ng * 0.2) & (np.abs(yy - ng / 2) < ng * 0.2)
    nz = 8
    np.savez(gd / "heatr" / "dopant_volume.npz",
             sat=np.where(mask2, 0.5, 1.0)[None].repeat(nz, 0).astype(np.float32),
             part_mask=mask2[None].repeat(nz, 0),
             z_mm=(np.arange(nz) + 0.5) * 2.5,
             area_mm2=np.full(nz, 100.0), method=np.array(["m"] * nz),
             gain=np.ones(nz), chamber_m=0.060)
    prov = build_correction(gd, str(box_stl), n,
                            registry_path=tmp_path / "missing.json")
    assert prov["engine"] == "heatr_25d_perslice"
    assert prov["transfer"]["state"] == "measured_and_passed"


# (superseded: the no-source case now falls back to the native inversion
# when a BEFORE arm exists; see test_error_only_when_no_source_at_all)


def test_null_correction_is_flagged_when_even_the_inversion_degenerates(
        box_stl, tmp_path):
    """Null 2.5-D map falls through to the native inversion; when the
    before-arm fields are UNIFORM (saturated run) the inversion itself
    degenerates to an unmodulated map and the null flag must fire."""
    from studio3d.runner import voxelize_stl
    n = 16
    gd = tmp_path / "grade"
    (gd / "heatr").mkdir(parents=True)
    ng, nz = 40, 8
    yy, xx = np.meshgrid(np.arange(ng), np.arange(ng), indexing="ij")
    mask2 = (np.abs(xx - ng / 2) < ng * 0.2) & (np.abs(yy - ng / 2) < ng * 0.2)
    np.savez(gd / "heatr" / "dopant_volume.npz",
             sat=np.ones((nz, ng, ng), np.float32),
             part_mask=mask2[None].repeat(nz, 0),
             z_mm=(np.arange(nz) + 0.5) * 2.5,
             area_mm2=np.full(nz, 100.0), method=np.array(["m0"] * nz),
             gain=np.ones(nz), chamber_m=0.060)
    part = voxelize_stl(str(box_stl), n)
    out = gd / "heatr3d" / "uncorrected"
    out.mkdir(parents=True)
    np.savez(out / "fields.npz", part=part,
             T_phi90=np.where(part, 200.0, 23.0).astype(np.float32),
             phi_final=np.where(part, 1.0, 0.0).astype(np.float32),
             Qrf=np.zeros(part.shape, np.float32),
             rho_final=np.where(part, 1.0, 0.0).astype(np.float32),
             sat=np.zeros((1,), np.float32), h=0.060 / n)
    prov = build_correction(gd, str(box_stl), n,
                            registry_path=tmp_path / "missing.json")
    assert prov["engine"] == "heatr3d_native_inversion"
    assert prov["null_correction"] is True
    assert "NULL" in prov["null_note"]


def test_real_correction_is_not_flagged_null(box_stl, tmp_path):
    prov_modulated = None
    n = 16
    gd = tmp_path / "grade"
    (gd / "heatr").mkdir(parents=True)
    ng, nz = 40, 8
    yy, xx = np.meshgrid(np.arange(ng), np.arange(ng), indexing="ij")
    mask2 = (np.abs(xx - ng / 2) < ng * 0.2) & (np.abs(yy - ng / 2) < ng * 0.2)
    np.savez(gd / "heatr" / "dopant_volume.npz",
             sat=np.where(mask2, 0.6, 1.0)[None].repeat(nz, 0).astype(np.float32),
             part_mask=mask2[None].repeat(nz, 0),
             z_mm=(np.arange(nz) + 0.5) * 2.5,
             area_mm2=np.full(nz, 100.0), method=np.array(["m"] * nz),
             gain=np.ones(nz), chamber_m=0.060)
    prov_modulated = build_correction(gd, str(box_stl), n,
                                      registry_path=tmp_path / "missing.json")
    assert prov_modulated["null_correction"] is False


def _fake_before_arm(gd, part, n):
    """A before-arm artifact set with STRUCTURED fields (as a stop-at-target
    run produces), for the native-inversion fallback."""
    import numpy as np
    out = gd / "heatr3d" / "uncorrected"
    out.mkdir(parents=True, exist_ok=True)
    idx = np.indices(part.shape).astype(float)
    grad = idx[0] / part.shape[0]
    T = np.where(part, 150.0 + 100.0 * grad, 23.0)
    rho = np.where(part, 0.6 + 0.35 * grad, 0.0)
    np.savez(out / "fields.npz", part=part,
             T_phi90=T.astype(np.float32),
             phi_final=np.where(part, 0.9, 0.0).astype(np.float32),
             Qrf=np.zeros_like(T, np.float32),
             rho_final=rho.astype(np.float32),
             sat=np.zeros((1,), np.float32), h=0.060 / n)


def test_native_inversion_fallback_when_25d_is_null(box_stl, tmp_path):
    """When the rulebook refuses a part (null 2.5-D map) and no solved map
    matches, the corrected arm must still get a REAL modulated FGM: the
    native density-targeted inversion from the before arm, labeled legacy
    (Matt 2026-08-03: 'that should be showing an FGM corrected
    simulation')."""
    from studio3d.runner import voxelize_stl
    n = 16
    gd = tmp_path / "grade"
    (gd / "heatr").mkdir(parents=True)
    ng, nz = 40, 8
    yy, xx = np.meshgrid(np.arange(ng), np.arange(ng), indexing="ij")
    mask2 = (np.abs(xx - ng / 2) < ng * 0.2) & (np.abs(yy - ng / 2) < ng * 0.2)
    np.savez(gd / "heatr" / "dopant_volume.npz",
             sat=np.ones((nz, ng, ng), np.float32),
             part_mask=mask2[None].repeat(nz, 0),
             z_mm=(np.arange(nz) + 0.5) * 2.5,
             area_mm2=np.full(nz, 100.0), method=np.array(["m0"] * nz),
             gain=np.ones(nz), chamber_m=0.060)
    part = voxelize_stl(str(box_stl), n)
    _fake_before_arm(gd, part, n)
    prov = build_correction(gd, str(box_stl), n,
                            registry_path=tmp_path / "missing.json")
    assert prov["engine"] == "heatr3d_native_inversion"
    assert "legacy" in prov["trust_badge"]
    assert prov["null_correction"] is False
    with np.load(gd / "heatr3d" / "correction_sat.npz") as d:
        sat = d["sat"]
    assert sat[part].std() > 0.05, "the map must actually be modulated"


def test_native_inversion_fallback_without_any_25d(box_stl, tmp_path):
    from studio3d.runner import voxelize_stl
    n = 16
    gd = tmp_path / "grade"
    part = voxelize_stl(str(box_stl), n)
    _fake_before_arm(gd, part, n)
    prov = build_correction(gd, str(box_stl), n,
                            registry_path=tmp_path / "missing.json")
    assert prov["engine"] == "heatr3d_native_inversion"


def test_error_only_when_no_source_at_all(box_stl, tmp_path):
    gd = tmp_path / "grade"
    with pytest.raises(FileNotFoundError, match="BEFORE"):
        build_correction(gd, str(box_stl), 16,
                         registry_path=tmp_path / "missing.json")


def test_correction_also_emits_the_meteor_stack(box_stl, tmp_path):
    """Whatever engine built the correction, the print path grades TIFFs
    from a meteor-convention stack (sat[k, iy, ix], z_mm from part bottom,
    1.0 = unmodulated outside the mask)."""
    from studio3d.runner import voxelize_stl
    n = 16
    gd = tmp_path / "grade"
    part = voxelize_stl(str(box_stl), n)
    _fake_before_arm(gd, part, n)
    build_correction(gd, str(box_stl), n,
                     registry_path=tmp_path / "missing.json")
    with np.load(gd / "heatr3d" / "correction_stack.npz") as d:
        sat, mask, z = d["sat"], d["part_mask"], d["z_mm"]
    zs = np.where(part.any(axis=(0, 1)))[0]
    assert sat.shape == (len(zs), n, n)
    assert mask.dtype == bool
    assert z[0] > 0 and np.all(np.diff(z) > 0)
    assert sat[~mask].min() == 1.0          # unmodulated outside the part
