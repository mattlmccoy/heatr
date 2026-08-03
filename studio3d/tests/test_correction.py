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


def test_no_correction_source_is_a_loud_error(box_stl, tmp_path):
    gd = tmp_path / "grade"
    with pytest.raises(FileNotFoundError, match="verification"):
        build_correction(gd, str(box_stl), 16,
                         registry_path=tmp_path / "missing.json")


def test_null_correction_is_flagged_loudly(box_stl, tmp_path):
    """A dopant volume with sat = 1.0 everywhere corrects nothing; the
    provenance must say so instead of letting two identical arms render
    silently (found live on the l_extrusion stored 2.5-D result)."""
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
    prov = build_correction(gd, str(box_stl), n,
                            registry_path=tmp_path / "missing.json")
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
