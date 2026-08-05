"""Level 0 pre-compensation wired into the pipeline (spec section 1).

Default ON with one honest switch. What the solver voxelizes, what the
verifier re-marches, and what the printer slices must all be the SAME
pre-compensated mesh -- otherwise the simulation and the print disagree.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

import studio3d.precomp as P
from studio3d.densify_job import run_job


@pytest.fixture()
def grade_env(tmp_path):
    mesh = tmp_path / "box20.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(mesh)
    gd = tmp_path / "grade"
    gd.mkdir(parents=True)
    return mesh, gd


# --------------------------------------------------------------------------- #
# prepare_mesh: the one helper both entry points call
# --------------------------------------------------------------------------- #
def test_prepare_mesh_default_on(tmp_path):
    src = tmp_path / "box.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(src)
    path, prov = P.prepare_mesh(src, tmp_path / "work")
    assert Path(path) == tmp_path / "work" / "precomp" / "box.stl"
    assert prov["enabled"] is True
    cfg = P.load_precomp_config()
    f_xy, f_z = P.compensation_factors(cfg)
    assert prov["f_xy"] == f_xy and prov["f_z"] == f_z
    assert prov["s_xy"] == cfg["s_xy"] and prov["s_z_mat"] == cfg["s_z_mat"]
    assert prov["applicability"] == cfg["applicability"]


def test_prepare_mesh_escape_hatch_records_disabled(tmp_path):
    src = tmp_path / "box.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(src)
    path, prov = P.prepare_mesh(src, tmp_path / "work", enabled=False)
    assert Path(path) == src                     # untouched geometry
    assert prov == {"enabled": False,
                    "reason": "precomp=False requested by the caller"}
    assert not (tmp_path / "work" / "precomp").exists()


def test_prepare_mesh_refuses_a_part_that_only_fits_before_compensation(
        tmp_path):
    """Chamber-fit ordering: the fit check sees the ENLARGED (green) part."""
    src = tmp_path / "wide.stl"
    trimesh.creation.box(extents=(58.5, 10.0, 10.0)).export(src)
    f_xy, _ = P.compensation_factors(P.load_precomp_config())
    assert 58.5 < 60.0 < 58.5 * f_xy              # the premise of the test
    with pytest.raises(ValueError) as e:
        P.prepare_mesh(src, tmp_path / "work")
    msg = str(e.value)
    assert "60.3" in msg                          # names the COMPENSATED bbox
    assert "58.5" in msg                          # and the nominal one
    assert "chamber" in msg.lower()


# --------------------------------------------------------------------------- #
# densify_job
# --------------------------------------------------------------------------- #
def test_run_job_marches_the_precompensated_mesh(grade_env, monkeypatch):
    mesh, gd = grade_env
    seen = {}

    def _fake_run_densify(mesh_path, out_dir, **kw):
        seen["mesh"] = mesh_path
        seen["precomp"] = kw.get("shrinkage_precomp")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {"arm": kw.get("arm"), "gates": {},
                "shrinkage_precomp": kw.get("shrinkage_precomp")}

    monkeypatch.setattr("studio3d.densify_job.run_densify", _fake_run_densify)
    res = run_job(str(mesh), gd, arm="uncorrected", n=16, max_time_s=1.0)
    assert Path(seen["mesh"]) == gd / "precomp" / "box20.stl"
    assert Path(seen["mesh"]).exists()
    assert res["shrinkage_precomp"]["enabled"] is True
    assert res["shrinkage_precomp"]["applicability"] == \
        P.load_precomp_config()["applicability"]


def test_run_job_precomp_false_records_it_explicitly(grade_env, monkeypatch):
    mesh, gd = grade_env
    seen = {}

    def _fake_run_densify(mesh_path, out_dir, **kw):
        seen["mesh"] = mesh_path
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {"arm": kw.get("arm"), "gates": {},
                "shrinkage_precomp": kw.get("shrinkage_precomp")}

    monkeypatch.setattr("studio3d.densify_job.run_densify", _fake_run_densify)
    res = run_job(str(mesh), gd, arm="uncorrected", n=16, max_time_s=1.0,
                  precomp=False)
    assert Path(seen["mesh"]) == mesh
    assert res["shrinkage_precomp"]["enabled"] is False
    assert not (gd / "precomp").exists()


def test_run_job_refuses_when_compensation_overflows_the_chamber(tmp_path):
    mesh = tmp_path / "wide.stl"
    trimesh.creation.box(extents=(58.5, 10.0, 10.0)).export(mesh)
    gd = tmp_path / "grade"
    gd.mkdir()
    with pytest.raises(ValueError, match="60.3"):
        run_job(str(mesh), gd, arm="uncorrected", n=16, max_time_s=1.0)


def test_run_job_writes_precomp_provenance_to_disk(grade_env):
    """Full march at n=16: results.json on disk carries the block."""
    mesh, gd = grade_env
    run_job(str(mesh), gd, arm="uncorrected", n=16, max_time_s=1.0)
    on_disk = json.loads(
        (gd / "heatr3d" / "uncorrected" / "results.json").read_text())
    blk = on_disk["shrinkage_precomp"]
    assert blk["enabled"] is True
    assert "pending P1" in blk["applicability"]
    assert (gd / "precomp" / "box20.stl.precomp.json").exists()


# --------------------------------------------------------------------------- #
# package_verify
# --------------------------------------------------------------------------- #
def test_verify_package_uses_the_precompensated_mesh(tmp_path, monkeypatch):
    from studio3d import package_verify as PV

    mesh = tmp_path / "box20.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(mesh)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    seen = {}

    def _fake_reconstruct(job_dir):
        n = 8
        sat = np.full((2, n, n), 0.7)
        mask = np.ones((2, n, n), bool)
        return sat, mask, np.array([0.5, 1.5]), 0.040

    def _fake_voxelize(mesh_path, n):
        seen["voxelized"] = mesh_path
        part = np.zeros((n, n, n), bool)
        part[2:6, 2:6, 2:6] = True
        return part

    def _fake_run_densify(mesh_path, out_dir, **kw):
        seen["marched"] = mesh_path
        return {"gates": {"energy_residual_ok": True, "T_ceiling_ok": True,
                          "clamp_bound": False},
                "sigma_T": 1.0, "rho_final_mean": 0.9}

    monkeypatch.setattr(PV, "reconstruct_sat_stack", _fake_reconstruct)
    monkeypatch.setattr(PV, "voxelize_stl", _fake_voxelize)
    monkeypatch.setattr(PV, "run_densify", _fake_run_densify)
    rec = PV.verify_package(pkg, str(mesh), n=8, tiff_job_dir=tmp_path / "job")
    assert "error" not in rec, rec.get("error")
    expect = str(pkg / "precomp" / "box20.stl")
    assert seen["voxelized"] == expect
    assert seen["marched"] == expect
    assert rec["shrinkage_precomp"]["enabled"] is True
    on_disk = json.loads(
        (pkg / "production_verify_summary.json").read_text())
    assert on_disk["shrinkage_precomp"]["enabled"] is True
