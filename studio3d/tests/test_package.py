"""Print package emitter vs the FROZEN schema 2.0.0 (spec 7b, e8600d1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from studio3d.package import (SCHEMA_VERSION, emit_package, is_sendable,
                              validate_manifest)


@pytest.fixture()
def grade_env(tmp_path):
    """A grade dir with everything a package needs, plus a fake TIFF job."""
    mesh = tmp_path / "part.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(mesh)
    gd = tmp_path / "grade"
    (gd / "heatr3d" / "uncorrected").mkdir(parents=True)
    (gd / "heatr3d" / "corrected").mkdir(parents=True)
    (gd / "intake.json").write_text(json.dumps(
        {"accepted": True, "error": None,
         "checks": {"chamber_fit": {"ok": True}}}))
    gates = {"reached_phi90": True, "MELT_ONSET_FALLBACK": False,
             "energy_residual_frac": 1e-13, "energy_residual_ok": True,
             "clamp_bound": False, "T_max_C": 200.0, "T_ceiling_C": 250.0,
             "T_ceiling_ok": True}
    for arm, eng in (("uncorrected", None),
                     ("corrected", "heatr3d_native_inversion")):
        (gd / "heatr3d" / arm / "results.json").write_text(json.dumps(
            {"engine": "heatr3d_native", "trust_badge": "badge", "arm": arm,
             "correction_engine": eng, "grid_n": 16, "sigma_T": 30.0,
             "t_phi90_s": 100.0, "sim_time_s": 200.0, "stop_mean_rho": 0.98,
             "rho_final_mean": 0.97, "rho_final_std": 0.02, "gates": gates}))
    (gd / "heatr3d" / "correction_provenance.json").write_text(json.dumps(
        {"engine": "heatr3d_native_inversion", "trust_badge": "legacy badge",
         "transfer": {"state": "transfer_not_applicable",
                      "method": "native_inversion_same_grid",
                      "dopant_mass_move_rel": 0.0},
         "null_correction": False, "grid_n": 16}))
    job = tmp_path / "tiff_job"
    job.mkdir()
    from PIL import Image
    for k in range(2):
        Image.new("L", (32, 32), 255).save(job / f"layer_{k:04d}.tif")
    (job / "job_info.json").write_text(json.dumps(
        {"job_name": "test_job", "layer_count": 2, "dpi": 720, "bpp": 4,
         "graded": True}))
    return mesh, gd, job


def test_power_settings_carries_the_drive_the_arms_used(grade_env, tmp_path):
    """The package must record the RECOMMENDED per-part drive the arms
    actually marched at (the ceiling-feasible power), not the nominal
    hardcode - otherwise the printed part cooks at a power the verification
    never simulated. Stays exactly-one-power, so 2.0.0 is unchanged."""
    mesh, gd, job = grade_env
    for arm in ("uncorrected", "corrected"):
        rp = gd / "heatr3d" / arm / "results.json"
        r = json.loads(rp.read_text())
        r["power_density_w_per_m3"] = 5.4e5      # backed-off ceiling-feasible
        r["drive_recommended"] = True
        rp.write_text(json.dumps(r))
    _, manifest = emit_package(
        tmp_path / "packages", grade_dir=gd, mesh_path=str(mesh),
        part_name="part", tiff_job_dir=job,
        options={"turntable_intent": False, "dwell_intent": False})
    pw = manifest["power_settings"]
    assert pw["power_density_w_per_m3"] == 5.4e5
    assert pw["drive_recommended"] is True
    assert "voltage_v" not in pw
    assert validate_manifest(manifest) == []


def test_manifest_conforms_to_the_frozen_schema(grade_env, tmp_path):
    mesh, gd, job = grade_env
    pkg_dir, manifest = emit_package(
        tmp_path / "packages", grade_dir=gd, mesh_path=str(mesh),
        part_name="part", tiff_job_dir=job,
        options={"turntable_intent": False, "dwell_intent": False})
    assert manifest["schema_version"] == SCHEMA_VERSION == "2.0.0"
    assert validate_manifest(manifest) == []
    # exactly-one power settings source
    assert "power_density_w_per_m3" in manifest["power_settings"]
    assert "voltage_v" not in manifest["power_settings"]
    # turntable NEVER silently absent: static record file exists
    assert manifest["turntable"]["mode"] == "static"
    assert (pkg_dir / manifest["turntable"]["file"]).exists()
    # verification explicit not-run until the real verify runs
    assert manifest["production_verify"]["run"] is False
    # every file hashed
    for f in manifest["files"]:
        assert len(f["sha256"]) == 64
    # provenance carried with a legal three-state transfer
    assert manifest["correction_provenance"]["transfer"]["state"] in (
        "measured_and_passed", "measured_and_failed",
        "transfer_not_applicable")
    assert "heatr3d" in manifest["engine_versions"]
    assert (pkg_dir / "manifest.json").exists()


def test_turntable_intent_is_recorded_not_fabricated(grade_env, tmp_path):
    mesh, gd, job = grade_env
    pkg_dir, manifest = emit_package(
        tmp_path / "packages", grade_dir=gd, mesh_path=str(mesh),
        part_name="part", tiff_job_dir=job,
        options={"turntable_intent": True, "dwell_intent": True})
    tt = manifest["turntable"]
    assert tt["mode"] == "static"          # no generator exists yet
    assert tt["requested_intent"] is True
    assert tt["advisory"] is True
    rec = json.loads((pkg_dir / tt["file"]).read_text())
    assert rec["requested_intent"] is True


def test_validator_rejects_both_power_sources(grade_env, tmp_path):
    mesh, gd, job = grade_env
    _, manifest = emit_package(
        tmp_path / "packages", grade_dir=gd, mesh_path=str(mesh),
        part_name="part", tiff_job_dir=job, options={})
    manifest["power_settings"]["voltage_v"] = 860.0
    errs = validate_manifest(manifest)
    assert any("exactly one" in e for e in errs)


def test_send_is_blocked_until_verification_passes(grade_env, tmp_path):
    mesh, gd, job = grade_env
    _, manifest = emit_package(
        tmp_path / "packages", grade_dir=gd, mesh_path=str(mesh),
        part_name="part", tiff_job_dir=job, options={})
    assert is_sendable(manifest) is False
    manifest["production_verify"] = {"run": True, "gates_ok": True}
    assert is_sendable(manifest) is True
    manifest["production_verify"] = {"run": True, "gates_ok": False}
    assert is_sendable(manifest) is False
