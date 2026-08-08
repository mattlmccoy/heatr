"""densify_job CLI: the one subprocess entry the Studio server calls."""
from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from studio3d.densify_job import recommended_drive, run_job


@pytest.fixture()
def grade_env(tmp_path):
    mesh = tmp_path / "box20.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(mesh)
    gd = tmp_path / "grade"
    (gd / "heatr").mkdir(parents=True)
    ng, nz = 40, 8
    yy, xx = np.meshgrid(np.arange(ng), np.arange(ng), indexing="ij")
    mask2 = (np.abs(xx - ng / 2) < ng * 0.2) & (np.abs(yy - ng / 2) < ng * 0.2)
    np.savez(gd / "heatr" / "dopant_volume.npz",
             sat=np.where(mask2, 0.5, 1.0)[None].repeat(nz, 0).astype(np.float32),
             part_mask=mask2[None].repeat(nz, 0),
             z_mm=(np.arange(nz) + 0.5) * 2.5,
             area_mm2=np.full(nz, 100.0), method=np.array(["m"] * nz),
             gain=np.ones(nz), chamber_m=0.060)
    return mesh, gd


def _write_solve_results(gd, **fields):
    out = gd / "heatr3d" / "solve"
    out.mkdir(parents=True, exist_ok=True)
    (out / "studio_solve_results.json").write_text(json.dumps(fields))


def test_recommended_drive_read_from_solve_artifact(tmp_path):
    """The joint solve's recommended ceiling-feasible drive is the standard
    print power (cross-lane contract 2026-08-07): studio_solve_results.json
    carries recommended_power_density_w_per_m3, and the Studio densifies at
    it instead of the nominal hardcode. Absent/null/no-file -> None (the
    caller falls back to nominal and records drive_recommended=false)."""
    gd = tmp_path / "grade"
    _write_solve_results(gd, recommended_power_density_w_per_m3=5.4e5)
    assert recommended_drive(gd) == 5.4e5
    # honest-null (drive-limited part): field present but null -> None
    _write_solve_results(gd, recommended_power_density_w_per_m3=None,
                         recommended_drive_reason="drive-limited")
    assert recommended_drive(gd) is None
    # field absent -> None
    _write_solve_results(gd, solved_label=True)
    assert recommended_drive(gd) is None
    # no solve artifact at all -> None
    assert recommended_drive(tmp_path / "empty") is None


def test_arms_densify_at_the_recommended_drive(grade_env):
    """When the solve recommends a ceiling-feasible drive, the densify arm
    runs at THAT power, not the nominal hardcode - so the print is simulated
    at the power it will actually use and the ceiling gate is meaningful."""
    mesh, gd = grade_env
    _write_solve_results(gd, recommended_power_density_w_per_m3=5.0e5)
    res = run_job(str(mesh), gd, arm="uncorrected", n=16, max_time_s=1.0)
    assert res["power_density_w_per_m3"] == 5.0e5
    assert res["drive_recommended"] is True
    # and it must be ON DISK: the package reads results.json, not the return
    on_disk = json.loads(
        (gd / "heatr3d" / "uncorrected" / "results.json").read_text())
    assert on_disk["drive_recommended"] is True
    assert on_disk["power_density_w_per_m3"] == 5.0e5


def test_no_recommended_drive_falls_back_to_nominal_and_says_so(grade_env):
    """No solve / honest-null: densify at nominal but RECORD drive_recommended
    false, so nothing silently ships at a cooking drive without it being
    visible (the ceiling gate is still the backstop)."""
    mesh, gd = grade_env
    res = run_job(str(mesh), gd, arm="uncorrected", n=16, max_time_s=1.0)
    assert res["drive_recommended"] is False
    assert res["power_density_w_per_m3"] > 0    # the heatr3d nominal default


def test_uncorrected_arm(grade_env):
    mesh, gd = grade_env
    res = run_job(str(mesh), gd, arm="uncorrected", n=16, max_time_s=1.0)
    assert res["arm"] == "uncorrected"
    assert res["correction_engine"] is None
    assert (gd / "heatr3d" / "uncorrected" / "results.json").exists()


def test_corrected_arm_carries_the_correction_engine(grade_env):
    mesh, gd = grade_env
    res = run_job(str(mesh), gd, arm="corrected", n=16, max_time_s=1.0)
    assert res["arm"] == "corrected"
    assert res["correction_engine"] == "heatr_25d_perslice"
    on_disk = json.loads(
        (gd / "heatr3d" / "corrected" / "results.json").read_text())
    assert on_disk["correction_engine"] == "heatr_25d_perslice"
    prov = json.loads(
        (gd / "heatr3d" / "correction_provenance.json").read_text())
    assert prov["transfer"]["state"] == "measured_and_passed"


def test_unknown_arm_refused(grade_env):
    mesh, gd = grade_env
    with pytest.raises(ValueError, match="arm"):
        run_job(str(mesh), gd, arm="mystery", n=16)
