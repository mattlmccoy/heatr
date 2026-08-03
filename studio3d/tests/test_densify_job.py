"""densify_job CLI: the one subprocess entry the Studio server calls."""
from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from studio3d.densify_job import run_job


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
