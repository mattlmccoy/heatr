"""Verify what prints (spec 7c): the sat map is reconstructed from the
EMITTED TIFFs (graded / ungraded level ratio), never from the in-memory
volume, then re-marched through heatr3d."""
from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from studio3d.package_verify import reconstruct_sat_stack, verify_package


def _write_tiff_pair(job, k, base_levels, sat):
    """Meteor convention: WhiteIsZero 8-bit gray = 255*(1 - level/15)."""
    (job / "_ungraded").mkdir(exist_ok=True)
    graded_levels = np.round(base_levels * sat).astype(int)
    for d, lv in ((job, graded_levels), (job / "_ungraded", base_levels)):
        gray = (255 * (1 - lv / 15.0)).astype(np.uint8)
        Image.fromarray(gray, "L").save(d / f"layer_{k:04d}.tif")


def test_reconstruct_recovers_the_applied_sat(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    ny = nx = 40
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ink = (np.abs(xx - nx / 2) < 10) & (np.abs(yy - ny / 2) < 10)
    base = np.where(ink, 15, 0)
    for k, s in enumerate((0.6, 1.0, 0.8)):
        _write_tiff_pair(job, k, base, s)
    # dpi 25.4 -> exactly 1 mm/px, so the 40 px canvas is a 40 mm frame:
    # the reconstruction must report the canvas's PHYSICAL size (the frame
    # the transfer needs as its source chamber), never assume 60 mm
    (job / "job_info.json").write_text(json.dumps(
        {"layer_count": 3, "dpi": 25.4, "bpp": 4, "layer_height_mm": 0.5}))
    sat, mask, z_mm, canvas_m = reconstruct_sat_stack(job)
    assert sat.shape == (3, ny, nx)
    assert np.allclose(sat[0][mask[0]].mean(), 0.6, atol=0.04)
    assert np.allclose(sat[1][mask[1]].mean(), 1.0, atol=0.04)
    assert list(np.round(z_mm, 2)) == [0.25, 0.75, 1.25]
    assert canvas_m == pytest.approx(0.040, rel=1e-6)


def test_verify_package_runs_the_march_and_records_gates(tmp_path):
    mesh = tmp_path / "part.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(mesh)
    job = tmp_path / "job"
    job.mkdir()
    ny = nx = 40
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ink = (np.abs(xx - nx / 2) < 7) & (np.abs(yy - ny / 2) < 7)
    base = np.where(ink, 15, 0)
    nz = 10
    for k in range(nz):
        _write_tiff_pair(job, k, base, 0.7)
    # 1 mm/px so the 40 px canvas is a physically consistent 40 mm frame
    # for a 14 px = 14 mm ink square (the old 720 dpi fixture put a 20 mm
    # part on a 1.4 mm canvas, which the frame fix rightly breaks)
    (job / "job_info.json").write_text(json.dumps(
        {"layer_count": nz, "dpi": 25.4, "bpp": 4, "layer_height_mm": 2.0}))
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "production_verify_summary.json").write_text(
        json.dumps({"run": False}))
    rec = verify_package(pkg, str(mesh), n=16, tiff_job_dir=job,
                         max_time_s=2.0)
    assert rec["run"] is True
    assert "gates_ok" in rec
    assert rec["source"] == "emitted_rasters"
    on_disk = json.loads((pkg / "production_verify_summary.json").read_text())
    assert on_disk["run"] is True
