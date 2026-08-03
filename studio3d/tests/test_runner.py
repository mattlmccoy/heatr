"""studio3d.runner: native heatr3d densify runs for the Studio (spec section 5).

Voxelization is mesh-driven (trimesh containment on the Grid cell centers),
no shape presets. Grid ceiling n <= 96 is enforced here as well as in the UI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from studio3d.runner import (ENGINE_LABEL, N_MAX, TRUST_BADGE, run_densify,
                             voxelize_stl)


@pytest.fixture()
def box20_stl(tmp_path) -> Path:
    """A 20 mm cube STL, the kind of small part the Studio imports."""
    p = tmp_path / "box20.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(p)
    return p


def test_voxelize_is_centered_and_volume_correct(box20_stl):
    part = voxelize_stl(str(box20_stl), n=16)
    assert part.shape == (16, 16, 16)
    assert part.dtype == bool
    # exact containment-at-cell-centers convention: for an axis-aligned
    # 20 mm cube the voxel count is (number of Grid centers inside)^3
    c = (np.arange(16) + 0.5) * (0.060 / 16) - 0.030
    n_axis = int(np.sum(np.abs(c) < 0.010))
    assert part.sum() == n_axis ** 3
    idx = np.argwhere(part)
    center = idx.mean(axis=0)
    assert np.allclose(center, (16 - 1) / 2.0, atol=1.0)


def test_oversize_part_is_refused(tmp_path):
    p = tmp_path / "big.stl"
    trimesh.creation.box(extents=(70.0, 20.0, 20.0)).export(p)
    with pytest.raises(ValueError, match="chamber"):
        voxelize_stl(str(p), n=16)


def test_grid_ceiling_enforced(box20_stl, tmp_path):
    with pytest.raises(ValueError, match="96"):
        run_densify(str(box20_stl), tmp_path / "out", n=N_MAX + 1)


def test_run_densify_writes_the_artifact_set(box20_stl, tmp_path):
    out = tmp_path / "out"
    res = run_densify(str(box20_stl), out, n=16, max_time_s=2.0)
    # engine + trust labeling (spec section 3)
    assert res["engine"] == ENGINE_LABEL == "heatr3d_native"
    assert res["trust_badge"] == TRUST_BADGE
    assert res["arm"] == "uncorrected"
    # standing gates surfaced, never buried
    for key in ("reached_phi90", "energy_residual_frac", "clamp_bound",
                "T_max_C"):
        assert key in res["gates"], key
    # artifact set on disk
    assert (out / "results.json").exists()
    assert json.loads((out / "results.json").read_text())["engine"] == ENGINE_LABEL
    with np.load(out / "fields.npz") as d:
        assert d["rho_final"].shape == (16, 16, 16)
        assert d["T_phi90"].shape == (16, 16, 16)
    meta = json.loads((out / "fieldmeta.json").read_text())
    assert meta["dims"] == [16, 16, 16]
    assert "rho_final" in meta["fields"]
    assert (out / "slices").is_dir()
    # densified-form 3-D model artifacts (spec 7d follow-on)
    assert (out / "warped_mesh.stl").exists()
    assert (out / "warped_geometry.json").exists()
    assert res["phi_hist_len"] > 0
    # results must be STRICT JSON: a 2 s horizon never reaches phi90, so
    # t_phi90_s is non-finite in the raw Result; NaN in the file breaks
    # every browser JSON.parse downstream (found live: /grade/status 500)
    on_disk = json.loads((out / "results.json").read_text())
    json.dumps(on_disk, allow_nan=False)
    assert on_disk["t_phi90_s"] is None
    json.dumps(res, allow_nan=False)


def test_voxelize_preserves_through_holes(tmp_path):
    """A tube's bore must stay empty: hole-true 3-D geometry (Matt
    2026-08-03: 'fails to do any geometry with holes')."""
    p = tmp_path / "tube.stl"
    outer = trimesh.creation.cylinder(radius=12.0, height=20.0, sections=64)
    inner = trimesh.creation.cylinder(radius=5.0, height=22.0, sections=64)
    tube = outer.difference(inner)
    assert tube.is_watertight
    tube.export(p)
    part = voxelize_stl(str(p), n=32)
    zc = np.where(part.any(axis=(0, 1)))[0]
    mid = zc[len(zc) // 2]
    c = 32 // 2
    assert part[:, :, mid].sum() > 0
    assert not part[c, c, mid], "bore center must be empty"
    assert not part[c - 1, c - 1, mid], "bore interior must be empty"


def test_stop_mean_rho_is_wired_and_recorded(box20_stl, tmp_path):
    """Studio densify runs stop at a target mean density instead of
    over-marching to saturation (Matt 2026-08-03: the before form must be
    a real volumetric output, not a uniform squash; a saturated field has
    no spatial structure left)."""
    out = tmp_path / "out"
    res = run_densify(str(box20_stl), out, n=16, max_time_s=5.0,
                      stop_mean_rho=0.9)
    assert res["stop_mean_rho"] == 0.9
    assert "sim_time_s" in res
    assert res["sim_time_s"] <= 5.0 + 1e-9


def test_voxelize_is_fast_on_real_size_meshes(tmp_path):
    """Perf gate (found live: a 125k-triangle part sat 34 minutes inside
    mesh.contains at n=64). An 82k-triangle sphere at n=64 must voxelize
    in seconds."""
    import time
    p = tmp_path / "sphere.stl"
    trimesh.creation.icosphere(subdivisions=6, radius=15.0).export(p)
    t0 = time.time()
    part = voxelize_stl(str(p), n=64)
    dt = time.time() - t0
    assert part.sum() > 1000
    assert dt < 20.0, f"voxelize took {dt:.1f} s"


def test_fast_march_optin_is_bit_identical_and_labeled(box20_stl, tmp_path):
    """march_fast opt-in (engine-lane blessing, adoption terms 2026-08-03):
    results must be BIT-IDENTICAL to the reference march, and the run must
    record the numba env provenance (the scipy-downgrade lesson)."""
    ref = run_densify(str(box20_stl), tmp_path / "ref", n=16, max_time_s=2.0)
    fast = run_densify(str(box20_stl), tmp_path / "fast", n=16,
                       max_time_s=2.0, fast_march=True)
    with np.load(tmp_path / "ref" / "fields.npz") as a, \
         np.load(tmp_path / "fast" / "fields.npz") as b:
        for key in ("T_phi90", "phi_final", "rho_final"):
            assert np.array_equal(a[key], b[key]), key
    assert fast["engine_march"] == "march_fast"
    assert "numba" in fast["env_provenance"]
    assert ref.get("engine_march", "heatr3d") == "heatr3d"
