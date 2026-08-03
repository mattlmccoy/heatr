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
    assert res["phi_hist_len"] > 0
    # results must be STRICT JSON: a 2 s horizon never reaches phi90, so
    # t_phi90_s is non-finite in the raw Result; NaN in the file breaks
    # every browser JSON.parse downstream (found live: /grade/status 500)
    on_disk = json.loads((out / "results.json").read_text())
    json.dumps(on_disk, allow_nan=False)
    assert on_disk["t_phi90_s"] is None
    json.dumps(res, allow_nan=False)
