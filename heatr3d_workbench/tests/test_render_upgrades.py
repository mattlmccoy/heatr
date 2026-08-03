"""Display-resolution and legibility upgrades (Matt iteration 2, 2026-08-03).

1. Colormap floors: never map data onto near-black on the dark UI - the
   colormap families stay (inferno/viridis/density) but their bottoms are
   clipped. Outside-part (bed) renders as a distinct muted slate, not the
   colormap floor.
2. Display smoothing per the visualization standard: gaussian sigma ~1.0 on
   DISPLAY arrays only + bilinear interpolation; phi contours stay unsmoothed.
3. Isosurface export: marching cubes (linear interpolation) on the phi/rho
   fields -> mesh JSON in mm for smooth three.js rendering.
4. --rerender mode: re-render views from an existing run dir without solving.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from heatr3d_workbench import workbench_job as WJ

ROOT = Path(__file__).resolve().parents[2]
CONE = ROOT / "outputs_eqs" / "_heatr3d" / "5a4e465fde1b"


def test_truncated_cmap_floor_is_not_near_black():
    cm = WJ.truncated_cmap("inferno", lo=0.15)
    r, g, b, _ = cm(0.0)
    assert (r + g + b) > 0.15, "clipped inferno floor must be visibly non-black"
    import matplotlib.pyplot as plt
    full = plt.get_cmap("inferno")
    assert cm(0.0) == pytest.approx(full(0.15), abs=1e-6)
    assert cm(1.0) == pytest.approx(full(1.0), abs=1e-6)


def test_field_cmaps_have_bad_color_distinct_slate():
    for name in ("T_phi90", "rho_final", "sat", "Qrf", "phi_final"):
        cm = WJ.field_cmap(name)
        bad = cm.get_bad()
        assert bad[3] > 0, f"{name}: outside-part must be a visible tone, not transparent"
        assert 0.05 < bad[0] < 0.35, f"{name}: bad color should be the muted slate"


def test_display_smooth_preserves_shape_and_mask():
    rng = np.random.default_rng(0)
    a = rng.random((20, 20))
    m = np.zeros((20, 20), bool); m[5:15, 5:15] = True
    out = WJ.display_smooth(np.where(m, a, np.nan), m)
    assert out.shape == a.shape
    assert np.isnan(out[0, 0])          # outside stays masked
    assert np.isfinite(out[10, 10])
    # smoothing reduces variance inside the mask (display-only smoothing)
    assert np.nanstd(out[m]) < np.nanstd(np.where(m, a, np.nan)[m])


def test_isosurface_export_sphere_field():
    n = 24
    x = np.linspace(-1, 1, n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    phi = np.clip(1.2 - np.sqrt(X**2 + Y**2 + Z**2), 0, 1)   # phi=0.9 at r=0.3
    surfs = WJ.isosurfaces_for({"phi_final": phi}, h_mm=1.0,
                               specs=[("phi_final", 0.9, "melt front phi=0.9")])
    assert len(surfs) == 1
    s = surfs[0]
    assert s["level"] == 0.9 and s["field"] == "phi_final"
    v = np.asarray(s["vertices_mm"]); f = np.asarray(s["faces"])
    assert v.ndim == 2 and v.shape[1] == 3 and len(f) > 0
    r = np.linalg.norm(v - v.mean(axis=0), axis=1)
    # radius in mm: 0.3 in unit coords -> 0.3/(2/(n-1)) grid cells * 1 mm
    expect = 0.3 / (2.0 / (n - 1))
    assert abs(r.mean() - expect) / expect < 0.05


def test_isosurface_absent_level_is_skipped_not_faked():
    phi = np.zeros((8, 8, 8))
    surfs = WJ.isosurfaces_for({"phi_final": phi}, h_mm=1.0,
                               specs=[("phi_final", 0.9, "melt front")])
    assert surfs == []


@pytest.mark.skipif(not (CONE / "fields.npz").exists(), reason="cone run absent")
def test_rerender_mode_writes_views_without_solving(tmp_path):
    d = tmp_path / "run"
    shutil.copytree(CONE, d)
    for p in list((d / "slices").glob("*.png")):
        p.unlink()
    WJ.main(["workbench_job.py", str(d / "config.json"), "--rerender"])
    assert (d / "slices").exists() and any((d / "slices").glob("T_phi90_z_*.png"))
    iso = json.loads((d / "isosurfaces.json").read_text())
    assert any(s["field"] == "phi_final" for s in iso["surfaces"])
    # results.json untouched by a pure re-render
    assert json.loads((d / "results.json").read_text()) == \
        json.loads((CONE / "results.json").read_text())
