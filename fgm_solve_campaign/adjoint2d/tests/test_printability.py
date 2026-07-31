"""Contract tests for the printer quantizer.

The fixtures here are CAPTURED FROM REAL STORED ARTIFACTS, not invented: the
`level_map` arrays inside the production FGM npz files that the historical runs
actually consumed. A quantizer that cannot reproduce those bit-for-bit is not
the production quantizer.

Production references:
  fgm_generator.py:583-586   sim-resolution quantization
                             `level_map_sim = np.round(sat_scaled * max_val)`
  fgm_generator.py:593-606   printer-resolution round trip
                             `px_m = 25.4e-3/dpi; zx = dx_m/px_m;`
                             `sat_dpi = zoom(sat_scaled, (zy, zx), order=1)`
                             `level_map_dpi = clip(round(sat_dpi*max_val),0,max_val)`
  rfam_eqs_coupled.py:366-380  the loader that inverts it
                             `sat = level_map/maxv`, zoom back to the sim grid,
                             clip to [0, 1]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adjoint2d import printability as pq

MAIN = Path("/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research"
            "/research/binderjet/code/geo-prewarp")
FCC = MAIN / "outputs_eqs/fgm_calibrated_control/runs"
GDR = MAIN / "outputs_eqs/geometry_dual_readstate/runs"

STORED = [
    FCC / "square/map_m0p9033/fgm_baseline_T_phi90_4bpp_mag0p90.npz",
    FCC / "triangle/map_m1p2164/fgm_baseline_T_phi90_4bpp_mag1p22.npz",
    FCC / "cross/map_m1p1855/fgm_baseline_T_phi90_4bpp_mag1p19.npz",
    FCC / "L_shape/map_m0p1110/fgm_baseline_T_phi90_4bpp_mag0p11.npz",
    GDR / "square/map_m0p85/fgm_baseline_T_phi90_4bpp_mag0p85.npz",
]


def test_quantize_levels_matches_generator_convention():
    """fgm_generator.py:586 is round(sat*max_val); the value grid is k/max_val."""
    sat = np.array([[0.0, 0.03, 0.0334, 0.5, 0.9667, 1.0]])
    got4 = pq.quantize_levels(sat, bpp=4)
    assert np.allclose(got4, np.round(sat * 15.0) / 15.0)
    got2 = pq.quantize_levels(sat, bpp=2)
    assert np.allclose(got2, np.round(sat * 3.0) / 3.0)
    # 4 bpp admits exactly 16 distinct values on [0, 1], 2 bpp exactly 4.
    grid = pq.quantize_levels(np.linspace(0.0, 1.0, 4001), bpp=2)
    assert len(np.unique(grid)) == 4


def test_quantize_levels_allows_double_pass_above_one():
    """Saturation above 1.0 is a second printing pass, not a clipped value.

    The level QUANTUM stays the printer's 1/max_val; only the number of levels
    grows. Clipping at 1.0 would silently convert a double-pass map into a
    single-pass one and hide the actual cost of the box constraint.
    """
    sat = np.array([1.0, 1.4667, 1.5])
    got = pq.quantize_levels(sat, bpp=4, sat_max=1.5)
    assert np.allclose(got, np.round(sat * 15.0) / 15.0)
    clipped = pq.quantize_levels(sat, bpp=4, sat_max=1.0)
    assert np.allclose(clipped, 1.0)


@pytest.mark.parametrize("path", STORED, ids=lambda p: p.parent.parent.name + "_" + p.parent.name)
def test_printer_level_map_reproduces_stored_artifact(path):
    """Captured-real-data gate: reproduce the historical level_map exactly."""
    d = np.load(path, allow_pickle=True)
    sat = np.asarray(d["sat_map"])
    x_m = np.asarray(d["x_mm"], dtype=float) / 1000.0
    dx = float(x_m[1] - x_m[0])
    got = pq.printer_level_map(sat, dx_m=dx, dy_m=dx, dpi=int(d["dpi"]), bpp=int(d["bpp"]))
    assert got.shape == d["level_map"].shape
    assert np.array_equal(got, d["level_map"])


def test_round_trip_matches_production_loader(tmp_path):
    """Write with my quantizer, read with the PRODUCTION loader, compare."""
    from adjoint2d.prod import rfam

    rng = np.random.default_rng(0)
    ny = nx = 40
    sat = np.clip(rng.random((ny, nx)), 0.0, 1.0)
    x = np.linspace(-0.03, 0.03, nx)
    y = np.linspace(-0.03, 0.03, ny)
    dx = float(x[1] - x[0])
    out = tmp_path / "map.npz"
    mine = pq.write_printed_map(sat, x, y, out, bpp=4, dpi=720)

    part_mask = np.ones((ny, nx), dtype=bool)
    cfg = {"fgm_feedback": {"enabled": True, "saturation_map_npz": str(out),
                            "magnitude": 1.0, "baseline_saturation": 0.5}}
    fb = rfam._FgmFeedback.from_config(cfg, x, y, part_mask)
    assert np.allclose(np.asarray(fb.sat_map, dtype=float), mine, atol=0.0, rtol=0.0)
