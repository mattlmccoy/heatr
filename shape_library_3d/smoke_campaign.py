"""Densification-visual smoke campaign for the 3-D shape library.

Runs heatr3d end-to-end with the STANDARD parameter set on a few Tier-1 shapes,
driven from the library's STL -> voxel-mask bridge, to prove the
STL -> stl_to_mask -> heatr3d.run -> standard-visuals path works. Renders the
existing per-run figure set (heatr3d_job) into smoke_out/<shape>/plots.

This is a SMOKE run (2-3 shapes within the n<=96 full-physics ceiling), NOT the
full 14-shape physics campaign (which gets its own plan).

    ../geo-prewarp/.venv312/bin/python -m shape_library_3d.smoke_campaign
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import heatr3d as H
import heatr3d_job as J

from shape_library_3d import load_part_stl
from shape_library_3d.constants import V_STAR_MM3
from shape_library_3d.normalize import scale_to_volume
from shape_library_3d.voxelize import stl_to_mask, voxel_volume_report

logger = logging.getLogger(__name__)
_PKG = Path(__file__).resolve().parent

# control / control / stressor spread, all within the n<=96 full-physics ceiling
SMOKE_SHAPES: List[str] = ["cube", "sphere", "cone"]
CEILING_C = 250.0
ENERGY_GATE = 1e-2


def run_one(name: str, n: int, max_time_s: float, stop_mean_rho: float,
            out_root: Path) -> Dict:
    """Voxelize the library STL, run heatr3d (densify), gate, and render."""
    grid = H.Grid(n=n)
    mesh = load_part_stl(name)
    scale_to_volume(mesh, V_STAR_MM3)                 # idempotent (STL already at V*)
    part = stl_to_mask(mesh, grid)
    vox = voxel_volume_report(part, grid, V_STAR_MM3)

    p = H.Params(phase_update="enthalpy")             # standard set (voltage-drive default)
    t0 = time.time()
    r = H.run(grid, part, p, sat=None, max_time_s=max_time_s,
              densify=True, stop_mean_rho=stop_mean_rho)
    wall_s = time.time() - t0

    energy_ok = abs(r.energy_residual_frac) < ENERGY_GATE and not r.clamp_bound
    ceiling_ok = bool(r.T_max_c <= CEILING_C)

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    fields = {
        "part": part, "T_phi90": r.T_phi90, "phi_final": r.phi_final, "Qrf": r.Qrf,
        "rho_final": (r.rho_final if r.rho_final is not None else np.zeros((1,), np.float32)),
        "sat": np.zeros((1,), np.float32),            # uniform drive (no FGM in smoke)
    }
    meta = J._field_meta(fields, grid.h)
    J._render_slices(out, fields, meta)
    J._render_summary_plots(out, r.phi_hist, p.dt_s, fields, meta)

    result = {
        "shape": name, "n": n,
        "voxel_vs_Vstar_frac": vox["voxel_vs_Vstar_frac"],
        "reached_phi90": bool(r.reached),
        "t_phi90_s": float(r.t_phi90_s),
        "sigma_T_C": float(r.sigma_T),
        "T_max_C": float(r.T_max_c),
        "mean_rho_final": (float(r.rho_final[part].mean()) if r.rho_final is not None else None),
        "energy_residual_frac": float(r.energy_residual_frac),
        "clamp_bound": bool(r.clamp_bound),
        "energy_gate_ok": bool(energy_ok),
        "ceiling_250C_ok": ceiling_ok,
        "wall_s": round(wall_s, 1),
        "plots_dir": str((out / "plots").relative_to(_PKG)),
    }
    (out / "smoke_result.json").write_text(json.dumps(result, indent=2))
    return result


def main(n: int = 48, max_time_s: float = 1200.0, stop_mean_rho: float = 0.90) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_root = _PKG / "smoke_out"
    results = []
    for name in SMOKE_SHAPES:
        logger.info("=== smoke: %s (n=%d) ===", name, n)
        results.append(run_one(name, n, max_time_s, stop_mean_rho, out_root))
    (out_root / "smoke_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nSmoke summary ({len(results)} shapes, n={n}):")
    for r in results:
        print(f"  {r['shape']:8s} reached={r['reached_phi90']!s:5s} "
              f"t90={r['t_phi90_s']:.1f}s sigmaT={r['sigma_T_C']:.2f}C "
              f"Tmax={r['T_max_C']:.1f}C rho={r['mean_rho_final']:.3f} "
              f"Eres={r['energy_residual_frac']:+.1e} "
              f"gate={'OK' if r['energy_gate_ok'] else 'FAIL'} "
              f"ceil={'OK' if r['ceiling_250C_ok'] else 'OVER'} "
              f"({r['wall_s']}s)")


if __name__ == "__main__":
    main()
