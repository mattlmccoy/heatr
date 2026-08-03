#!/usr/bin/env python3
"""Pyramid FGM demo runner: voxelize the library pyramid, apply the heuristic
3-axis graded dopant map, and run REAL heatr3d physics with densification.

Usage (under ./.venv312/bin/python, from the repo root):
    python demo_pyramid_fgm/run_demo.py graded
    python demo_pyramid_fgm/run_demo.py uniform

Each arm writes demo_pyramid_fgm/out_<arm>/fields.npz + results.json.
The simulation is REAL heatr3d (no fabricated fields); only the dopant MAP is
heuristic (see grading.py for the stated law and honesty scope).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import heatr3d as H  # noqa: E402
from demo_pyramid_fgm.grading import graded_sat, graded_sat_strong  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

STL = REPO / "shape_library_3d" / "stl" / "pyramid.stl"
N_GRID = 48
EXPOSURE_S = 1200.0


def voxelize_pyramid(grid: H.Grid) -> np.ndarray:
    """Voxelize the library pyramid onto the centered heatr3d grid.

    Dimensions are read from the library STL (mm, apex already +z), but the
    occupancy is evaluated ANALYTICALLY at voxel centers: the pyramid is an
    exact primitive (|x|,|y| <= half_width * (1 - z_hat)), and trimesh's
    voxelizer under-fills this minimal 6-face mesh (base read 13 cells instead
    of the true ~18.6 at h = 1.25 mm; verified 2026-08-03)."""
    import trimesh

    mesh = trimesh.load(str(STL))
    (x0, y0, z0), (x1, y1, z1) = mesh.bounds
    half_w = float(x1 - x0) / 2.0 * 1e-3                      # m
    height = float(z1 - z0) * 1e-3                            # m
    centers = (np.arange(grid.n) + 0.5) * grid.h - grid.L / 2.0
    X, Y, Z = np.meshgrid(centers, centers, centers, indexing="ij")
    zb = Z + height / 2.0                                     # height above base
    frac = 1.0 - zb / height                                  # taper toward apex
    inside_z = (zb >= 0.0) & (zb <= height)
    hw = half_w * np.clip(frac, 0.0, 1.0)
    part = inside_z & (np.abs(X) <= hw) & (np.abs(Y) <= hw)
    return part


_SAT_BUILDERS = {"graded": graded_sat, "strong": graded_sat_strong,
                 "uniform": lambda part: None}


def main(arm: str, exposure_s: float = EXPOSURE_S) -> None:
    if arm not in _SAT_BUILDERS:
        raise SystemExit("arm must be 'graded', 'strong', or 'uniform'")
    tag = f"_{int(exposure_s)}" if exposure_s != EXPOSURE_S else ""
    out = Path(__file__).resolve().parent / f"out_{arm}{tag}"
    out.mkdir(parents=True, exist_ok=True)

    grid = H.Grid(n=N_GRID)
    p = H.Params(phase_update="enthalpy")
    part = voxelize_pyramid(grid)
    logger.info("arm=%s exposure=%.0f voxels=%d h=%.3f mm",
                arm, exposure_s, int(part.sum()), grid.h * 1e3)

    sat = _SAT_BUILDERS[arm](part)

    t0 = time.time()
    r = H.run(grid, part, p, sat=sat, max_time_s=exposure_s,
              densify=True, verbose=False)
    wall = time.time() - t0

    results = {
        "arm": arm, "grid_n": grid.n, "exposure_s": exposure_s,
        "densify": True, "phase_update": "enthalpy",
        "sigma_T_stdT_phi90": round(float(r.sigma_T), 3),
        "t_phi90_s": round(float(r.t_phi90_s), 1),
        "reached_phi90": bool(r.reached),
        "T_max_C": round(float(r.T_max_c), 1),
        "wall_s": round(wall, 1),
        "voxels": int(part.sum()),
        "honesty": "heuristic graded map (not solved), heatr3d n=48, simulation-only",
    }
    results.update(H.sinter_metrics(r))
    (out / "results.json").write_text(json.dumps(
        {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
         for k, v in results.items()}, indent=2, default=float))

    np.savez_compressed(
        out / "fields.npz", part=part,
        sat=(sat.astype(np.float32) if sat is not None else np.zeros((1,), np.float32)),
        T_phi90=r.T_phi90.astype(np.float32),
        phi_final=r.phi_final.astype(np.float32),
        Qrf=r.Qrf.astype(np.float32),
        rho_final=(r.rho_final.astype(np.float32) if r.rho_final is not None
                   else np.zeros((1,), np.float32)),
        phi_hist=(np.asarray(r.phi_hist, np.float32) if r.phi_hist is not None
                  else np.zeros((1,), np.float32)),
        dt_s=p.dt_s, h=grid.h, L=grid.L)
    logger.info("arm=%s DONE wall=%.1f s sigma_T=%.3f", arm, wall, r.sigma_T)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "graded",
         float(sys.argv[2]) if len(sys.argv) > 2 else EXPOSURE_S)
