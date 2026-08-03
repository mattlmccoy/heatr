#!/usr/bin/env python3
"""How exactly does each ladder grid represent the cross and the keyhole?

The rotating forward grid ladder found a NON-MONOTONE intersection-over-union
sequence on the cross and a monotone one on the keyhole. The cross is a
RECTILINEAR polygon whose boundaries sit at fixed physical distances, so the
number of whole cells its limbs and its arms occupy jumps discontinuously with
the grid number. This measures that jump directly, from the SAME production
rasterizer the forward uses, so the ladder's scatter can be attributed rather
than guessed at.

MEASURED per grid, on the part mask the production domain builder produces:
  * the cell size;
  * the limb half-length and the arm half-width in cells, exactly and as the
    raster realizes them, with the relative error of each;
  * the raster area against the sub-cell area fill.

Run:
  ./.venv312/bin/python scripts/analysis/rot_ladder_raster_geometry.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from adjoint2d import chi_area, robust_rot as rr          # noqa: E402
from adjoint2d.library_solve import shape_config          # noqa: E402
from adjoint2d.pins import build_case, load_cfg           # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_rot_ladder"
GRIDS = (96, 120, 160, 180, 200, 240, 360)

# The cross polygon, from the production shape builder at the pinned
# 0.022 m width: unique coordinates +-11.0 mm and +-11/3 mm. The limb half
# length is 11.0 mm and the arm HALF width is 11/3 mm.
CROSS_LIMB_HALF_M = 0.011
CROSS_ARM_HALF_M = 0.011 / 3.0


def main() -> None:
    cfg = load_cfg(shape_config("cross"))
    rows = []
    for n in GRIDS:
        c = rr.cfg_at_grid(cfg, n)
        case = build_case(c)
        pm = np.asarray(case.part_mask, dtype=bool)
        chi, info = chi_area.chi_from_cfg(c, case.x, case.y)
        dx = float(case.dx)
        ny, nx = pm.shape
        # the raster's own limb half length and arm half width, read off the
        # mask along its central row and its central column
        row = pm[ny // 2, :]
        col = pm[:, nx // 2]
        limb_cells = int(row.sum()) / 2.0          # half length, in cells
        # the arm half width is the run of in-part cells on a column far from
        # the centre, taken at the row where only the vertical arm exists
        j = 2                                       # near the top edge of the mask
        rows_with_part = np.flatnonzero(col)
        j = int(rows_with_part[0]) + 1              # one row inside the limb tip
        arm_cells = int(pm[j, :].sum()) / 2.0
        exact_limb = CROSS_LIMB_HALF_M / dx
        exact_arm = CROSS_ARM_HALF_M / dx
        rows.append({
            "n_grid": n, "dx_mm": dx * 1e3,
            "n_part_cells": int(pm.sum()),
            "limb_half_cells_exact": exact_limb,
            "limb_half_cells_raster": limb_cells,
            "limb_rel_err_pct": 100.0 * (limb_cells - exact_limb) / exact_limb,
            "limb_boundary_on_a_cell_edge":
                bool(abs(exact_limb - round(exact_limb)) < 1e-9),
            "arm_half_cells_exact": exact_arm,
            "arm_half_cells_raster": arm_cells,
            "arm_rel_err_pct": 100.0 * (arm_cells - exact_arm) / exact_arm,
            "arm_boundary_on_a_cell_edge":
                bool(abs(exact_arm - round(exact_arm)) < 1e-9),
            "raster_minus_area_pct": 100.0 * chi_area.raster_vs_area_delta(
                pm, chi, case.dx, case.dy)["area_rel_delta"],
            "chi_area_m2": float(info["area_m2"]),
        })
        r = rows[-1]
        print(f"n={n:4d} dx {r['dx_mm']:.4f} mm  limb {r['limb_half_cells_exact']:7.3f} "
              f"-> {r['limb_half_cells_raster']:5.1f} ({r['limb_rel_err_pct']:+6.2f} %)"
              f"{'  EDGE' if r['limb_boundary_on_a_cell_edge'] else '      '}  "
              f"arm {r['arm_half_cells_exact']:7.3f} -> "
              f"{r['arm_half_cells_raster']:5.1f} ({r['arm_rel_err_pct']:+6.2f} %)"
              f"{'  EDGE' if r['arm_boundary_on_a_cell_edge'] else '      '}  "
              f"raster-area {r['raster_minus_area_pct']:+6.3f} %", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cross_raster_geometry.json").write_text(json.dumps(
        {"shape": "cross",
         "limb_half_m": CROSS_LIMB_HALF_M, "arm_half_m": CROSS_ARM_HALF_M,
         "note": "the limb boundary sits on a cell edge when the grid number is "
                 "a multiple of 60; the ARM boundary needs a multiple of 180, "
                 "so only grids that are multiples of 180 represent BOTH cross "
                 "boundaries exactly",
         "rows": rows}, indent=2, default=float))


if __name__ == "__main__":
    main()
