"""Heavy heatr3d densify-march forward for the green-geometry pre-warp loop, plus
the CLI that runs the loop on a saved densify baseline and emits the spec. Imports
heatr3d (scipy); runs in .venv312. Kept separate from geom_prewarp.py so the pure
logic tests never import the solver."""
from __future__ import annotations

import numpy as np

import heatr3d as H
from solve3d import geom_prewarp as gp


def embed_in_grid(green_mask: np.ndarray, green_dop: np.ndarray, n: int,
                  z0: int = 0):
    """Place a (nx,ny,nz) green volume into a cubic (n,n,n) grid: footprint
    centred in x,y, base at z=z0. Returns (part, sat), sat zeroed outside part."""
    gx, gy, gz = green_mask.shape
    if gx > n or gy > n or z0 + gz > n:
        raise ValueError(f"green volume {green_mask.shape} + z0={z0} does not fit "
                         f"in a {n}^3 grid; raise n or scale the part down")
    part = np.zeros((n, n, n), bool)
    sat = np.zeros((n, n, n), float)
    ox, oy = (n - gx) // 2, (n - gy) // 2
    part[ox:ox + gx, oy:oy + gy, z0:z0 + gz] = green_mask
    sat[ox:ox + gx, oy:oy + gy, z0:z0 + gz] = np.where(green_mask, green_dop, 0.0)
    return part, sat


def march_dense_heights(part: np.ndarray, sat: np.ndarray, p: "H.Params",
                        grid: "H.Grid", max_time_s: float = 1200.0):
    """One densify march + shrinkage read. Returns (H_measured(nx,ny) in metres
    over the FULL grid footprint, warp_std_pct, Result)."""
    res = H.run(grid, part, p, sat=sat, max_time_s=max_time_s, densify=True)
    sh = H.shrinkage_analysis(res, p, grid.h)
    return np.asarray(sh["_H_final"], float), float(sh["warp_std_pct"]), res
