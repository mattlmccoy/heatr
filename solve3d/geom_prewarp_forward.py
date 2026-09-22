"""Heavy heatr3d densify-march forward for the green-geometry pre-warp loop, plus
the CLI that runs the loop on a saved densify baseline and emits the spec. Imports
heatr3d (scipy); runs in .venv312. Kept separate from geom_prewarp.py so the pure
logic tests never import the solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def crop_to_footprint(full_xy: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Crop a full-grid (n,n) column-map back to the centred (nx,ny) footprint."""
    n = full_xy.shape[0]
    ox, oy = (n - nx) // 2, (n - ny) // 2
    return np.asarray(full_xy)[ox:ox + nx, oy:oy + ny]


def run_prewarp(fields_npz: str, out_spec: str, *, bulk_factor: float | None = None,
                tol: float = 0.01, k_max: int = 5, grid_n: int = 48,
                z0: int = 1, max_time_s: float = 1200.0) -> dict:
    """Load a densify baseline, run the green-geometry backsolve on the real
    march, emit the pre-warped spec + a JSON record next to it."""
    d = np.load(fields_npz, allow_pickle=True)
    mask0 = np.asarray(d["part"], bool)          # (nx,ny,nz) nominal/target
    dop0 = np.where(mask0, np.asarray(d["sat"], float), 0.0)
    h = float(d["h"])
    p = H.Params()
    grid = H.Grid(n=grid_n, L=grid_n * h)
    nx, ny, _ = mask0.shape
    if bulk_factor is None:            # warm start: bulk factor from the baseline march
        part0, sat0 = embed_in_grid(mask0, dop0, grid_n, z0)
        sh0 = H.shrinkage_analysis(
            H.run(grid, part0, p, sat=sat0, max_time_s=max_time_s, densify=True),
            p, h)
        bulk_factor = float(sh0["layer_multiplier"])

    def forward_fn(green_mask, green_dop):
        part, sat = embed_in_grid(green_mask, green_dop, grid_n, z0)
        Hm_full, warp, _res = march_dense_heights(part, sat, p, grid, max_time_s)
        gx, gy, _ = green_mask.shape
        return crop_to_footprint(Hm_full, gx, gy), warp, {}

    res = gp.prewarp_solve(mask0, dop0, h, forward_fn,
                           bulk_factor=bulk_factor, tol=tol, k_max=k_max)
    prov = {"enabled": True, "iters": res["iters"], "converged": res["converged"],
            "tol": tol, "bulk_factor": bulk_factor,
            "warp_std_history": res["warp_history"],
            "err_history": res["err_history"],
            "source_densify": Path(fields_npz).parent.name}
    gp.emit_prewarped_spec(res["green_mask"], res["green_dop"], out_spec, prov)
    Path(str(out_spec) + ".record.json").write_text(json.dumps(prov, indent=2))
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="green-geometry pre-warp backsolve")
    ap.add_argument("fields_npz", help="densify baseline fields.npz (part/sat/h)")
    ap.add_argument("out_spec", help="output *_prewarped_green_spec.npz")
    ap.add_argument("--bulk-factor", type=float, default=None)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--grid-n", type=int, default=48)
    args = ap.parse_args(argv)
    prov = run_prewarp(args.fields_npz, args.out_spec, bulk_factor=args.bulk_factor,
                       tol=args.tol, k_max=args.k_max, grid_n=args.grid_n)
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
