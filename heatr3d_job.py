#!/usr/bin/env python3
"""heatr3d_job.py — CLI job wrapper for GUI integration (run under .venv312).

The HEATR GUI server (system Python 3.14, buggy numpy) spawns this as a SUBPROCESS
under ./.venv312/bin/python so the solver runs on a known-good numpy. It reads a
JSON config, optionally builds a 3-D FGM, runs the sim, and writes machine-readable
outputs the GUI can render. Progress is printed as `PROGRESS <pct>` lines on stdout
for the GUI to stream.

Usage:
  ./.venv312/bin/python heatr3d_job.py config.json
  ./.venv312/bin/python heatr3d_job.py config.json --preview   # geometry only, no solve

config.json (all keys optional except shape OR stl):
  {"shape":"sphere","diam":0.028,"zspan":0.03,"n":48,
   "exposure_s":1200,"densify":true,"stop_mean_rho":0.85,
   "fgm":"melt",            # none|melt|density
   "magnitude":1.0,"out_dir":"job_out"}

Outputs in out_dir:
  geometry.json  — grid dims, spacing, surface-voxel centers (for three.js render)
  results.json   — scalar metrics (sigma_T, shrinkage, layer count, ...)
  fields.npz     — part, T_phi90, rho_final, phi_final, Qrf, sat (for 3-D views)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import heatr3d as H


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Indices of part voxels exposed on at least one 6-face (the renderable shell)."""
    m = mask
    exposed = np.zeros_like(m)
    for ax in range(3):
        for s in (-1, 1):
            nb = np.roll(m, s, axis=ax)
            # voxels whose neighbor is outside (rolled-in edge counts as outside)
            sl = [slice(None)] * 3
            sl[ax] = (0 if s == 1 else -1)
            nb[tuple(sl)] = False
            exposed |= m & ~nb
    return np.argwhere(exposed)


def _write_geometry(out: Path, grid: H.Grid, part: np.ndarray, sat=None) -> None:
    surf = _surface_voxels(part)
    centers = (surf + 0.5) * grid.h - grid.L / 2.0      # physical (m), centered
    payload = {
        "dims": list(part.shape), "h_mm": grid.h * 1e3, "L_mm": grid.L * 1e3,
        "n_voxels": int(part.sum()), "n_surface": int(len(surf)),
        "surface_xyz_mm": (centers * 1e3).round(3).tolist(),
    }
    if sat is not None:
        payload["surface_sat"] = sat[surf[:, 0], surf[:, 1], surf[:, 2]].round(3).tolist()
    (out / "geometry.json").write_text(json.dumps(payload))


def build_part(grid: H.Grid, cfg: dict) -> np.ndarray:
    stl = cfg.get("stl")
    if stl:
        # STL import (P2): voxelize via trimesh if available.
        try:
            import trimesh
        except ImportError:
            raise SystemExit("STL import needs trimesh: ./.venv312/bin/pip install trimesh")
        mesh = trimesh.load(stl)
        pitch = grid.h
        vg = mesh.voxelized(pitch=pitch).fill()
        # resample onto our centered grid
        part = np.zeros((grid.n,) * 3, bool)
        pts = vg.points  # occupied cell centers (m if STL in m; assume mm -> scale)
        scale = 1e-3 if cfg.get("stl_units", "mm") == "mm" else 1.0
        idx = np.round((pts * scale + grid.L / 2.0) / grid.h - 0.5).astype(int)
        ok = ((idx >= 0) & (idx < grid.n)).all(axis=1)
        idx = idx[ok]
        part[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return part
    return H.make_geometry(grid, cfg.get("shape", "sphere"),
                           diam=cfg.get("diam", 0.024), zspan=cfg.get("zspan", 0.024))


# Fields worth coloring as slices (name -> human label). `part` is the mask, not a field.
_SLICE_FIELDS = {
    "sat": "FGM dopant fraction",
    "T_phi90": "Temperature at phi=0.90 (C)",
    "phi_final": "Melt fraction",
    "rho_final": "Relative density",
    "Qrf": "Absorbed RF power (W/m^3)",
}


def _field_meta(fields: dict, h: float) -> dict:
    """Metadata for the per-layer slice viewer. Only real (n,n,n) volumes are reported;
    shape-(1,) sentinels (absent fields) and the boolean `part` mask are skipped."""
    part = fields.get("part")
    dims = list(part.shape) if getattr(part, "ndim", 0) == 3 else None
    out_fields = {}
    for name in _SLICE_FIELDS:
        a = fields.get(name)
        if a is None or getattr(a, "ndim", 0) != 3:
            continue
        if dims is None:
            dims = list(a.shape)
        # shape disagrees with the grid: drop rather than mis-slice (should not happen for a valid run)
        if list(a.shape) != dims:
            continue
        out_fields[name] = {
            "label": _SLICE_FIELDS[name],
            "min": float(a.min()),
            "max": float(a.max()),
            "slices": int(a.shape[2]),  # z-axis
        }
    return {"dims": dims or [0, 0, 0], "h_mm": float(h) * 1000.0, "axis": "z", "fields": out_fields}


def _write_summary(out: Path, results: dict, cfg: dict) -> None:
    """Write summary.json so the run is picked up by the Results browser
    (rfam_gui_server _collect_results accepts any dir with summary.json). Carries a
    `run_type: heatr3d` tag plus the scalar metrics and the input config for the run card."""
    summary = {"run_type": "heatr3d"}
    summary.update(results)
    summary["config"] = {k: cfg.get(k) for k in ("shape", "n", "fgm", "magnitude", "densify",
                                                  "exposure_s", "stop_mean_rho", "diam", "zspan")
                         if cfg.get(k) is not None}
    # Also promote shape to top level so it shows on the run card without digging into config.
    if "shape" in cfg:
        summary["shape"] = cfg["shape"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def _render_slices(out: Path, fields: dict, meta: dict) -> None:
    """Pre-render one PNG per z-slice for each real field (viridis, per-field global min/max so
    the colormap is stable across layers), plus a preview.png, plus fieldmeta.json. Done in-job
    because the GUI server's numpy is unreliable; the server only serves these static files."""
    (out / "fieldmeta.json").write_text(json.dumps(meta, indent=2))
    sl = out / "slices"; sl.mkdir(parents=True, exist_ok=True)
    part = fields.get("part")
    mask3d = part if getattr(part, "ndim", 0) == 3 else None
    preview_written = False
    for name, info in meta["fields"].items():
        a = fields[name]
        vmin, vmax = info["min"], info["max"]
        if vmax <= vmin:
            vmax = vmin + 1e-9
        nz = a.shape[2]
        for k in range(nz):
            img = a[:, :, k].astype(float)
            if mask3d is not None:
                img = np.where(mask3d[:, :, k], img, np.nan)  # outside-part = transparent, not 0
            fig = plt.figure(figsize=(2.2, 2.2), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.imshow(img.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            fig.savefig(sl / f"{name}_z_{k:03d}.png", transparent=True)
            plt.close(fig)
        if not preview_written:
            _save_preview(out / "preview.png", a, mask3d, vmin, vmax)
            preview_written = True
    if not preview_written:  # no real fields (e.g. no-FGM, no-densify run): preview the geometry mask
        _save_preview(out / "preview.png", (mask3d.astype(float) if mask3d is not None
                      else np.zeros((1, 1, 1))), mask3d, 0.0, 1.0)

def _save_preview(path: Path, a, mask3d, vmin, vmax):
    kmid = a.shape[2] // 2
    img = a[:, :, kmid].astype(float)
    if mask3d is not None and mask3d.shape == a.shape:
        img = np.where(mask3d[:, :, kmid], img, np.nan)
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(img.T, origin="lower", cmap="viridis", vmin=vmin, vmax=(vmax if vmax > vmin else vmin + 1e-9),
              interpolation="nearest")
    fig.savefig(path, transparent=True)
    plt.close(fig)


def main(argv: list[str]) -> None:
    cfg_path = Path(argv[1])
    preview = "--preview" in argv
    cfg = json.loads(cfg_path.read_text())
    out = Path(cfg.get("out_dir", "job_out")); out.mkdir(parents=True, exist_ok=True)
    grid = H.Grid(n=int(cfg.get("n", 48)))
    p = H.Params()
    part = build_part(grid, cfg)

    if preview:
        _write_geometry(out, grid, part)
        print("PROGRESS 100"); print(f"PREVIEW_OK voxels={int(part.sum())}")
        return

    densify = bool(cfg.get("densify", False))
    stop_rho = cfg.get("stop_mean_rho")
    fgm = str(cfg.get("fgm", "none")).lower()
    mag = float(cfg.get("magnitude", 1.0))
    expo = float(cfg.get("exposure_s", 1200.0 if densify else 1500.0))

    sat = None
    if fgm in ("melt", "density"):
        print("PROGRESS 5  # FGM probe")
        if fgm == "melt":
            probe = H.run(grid, part, p, max_time_s=1500.0)
            sat = H.make_fgm(probe, magnitude=mag)
        else:  # density-targeted needs a baseline densify first
            base = H.run(grid, part, p, max_time_s=expo, densify=True, stop_mean_rho=stop_rho)
            sat = H.make_fgm(base, magnitude=mag, proxy=base.rho_final)

    print("PROGRESS 30  # main solve")
    t0 = time.time()
    r = H.run(grid, part, p, sat=sat, max_time_s=expo, densify=densify, stop_mean_rho=stop_rho)
    print("PROGRESS 90  # post-processing")

    results = {"sigma_T": round(r.sigma_T, 3), "t_phi90_s": round(r.t_phi90_s, 1),
               "reached_phi90": bool(r.reached), "T_max_C": round(r.T_max_c, 1),
               "fgm": fgm, "densify": densify, "grid_n": grid.n, "solve_s": round(time.time() - t0, 1)}
    results.update({k: v for k, v in H.sinter_metrics(r).items()})
    if densify and r.rho_final is not None:
        sh = {k: v for k, v in H.shrinkage_analysis(r, p, grid.h).items() if not k.startswith("_")}
        results.update(sh)

    np.savez_compressed(out / "fields.npz", part=part,
                        T_phi90=r.T_phi90.astype(np.float32),
                        phi_final=r.phi_final.astype(np.float32),
                        Qrf=r.Qrf.astype(np.float32),
                        rho_final=(r.rho_final.astype(np.float32) if r.rho_final is not None
                                   else np.zeros((1,), np.float32)),
                        sat=(sat.astype(np.float32) if sat is not None else np.zeros((1,), np.float32)),
                        h=grid.h)
    _write_geometry(out, grid, part, sat=sat)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("PROGRESS 100")
    print("RESULTS " + json.dumps(results))


if __name__ == "__main__":
    main(sys.argv)
