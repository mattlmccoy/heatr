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
import math
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


def _validate_dims(diam, zspan, L):
    """Reject parametric sizes that don't fit the chamber. diam/zspan are METERS
    (the UI divides mm by 1000); passing raw mm (e.g. 28) silently fills the whole
    grid. Fail loudly with a units-aware message instead of producing a bad part."""
    for name, v in (("diam", diam), ("zspan", zspan)):
        if v is None:
            continue
        if not (0 < v < L):
            raise SystemExit(
                f"{name}={v} is out of range for a {L * 1e3:.0f} mm chamber. "
                f"diam/zspan are in METERS (e.g. 0.028 for 28 mm), not mm.")


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
    diam = cfg.get("diam", 0.024); zspan = cfg.get("zspan", 0.024)
    _validate_dims(diam, zspan, grid.L)
    return H.make_geometry(grid, cfg.get("shape", "sphere"), diam=diam, zspan=zspan)


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


def _shrink_factor_fields(rho_final, part, p, xy_frac: float = 0.04):
    """Per-voxel anisotropic linear-shrink factors (lam_xy, lam_z) from the final
    density field, using the solver's own single-source-of-truth shrink law
    (H.shrinkage_factors). Both are zeroed outside the part."""
    rho_f = np.clip(np.asarray(rho_final, float), 1e-6, 1.0)
    S_V = np.clip(p.rho_rel / rho_f, 1e-6, 1.0)   # local volume shrink ratio (<=1)
    lam_xy, lam_z = H.shrinkage_factors(S_V, xy_frac=xy_frac)
    m = part.astype(bool)
    return np.where(m, lam_xy, 0.0), np.where(m, lam_z, 0.0)


def _warped_centers(part, lam_xy, lam_z, h, L):
    """Displaced voxel-center coordinates (m, domain-centered) after sintering shrink.
    Z compacts from the build plate: each column integrates h*lam_z, anchored at the
    part's nominal bottom face. X/Y shrink toward the part centroid by lam_xy. No new
    physics — a closed-form integration of the shrink fields the solver produces."""
    m = part.astype(bool)
    idx = np.indices(part.shape)
    xc = (idx[0] + 0.5) * h - L / 2.0
    yc = (idx[1] + 0.5) * h - L / 2.0
    cx = float(xc[m].mean()); cy = float(yc[m].mean())
    wx = cx + (xc - cx) * lam_xy
    wy = cy + (yc - cy) * lam_xy
    hz = h * lam_z                                  # compacted voxel heights (0 outside part)
    cfb = np.cumsum(hz, axis=2) - 0.5 * hz          # compacted center height above part-bottom face
    first = np.argmax(m, axis=2)                    # first part layer per column (0 if none)
    anchor = first * h - L / 2.0                    # nominal z of that bottom face
    wz = anchor[:, :, None] + cfb
    return wx, wy, wz


def _write_warped_geometry(out: Path, part, rho_final, p, grid) -> None:
    """Post-sinter warped geometry (F4): emit the deformed SURFACE shell + per-voxel
    displacement (mm) for the viewer. In the job wrapper — heatr3d.py is a synced
    copy — and adds no physics; it integrates the existing shrink fields."""
    lam_xy, lam_z = _shrink_factor_fields(rho_final, part, p)
    wx, wy, wz = _warped_centers(part, lam_xy, lam_z, grid.h, grid.L)
    surf = _surface_voxels(part)
    ix, iy, iz = surf[:, 0], surf[:, 1], surf[:, 2]
    warped = np.stack([wx[ix, iy, iz], wy[ix, iy, iz], wz[ix, iy, iz]], axis=1)   # (S,3) m
    nominal = (surf + 0.5) * grid.h - grid.L / 2.0
    disp = np.linalg.norm(warped - nominal, axis=1)
    payload = {
        "dims": list(part.shape), "h_mm": grid.h * 1e3, "n_surface": int(len(surf)),
        "warped_xyz_mm": (warped * 1e3).round(3).tolist(),
        "disp_mm": (disp * 1e3).round(4).tolist(),
        "disp_max_mm": (round(float(disp.max()) * 1e3, 4) if len(disp) else 0.0),
    }
    (out / "warped_geometry.json").write_text(json.dumps(payload))


def _fgm_z_profile(sat, part):
    """Mean dopant fraction over part voxels in each build layer (z index), NaN where
    a layer has no part voxels. The 1-D 'how it was functionally graded' curve."""
    nz = sat.shape[2]
    prof = np.full(nz, np.nan, dtype=float)
    for k in range(nz):
        col = sat[:, :, k][part[:, :, k].astype(bool)]
        if col.size:
            prof[k] = float(col.mean())
    return prof


def _melt_classes(phi, part, thresh: float = 0.5):
    """Per-voxel melt-vs-CAD class: 0 = outside the CAD part, 1 = the CAD wants it but it
    stayed cold (under-melt), 2 = correctly sintered (phi >= thresh). The 'did it build
    the intended shape' map."""
    m = part.astype(bool)
    cls = np.zeros(phi.shape, dtype=np.int8)
    cls[m & (phi >= thresh)] = 2
    cls[m & (phi < thresh)] = 1
    return cls


def _radial_density_profile(rho, part, h, nbins: int = 12):
    """Mean relative density vs radial distance from the part centroid (mm). Reveals
    core-dense / rim-porous structure. Empty radial bins are NaN."""
    m = part.astype(bool)
    idx = np.argwhere(m)
    if len(idx) == 0:
        return np.array([]), np.array([])
    r = np.linalg.norm(idx - idx.mean(axis=0), axis=1) * float(h)   # meters
    vals = np.asarray(rho, float)[m]
    rmax = r.max() if r.max() > 0 else 1e-9
    edges = np.linspace(0.0, rmax, nbins + 1)
    which = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:]) * 1e3   # mm
    means = np.full(nbins, np.nan)
    for b in range(nbins):
        sel = which == b
        if sel.any():
            means[b] = float(vals[sel].mean())
    return centers, means


def _cad_outline(ax, mask2d):
    """Overlay the CAD (part) boundary as a thin outline, aligned with an
    imshow(X.T, origin='lower'). Pass the mask slice already transposed like X.T.
    Drawn as a dark halo + white core so it reads on any background (dark viewport,
    white plot, or the field itself)."""
    if mask2d is None:
        return
    m = np.asarray(mask2d, dtype=float)
    if m.any() and not m.all():
        ax.contour(m, levels=[0.5], colors="#101010", linewidths=1.4, alpha=0.6)
        ax.contour(m, levels=[0.5], colors="#ffffff", linewidths=0.6, alpha=0.95)


def _render_summary_plots(out: Path, phi_hist, dt_s: float, fields: dict, meta: dict) -> None:
    """Per-run summary figures (matplotlib, in-job) under plots/: melt progression,
    FGM grading profile, temperature & density histograms, and an orthogonal
    center-slice montage. Each plot is written only when its data exists."""
    pdir = out / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    part = fields.get("part")
    mask = part.astype(bool) if getattr(part, "ndim", 0) == 3 else None
    real = meta.get("fields", {})

    if phi_hist is not None and len(phi_hist) > 1:
        t = np.arange(len(phi_hist)) * float(dt_s)
        fig, ax = plt.subplots(figsize=(4, 2.6), dpi=150)
        ax.plot(t, phi_hist, color="#d1495b", lw=1.6)
        ax.axhline(0.90, ls="--", lw=0.8, color="#666")
        ax.set_xlabel("exposure time (s)"); ax.set_ylabel("mean melt fraction phi")
        ax.set_title("Melt progression"); ax.set_ylim(0, 1); fig.tight_layout()
        fig.savefig(pdir / "melt_progression.png"); plt.close(fig)

    if "sat" in real and mask is not None:
        prof = _fgm_z_profile(fields["sat"], mask)
        fig, ax = plt.subplots(figsize=(4, 2.6), dpi=150)
        ax.plot(np.arange(len(prof)), prof, color="#2e8b57", lw=1.6, marker="o", ms=2)
        ax.set_xlabel("build layer (z index)"); ax.set_ylabel("mean dopant fraction")
        ax.set_title("FGM grading profile"); fig.tight_layout()
        fig.savefig(pdir / "fgm_z_profile.png"); plt.close(fig)

    if "T_phi90" in real and mask is not None:
        fig, ax = plt.subplots(figsize=(4, 2.6), dpi=150)
        ax.hist(fields["T_phi90"][mask], bins=40, color="#e07a3f")
        ax.set_xlabel("temperature at phi=0.90 (C)"); ax.set_ylabel("voxels")
        ax.set_title("Temperature distribution"); fig.tight_layout()
        fig.savefig(pdir / "temperature_hist.png"); plt.close(fig)

    if "rho_final" in real and mask is not None:
        fig, ax = plt.subplots(figsize=(4, 2.6), dpi=150)
        ax.hist(fields["rho_final"][mask], bins=40, color="#4c8dff")
        ax.set_xlabel("relative density"); ax.set_ylabel("voxels")
        ax.set_title("Density distribution"); fig.tight_layout()
        fig.savefig(pdir / "density_hist.png"); plt.close(fig)

        # radial density profile — core-dense vs rim-porous
        h_m = float(meta.get("h_mm", 1.0)) / 1000.0
        centers, means = _radial_density_profile(fields["rho_final"], mask, h_m, nbins=12)
        good = ~np.isnan(means)
        if good.sum() >= 2:
            fig, ax = plt.subplots(figsize=(4, 2.6), dpi=150)
            ax.plot(centers[good], means[good], color="#d97706", lw=1.6, marker="o", ms=3)
            ax.set_xlabel("radius from centroid (mm)"); ax.set_ylabel("mean relative density")
            ax.set_title("Radial density profile"); fig.tight_layout()
            fig.savefig(pdir / "radial_density.png"); plt.close(fig)

    prim = "sat" if "sat" in real else ("T_phi90" if "T_phi90" in real else None)
    if prim is not None and mask is not None:
        a = fields[prim]; nx, ny, nz = a.shape
        info = real[prim]; vmin, vmax = info["min"], info["max"]
        if vmax <= vmin:
            vmax = vmin + 1e-9
        planes = [
            ("XY (z mid)", np.where(mask[:, :, nz // 2], a[:, :, nz // 2].astype(float), np.nan).T, mask[:, :, nz // 2].T),
            ("XZ (y mid)", np.where(mask[:, ny // 2, :], a[:, ny // 2, :].astype(float), np.nan).T, mask[:, ny // 2, :].T),
            ("YZ (x mid)", np.where(mask[nx // 2, :, :], a[nx // 2, :, :].astype(float), np.nan).T, mask[nx // 2, :, :].T),
        ]
        fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.7), dpi=150)
        for ax, (ttl, img, mk) in zip(axs, planes):
            ax.imshow(img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            _cad_outline(ax, mk)
            ax.set_title(ttl, fontsize=9); ax.axis("off")
        fig.suptitle(f"{prim} - center slices  (white = CAD outline)", fontsize=10); fig.tight_layout()
        fig.savefig(pdir / "ortho_slices.png"); plt.close(fig)

    # Melt-vs-CAD overlay: green = correctly sintered, red = CAD wanted it but it stayed
    # cold (under-melt). The "did it build the intended shape" money-shot.
    if "phi_final" in real and mask is not None and getattr(fields.get("phi_final"), "ndim", 0) == 3:
        from matplotlib.colors import ListedColormap
        cls = _melt_classes(fields["phi_final"], mask, 0.5)
        cmap = ListedColormap([[0.86, 0.24, 0.24], [0.29, 0.74, 0.36]])   # 0->red (cold), 1->green (sintered)
        nx, ny, nz = cls.shape
        def _cp(a2):
            return np.where(a2 > 0, a2.astype(float) - 1.0, np.nan)       # 0 outside -> NaN; 1->0 red; 2->1 green
        planes = [
            ("XY (z mid)", _cp(cls[:, :, nz // 2]).T),
            ("XZ (y mid)", _cp(cls[:, ny // 2, :]).T),
            ("YZ (x mid)", _cp(cls[nx // 2, :, :]).T),
        ]
        fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.7), dpi=150)
        for ax, (ttl, img) in zip(axs, planes):
            ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(ttl, fontsize=9); ax.axis("off")
        fig.suptitle("Melt vs CAD  —  green: sintered   red: cold (under-melt)", fontsize=10)
        fig.tight_layout()
        fig.savefig(pdir / "melt_vs_cad.png"); plt.close(fig)


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
            fig = plt.figure(figsize=(2.6, 2.6), dpi=150)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.imshow(img.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            if mask3d is not None:
                _cad_outline(ax, mask3d[:, :, k].T)
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
    fig = plt.figure(figsize=(3.2, 3.2), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(img.T, origin="lower", cmap="viridis", vmin=vmin, vmax=(vmax if vmax > vmin else vmin + 1e-9),
              interpolation="nearest")
    if mask3d is not None and mask3d.shape == a.shape:
        _cad_outline(ax, mask3d[:, :, kmid].T)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def _finite_results(results: dict) -> dict:
    """Replace non-finite floats (NaN/Inf) with None so the JSON is strict-valid.
    Python's json.dumps emits bare NaN/Infinity, which browsers' JSON.parse reject —
    that silently breaks the results grid and the Results-detail view."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in results.items()}


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

    results = _finite_results(results)

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

    fields_for_view = {"part": part, "T_phi90": r.T_phi90, "phi_final": r.phi_final, "Qrf": r.Qrf,
                       "rho_final": (r.rho_final if r.rho_final is not None else np.zeros((1,), np.float32)),
                       "sat": (sat if sat is not None else np.zeros((1,), np.float32))}
    meta = _field_meta(fields_for_view, grid.h)
    _write_summary(out, results, cfg)
    _render_slices(out, fields_for_view, meta)
    _render_summary_plots(out, r.phi_hist, getattr(p, "dt_s", 0.05), fields_for_view, meta)
    if densify and r.rho_final is not None:
        _write_warped_geometry(out, part, r.rho_final, p, grid)

    print("PROGRESS 100")
    print("RESULTS " + json.dumps(results))


if __name__ == "__main__":
    main(sys.argv)
