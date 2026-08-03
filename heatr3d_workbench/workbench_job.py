#!/usr/bin/env python3
"""workbench_job.py - the heatr3d workbench run executor (solver venv only).

Spawned by the GUI server as a SUBPROCESS under the interpreter resolved by
_h3d_python() (numpy 2.2 buffer-elision bug: heatr3d is never imported into
the server process). Replaces heatr3d_job.py for NEW runs; the legacy wrapper
stays untouched as the executor of record for historical configs.

Differences from heatr3d_job.py (spec section 5 + Appendix B):
  - Params default to the ENTHALPY standard (phase_update="enthalpy"), with
    the legacy apparent_cp behind an explicit config key.
  - geometry sources: parametric (legacy shapes), library (shape_library_3d),
    stl (gated by heatr3d_workbench.intake before any solve).
  - grid ceiling enforced (n <= 96 full physics; 128 EQS-only mode deferred).
  - live march series: run(verbose=True) plus wrapper PHASE/PROGRESS lines.
  - Tier 2 time snapshots (densify=False only): chained run() segments via
    T0_override + t_start_s; equivalence pinned by test red-first.
  - results.json adds the standing gates (energy_residual_frac, clamp_bound,
    cfl_violated, n_substeps_used, n_eqs_solves) and the engine version stamp.
  - shape_metrics.json: IoU / out-of-bounds melt headline (vox_metrics).
  - x/y slice renders in addition to z.

Usage:
  workbench_job.py config.json                # full run
  workbench_job.py config.json --preview      # geometry only
  workbench_job.py --intake part.stl out.json # STL gate verdict only
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")        # S4 lesson: 26 s -> 18 min
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # without these pins

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np                                       # noqa: E402
import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402,F401

import heatr3d as H                                      # noqa: E402
import heatr3d_job as LJ                                 # legacy writers, reused
from heatr3d_workbench import vox_metrics                # noqa: E402

logger = logging.getLogger(__name__)

GRID_CEILING_FULL = 96
SNAPSHOT_SEGMENTS = 10


# --------------------------------------------------------------------------- #
# Pure logic (TDD'd)
# --------------------------------------------------------------------------- #
def params_from_cfg(cfg: Dict[str, Any]) -> "H.Params":
    """Config -> heatr3d Params. Enthalpy is the workbench standard."""
    kwargs: Dict[str, Any] = {"phase_update": str(cfg.get("phase_update", "enthalpy"))}
    for key in ("power_density_w_per_m3", "eqs_update_interval_s",
                "sigma_temp_coeff_per_K", "sigma_density_coeff",
                "eqs_resolve_drift_rtol"):
        if cfg.get(key) is not None:
            kwargs[key] = float(cfg[key])
    return H.Params(**kwargs)


def check_grid_n(n: int) -> int:
    """Enforce the full-physics ceiling (spec: n<=96; 128 EQS-only deferred)."""
    n = int(n)
    if n > GRID_CEILING_FULL:
        raise SystemExit(
            f"grid n={n} exceeds the full-physics ceiling n<={GRID_CEILING_FULL} "
            f"(EQS-01). The n<=128 EQS-only diagnostic mode is not built yet.")
    return n


def segment_plan(total_s: float, k: int) -> List[Tuple[float, float]]:
    """K equal march segments covering [0, total_s]."""
    k = max(1, int(k))
    edges = [total_s * i / k for i in range(k + 1)]
    return [(edges[i], edges[i + 1]) for i in range(k)]


def run_segment(grid, part, p, t0: float, t1: float, T_prev):
    """One chained segment: duration t1-t0 starting at absolute t0."""
    return H.run(grid, part, p, max_time_s=(t1 - t0), t_start_s=t0,
                 T0_override=T_prev, verbose=False)


def engine_version() -> str:
    return "heatr3d-" + H.SYNCED_FROM_SHA256[:8]


def build_part(grid, cfg: Dict[str, Any]) -> np.ndarray:
    """Build the boolean part mask from the configured geometry source."""
    src = str(cfg.get("source", "parametric"))
    if src == "library":
        name = str(cfg.get("library_shape", ""))
        import shape_library_3d as SL
        from shape_library_3d.voxelize import stl_to_mask
        loadable = {n for n, _ in SL.iter_parts()}
        if name not in loadable:
            raise SystemExit(
                f"library shape '{name}' is not a loadable part "
                f"(Tier-3 rejection fixture or unknown). Loadable: {sorted(loadable)}")
        mesh = SL.load_part_stl(name)
        return stl_to_mask(mesh, grid)
    if src == "stl":
        import trimesh
        from shape_library_3d.voxelize import stl_to_mask
        from heatr3d_workbench import intake
        stl = cfg.get("stl")
        if not stl:
            raise SystemExit("source=stl but no 'stl' path in config")
        verdict = intake.gate_stl(stl)
        if not verdict["accepted"]:
            raise SystemExit(f"STL REFUSED: {verdict['error_type']}: {verdict['error']}")
        mesh = trimesh.load(str(stl), force="mesh")
        return stl_to_mask(mesh, grid)
    # parametric (legacy shapes, legacy validation)
    return LJ.build_part(grid, cfg)


# --------------------------------------------------------------------------- #
# Rendering (all three axes, locked colormaps, phi-front contours, DPI 180)
# --------------------------------------------------------------------------- #
_AXES = {"x": 0, "y": 1, "z": 2}

# Visualization standard (memory: visualization-standard): thermal = inferno,
# dopant/sat = viridis, density = powder-gray -> melt-purple -> dense-gold.
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "rfam_density", ["#8f8f8f", "#6b3fa0", "#d4a017"])
_FIELD_CMAPS = {"T_phi90": "inferno", "rho_final": _DENSITY_CMAP,
                "sat": "viridis", "Qrf": "viridis", "phi_final": "viridis"}


def _render_slices_all(out: Path, fields: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Pre-render x/y/z slices for each real field: per-field locked colormap,
    per-field GLOBAL color range (scrub-stable), outside-part transparent,
    CAD outline, and the phi = 0.9 / 0.5 front contours from the ORIGINAL
    (unsmoothed) phi overlaid on every field. phi itself renders as a
    gradient with contours, never a saturated indicator (spec 4.2)."""
    sl = out / "slices"
    sl.mkdir(parents=True, exist_ok=True)
    part = fields.get("part")
    mask3d = part if getattr(part, "ndim", 0) == 3 else None
    phi3d = fields.get("phi_final")
    phi3d = phi3d if getattr(phi3d, "ndim", 0) == 3 else None
    for name, info in meta["fields"].items():
        a = fields[name]
        vmin, vmax = info["min"], info["max"]
        if vmax <= vmin:
            vmax = vmin + 1e-9
        cmap = _FIELD_CMAPS.get(name, "viridis")
        for ax_name, ax_idx in _AXES.items():
            nk = a.shape[ax_idx]
            for k in range(nk):
                img = np.take(a, k, axis=ax_idx).astype(float)
                mk = (np.take(mask3d, k, axis=ax_idx) if mask3d is not None else None)
                if mk is not None:
                    img = np.where(mk, img, np.nan)
                fig = plt.figure(figsize=(2.6, 2.6), dpi=180)
                axp = fig.add_axes([0, 0, 1, 1]); axp.axis("off")
                axp.imshow(img.T, origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
                if mk is not None:
                    LJ._cad_outline(axp, mk.T)
                if phi3d is not None:
                    pm = np.take(phi3d, k, axis=ax_idx).astype(float)
                    for lev, ls, lw in ((0.9, "-", 1.1), (0.5, "--", 0.7)):
                        if (pm >= lev).any() and not (pm >= lev).all():
                            axp.contour(pm.T, levels=[lev], colors="#ffffff",
                                        linewidths=lw, linestyles=ls)
                fig.savefig(sl / f"{name}_{ax_name}_{k:03d}.png", transparent=True)
                plt.close(fig)
    # preview.png: first real field's mid-z slice via the legacy helper
    prim = next(iter(meta["fields"]), None)
    if prim is not None:
        info = meta["fields"][prim]
        LJ._save_preview(out / "preview.png", fields[prim], mask3d,
                         info["min"], info["max"])
    elif mask3d is not None:
        LJ._save_preview(out / "preview.png", mask3d.astype(float), mask3d, 0.0, 1.0)


def _write_snapshots(out: Path, snaps: List[Tuple[float, np.ndarray, np.ndarray]],
                     part: np.ndarray, h: float) -> None:
    """Persist Tier 2 time snapshots + a mid-z T slice render per snapshot."""
    sdir = out / "snapshots"
    sdir.mkdir(parents=True, exist_ok=True)
    index = []
    t_all = np.array([s[0] for s in snaps])
    T_stack = [s[1] for s in snaps]
    vmin = float(min(T[part].min() for T in T_stack)) if part.any() else 0.0
    vmax = float(max(T[part].max() for T in T_stack)) if part.any() else 1.0
    kmid = part.shape[2] // 2
    for i, (t_s, T, phi) in enumerate(snaps):
        np.savez_compressed(sdir / f"snap_{i:03d}.npz", t_s=t_s,
                            T=T.astype(np.float32), phi=phi.astype(np.float32))
        img = np.where(part[:, :, kmid], T[:, :, kmid].astype(float), np.nan)
        fig = plt.figure(figsize=(2.6, 2.6), dpi=150)
        axp = fig.add_axes([0, 0, 1, 1]); axp.axis("off")
        axp.imshow(img.T, origin="lower", cmap="inferno", vmin=vmin,
                   vmax=(vmax if vmax > vmin else vmin + 1e-9),
                   interpolation="nearest")
        LJ._cad_outline(axp, part[:, :, kmid].T)
        # phi=0.9 front from the ORIGINAL phi, never smoothed
        pm = phi[:, :, kmid]
        if (pm >= 0.9).any() and not (pm >= 0.9).all():
            axp.contour(pm.T, levels=[0.9], colors="#ffffff", linewidths=1.0)
        fig.savefig(sdir / f"T_zmid_{i:03d}.png", transparent=True)
        plt.close(fig)
        index.append({"i": i, "t_s": float(t_s),
                      "T_png": f"snapshots/T_zmid_{i:03d}.png"})
    (sdir / "index.json").write_text(json.dumps(
        {"count": len(index), "t_min": float(t_all.min()), "t_max": float(t_all.max()),
         "vmin_c": vmin, "vmax_c": vmax, "snaps": index}, indent=1))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _phase(name: str, prog: Optional[float] = None) -> None:
    print(f"PHASE {name}", flush=True)
    if prog is not None:
        print(f"PROGRESS {prog:.0f}", flush=True)


def _do_intake(stl_path: str, out_json: str) -> None:
    from heatr3d_workbench import intake
    verdict = intake.gate_stl(stl_path)
    Path(out_json).write_text(json.dumps(verdict, indent=1))
    print(("INTAKE_OK" if verdict["accepted"] else "INTAKE_REFUSED"), flush=True)


def main(argv: List[str]) -> None:
    if "--intake" in argv:
        i = argv.index("--intake")
        _do_intake(argv[i + 1], argv[i + 2])
        return

    cfg_path = Path(argv[1])
    preview = "--preview" in argv
    cfg = json.loads(cfg_path.read_text())
    out = Path(cfg.get("out_dir", "job_out"))
    out.mkdir(parents=True, exist_ok=True)

    _phase("build", 2)
    grid = H.Grid(n=check_grid_n(cfg.get("n", 64)))
    p = params_from_cfg(cfg)
    part = build_part(grid, cfg)

    if preview:
        LJ._write_geometry(out, grid, part)
        from shape_library_3d.voxelize import voxel_volume_report
        try:
            h_mm = grid.h * 1e3
            rep = {"voxel_volume_mm3": float(part.sum()) * h_mm ** 3}
        except Exception:  # report is best-effort  # noqa: BLE001
            rep = {}
        print("PROGRESS 100", flush=True)
        print(f"PREVIEW_OK voxels={int(part.sum())} {json.dumps(rep)}", flush=True)
        return

    densify = bool(cfg.get("densify", False))
    stop_rho = cfg.get("stop_mean_rho")
    fgm = str(cfg.get("fgm", "none")).lower()
    mag = float(cfg.get("magnitude", 1.0))
    expo = float(cfg.get("exposure_s", 1200.0 if densify else 1500.0))
    want_snaps = bool(cfg.get("snapshots", False)) and not densify

    sat = None
    if fgm in ("melt", "density"):
        _phase("fgm_probe", 5)
        if fgm == "melt":
            probe = H.run(grid, part, p, max_time_s=1500.0, verbose=True)
            sat = H.make_fgm(probe, magnitude=mag)
        else:
            base = H.run(grid, part, p, max_time_s=expo, densify=True,
                         stop_mean_rho=stop_rho, verbose=True)
            sat = H.make_fgm(base, magnitude=mag, proxy=base.rho_final)

    _phase("march", 10)
    t0 = time.time()
    snaps: List[Tuple[float, np.ndarray, np.ndarray]] = []
    if want_snaps:
        T_prev = None
        r = None
        for (s0, s1) in segment_plan(expo, SNAPSHOT_SEGMENTS):
            r = H.run(grid, part, p, sat=sat, max_time_s=(s1 - s0), t_start_s=s0,
                      T0_override=T_prev, verbose=True)
            T_prev = r.T_final
            t_now = s0 + (r.t_phi90_s if r.reached else (s1 - s0))
            snaps.append((min(t_now, s1), r.T_final.copy(), r.phi_final.copy()))
            phi_bar = float(r.phi_final[part].mean()) if part.any() else float("nan")
            rho_bar = float(p.rho_rel)
            print(f"  t={min(t_now, s1):6.1f}s  Tmax={r.T_max_c:6.1f}  "
                  f"phi={phi_bar:.3f}  rho={rho_bar:.3f}", flush=True)
            if r.reached:
                break
        assert r is not None
    else:
        r = H.run(grid, part, p, sat=sat, max_time_s=expo, densify=densify,
                  stop_mean_rho=stop_rho, verbose=True)
    _phase("post", 92)

    results = {
        "sigma_T": round(r.sigma_T, 3), "t_phi90_s": round(r.t_phi90_s, 1),
        "reached_phi90": bool(r.reached),
        "MELT_ONSET_FALLBACK": (not bool(r.reached)),
        "T_max_C": round(r.T_max_c, 1),
        "fgm": fgm, "densify": densify, "grid_n": grid.n,
        "solve_s": round(time.time() - t0, 1),
        # standing gates + provenance (spec section 6)
        "energy_residual_frac": float(r.energy_residual_frac),
        "clamp_bound": bool(r.clamp_bound),
        "cfl_violated": bool(r.cfl_violated),
        "n_substeps_used": int(r.n_substeps_used),
        "n_eqs_solves": int(r.n_eqs_solves),
        "phase_update": p.phase_update,
        "heatr3d_engine_version": engine_version(),
        "workbench_run": True,
    }
    results.update({k: v for k, v in H.sinter_metrics(r).items()})
    if densify and r.rho_final is not None:
        sh = {k: v for k, v in H.shrinkage_analysis(r, p, grid.h).items()
              if not k.startswith("_")}
        results.update(sh)

    # shape-fidelity headline (Q3: run-grid nominal mask)
    sm = vox_metrics.shape_metrics(r.phi_final, part, h_mm=grid.h * 1e3)
    (out / "shape_metrics.json").write_text(json.dumps(sm, indent=1))
    results.update({f"shape_{k}": v for k, v in sm.items()})

    results = LJ._finite_results(results)

    np.savez_compressed(out / "fields.npz", part=part,
                        T_phi90=r.T_phi90.astype(np.float32),
                        phi_final=r.phi_final.astype(np.float32),
                        Qrf=r.Qrf.astype(np.float32),
                        rho_final=(r.rho_final.astype(np.float32)
                                   if r.rho_final is not None
                                   else np.zeros((1,), np.float32)),
                        sat=(sat.astype(np.float32) if sat is not None
                             else np.zeros((1,), np.float32)),
                        h=grid.h)
    LJ._write_geometry(out, grid, part, sat=sat)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    fields_for_view = {"part": part, "T_phi90": r.T_phi90,
                       "phi_final": r.phi_final, "Qrf": r.Qrf,
                       "rho_final": (r.rho_final if r.rho_final is not None
                                     else np.zeros((1,), np.float32)),
                       "sat": (sat if sat is not None
                               else np.zeros((1,), np.float32))}
    meta = LJ._field_meta(fields_for_view, grid.h)
    meta["axes"] = {"x": int(part.shape[0]), "y": int(part.shape[1]),
                    "z": int(part.shape[2])}
    LJ._write_summary(out, results, cfg)
    _render_slices_all(out, fields_for_view, meta)   # x/y/z, locked colormaps
    (out / "fieldmeta.json").write_text(json.dumps(meta, indent=2))  # with axes
    LJ._render_summary_plots(out, r.phi_hist, getattr(p, "dt_s", 0.05),
                             fields_for_view, meta)
    if densify and r.rho_final is not None:
        LJ._write_warped_geometry(out, part, r.rho_final, p, grid)
    if snaps:
        _write_snapshots(out, snaps, part, grid.h)

    print("PROGRESS 100", flush=True)
    print("RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main(sys.argv)
