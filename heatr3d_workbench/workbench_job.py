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
SNAPSHOT_SEGMENTS = 10        # legacy equal-split plan (kept for the gate test)
SNAPSHOT_MAX = 24             # dyadic thinning cap for the adaptive driver
SNAPSHOT_MIN_SEGS = 40        # target time resolution: expo / this per segment


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


def run_snapshot_march(grid, part, p, sat, expo: float, densify: bool = False):
    """Tier 2 time-snapshot march (densify=False only): chained run() segments.

    The FIRST segment performs the one real EQS solve; every later segment
    passes its Qrf back as qrf_override (used verbatim - identical to the
    frozen-Q legacy march, so the chain is exact; pinned by test). Segment
    duration is expo/SNAPSHOT_MIN_SEGS; the march breaks at the phi_target
    crossing; snapshots are dyadically thinned to at most SNAPSHOT_MAX so an
    early melt still yields a well-covered sequence.

    Returns (last_result, snaps) with snaps = [(t_abs_s, T, phi), ...].
    """
    if densify:
        raise SystemExit("snapshot march is melt-onset only (spec 4.2 Tier 3)")
    seg = max(float(expo) / SNAPSHOT_MIN_SEGS, 2.0 * p.dt_s)
    T_prev = None
    qrf = None
    snaps: List[Tuple[float, np.ndarray, np.ndarray]] = []
    stride = 1
    i = 0
    r = None
    t0 = 0.0
    n_eqs_total = 0
    while t0 < expo - 1e-9:
        t1 = min(t0 + seg, expo)
        r = H.run(grid, part, p, sat=sat, max_time_s=(t1 - t0), t_start_s=t0,
                  T0_override=T_prev, qrf_override=qrf, verbose=False)
        n_eqs_total += int(r.n_eqs_solves)
        if qrf is None:
            qrf = r.Qrf
        T_prev = r.T_final
        # run() reports t90 LOCAL to the call (heatr3d.py:1224); make absolute.
        if r.reached:
            r.t_phi90_s = t0 + float(r.t_phi90_s)
        t_now = r.t_phi90_s if r.reached else t1
        if i % stride == 0:
            snaps.append((min(t_now, t1), r.T_final.copy(), r.phi_final.copy()))
            if len(snaps) > SNAPSHOT_MAX:
                snaps = snaps[::2]         # dyadic thinning: halve, keep coverage
                stride *= 2
        i += 1
        phi_bar = float(r.phi_final[part].mean()) if part.any() else float("nan")
        print(f"  t={min(t_now, t1):6.1f}s  Tmax={r.T_max_c:6.1f}  "
              f"phi={phi_bar:.3f}  rho={float(p.rho_rel):.3f}", flush=True)
        if r.reached:
            break
        t0 = t1
    if r is not None:
        r.n_eqs_solves = n_eqs_total       # total for the chain, not last segment
    return r, snaps


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
# Rendering (all three axes, locked colormap FAMILIES, phi-front contours,
# DPI 180, display smoothing per the visualization standard)
# --------------------------------------------------------------------------- #
_AXES = {"x": 0, "y": 1, "z": 2}

from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

# Visualization standard (memory: visualization-standard): thermal = inferno,
# dopant/sat = viridis, density = powder-gray -> melt-purple -> dense-gold.
# DEVIATION FROM THE STANDARD, recorded (Matt, 2026-08-03): the colormap
# FAMILIES are kept but their BOTTOMS are clipped (inferno from 0.25, viridis
# from 0.20) because the untruncated floors are near-black and vanish on the
# dark UI ("so hard to see with the pixels being black"). First pass at
# 0.15/0.12 was still near-black at the cold rim voxels (checked on the cone
# renders); raised after viewing. Outside-part (bed) voxels render as a
# distinct muted slate via set_bad, never the colormap floor, so bed vs
# cold-part is visually distinct.
_BED_SLATE = (0.145, 0.165, 0.205, 1.0)
DISPLAY_SMOOTH_SIGMA = 1.0   # standard: field sigma ~1.0-1.2, DISPLAY ONLY


def truncated_cmap(name: str, lo: float, n: int = 256) -> ListedColormap:
    """The named colormap restricted to [lo, 1] (floor never near-black)."""
    base = plt.get_cmap(name)
    return ListedColormap(base(np.linspace(lo, 1.0, n)), name=f"{name}_{lo:g}")


def _density_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "rfam_density", ["#8f8f8f", "#6b3fa0", "#d4a017"])


def field_cmap(field: str):
    """Per-field display colormap with the clipped floor + slate bad color."""
    if field == "T_phi90":
        cm = truncated_cmap("inferno", 0.25)
    elif field == "rho_final":
        cm = _density_cmap()
    else:                       # sat, Qrf, phi_final
        cm = truncated_cmap("viridis", 0.20)
    cm.set_bad(_BED_SLATE)
    return cm


def display_smooth(img: np.ndarray, mask: np.ndarray,
                   sigma: float = DISPLAY_SMOOTH_SIGMA) -> np.ndarray:
    """Mask-aware gaussian smoothing of a DISPLAY array (never the data).

    Normalized convolution: smooth field*mask and mask separately, divide,
    keep outside-mask NaN so the slate bad color renders there."""
    m = mask.astype(float)
    filled = np.nan_to_num(np.where(mask, img, 0.0))
    num = gaussian_filter(filled, sigma)
    den = gaussian_filter(m, sigma)
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-12), np.nan)
    return np.where(mask, out, np.nan)


_ISO_SPECS = [("phi_final", 0.9, "melt front phi = 0.9"),
              ("phi_final", 0.5, "melt front phi = 0.5"),
              ("rho_final", 0.9, "density rho = 0.9")]


def isosurfaces_for(fields: Dict[str, Any], h_mm: float,
                    specs=None) -> List[Dict[str, Any]]:
    """Marching-cubes isosurfaces (trilinear in-cell interpolation) of the
    computed fields, in mm, for smooth-shaded viewer rendering. Levels outside
    a field's range are SKIPPED, never faked. Display interpolation only -
    the physics stays at the run grid."""
    from skimage import measure
    out: List[Dict[str, Any]] = []
    for field, level, label in (specs if specs is not None else _ISO_SPECS):
        a = fields.get(field)
        if a is None or getattr(a, "ndim", 0) != 3:
            continue
        a = np.asarray(a, dtype=float)
        if not (float(a.min()) < level < float(a.max())):
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(
                a, level=level, spacing=(h_mm, h_mm, h_mm))
        except (ValueError, RuntimeError) as e:
            logger.warning("marching cubes %s@%s failed: %s", field, level, e)
            continue
        out.append({
            "field": field, "level": float(level), "label": label,
            "vertices_mm": np.round(verts, 3).tolist(),
            "faces": faces.astype(int).tolist(),
        })
    return out


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
        cmap = field_cmap(name)
        for ax_name, ax_idx in _AXES.items():
            nk = a.shape[ax_idx]
            for k in range(nk):
                img = np.take(a, k, axis=ax_idx).astype(float)
                mk = (np.take(mask3d, k, axis=ax_idx) if mask3d is not None else None)
                if mk is not None:
                    # display smoothing (standard: sigma ~1, DISPLAY only)
                    img = display_smooth(img, mk)
                fig = plt.figure(figsize=(2.6, 2.6), dpi=180)
                fig.patch.set_alpha(0.0)
                axp = fig.add_axes([0, 0, 1, 1]); axp.axis("off")
                axp.imshow(img.T, origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="bilinear")
                if mk is not None:
                    # outline from the display-smoothed mask (standard)
                    LJ._cad_outline(axp, gaussian_filter(mk.astype(float),
                                                         DISPLAY_SMOOTH_SIGMA).T)
                if phi3d is not None:
                    # melt-front contours from the ORIGINAL phi, never smoothed
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
    snap_cmap = field_cmap("T_phi90")
    for i, (t_s, T, phi) in enumerate(snaps):
        np.savez_compressed(sdir / f"snap_{i:03d}.npz", t_s=t_s,
                            T=T.astype(np.float32), phi=phi.astype(np.float32))
        img = display_smooth(T[:, :, kmid].astype(float), part[:, :, kmid])
        fig = plt.figure(figsize=(2.6, 2.6), dpi=180)
        fig.patch.set_alpha(0.0)
        axp = fig.add_axes([0, 0, 1, 1]); axp.axis("off")
        axp.imshow(img.T, origin="lower", cmap=snap_cmap, vmin=vmin,
                   vmax=(vmax if vmax > vmin else vmin + 1e-9),
                   interpolation="bilinear")
        LJ._cad_outline(axp, gaussian_filter(
            part[:, :, kmid].astype(float), DISPLAY_SMOOTH_SIGMA).T)
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
def _write_isosurfaces(out: Path, fields: Dict[str, Any], h_mm: float) -> None:
    surfs = isosurfaces_for(fields, h_mm)
    dims = list(fields["part"].shape)
    payload = {
        "h_mm": h_mm, "dims": dims,
        # vertex mm -> domain-centered mm: add offset (cell centers at
        # (i+0.5)*h - L/2 with L = n*h; marching cubes vertices are i*h)
        "offset_mm": [h_mm / 2.0 - d * h_mm / 2.0 for d in dims],
        "note": ("display interpolation (marching cubes, in-cell trilinear) "
                 "of the computed field; physics resolution is the run grid"),
        "surfaces": surfs,
    }
    (out / "isosurfaces.json").write_text(json.dumps(payload))


def render_views(out: Path, fields_for_view: Dict[str, Any], meta: Dict[str, Any],
                 phi_hist, dt_s: float, h_m: float) -> None:
    """All display artifacts for a run dir (slices, plots, isosurfaces)."""
    _render_slices_all(out, fields_for_view, meta)
    (out / "fieldmeta.json").write_text(json.dumps(meta, indent=2))
    LJ._render_summary_plots(out, phi_hist, dt_s, fields_for_view, meta)
    _write_isosurfaces(out, fields_for_view, h_m * 1000.0)


def _do_rerender(cfg_path: Path) -> None:
    """Re-render display artifacts from an existing run dir (no solving).
    Renders into the CONFIG FILE'S OWN directory (never the recorded out_dir,
    which may point at the original run when a dir was copied)."""
    out = cfg_path.parent
    z = np.load(out / "fields.npz")
    part = z["part"].astype(bool)
    fields_for_view = {"part": part}
    for k in ("T_phi90", "phi_final", "Qrf", "rho_final", "sat"):
        fields_for_view[k] = z[k]
    h_m = float(z["h"])
    meta = LJ._field_meta(fields_for_view, h_m)
    meta["axes"] = {"x": int(part.shape[0]), "y": int(part.shape[1]),
                    "z": int(part.shape[2])}
    render_views(out, fields_for_view, meta, None, 0.05, h_m)
    # refresh snapshot frames from their stored volumes, when present
    sdir = out / "snapshots"
    if (sdir / "index.json").exists():
        snaps = []
        for p in sorted(sdir.glob("snap_*.npz")):
            zz = np.load(p)
            snaps.append((float(zz["t_s"]), zz["T"].astype(float),
                          zz["phi"].astype(float)))
        if snaps:
            _write_snapshots(out, snaps, part, h_m)
    print("RERENDER_OK", flush=True)


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
    if "--rerender" in argv:
        _do_rerender(cfg_path)
        return
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
        r, snaps = run_snapshot_march(grid, part, p, sat, expo, densify=False)
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
    # Promote a display shape so the run pickers group library/STL runs
    # correctly (the legacy writer only reads cfg["shape"]).
    if not cfg.get("shape"):
        cfg["shape"] = (cfg.get("library_shape")
                        or (Path(cfg["stl"]).stem if cfg.get("stl") else None)
                        or "unknown")
    LJ._write_summary(out, results, cfg)
    render_views(out, fields_for_view, meta, r.phi_hist,
                 getattr(p, "dt_s", 0.05), grid.h)
    if densify and r.rho_final is not None:
        LJ._write_warped_geometry(out, part, r.rho_final, p, grid)
    if snaps:
        _write_snapshots(out, snaps, part, grid.h)

    print("PROGRESS 100", flush=True)
    print("RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main(sys.argv)
