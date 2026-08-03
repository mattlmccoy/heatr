"""Native heatr3d densification runs for RFAM Print Studio (spec section 5).

Orchestrates heatr3d WITHOUT modifying it (frozen during the S2 campaign):
voxelize the accepted mesh onto the Grid lattice, run densify=True with the
enthalpy standard, and write the heatr3d_job-standard artifact set by calling
heatr3d_job's render helpers directly (never heatr3d_job.main, which
hardcodes apparent_cp Params).

Every result carries its engine label and trust badge (spec section 3).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import trimesh

import heatr3d as H
import heatr3d_job as J

logger = logging.getLogger(__name__)

ENGINE_LABEL = "heatr3d_native"
TRUST_BADGE = ("heatr3d native | S1 passed within validity domain | "
               "S4 not passed | sim-only")
N_MAX = 96                     # full-physics ceiling (EQS-01 guard)
CHAMBER_M = 0.060
_MM_TO_M = 1e-3


def voxelize_stl(mesh_path: str, n: int, chamber_m: float = CHAMBER_M
                 ) -> np.ndarray:
    """Part mask on the heatr3d Grid lattice from an STL in millimetres.

    Mesh-driven (containment test at the Grid cell centers), centered on the
    chamber center. Refuses parts that do not fit the chamber; never scales.
    """
    if not 2 <= n <= N_MAX:
        raise ValueError(f"grid n={n} outside [2, {N_MAX}] (full-physics "
                         "ceiling)")
    mesh = trimesh.load_mesh(mesh_path)
    mesh = mesh.copy()
    mesh.apply_scale(_MM_TO_M)
    lo, hi = mesh.bounds
    size = hi - lo
    if np.any(size >= chamber_m):
        raise ValueError(
            f"part bbox {np.round(size / _MM_TO_M, 1).tolist()} mm does not "
            f"fit the {chamber_m / _MM_TO_M:.0f} mm chamber")
    mesh.apply_translation(-(lo + hi) / 2.0)      # center on chamber center
    grid = H.Grid(n=n)
    part = _fill_by_slicing(mesh, grid)
    if not part.any():
        raise ValueError("voxelization produced an empty part (mesh thinner "
                         f"than the n={n} cell size?)")
    return part


def _fill_by_slicing(mesh: "trimesh.Trimesh", grid) -> np.ndarray:
    """Slice-based solid fill at the Grid cell centers.

    mesh.contains ray-casts every cell center against every triangle
    (found live: 34 minutes at n=64 on a 125k-triangle part). This path
    sections the mesh once per z layer (C-backed) and tests the layer's
    cell centers against the section polygons (shapely vectorized,
    hole-correct via interiors). Same strict containment-at-cell-centers
    convention.
    """
    import shapely

    n = grid.n
    part = np.zeros((n, n, n), dtype=bool)
    zmin = float(mesh.bounds[0][2])
    heights = np.asarray(grid.z, float) - zmin
    in_range = (heights > 0) & (grid.z < mesh.bounds[1][2])
    sections = mesh.section_multiplane(
        plane_origin=[0.0, 0.0, zmin], plane_normal=[0.0, 0.0, 1.0],
        heights=heights[in_range])
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    xf, yf = X.ravel(), Y.ravel()
    for k, sec in zip(np.where(in_range)[0], sections):
        if sec is None:
            continue
        layer = np.zeros(n * n, dtype=bool)
        for poly in sec.polygons_full:      # exteriors WITH their holes
            layer |= shapely.contains_xy(poly, xf, yf)
        part[:, :, k] = layer.reshape(n, n)
    return part


def _finite(v: Any) -> Any:
    """Strict-JSON sanitizer: non-finite floats become None, recursively.

    NaN survives Python's json.dumps but breaks every browser JSON.parse;
    a never-reached t_phi90_s must arrive as null, never NaN."""
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def run_densify(mesh_path: str, out_dir: str | Path, n: int = 64,
                sat_path: Optional[str] = None, arm: str = "uncorrected",
                max_time_s: float = 1500.0,
                stop_mean_rho: Optional[float] = 0.98,
                power_density_w_per_m3: Optional[float] = None,
                correction_engine: Optional[str] = None,
                fast_march: bool = False,
                eqs_store_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """One densify=True heatr3d march + the standard artifact set.

    sat_path: optional npz with a (n, n, n) ``sat`` array (the corrected
    arm's dopant volume, already on this grid). correction_engine labels
    which engine produced that map; required when sat_path is given.

    eqs_store_dir: optional per-job directory for the EQS solution store
    (the Studio passes ``<grade_dir>/heatr3d/eqs_store``). Only meaningful
    together with fast_march=True, which is the only path that takes an
    EqsCache; it is IGNORED with an explicit warning otherwise rather than
    silently pretending to accelerate. Default None everywhere.

    Recorded acceleration: results["eqs_cache"] always states whether a cache
    was active and, when it was, how many solves it hit and missed. Silent
    acceleration is fine; unrecorded acceleration is not.
    """
    if n > N_MAX:
        raise ValueError(f"grid n={n} exceeds the enforced ceiling {N_MAX}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    part = voxelize_stl(mesh_path, n)
    grid = H.Grid(n=n)
    p = H.Params(phase_update="enthalpy")
    if power_density_w_per_m3 is not None:
        p.power_density_w_per_m3 = float(power_density_w_per_m3)

    sat = None
    if sat_path is not None:
        if not correction_engine:
            raise ValueError("a corrected arm must name its "
                             "correction_engine; never blend engines "
                             "silently")
        with np.load(sat_path) as d:
            sat = np.asarray(d["sat"], dtype=float)
        if sat.shape != part.shape:
            raise ValueError(f"sat volume {sat.shape} does not match the "
                             f"grid {part.shape}")

    t0 = time.time()
    env_provenance = None
    engine_march = "heatr3d"
    # Recorded acceleration: explicitly DISABLED unless a cache is actually
    # constructed below. "absent" and "off" must never be confusable.
    eqs_cache_record: Dict[str, Any] = {"enabled": False}
    if fast_march:
        # engine-lane blessed opt-in (2026-08-03): bit-identical by gate,
        # so results carry no caveat; env pins recorded (the
        # scipy-downgrade lesson)
        import llvmlite
        import numba
        from engine_speed.eqs_cache import EqsCache
        from engine_speed.march_fast import march_fast as _march
        engine_march = "march_fast"
        env_provenance = {"numba": numba.__version__,
                          "llvmlite": llvmlite.__version__}
        cache = EqsCache(store_dir=eqs_store_dir)
        r = _march(grid, part, p, sat=sat, max_time_s=float(max_time_s),
                   densify=True, stop_mean_rho=stop_mean_rho, verbose=True,
                   eqs_cache=cache)
        st = cache.stats
        eqs_cache_record = {
            "enabled": True,
            "store": (str(eqs_store_dir) if eqs_store_dir is not None else None),
            # hits = solves avoided, from either tier; misses = solves actually
            # performed. The two always sum to the number of EQS solves asked for.
            "hits": int(st["solution_hits"] + st["disk_hits"]),
            "misses": int(st["solution_misses"]),
            "memory_hits": int(st["solution_hits"]),
            "disk_hits": int(st["disk_hits"]),
            # a nonzero count here means a stored field was REFUSED as corrupt
            # and re-solved -- visible, never silent
            "disk_corrupt": int(st["disk_corrupt"]),
        }
    else:
        if eqs_store_dir is not None:
            logger.warning(
                "eqs_store_dir=%s was given without fast_march=True; the "
                "reference march does not take a cache, so it is IGNORED "
                "and results.json records eqs_cache disabled.", eqs_store_dir)
        # verbose march lines feed the Studio's live progress bars
        r = H.run(grid, part, p, sat=sat, max_time_s=float(max_time_s),
                  densify=True, stop_mean_rho=stop_mean_rho, verbose=True)
    wall_s = time.time() - t0

    gates = {
        "reached_phi90": bool(r.reached),
        "MELT_ONSET_FALLBACK": (not bool(r.reached)),
        "energy_residual_frac": float(r.energy_residual_frac),
        "energy_residual_ok": bool(abs(r.energy_residual_frac) < 1e-2),
        "clamp_bound": bool(r.clamp_bound),
        "T_max_C": float(round(r.T_max_c, 1)),
        "T_ceiling_C": 250.0,
        "T_ceiling_ok": bool(r.T_max_c <= 250.0),
    }
    results: Dict[str, Any] = {
        "engine": ENGINE_LABEL,
        "engine_march": engine_march,
        "env_provenance": env_provenance,
        "eqs_cache": eqs_cache_record,
        "trust_badge": TRUST_BADGE,
        "arm": str(arm),
        "correction_engine": correction_engine,
        "grid_n": n,
        "sigma_T": float(round(r.sigma_T, 3)),
        "t_phi90_s": float(round(r.t_phi90_s, 1)),
        "max_time_s": float(max_time_s),
        "stop_mean_rho": (float(stop_mean_rho)
                          if stop_mean_rho is not None else None),
        "sim_time_s": float(round(len(r.phi_hist)
                                  * getattr(p, "dt_s", 0.05), 2)),
        "solve_wall_s": float(round(wall_s, 1)),
        "phi_hist_len": len(r.phi_hist),
        "dt_s": float(getattr(p, "dt_s", 0.05)),
        "gates": gates,
    }
    results.update({k: v for k, v in H.sinter_metrics(r).items()})
    if r.rho_final is not None:
        sh = H.shrinkage_analysis(r, p, grid.h)
        results.update({k: v for k, v in sh.items()
                        if not k.startswith("_")})

    results = _finite(results)

    np.savez_compressed(
        out / "fields.npz", part=part,
        T_phi90=r.T_phi90.astype(np.float32),
        phi_final=r.phi_final.astype(np.float32),
        Qrf=r.Qrf.astype(np.float32),
        rho_final=(r.rho_final.astype(np.float32) if r.rho_final is not None
                   else np.zeros((1,), np.float32)),
        sat=(np.asarray(sat, np.float32) if sat is not None
             else np.zeros((1,), np.float32)),
        h=grid.h)
    np.save(out / "phi_hist.npy", np.asarray(r.phi_hist, np.float32))
    (out / "results.json").write_text(json.dumps(results, indent=2,
                                                 default=float))

    fields_for_view = {
        "part": part, "T_phi90": r.T_phi90, "phi_final": r.phi_final,
        "Qrf": r.Qrf,
        "rho_final": (r.rho_final if r.rho_final is not None
                      else np.zeros((1,), np.float32)),
        "sat": (sat if sat is not None else np.zeros((1,), np.float32)),
    }
    meta = J._field_meta(fields_for_view, grid.h)
    (out / "fieldmeta.json").write_text(json.dumps(meta, indent=2))
    J._render_slices(out, fields_for_view, meta)
    J._render_summary_plots(out, r.phi_hist, results["dt_s"],
                            fields_for_view, meta)
    if r.rho_final is not None:
        from studio3d.warped_mesh import build_densified_meshes
        J._write_warped_geometry(out, part, r.rho_final, p, grid)
        solid, powder, winfo = build_densified_meshes(
            part, r.rho_final, r.phi_final, p, grid)
        solid.export(out / "warped_mesh.stl")
        if len(powder.vertices):
            powder.export(out / "loose_powder.stl")
        (out / "warped_info.json").write_text(
            json.dumps(winfo, indent=2, default=float))
    logger.info("densify %s arm done in %.1f s (n=%d)", arm, wall_s, n)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Studio densify run")
    ap.add_argument("mesh")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--arm", default="uncorrected")
    ap.add_argument("--sat", default=None)
    ap.add_argument("--correction-engine", default=None)
    ap.add_argument("--max-time-s", type=float, default=1500.0)
    args = ap.parse_args()
    res = run_densify(args.mesh, args.out_dir, n=args.n, arm=args.arm,
                      sat_path=args.sat,
                      correction_engine=args.correction_engine,
                      max_time_s=args.max_time_s)
    print("RESULTS " + json.dumps(res, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def export_part(mesh_path: str, out_npz: str | Path, n: int = 64) -> None:
    """Write the voxelized part npz the solve service consumes."""
    part = voxelize_stl(mesh_path, n)
    np.savez_compressed(out_npz, part=part, n=n, h=CHAMBER_M / n)


def _export_main() -> int:
    ap = argparse.ArgumentParser(description="export part voxel npz")
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    export_part(a.mesh, a.out, a.n)
    print("EXPORTED")
    return 0
