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
import dataclasses
import json
import logging
import os
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

# SHARED thermal-ceiling config (solve3d/thermal_config.json). SINGLE SOURCE
# OF TRUTH read by BOTH lanes: the solve3d constrained solve and this Studio
# heatr3d verify. The cross-engine verify gate on a recommended drive is only
# meaningful if both read the IDENTICAL ceiling. Env override for tests /
# campaigns; mirrors the shrinkage_precomp.json loader in precomp.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
THERMAL_CONFIG_ENV = "RFAM_THERMAL_CONFIG"
THERMAL_CONFIG_PATH = _REPO_ROOT / "solve3d" / "thermal_config.json"
_THERMAL_SCHEMA = "1.0"


def load_thermal_config(path: str | Path | None = None) -> Dict[str, Any]:
    """The shared per-material thermal config. Refuses a missing file or an
    unsupported schema_version rather than silently falling back to a literal
    (a divergent ceiling would make the cross-engine verify meaningless)."""
    if path is None:
        env = os.environ.get(THERMAL_CONFIG_ENV)
        path = Path(env) if env else THERMAL_CONFIG_PATH
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"shared thermal config not found: {p} (set {THERMAL_CONFIG_ENV} "
            "or restore solve3d/thermal_config.json)")
    cfg = json.loads(p.read_text())
    ver = str(cfg.get("schema_version"))
    if ver != _THERMAL_SCHEMA:
        raise ValueError(f"unsupported thermal-config schema_version {ver!r} "
                         f"in {p} (supported: {_THERMAL_SCHEMA})")
    if "T_ceiling_C" not in cfg:
        raise ValueError(f"thermal config {p} missing T_ceiling_C")
    cfg["_path"] = str(p)
    return cfg


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


def _out_of_part_melt(phi_final: np.ndarray, part: np.ndarray) -> Optional[float]:
    """Mean melt fraction outside the part mask (bed spill), or None.

    Thin wrapper so the runner and the gate share ONE definition; importing
    lazily keeps studio3d.runner free of a hard dependency on the gate module.
    """
    from studio3d.correction_gate import out_of_part_melt_frac
    return out_of_part_melt_frac(phi_final, part)


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
                eqs_store_dir: Optional[str | Path] = None,
                shrinkage_precomp: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
    """One densify=True heatr3d march + the standard artifact set.

    sat_path: optional npz with a (n, n, n) ``sat`` array (the corrected
    arm's dopant volume, already on this grid). correction_engine labels
    which engine produced that map; required when sat_path is given.

    eqs_store_dir: optional per-job directory for the EQS solution store
    (the Studio passes ``<grade_dir>/heatr3d/eqs_store``). Only meaningful
    together with fast_march=True, which is the only path that takes an
    EqsCache; it is IGNORED with an explicit warning otherwise rather than
    silently pretending to accelerate. Default None everywhere.

    shrinkage_precomp: the Level 0 provenance block from
    studio3d.precomp.prepare_mesh, describing whether mesh_path is a
    pre-compensated mesh and with which coefficients. Recorded verbatim in
    results.json; None means the caller did not go through the Level 0 path
    at all (distinct from an explicit enabled=false).

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
        # Params is a frozen dataclass; an in-place assignment raised
        # FrozenInstanceError, so this drive-backoff knob (the ceiling lever)
        # was silently broken. Rebuild the frozen instance with the new drive.
        p = dataclasses.replace(p, power_density_w_per_m3=float(
            power_density_w_per_m3))

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

    # STALE-SNAPSHOT FIX (Tamper incident 2026-08-05): heatr3d's T_max_c is a
    # MELT-ONSET read (T_phi90 snapshot at t90, heatr3d.py:1248-1263). For
    # densify runs the march keeps heating long past t90 - on the Tamper the
    # snapshot said 239 C while the march log showed ~317 C at the density
    # stop. The ceiling is a material-degradation limit, so it must gate on
    # the run's TRUE peak. T_final is the end-of-run field; with the drive on
    # for the whole march the temperature rise is monotone, so its max is the
    # run peak. Both reads are recorded; the ok flag uses the worse one.
    t_end_max = (float(r.T_final[part].max())
                 if getattr(r, "T_final", None) is not None and part.any()
                 else None)
    t_peak = max(float(r.T_max_c), t_end_max) if t_end_max is not None \
        else float(r.T_max_c)
    # the ceiling comes from the SHARED config, NOT a literal, so this verify
    # and the solve3d constrained solve gate against one identical number.
    thermal_cfg = load_thermal_config()
    ceiling_c = float(thermal_cfg["T_ceiling_C"])
    gates = {
        "reached_phi90": bool(r.reached),
        "MELT_ONSET_FALLBACK": (not bool(r.reached)),
        "energy_residual_frac": float(r.energy_residual_frac),
        "energy_residual_ok": bool(abs(r.energy_residual_frac) < 1e-2),
        "clamp_bound": bool(r.clamp_bound),
        "T_max_C": float(round(r.T_max_c, 1)),
        "T_max_C_read": "melt_onset_snapshot",
        "T_end_max_C": (float(round(t_end_max, 1))
                        if t_end_max is not None else None),
        "T_ceiling_C": ceiling_c,
        "T_ceiling_ok": bool(t_peak <= ceiling_c),
        "T_ceiling_read": ("end_state_peak" if t_end_max is not None
                           else "melt_onset_snapshot"),
    }
    results: Dict[str, Any] = {
        "engine": ENGINE_LABEL,
        "engine_march": engine_march,
        "env_provenance": env_provenance,
        "eqs_cache": eqs_cache_record,
        # Level 0 material-shrinkage pre-compensation provenance (spec 1):
        # None = this caller never went through the Level 0 path; a block with
        # enabled false = it was explicitly switched off.
        "shrinkage_precomp": shrinkage_precomp,
        "trust_badge": TRUST_BADGE,
        "arm": str(arm),
        # HARD guard for the predicted-benefit gate (studio3d/correction_gate):
        # melt OUTSIDE the part mask is bed spill -- the hard-failure side of
        # the dense-iff-in-bounds hierarchy. Recorded for EVERY arm so the gate
        # can compare corrected against uniform. A missing value is read by the
        # gate as UNKNOWN and REJECTS, so this is not cosmetic.
        "out_of_part_melt_frac": _out_of_part_melt(r.phi_final, part),
        "correction_engine": correction_engine,
        "grid_n": n,
        # chamber tag (adaptive-chamber convention, solve3d 2026-08-05):
        # heatr3d's chamber is FROZEN at 60 mm (Grid L, owned by the
        # graduation lane); recorded so job records are comparable-by-tag
        # and never averaged/ranked across chamber sizes.
        "chamber_m": CHAMBER_M,
        "chamber_mode": "frozen_60mm_heatr3d",
        # which shared thermal config supplied the ceiling (auditable; both
        # lanes must show the same source for the cross-engine gate to mean
        # anything).
        "thermal_config": {"T_ceiling_C": ceiling_c,
                           "source": thermal_cfg.get("source"),
                           "material": thermal_cfg.get("material"),
                           "path": thermal_cfg.get("_path")},
        # the effective power-density drive (the thermal-ceiling lever): a
        # backed-off drive lowers the peak temperature at target density.
        # Recorded always, so a package can never hide which drive it used.
        "power_density_w_per_m3": float(p.power_density_w_per_m3),
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
    # END-STATE completeness from rho_final (the true end state; the
    # sinter_metrics above classify by the stale melt-onset phi snapshot and
    # stay recorded for continuity). fused = consolidation began (implies
    # melt); consolidated = essentially finished. The under-consolidated
    # count is the number a mean-based stop can hide - it is recorded so no
    # green summary can claim a complete part while the rim is porous.
    if r.rho_final is not None and part.any():
        from studio3d.warped_mesh import CONSOLIDATED_RHO, FUSED_RHO
        rho_in = r.rho_final[part]
        results["end_state"] = {
            "classification": "rho_final (end of run)",
            "fused_frac": float((rho_in >= FUSED_RHO).mean()),
            "consolidated_frac": float((rho_in >= CONSOLIDATED_RHO).mean()),
            "n_under_consolidated": int(((rho_in >= FUSED_RHO)
                                         & (rho_in < CONSOLIDATED_RHO)).sum()),
            "n_unfused": int((rho_in < FUSED_RHO).sum()),
            "min_rho_in_part": float(rho_in.min()),
            "melt_onset_note": ("sintered_frac / dice above are melt-onset "
                                "snapshot reads (t90), not end-state"),
        }
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
        T_final=(r.T_final.astype(np.float32)
                 if getattr(r, "T_final", None) is not None
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
