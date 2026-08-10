"""Mandatory pre-send verification (spec 7c): verify what PRINTS.

The sat map is reconstructed from the emitted TIFF job itself, per layer,
as the graded/ungraded level ratio (Meteor WhiteIsZero convention), then
transferred support-aware onto the solver grid and re-marched through the
native heatr3d densify. The record lands in the package's
production_verify_summary.json; a package whose gates fail (or whose
verification never ran) is not sendable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from studio3d.precomp import prepare_mesh
from studio3d.runner import run_densify, voxelize_stl
from studio3d.transfer import stack_to_voxel


def _decode_levels(path: Path, bpp: int) -> np.ndarray:
    """Meteor TIFF gray -> ink level (WhiteIsZero, black = max ink)."""
    gray = np.asarray(Image.open(path).convert("L"), float)
    max_val = (1 << bpp) - 1
    return np.round((255.0 - gray) / 255.0 * max_val)


def reconstruct_sat_stack(job_dir: str | Path
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sat_stack, mask_stack, z_mm) from graded + _ungraded TIFF pairs."""
    job = Path(job_dir)
    info = json.loads((job / "job_info.json").read_text())
    bpp = int(info.get("bpp", 4))
    lh = float(info.get("layer_height_mm", 0.1))
    graded = sorted(q for q in job.iterdir()
                    if q.suffix.lower() in (".tif", ".tiff"))
    if not graded:
        raise FileNotFoundError(f"no TIFFs in {job}")
    base_dir = job / "_ungraded"
    if not base_dir.is_dir():
        raise FileNotFoundError(
            "no _ungraded reference TIFFs: the packaging step must retain "
            "the pre-grading rasters so verification can reconstruct the "
            "applied map from what actually prints")
    sats, masks = [], []
    for g in graded:
        b = base_dir / g.name
        Lg = _decode_levels(g, bpp)
        Lb = _decode_levels(b, bpp)
        mask = Lb > 0
        sat = np.ones_like(Lg)
        sat[mask] = np.clip(Lg[mask] / Lb[mask], 0.0, 1.0)
        sats.append(sat)
        masks.append(mask)
    z_mm = (np.arange(len(graded)) + 0.5) * lh
    # the canvas's PHYSICAL frame (the transfer's SOURCE chamber): pixels
    # over dpi. Assuming 60 mm here is the same frame-bug class the Tamper
    # exposed in the 2.5-D path (TAMPER_DIAGNOSIS.md 3f). The part is
    # assumed centered on the canvas (the slicer's convention); the 2
    # percent gate is the backstop if that assumption breaks.
    dpi = float(info.get("dpi", 720))
    canvas_m = sats[0].shape[1] / (dpi / 25.4) / 1000.0
    return np.stack(sats), np.stack(masks), z_mm, canvas_m


def verify_package(pkg_dir: str | Path, mesh_path: str, n: int,
                   tiff_job_dir: str | Path,
                   max_time_s: float = 1500.0,
                   stop_mean_rho: float | None = 0.98,
                   fast_march: bool = True,
                   eqs_store_dir: str | Path | None = None,
                   precomp: bool = True,
                   grade_dir: str | Path | None = None) -> Dict[str, Any]:
    """Reconstruct -> transfer -> densify -> record. Never silent.

    eqs_store_dir: the originating job's <grade_dir>/heatr3d/eqs_store. A
    re-verify of the SAME emitted rasters reproduces the same sat volume and
    therefore the same gamma, so the EQS solve is served from the store even
    though this is a fresh process. Only takes effect with fast_march=True
    (the only path that accepts a cache); both default off.

    Note this verify normally MISSES on a first run: the rasters are
    re-quantized, so its sat differs from the corrected arm's. That is the
    intended behaviour -- a hit there would mean the cache had ignored a real
    change to the dopant map.
    """
    pkg = Path(pkg_dir)
    rec: Dict[str, Any]
    # Flag 1 (chamber guard): the recommended drive is valid ONLY in the
    # chamber it was solved in; heatr3d verify is frozen at 60 mm. If the
    # correction was solved in a different chamber, verifying at 60 mm would
    # compare unlike chambers - a false-green. Refuse BEFORE marching.
    man_p0 = pkg / "manifest.json"
    if man_p0.exists():
        note = (json.loads(man_p0.read_text())
                .get("correction_provenance", {}).get("chamber_mismatch_note"))
        if note:
            rec = {"run": True, "source": "emitted_rasters", "gates_ok": False,
                   "refusal": ("chamber_mismatch: the drive/map was solved in a "
                               "non-60 mm chamber; heatr3d is_sendable verify is "
                               "frozen at 60 mm and cannot certify it here. " + str(note))}
            (pkg / "production_verify_summary.json").write_text(
                json.dumps(rec, indent=2, default=float))
            man = json.loads(man_p0.read_text())
            man["production_verify"] = rec
            man_p0.write_text(json.dumps(man, indent=2, default=float))
            return rec
    try:
        # verify the mesh that PRINTS: the same Level 0 pre-compensated STL
        # the densify arms marched (reused from grade_dir when given).
        mesh_path, precomp_prov = prepare_mesh(
            mesh_path, grade_dir if grade_dir is not None else pkg,
            enabled=precomp)
        sat_stack, mask_stack, z_mm, canvas_m = \
            reconstruct_sat_stack(tiff_job_dir)
        part = voxelize_stl(mesh_path, n)
        # source frame = the canvas's physical extent; target frame defaults
        # to the voxel arm's 60 mm chamber (two frames, never one)
        tr = stack_to_voxel(sat_stack, mask_stack, z_mm, part,
                            chamber_m=canvas_m)
        sat_path = pkg / "verify_sat.npz"
        np.savez_compressed(sat_path, sat=tr["sat"].astype(np.float32),
                            part=part)
        # verify at the SAME power the package declares it will print at (the
        # recommended ceiling-feasible drive), not the nominal - else the
        # cross-engine ceiling gate certifies a power the part will not use.
        # None -> run_densify uses its nominal default.
        man_p = pkg / "manifest.json"
        verify_pd = None
        if man_p.exists():
            verify_pd = (json.loads(man_p.read_text())
                         .get("power_settings", {})
                         .get("power_density_w_per_m3"))
        res = run_densify(mesh_path, pkg / "verify_run", n=n,
                          arm="corrected", sat_path=str(sat_path),
                          power_density_w_per_m3=verify_pd,
                          correction_engine="emitted_rasters",
                          max_time_s=max_time_s,
                          stop_mean_rho=stop_mean_rho,
                          fast_march=fast_march,
                          eqs_store_dir=(eqs_store_dir if fast_march else None),
                          shrinkage_precomp=precomp_prov)
        g = res["gates"]
        gates_ok = bool(g["energy_residual_ok"] and g["T_ceiling_ok"]
                        and not g["clamp_bound"])
        rec = {"run": True, "source": "emitted_rasters",
               "shrinkage_precomp": precomp_prov,
               "transfer": {k: v for k, v in tr.items() if k != "sat"},
               "grid_n": n, "gates": g, "gates_ok": gates_ok,
               "power_density_w_per_m3": res.get("power_density_w_per_m3"),
               "sigma_T": res["sigma_T"],
               "rho_final_mean": res.get("rho_final_mean"),
               # recorded acceleration travels with the verify record too
               "engine_march": res.get("engine_march"),
               "env_provenance": res.get("env_provenance"),
               "eqs_cache": res.get("eqs_cache")}
    except Exception as e:
        rec = {"run": True, "source": "emitted_rasters", "gates_ok": False,
               "shrinkage_precomp": locals().get("precomp_prov"),
               "error": f"verification failed: {e}"}
    (pkg / "production_verify_summary.json").write_text(
        json.dumps(rec, indent=2, default=float))
    # keep the manifest in sync when it exists
    man_p = pkg / "manifest.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())
        man["production_verify"] = rec
        man_p.write_text(json.dumps(man, indent=2, default=float))
    return rec


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="verify a print package")
    ap.add_argument("pkg_dir")
    ap.add_argument("mesh")
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-time-s", type=float, default=1500.0)
    ap.add_argument("--grade-dir", default=None,
                    help="the job's grade dir, so the SAME pre-compensated "
                         "mesh the densify arms used is reused here")
    ap.add_argument("--no-precomp", action="store_true",
                    help="escape hatch: verify the nominal mesh (recorded as "
                         "enabled false)")
    ap.add_argument("--no-fast-march", action="store_true",
                    help="escape hatch: reference march instead of the "
                         "bit-identical numba march (default: fast on)")
    args = ap.parse_args()
    rec = verify_package(args.pkg_dir, args.mesh, n=args.n,
                         tiff_job_dir=args.job_dir,
                         max_time_s=args.max_time_s,
                         fast_march=not args.no_fast_march,
                         precomp=not args.no_precomp,
                         grade_dir=args.grade_dir)
    print("VERIFY " + json.dumps({"run": rec["run"],
                                  "gates_ok": rec.get("gates_ok")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
