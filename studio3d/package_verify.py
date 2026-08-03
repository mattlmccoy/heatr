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
    return np.stack(sats), np.stack(masks), z_mm


def verify_package(pkg_dir: str | Path, mesh_path: str, n: int,
                   tiff_job_dir: str | Path,
                   max_time_s: float = 1500.0,
                   stop_mean_rho: float | None = 0.98,
                   fast_march: bool = False,
                   eqs_store_dir: str | Path | None = None) -> Dict[str, Any]:
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
    try:
        sat_stack, mask_stack, z_mm = reconstruct_sat_stack(tiff_job_dir)
        part = voxelize_stl(mesh_path, n)
        tr = stack_to_voxel(sat_stack, mask_stack, z_mm, part,
                            chamber_m=0.060)
        sat_path = pkg / "verify_sat.npz"
        np.savez_compressed(sat_path, sat=tr["sat"].astype(np.float32),
                            part=part)
        res = run_densify(mesh_path, pkg / "verify_run", n=n,
                          arm="corrected", sat_path=str(sat_path),
                          correction_engine="emitted_rasters",
                          max_time_s=max_time_s,
                          stop_mean_rho=stop_mean_rho,
                          fast_march=fast_march,
                          eqs_store_dir=(eqs_store_dir if fast_march else None))
        g = res["gates"]
        gates_ok = bool(g["energy_residual_ok"] and g["T_ceiling_ok"]
                        and not g["clamp_bound"])
        rec = {"run": True, "source": "emitted_rasters",
               "transfer": {k: v for k, v in tr.items() if k != "sat"},
               "grid_n": n, "gates": g, "gates_ok": gates_ok,
               "sigma_T": res["sigma_T"],
               "rho_final_mean": res.get("rho_final_mean"),
               # recorded acceleration travels with the verify record too
               "engine_march": res.get("engine_march"),
               "env_provenance": res.get("env_provenance"),
               "eqs_cache": res.get("eqs_cache")}
    except Exception as e:
        rec = {"run": True, "source": "emitted_rasters", "gates_ok": False,
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
    args = ap.parse_args()
    rec = verify_package(args.pkg_dir, args.mesh, n=args.n,
                         tiff_job_dir=args.job_dir,
                         max_time_s=args.max_time_s)
    print("VERIFY " + json.dumps({"run": rec["run"],
                                  "gates_ok": rec.get("gates_ok")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
