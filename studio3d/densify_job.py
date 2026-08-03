"""The one subprocess entry the Studio server calls for 3-D densify arms.

uncorrected: voxelize + native heatr3d densify march.
corrected:   build the correction (solved registry or 2.5-D fallback, engine
             labeled) then the same march with the sat volume applied.

Progress lines "STUDIO3D_PROGRESS stage=<name>" are printed for the server's
log tail. Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 (the caller's
job; see the S4 oversubscription note).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from studio3d.correction import build_correction
from studio3d.runner import run_densify

ARMS = ("uncorrected", "corrected")


def run_job(mesh_path: str, grade_dir: str | Path, arm: str = "uncorrected",
            n: int = 64, max_time_s: float = 1500.0,
            stop_mean_rho: float | None = 0.98,
            fast_march: bool = False) -> Dict[str, Any]:
    """One densify arm. fast_march defaults OFF (the blessed opt-in terms).

    When fast_march is on, the job's EQS solution store lives at
    <grade_dir>/heatr3d/eqs_store, so a later package-verify of the SAME
    dopant map hits it from a fresh process instead of re-solving. The two
    arms have different sat maps, so the corrected arm always misses -- that
    is correct, not a cache failure.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    grade_dir = Path(grade_dir)
    out = grade_dir / "heatr3d" / arm
    eqs_store = grade_dir / "heatr3d" / "eqs_store"

    sat_path = None
    correction_engine = None
    if arm == "corrected":
        print("STUDIO3D_PROGRESS stage=correction", flush=True)
        prov = build_correction(grade_dir, mesh_path, n)
        sat_path = str(grade_dir / "heatr3d" / "correction_sat.npz")
        correction_engine = prov["engine"]

    print("STUDIO3D_PROGRESS stage=march", flush=True)
    res = run_densify(mesh_path, out, n=n, arm=arm, sat_path=sat_path,
                      correction_engine=correction_engine,
                      max_time_s=max_time_s, stop_mean_rho=stop_mean_rho,
                      fast_march=fast_march,
                      eqs_store_dir=(eqs_store if fast_march else None))
    print("STUDIO3D_PROGRESS stage=done", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Studio 3-D densify job")
    ap.add_argument("mesh")
    ap.add_argument("--grade-dir", required=True)
    ap.add_argument("--arm", default="uncorrected", choices=ARMS)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-time-s", type=float, default=1500.0)
    ap.add_argument("--stop-mean-rho", type=float, default=0.98,
                    help="stop the march at this mean part density "
                         "(<= 0 disables; horizon then rules)")
    ap.add_argument("--fast-march", action="store_true",
                    help="opt in to the bit-identical numba march + the "
                         "per-job EQS solution store (default off)")
    args = ap.parse_args()
    stop = args.stop_mean_rho if args.stop_mean_rho > 0 else None
    try:
        res = run_job(args.mesh, args.grade_dir, arm=args.arm, n=args.n,
                      max_time_s=args.max_time_s, stop_mean_rho=stop,
                      fast_march=args.fast_march)
    except Exception as e:
        # one clean line for the UI; the traceback stays in the log
        import traceback
        traceback.print_exc()
        print(f"STUDIO3D_ERROR {e}", flush=True)
        return 1
    print("RESULTS " + json.dumps(res, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
