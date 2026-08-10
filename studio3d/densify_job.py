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
from studio3d.precomp import prepare_mesh
from studio3d.runner import run_densify

ARMS = ("uncorrected", "corrected")


def recommended_drive(grade_dir: str | Path) -> float | None:
    """The joint solve's recommended ceiling-feasible print power, or None.

    Cross-lane contract (2026-08-07): the Studio's production solve
    (solve3d.studio_solve, Stage A/B4) writes the ceiling-respecting drive
    into studio_solve_results.json as recommended_power_density_w_per_m3
    (absolute W/m3). Both densify arms run at THIS drive so the print is
    simulated at the power it will actually use, and the benefit gate
    compares arms at a matched drive. None means the field is absent, null
    (an honest-null / drive-limited part), or there is no solve artifact -
    the caller then falls back to the nominal drive and records that it did.
    """
    p = Path(grade_dir) / "heatr3d" / "solve" / "studio_solve_results.json"
    if not p.exists():
        return None
    try:
        val = json.loads(p.read_text()).get("recommended_power_density_w_per_m3")
    except (ValueError, OSError):
        return None
    return float(val) if val is not None else None


def run_job(mesh_path: str, grade_dir: str | Path, arm: str = "uncorrected",
            n: int = 64, max_time_s: float = 1500.0,
            stop_mean_rho: float | None = 0.98,
            fast_march: bool = True,
            precomp: bool = True) -> Dict[str, Any]:
    """One densify arm. fast_march defaults OFF (the blessed opt-in terms).

    precomp defaults ON (spec section 1, Level 0): the mesh is affinely
    pre-compensated for MATERIAL shrinkage into <grade_dir>/precomp/<name>
    and everything downstream -- correction build, voxelization, march --
    consumes THAT file. The coefficients are borrowed SLS literature values,
    unmeasured for RFAM; the applicability caveat travels in results
    ["shrinkage_precomp"] and must be quoted with every dimensional claim.

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

    # Level 0 pre-compensation FIRST: the chamber-fit refusal must see the
    # enlarged part, and every downstream consumer must see the same file.
    print("STUDIO3D_PROGRESS stage=precomp", flush=True)
    mesh_path, precomp_prov = prepare_mesh(mesh_path, grade_dir,
                                           enabled=precomp)

    sat_path = None
    correction_engine = None
    if arm == "corrected":
        print("STUDIO3D_PROGRESS stage=correction", flush=True)
        prov = build_correction(grade_dir, mesh_path, n)
        sat_path = str(grade_dir / "heatr3d" / "correction_sat.npz")
        correction_engine = prov["engine"]

    # The joint solve's recommended ceiling-feasible drive is the standard
    # print power: BOTH arms march at it (matched, so the benefit gate is
    # like-for-like), and the ceiling gate then reads the part at the power
    # it will actually print. Absent/honest-null -> nominal, recorded.
    rec_drive = recommended_drive(grade_dir)
    print("STUDIO3D_PROGRESS stage=march", flush=True)
    res = run_densify(mesh_path, out, n=n, arm=arm, sat_path=sat_path,
                      power_density_w_per_m3=rec_drive,
                      correction_engine=correction_engine,
                      max_time_s=max_time_s, stop_mean_rho=stop_mean_rho,
                      fast_march=fast_march,
                      eqs_store_dir=(eqs_store if fast_march else None),
                      shrinkage_precomp=precomp_prov)
    res["drive_recommended"] = rec_drive is not None
    # persist the job-level flag onto the on-disk results.json (run_densify
    # wrote the file before this flag existed; the package reads from disk).
    rp = out / "results.json"
    if rp.exists():
        disk = json.loads(rp.read_text())
        disk["drive_recommended"] = res["drive_recommended"]
        rp.write_text(json.dumps(disk))

    # ---- predicted-benefit gate (TAMPER_DIAGNOSIS.md fix 2c) ------------- #
    # The corrected arm must BEAT uniform. On the Tamper nothing compared the
    # two arms, so a map that zeroed 79 % of the dopant shipped. A rejection
    # reverts the ACTIVE packaging artifacts to uniform and raises the
    # no_correction_applied banner.
    if arm == "corrected":
        print("STUDIO3D_PROGRESS stage=benefit_gate", flush=True)
        res["correction_gate"] = _gate_against_before(grade_dir, res)

    print("STUDIO3D_PROGRESS stage=done", flush=True)
    return res


def _gate_against_before(grade_dir: Path,
                         after: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the corrected arm with the BEFORE arm and ENFORCE the verdict."""
    from studio3d.correction_gate import apply_verdict, evaluate_correction

    bp = grade_dir / "heatr3d" / "uncorrected" / "results.json"
    if not bp.exists():
        # Benefit cannot be measured without the baseline. Refuse rather than
        # let an unmeasured correction through (false-green rule).
        verdict = {"verdict": "REJECTED", "failed_guards": ["no_before_arm"],
                   "note": ("the uniform BEFORE arm's results.json is missing, "
                            "so predicted benefit cannot be measured; the "
                            "correction is not shipped unmeasured")}
    else:
        verdict = evaluate_correction(json.loads(bp.read_text()), after)
    prov = apply_verdict(grade_dir, verdict)
    if verdict.get("verdict") == "REJECTED":
        print("STUDIO3D_PROGRESS stage=correction_rejected", flush=True)
        print("STUDIO3D_BANNER " + str(prov.get("banner", "")), flush=True)
    return verdict


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
    ap.add_argument("--no-precomp", action="store_true",
                    help="escape hatch: skip the Level 0 material-shrinkage "
                         "pre-compensation (recorded as enabled false)")
    ap.add_argument("--no-fast-march", action="store_true",
                    help="escape hatch: use the slower reference march instead "
                         "of the bit-identical numba march (default: fast on)")
    args = ap.parse_args()
    stop = args.stop_mean_rho if args.stop_mean_rho > 0 else None
    try:
        res = run_job(args.mesh, args.grade_dir, arm=args.arm, n=args.n,
                      max_time_s=args.max_time_s, stop_mean_rho=stop,
                      fast_march=not args.no_fast_march,
                      precomp=not args.no_precomp)
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
