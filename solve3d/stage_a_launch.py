"""solve3d Stage A Task 4: the HEAVY drive-selection run at the physical target.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.stage_a_launch \
        --part square --rho-target 0.98 --drives 0.40,0.55,0.70,0.85,1.00 \
        --max-time 3000 --target-nodes 5600

Selects the best-part drive on a real deliverable part at the physical
rho_target by running a densify forward to the target at each drive, under the
degradation ceiling read from the SHARED solve3d/thermal_config.json. This is
the well-defined, no-rho-adjoint core of Task 4 (the drive is a scalar found by
forward evaluation; the dopant shape-solve at the chosen drive is the second
heavy phase and carries the objective-at-end-state scope note in
STAGE_A_REPORT.md).

CHECKPOINTED + RESUMABLE + MONITORABLE (compute-schedule convention): each
drive's record is appended to a checkpoint JSON as it completes, so a killed run
resumes where it stopped and a watcher can read progress at any time:

    cat solve3d/results/stage_a_task4_<part>_status.json

The square anchor is a real full-height part, over-ceiling at 1.0x like the
cube; the cube/pyramid run is identical on the Phase E conforming mesh
(geo.build_mesh) and is the follow-up.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def _write(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=float))
    tmp.replace(path)                       # atomic; no torn reads


def run(part: str, rho_target: float, drives: list[float], target_nodes: int,
        max_time_s: float, ckpt: Path, status: Path) -> dict:
    from solve3d import densify_forward as df
    from solve3d import forward as fwd
    from solve3d import stage_a

    tc = stage_a.thermal_config()
    ceiling_c = float(tc["T_ceiling_C"])
    melt_onset_c = float(tc["T_melt_onset_C"])
    rho_floor = float(tc["rho_target"]["floor"])
    rho_ideal = float(tc["rho_target"]["practical_ideal"])
    q = stage_a.prereg()["best_part_quality_metric"]
    base = fwd.ForwardParams()
    lc0 = 0.060 / 48.0

    done = {}
    if ckpt.exists():
        prev = json.loads(ckpt.read_text())
        # only resume a checkpoint written for the SAME part + target, else the
        # stale records would pollute this run's sweep
        if (prev.get("part") == part
                and abs(float(prev.get("rho_target", -1)) - rho_target) < 1e-9):
            done = {round(float(r["drive_a"]), 4): r
                    for r in prev.get("records", [])}

    records = list(done.values())
    for a in drives:
        key = round(float(a), 4)
        if key in done:
            print(f"[task4] a={a:.2f} already done (resume), skipping", flush=True)
            continue
        _write(status, {"part": part, "pid": os.getpid(), "state": "running",
                        "current_drive_a": a, "drives_total": len(drives),
                        "drives_done": len(records), "rho_target": rho_target,
                        "ceiling_c": ceiling_c,
                        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime())})
        t0 = time.perf_counter()
        pw = base.power_density_w_per_m3 * float(a)
        p = fwd.ForwardParams(power_density_w_per_m3=pw)
        m = df.run_densify_forward(part, target_nodes, lc0, rho_target, p=p,
                                   max_time_s=max_time_s, sample_dt_s=20.0)
        rec = {
            "drive_a": float(a), "power_density_w_per_m3": pw,
            "true_peak_c": float(m["true_peak_T_c"]),
            "achieved_rho": float(m["part_mean_rho"]),
            "reached_rho": bool(m["reached_rho"]),
            "shape_iou": stage_a._shape_iou_of_march(m, melt_onset_c),
            "exposure_s": float(m["exposure_s"]),
            "energy_residual_frac": float(m["energy_residual_frac"]),
            "clamp_bound": bool(m["clamp_bound"]),
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        records.append(rec)
        _write(ckpt, {"part": part, "rho_target": rho_target,
                      "ceiling_c": ceiling_c, "records": records})
        print(f"[task4] a={a:.2f} peak={rec['true_peak_c']:.1f}C "
              f"rho={rec['achieved_rho']:.3f} reached={rec['reached_rho']} "
              f"iou={rec['shape_iou']:.3f} wall={rec['wall_s']}s", flush=True)

    verdict = stage_a.select_from_sweep(
        records, ceiling_c, rho_floor, rho_ideal,
        float(q["w_density"]), float(q["w_shape"]), float(q["tie_break_tol"]))
    out = {
        "what": "Stage A Task 4 drive selection on a real part at the physical "
                "rho_target, under the shared-config degradation ceiling.",
        "part": part, "rho_target": rho_target, "ceiling_c": ceiling_c,
        "target_nodes_in_part": target_nodes, "max_time_s": max_time_s,
        "records": records, "verdict": verdict,
    }
    if verdict["verdict"] == "ok":
        out["recommended_power_settings"] = stage_a.recommended_power_settings(
            verdict["chosen_drive_a"])
    final = RESULTS / f"stage_a_task4_{part}.json"
    _write(final, out)
    _write(status, {"part": part, "pid": os.getpid(), "state": "done",
                    "verdict": verdict["verdict"],
                    "chosen_drive_a": verdict.get("chosen_drive_a"),
                    "drives_done": len(records), "final": final.name,
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime())})
    print(json.dumps({"verdict": verdict["verdict"],
                      "chosen_drive_a": verdict.get("chosen_drive_a"),
                      "final": final.name}, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="square", choices=("square", "circle"))
    ap.add_argument("--rho-target", type=float, default=0.98)
    ap.add_argument("--drives", default="0.40,0.55,0.70,0.85,1.00")
    ap.add_argument("--target-nodes", type=int, default=5600)
    ap.add_argument("--max-time", type=float, default=3000.0)
    args = ap.parse_args()
    drives = [float(x) for x in args.drives.split(",")]
    ckpt = RESULTS / f"stage_a_task4_{args.part}_ckpt.json"
    status = RESULTS / f"stage_a_task4_{args.part}_status.json"
    run(args.part, args.rho_target, drives, args.target_nodes, args.max_time,
        ckpt, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
