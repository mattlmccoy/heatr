"""solve3d Stage A: best-part drive selection under the degradation ceiling.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.stage_a --demo-sweep

DRIVE OBJECTIVE = BEST PART, NOT SPEED (Matt 2026-08-06, spec Stage A). The
drive is NOT the largest-under-ceiling (fastest print); it is the one that
MAXIMIZES part quality (density completeness toward rho_target + shape fidelity)
subject to the degradation ceiling, and on a quality tie the COOLER drive wins
for margin.

The ceiling is nearly independent of the dopant map (conservation: the dopant
relocates the peak, it does not lower its magnitude), so the drive sweep runs a
FIXED / uniform map -- no dopant solve inside the sweep. One dopant shape-solve
then runs at the chosen drive (Task 4).

HONEST-NULL (mandatory, spec sec 6): if no feasible drive both stays under the
ceiling and reaches rho_target within the exposure cap, select_from_sweep
returns "cannot make this part under this ceiling/chamber" -- never a
best-effort over-ceiling or under-dense map.

All thresholds are READ from solve3d/results/stage_a_preregistration.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from solve3d import gates, shape_metrics

RESULTS = Path(__file__).resolve().parent / "results"


def prereg() -> dict:
    return json.loads((RESULTS / "stage_a_preregistration.json").read_text())


# --------------------------------------------------------------------------- #
# Scoring + selection (pure logic; the tested core)
# --------------------------------------------------------------------------- #
def score_drive(achieved_rho: float, shape_iou: float, rho_ideal: float,
                w_density: float, w_shape: float) -> dict:
    """Best-part quality: Q = w_density * q_density + w_shape * q_shape.

    q_density = clip(achieved_rho / rho_ideal, 0, 1) -- completeness toward the
    practical-ideal density; q_shape = clip(shape_iou, 0, 1) -- fidelity of the
    fused body to the nominal part.
    """
    q_density = float(np.clip(achieved_rho / rho_ideal, 0.0, 1.0))
    q_shape = float(np.clip(shape_iou, 0.0, 1.0))
    return {"q_density": q_density, "q_shape": q_shape,
            "Q": float(w_density * q_density + w_shape * q_shape)}


def select_from_sweep(records: list[dict], ceiling_c: float, rho_floor: float,
                      rho_ideal: float, w_density: float, w_shape: float,
                      tie_tol: float) -> dict:
    """Choose the best-part feasible drive from a completed sweep.

    Each record: {drive_a, true_peak_c, achieved_rho, shape_iou}. A drive is
    FEASIBLE iff true_peak_c <= ceiling_c. A feasible drive is a VALID solution
    iff it also reaches the density floor. Among valid drives the highest Q
    wins; ties within tie_tol go to the COOLER (lowest drive_a). If no valid
    drive exists, the verdict is the honest null.
    """
    scored = []
    for r in records:
        feasible = bool(float(r["true_peak_c"]) <= float(ceiling_c))
        s = score_drive(float(r["achieved_rho"]), float(r["shape_iou"]),
                        rho_ideal, w_density, w_shape)
        scored.append({**r, "feasible": feasible, **s})
    scored.sort(key=lambda r: r["drive_a"])
    feasible = [r for r in scored if r["feasible"]]
    valid = [r for r in feasible if float(r["achieved_rho"]) >= float(rho_floor)]

    if not valid:
        best_density = (max(feasible, key=lambda r: r["achieved_rho"])
                        if feasible else None)
        return {
            "verdict": "cannot_make_under_ceiling",
            "chosen_drive_a": None,
            "reason": ("no drive both stays under the %.0f C degradation "
                       "ceiling and reaches the rho_target floor %.2f within "
                       "the sweep; the part cannot be made under this "
                       "ceiling/chamber" % (ceiling_c, rho_floor)),
            "n_feasible": len(feasible),
            "best_feasible_density": (
                None if best_density is None else
                {"drive_a": best_density["drive_a"],
                 "achieved_rho": best_density["achieved_rho"],
                 "true_peak_c": best_density["true_peak_c"],
                 "note": "reported as evidence only; NOT a shippable solution "
                         "(under the density floor)"}),
            "scored": scored,
        }

    best_q = max(r["Q"] for r in valid)
    tied = [r for r in valid if (best_q - r["Q"]) <= tie_tol]
    chosen = min(tied, key=lambda r: r["drive_a"])          # cooler wins ties
    return {
        "verdict": "ok",
        "chosen_drive_a": float(chosen["drive_a"]),
        "chosen": chosen,
        "n_feasible": len(feasible),
        "n_valid": len(valid),
        "tie_broken_by_cooler": bool(len(tied) > 1),
        "selection_rule": "max Q among feasible drives that reach the density "
                          "floor; ties within %.3f go to the cooler drive" % tie_tol,
        "scored": scored,
    }


# --------------------------------------------------------------------------- #
# Sweep driver (runs the forwards; the SHORT small-case runs, spec Stage A)
# --------------------------------------------------------------------------- #
def _shape_iou_of_march(march: dict, melt_onset_c: float) -> float:
    """Shape fidelity of the fused body: IoU between the region that fused
    (per-node trajectory peak >= melt onset) and the nominal part, on the solve
    mesh nodes. Captures under-fusion (fused inside the part) AND bed-melt /
    over-fusion (fused outside the part)."""
    fused = np.asarray(march["T_peak_nodal"], dtype=float) >= float(melt_onset_c)
    part = np.asarray(march["part_peak_mask"], dtype=bool)
    return shape_metrics.iou(fused, part)


def run_drive_sweep(shape: str, target_nodes_in_part: int, lc0: float,
                    drive_grid, stop_mean_rho: float,
                    max_time_s: float = 1500.0,
                    sample_dt_s: float = 10.0) -> list[dict]:
    """Run a densify forward at each drive on a fixed/uniform map and record the
    ceiling-relevant readings. Imported lazily so the pure selection logic (and
    its tests) never need dolfinx."""
    from solve3d import densify_forward as df
    from solve3d import forward as fwd

    base = fwd.ForwardParams()
    pr = prereg()
    melt_onset_c = float(pr["T_config"]["melt_onset_c"])
    records = []
    for a in drive_grid:
        pw = base.power_density_w_per_m3 * float(a)
        p = fwd.ForwardParams(power_density_w_per_m3=pw)
        m = df.run_densify_forward(shape, target_nodes_in_part, lc0,
                                   stop_mean_rho, p=p, max_time_s=max_time_s,
                                   sample_dt_s=sample_dt_s)
        records.append({
            "drive_a": float(a),
            "power_density_w_per_m3": pw,
            "true_peak_c": float(m["true_peak_T_c"]),
            "achieved_rho": float(m["part_mean_rho"]),
            "reached_rho": bool(m["reached_rho"]),
            "shape_iou": _shape_iou_of_march(m, melt_onset_c),
            "exposure_s": float(m["exposure_s"]),
            "energy_residual_frac": float(m["energy_residual_frac"]),
            "clamp_bound": bool(m["clamp_bound"]),
        })
        print(f"[drive-sweep] a={a:.2f} peak={m['true_peak_T_c']:.1f}C "
              f"rho={m['part_mean_rho']:.3f} iou={records[-1]['shape_iou']:.3f}",
              flush=True)
    return records


def select_drive(shape: str, target_nodes_in_part: int, lc0: float,
                 drive_grid, stop_mean_rho: float, **sweep_kw) -> dict:
    """Full Stage A drive selection: run the sweep, then choose the best-part
    feasible drive. Thresholds READ from the pre-registration."""
    pr = prereg()
    ceiling_c = float(pr["T_config"]["degradation_ceiling_c"])
    rho_floor = float(pr["rho_target"]["floor"])
    rho_ideal = float(pr["rho_target"]["practical_ideal"])
    q = pr["best_part_quality_metric"]
    records = run_drive_sweep(shape, target_nodes_in_part, lc0, drive_grid,
                              stop_mean_rho, **sweep_kw)
    verdict = select_from_sweep(records, ceiling_c, rho_floor, rho_ideal,
                                float(q["w_density"]), float(q["w_shape"]),
                                float(q["tie_break_tol"]))
    verdict["config"] = {"shape": shape, "ceiling_c": ceiling_c,
                         "rho_floor": rho_floor, "rho_ideal": rho_ideal,
                         "drive_grid": list(map(float, drive_grid)),
                         "stop_mean_rho": stop_mean_rho}
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-sweep", action="store_true",
                    help="short circle-anchor drive sweep demonstrating the "
                         "feasible/infeasible split and the best-part selection")
    args = ap.parse_args()
    if args.demo_sweep:
        # small, warm-driven, modest densification target: a SHORT run that
        # exercises the end-to-end wiring and produces a real ceiling crossing.
        # One sweep, TWO verdicts from the SAME runs so both paths are shown:
        #  - at the pre-registered rho floor (0.90) the modest 0.66 demo target
        #    is unreachable -> HONEST NULL (the mechanism, faithfully fired);
        #  - at a reachable demo floor (0.60) the feasible drives are valid ->
        #    OK, and best-part+cooler selection picks the coolest feasible drive.
        pr = prereg()
        ceiling_c = float(pr["T_config"]["degradation_ceiling_c"])
        rho_ideal = float(pr["rho_target"]["practical_ideal"])
        q = pr["best_part_quality_metric"]
        records = run_drive_sweep("circle", 2900, 0.060 / 32.0,
                                  drive_grid=(2.0, 4.0, 6.0, 8.0),
                                  stop_mean_rho=0.66, max_time_s=400.0)
        v_real = select_from_sweep(records, ceiling_c,
                                   float(pr["rho_target"]["floor"]), rho_ideal,
                                   float(q["w_density"]), float(q["w_shape"]),
                                   float(q["tie_break_tol"]))
        v_demo = select_from_sweep(records, ceiling_c, 0.60, rho_ideal,
                                   float(q["w_density"]), float(q["w_shape"]),
                                   float(q["tie_break_tol"]))
        doc = {
            "what": "Stage A Task 3 end-to-end drive-selection demonstration on "
                    "the extruded-circle anchor (SHORT small case). A real "
                    "ceiling crossing (a=6,8 over 250 C) and both selection "
                    "paths from one sweep.",
            "records": records,
            "verdict_at_prereg_floor_0p90": v_real,
            "verdict_at_reachable_demo_floor_0p60": v_demo,
            "note": "the physical rho_target=0.98 selection on the deliverable "
                    "part is the Task 4 heavy run; this demo target (0.66) is "
                    "kept low so the sweep is short.",
        }
        gates.write_json("stage_a_drive_sweep.json", doc)
        print(json.dumps({
            "at_prereg_floor_0.90": {"verdict": v_real["verdict"],
                                     "chosen": v_real.get("chosen_drive_a")},
            "at_demo_floor_0.60": {"verdict": v_demo["verdict"],
                                   "chosen": v_demo.get("chosen_drive_a")},
            "scored": [{"a": r["drive_a"], "peak": round(r["true_peak_c"], 1),
                        "rho": round(r["achieved_rho"], 3),
                        "feasible": r["feasible"]} for r in v_real["scored"]],
        }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
