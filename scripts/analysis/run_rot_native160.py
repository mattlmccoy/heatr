#!/usr/bin/env python3
"""THE DECISIVE RE-SOLVE: solve the rotating arm NATIVELY at the hold-out grid.

THE QUESTION. `ROTATING_HOLDOUT_REPORT.md` moved a dopant map solved at grid 120
to grid 160 and watched the cross's four-angle indexed arm fall out of the
SOLVED class, intersection over union 0.9700 to 0.8052. That pass could not say
whether the loss belongs to

  MAP TRANSFER          the solved map is entangled with the grid it was solved
                        on, even filtered and even under rotation, so moving it
                        is what costs the fidelity; or
  FORWARD NON-CONVERGENCE
                        the two-dimensional forward model itself is not grid
                        converged on this rotating actuator, so no map solved at
                        any grid reaches 0.95 at 160.

The experiment that separates them is one re-solve. Solve the SAME arm natively
at grid 160 under the production recipe, score it at 160 through the SAME
hold-out harness, and read the intersection over union against the same 0.95
line. Recovering the class means transfer. Not recovering it means
non-convergence.

WHAT IS HELD FIXED AGAINST THE GRID-120 SOLVE.
  * the recipe: physical-length design filter, box [0, 1], L-BFGS-B on the
    finite-difference gated filtered averaged-kernel gradient, conductivity
    channel only, 4 bits per pixel quantization for the deliverable map;
  * the budget class: 16 gradient evaluations per start, two starts (cold at
    saturation 1, warm from the best known static map), which is the campaign's
    40 forward-equivalent depth budget;
  * the actuator: 90-degree indexing, one quarter turn every 2.0 s, the same
    four candidate angles, the same 1500-step 750 s horizon;
  * the averaged kernel: equal angle weights, the model every map in this
    campaign was solved against.

WHAT IS REGENERATED AT 160, and it must be.
  * the part mask and the target, by the production domain builder at 160;
  * the drive, recalibrated so the uniform static arm absorbs 500.0 W/m in
    electrical state B (`FROZEN_CONVENTIONS_2D` Section 3);
  * the design filter width. sigma is a PHYSICAL length of 0.75 mm
    (`design_filter` radius convention), which is 1.5 cells at grid 120 and
    2.0 cells at grid 160. Holding the CELL count instead would shrink the
    design length with the grid and would confound the answer.

The warm start is the grid-120 static map moved to 160 by the production
resample. That is a STARTING POINT for the optimizer, not a transferred answer;
the cold start is run alongside and the better of the two wins, exactly as at
grid 120.

Run:
  ./.venv312/bin/python scripts/analysis/run_rot_native160.py cross
  ./.venv312/bin/python scripts/analysis/run_rot_native160.py keyhole
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from adjoint2d import design_filter as df, gradops                 # noqa: E402
from adjoint2d import printability as pq                           # noqa: E402
from adjoint2d import robust_rot as rr                             # noqa: E402
from adjoint2d.library_solve import shape_config                   # noqa: E402
from adjoint2d.pins import load_cfg                                # noqa: E402
from adjoint2d.rot_kernel import AveragedKernel                    # noqa: E402
from run_rot_avg_solve import solve_start                          # noqa: E402
from run_rot_holdout import ARMS, base_cfg, program_for            # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_rot_native160"
SOLVED_CLASS_IOU = 0.95
N_EVALS = 16                        # 40 forward-equivalent budget class
BOX = (0.0, 1.0)
SIGMA_MM_AT_120 = 0.75              # 1.5 cells at grid 120, a physical length
GRID_120 = 120
GRID = 160

CASES: dict[str, dict] = {
    "cross": {
        "arm": "cross_index90",
        "label": "cross, 90-degree indexing at 2.0 s, four-angle averaged kernel",
        "holdout_json": "cross_index90.json",
    },
    "keyhole": {
        "arm": "keyhole_cont_fixedprog",
        "label": "keyhole, continuous rotation, twelve positions, "
                 "divisor-aware re-emitted program",
        "holdout_json": "keyhole_cont_fixedprog.json",
    },
}


def sigma_cells_at(n_grid: int) -> float:
    """The design filter width in cells that holds the physical length fixed."""
    return float(df.DEFAULT_SIGMA_CELLS) * float(n_grid) / float(GRID_120)


def main(name: str) -> None:
    if name not in CASES:
        raise SystemExit(f"unknown case {name!r}; have {sorted(CASES)}")
    spec_case = CASES[name]
    spec = ARMS[spec_case["arm"]]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{name}/native160] {msg}", flush=True)

    cfg = base_cfg(spec, GRID)

    from adjoint2d.pins import build_case
    probe = build_case(rr.cfg_at_grid(cfg, GRID))
    dt = float(probe.pins.dt)
    n_steps = int(probe.pins.n_steps)
    del probe
    ang, pos, prog_info = program_for(spec, dt, n_steps)

    # ONE case build at 160, shared by the solve and by every scored arm, so the
    # forward the optimizer differentiates and the forward that scores it are
    # the same discrete operators.
    rc = rr.build_rot_case(cfg, GRID, ang, recalibrate=True)
    cal = rc.calibration
    pm = rc.part_mask
    sigma = sigma_cells_at(GRID)
    log(f"grid {GRID}, {int(pm.sum())} part cells, drive "
        f"{cal['voltage_v_pinned']:.1f} V gives {cal['p_at_pinned_v_w_per_m']:.1f} "
        f"W/m here, recalibrated to {cal['voltage_v']:.1f} V, verified "
        f"{cal['p_verified_w_per_m']:.2f} W/m")
    log(f"design filter sigma {sigma:.4f} cells at 160 = "
        f"{SIGMA_MM_AT_120} mm, the same physical length as 1.5 cells at 120")
    log(f"program: {prog_info['kind']}, {len(ang)} positions, "
        f"{prog_info.get('n_moves')} moves")

    kern = AveragedKernel(case0=rc.kernel.case0, angles=rc.kernel.angles,
                          per_angle=rc.kernel.per_angle)
    ops = gradops.gradient_matrices(rc.case.x, rc.case.y)

    # --- the solve, natively at 160 -----------------------------------------
    warm120 = np.asarray(np.load(spec["static_map"][0])[spec["static_map"][1]],
                         dtype=float)
    warm = rr.transfer_map(warm120, pm, bpp=None)
    starts = {"cold": np.ones(pm.shape),
              "warm": np.where(pm, np.clip(warm, *BOX), 1.0)}
    per_start: dict[str, dict] = {}
    for sname, v0 in starts.items():
        rows, store = solve_start(kern, ops, v0, N_EVALS, log, sname,
                                  sigma_cells=sigma)
        if not rows:
            log(f"  start {sname}: NO EVALUATIONS, skipped loudly")
            continue
        b = min(rows, key=lambda r: r["J"])
        per_start[sname] = {"rows": rows, "best_J": float(b["J"]),
                            "v": store[int(b["eval_index"])]}
    if not per_start:
        raise RuntimeError(f"{name}: no start produced an evaluation")
    winner = min(per_start, key=lambda k: per_start[k]["best_J"])
    v_best = per_start[winner]["v"]
    s_native_cont = df.apply_filter(v_best, pm, sigma)
    s_native = pq.quantize_in_part(s_native_cont, pm, bpp=rr.BPP, sat_max=1.0)
    log(f"solve done, winner start {winner}, best filtered J "
        f"{per_start[winner]['best_J']:.2f} on the solve objective")

    # --- the transferred grid-120 map, the thing the hold-out scored ---------
    s_transferred = rr.transfer_map(
        np.asarray(np.load(spec["map"][0])[spec["map"][1]], dtype=float),
        pm, bpp=rr.BPP)
    s_static = rr.transfer_map(warm120, pm, bpp=rr.BPP)
    ones = np.ones(pm.shape)

    # --- scoring, all through the hold-out harness at 160 --------------------
    arms: dict[str, dict] = {}
    arms["ROT_uniform"] = rr.score_program(rc, ones, pos, n_steps)
    rr.log_row(name, "ROT_uniform", arms["ROT_uniform"])
    arms["ROT_transferred120"] = rr.score_program(rc, s_transferred, pos, n_steps)
    rr.log_row(name, "ROT_transferred120", arms["ROT_transferred120"])
    arms["ROT_native160"] = rr.score_program(rc, s_native, pos, n_steps)
    rr.log_row(name, "ROT_native160", arms["ROT_native160"])
    arms["ROT_native160_cont"] = rr.score_program(rc, s_native_cont, pos, n_steps)
    rr.log_row(name, "ROT_native160_cont", arms["ROT_native160_cont"])
    arms["QS_transferred120_equalW"] = rr.score_quasistatic(rc, s_transferred)
    rr.log_row(name, "QS_transferred120_equalW", arms["QS_transferred120_equalW"])
    arms["QS_native160_equalW"] = rr.score_quasistatic(rc, s_native)
    rr.log_row(name, "QS_native160_equalW", arms["QS_native160_equalW"])
    arms["STATIC_uniform"] = rr.score_static(rc, ones)
    rr.log_row(name, "STATIC_uniform", arms["STATIC_uniform"])
    arms["STATIC_solved"] = rr.score_static(rc, s_static)
    rr.log_row(name, "STATIC_solved", arms["STATIC_solved"])

    fields = {}
    for k, a in arms.items():
        a.pop("J_at_first_step", None)
        ph = a.pop("_phi_at_stop", None)
        if ph is not None:
            fields[f"phi_{k}"] = np.asarray(ph, dtype=np.float32)

    # --- the verdict, in the two words the dissertation needs ----------------
    iou_n = float(arms["ROT_native160"]["IoU"])
    iou_t = float(arms["ROT_transferred120"]["IoU"])
    recovers = bool(iou_n >= SOLVED_CLASS_IOU)
    hold = json.loads((REPO / "fgm_solve_campaign/out_rot_holdout"
                       / spec_case["holdout_json"]).read_text())
    iou_120 = float(hold["grids"]["120"]["arms"]["ROT_solved"]["IoU"])
    verdict = {
        "question": "does a map solved NATIVELY at grid 160 recover the SOLVED "
                    "class (intersection over union at least 0.95) at grid 160?",
        "IoU_at_120_solved_at_120": iou_120,
        "IoU_at_160_map_transferred_from_120": iou_t,
        "IoU_at_160_map_solved_at_160": iou_n,
        "IoU_at_160_map_solved_at_160_unquantized": float(
            arms["ROT_native160_cont"]["IoU"]),
        "solved_class_recovered_by_the_native_solve": recovers,
        "native_minus_transferred_IoU": iou_n - iou_t,
        "native_minus_transferred_J_pct": 100.0 * (
            arms["ROT_transferred120"]["J"] - arms["ROT_native160"]["J"])
            / max(abs(arms["ROT_transferred120"]["J"]), 1e-30),
        "fraction_of_the_grid_120_to_160_IoU_loss_recovered": (
            (iou_n - iou_t) / (iou_120 - iou_t)) if abs(iou_120 - iou_t) > 1e-12
            else float("nan"),
        "VERDICT": ("MAP TRANSFER" if recovers else "FORWARD NON-CONVERGENCE"),
        "verdict_meaning": (
            "MAP TRANSFER: solved maps are entangled with the grid they were "
            "solved on, even filtered and even under rotation, and the fix is a "
            "better transfer convention or a solve at the deployment grid. "
            "FORWARD NON-CONVERGENCE: the rotating forward itself is not grid "
            "converged in this metric, and no absolute intersection over union "
            "at any grid should be quoted without a convergence study behind it."
        ),
    }

    res = {
        "case": name, "label": spec_case["label"], "shape": spec["shape"],
        "task": "native re-solve at the hold-out grid, scored by the hold-out "
                "harness at that grid",
        "grid": GRID, "solved_at_grid": GRID,
        "compared_against": str(REPO / "fgm_solve_campaign/out_rot_holdout"
                                / spec_case["holdout_json"]),
        "recipe": {
            "n_gradient_evals_per_start": N_EVALS, "starts": list(starts),
            "winner_start": winner,
            "start_best_J": {k: v["best_J"] for k, v in per_start.items()},
            "n_evals_by_start": {k: len(v["rows"]) for k, v in per_start.items()},
            "rows_by_start": {k: v["rows"] for k, v in per_start.items()},
            "sigma_cells_at_160": sigma,
            "sigma_cells_at_120": float(df.DEFAULT_SIGMA_CELLS),
            "sigma_physical_mm": SIGMA_MM_AT_120,
            "box": list(BOX), "bpp": rr.BPP, "channel": "conductivity only",
            "solve_objective": "shape_objective, binary raster target, the "
                               "production solve convention at grid 120",
            "score_objective": "topopt_objective against the sub-cell area fill, "
                               "the hold-out harness convention",
            "warm_start": {"npz": str(spec["static_map"][0]),
                           "key": spec["static_map"][1],
                           "note": "the grid-120 static map moved to 160 by the "
                                   "production resample, used as a STARTING "
                                   "POINT only; the cold start is run alongside"},
        },
        "n_part_cells": int(pm.sum()), "dt_s": dt, "n_steps": n_steps,
        "calibration": rc.calibration, "chi": rc.chi_info,
        "raster_vs_area": rc.raster_vs_area, "program": prog_info,
        "angles_deg": [float(a) for a in ang],
        "executed_position_histogram": [int(np.sum(pos == k))
                                        for k in range(len(ang))],
        "arms": arms, "verdict": verdict,
        "energy_gate_violations": [a for a, m in arms.items()
                                   if not m["energy_gate"]["PASS"]],
        "wall_s": time.perf_counter() - t0,
    }
    (OUT / f"{name}_native160.json").write_text(
        json.dumps(res, indent=2, default=float))
    np.savez_compressed(
        OUT / f"{name}_native160_maps.npz", part_mask=pm, chi=rc.chi,
        sat_native160=s_native.astype(np.float32),
        sat_native160_cont=s_native_cont.astype(np.float32),
        sat_transferred120=s_transferred.astype(np.float32),
        sat_static=s_static.astype(np.float32),
        v_best=v_best.astype(np.float32), x=rc.case.x, y=rc.case.y, **fields)

    log(f"VERDICT {verdict['VERDICT']}: IoU at 160 is {iou_n:.4f} solved at 160 "
        f"against {iou_t:.4f} transferred from 120 (and {iou_120:.4f} at grid "
        f"120); SOLVED class recovered {recovers}; wall {res['wall_s']:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cross")
