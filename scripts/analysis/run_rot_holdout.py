#!/usr/bin/env python3
"""GRID HOLD-OUT of the ROTATING, INDEXED and DWELL-SCHEDULED solved arms.

Solve at grid 120, score at grid 160. Nothing is re-solved: every arm takes a
dopant map that was already solved on the 120 grid, moves it to the hold-out
grid in the production convention, executes the arm's OWN stored turntable
program there, and re-optimizes only the stop.

FOUR SOLVED-CLASS ARMS, one per invocation:

  cross_index90   90-degree indexing, one quarter turn every 2.0 s, against the
                  matched four-angle solved map. Grid-120 headline
                  IoU 0.9866 (`CONTINUOUS_ROTATION_REPORT.md` Section 7).
  cross_dwell     the co-solved asymmetric-dwell deliverable, its stored
                  150-move 20 s-cycle program. Grid-120 headline IoU 0.9829
                  (`DWELL_SCHEDULE_REPORT.md`).
  star_index90    the same indexing actuator on the star. Grid-120 IoU 0.9527.
  keyhole_cont    the imported keyhole's continuous-rotation co-solve, its
                  stored 448-move twelve-position program. Grid-120 IoU 0.9753
                  (`GEOMETRY_GENERALIZATION_REPORT.md` Section 6.1).

WHAT EACH INVOCATION RUNS. At BOTH grids, so that the grid is the only thing
that changes and the RANKINGS are testable and not just the absolutes:

  ROT_uniform     saturation 1 everywhere, same program, same grid
  ROT_solved      the transferred solved map, same program
  STATIC_uniform  no rotation at all, same grid, same chi, same drive
  STATIC_solved   that shape's best static solved map, transferred, no rotation

Eight runs per arm, four at grid 120 and four at grid 160. The grid-120 runs are
NOT copied from the stored reports: they are re-run here so that the objective,
the target indicator, the drive calibration and the execution model are
identical at the two grids and the only difference is the discretization. Their
agreement with the stored numbers is reported as a separate check.

CONVENTIONS, on every number.
  * Objective J(s, t_stop) = sum over the WHOLE domain of (phi - chi)^2 with chi
    the sub-cell AREA FILL indicator (`adjoint2d.chi_area`), which is grid
    independent to the sampling error it quotes. J is a sum over cells and is
    therefore NOT comparable between grids; read rankings and margins.
    `J_raster_chi` against the binary raster is reported alongside for
    comparison with the stored grid-120 reports.
  * t_stop = argmin of J over that arm's OWN trajectory, horizon flagged.
  * Melted region for intersection over union is phi >= 0.5.
  * Drive recalibrated AT EACH GRID so the UNIFORM STATIC arm absorbs
    500.0 W/m in electrical state B (`FROZEN_CONVENTIONS_2D` Section 3), by one
    electro-quasi-static solve and one exact quadratic rescale, verified.
    Rotating arms are NOT dose matched against each other.
  * Conductivity channel only, 4 bits per pixel inside the part.
  * Execution is the PART-frame march, which interpolates no field and so
    carries none of the production engine's rotation remap error.

Run:
  ./.venv312/bin/python scripts/analysis/run_rot_holdout.py cross_index90
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

from adjoint2d import robust_rot as rr                    # noqa: E402
from adjoint2d.library_solve import shape_config          # noqa: E402
from adjoint2d.pins import load_cfg                       # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_rot_holdout"
OUT_ROT = REPO / "fgm_solve_campaign/out_rot"
OUT_DWELL = REPO / "fgm_solve_campaign/out_dwell"
OUT_LIB = REPO / "fgm_solve_campaign/out_lib"
OUT_INTAKE = REPO / "fgm_solve_campaign/out_intake"

GRIDS = (120, 160)
SOLVED_AT = 120
SOLVED_CLASS_IOU = 0.95


# ---------------------------------------------------------------------------
# the arm registry: what was solved, and the program the machine would run
# ---------------------------------------------------------------------------

def _dwell_program(path: Path, key: str | None = None) -> dict:
    d = json.loads(path.read_text())
    if key is not None:
        d = d["turntable_programs"][key]
    return d


ARMS: dict[str, dict] = {
    "cross_index90": {
        "shape": "cross",
        "label": "cross, 90-degree indexing at 2.0 s, solved four-angle map",
        "source": "library",
        "angles_deg": [0.0, 90.0, 180.0, 270.0],
        "program": {"kind": "index", "interval_s": 2.0},
        "map": (OUT_ROT / "cross_rotavg_step90_maps.npz", "sat_cont"),
        "static_map": (OUT_LIB / "cross_maps.npz", "A1_cont"),
        "stored_120": {"report": "CONTINUOUS_ROTATION_REPORT.md Section 7, "
                                 "production engine, binary raster chi",
                       "J": 34.8177, "IoU": 0.9866, "arm": "I90_avg4angle"},
    },
    "cross_dwell": {
        "shape": "cross",
        "label": "cross, co-solved asymmetric dwell, stored 20 s-cycle program",
        "source": "library",
        "angles_deg": None,          # taken from the stored program
        "program": {"kind": "stored",
                    "path": OUT_DWELL / "cross_turntable_deliverable.json"},
        "map": (OUT_DWELL / "cross_dwell_maps.npz",
                "sat_D_refined_4bpp_discovered_lib"),
        "static_map": (OUT_LIB / "cross_maps.npz", "A1_cont"),
        "stored_120": {"report": "DWELL_SCHEDULE_REPORT.md Section 5, "
                                 "time-resolved deliverable, binary raster chi",
                       "J": 34.04, "IoU": 0.9829, "arm": "DELIVERABLE"},
    },
    "star_index90": {
        "shape": "star",
        "label": "star, 90-degree indexing at 2.0 s, solved four-angle map",
        "source": "library",
        "angles_deg": [0.0, 90.0, 180.0, 270.0],
        "program": {"kind": "index", "interval_s": 2.0},
        "map": (OUT_ROT / "star_rotavg_step90_maps.npz", "sat_cont"),
        "static_map": (OUT_LIB / "star_maps.npz", "A1_cont"),
        "stored_120": {"report": "CONTINUOUS_ROTATION_REPORT.md Section 7, "
                                 "production engine, binary raster chi",
                       "J": 24.46, "IoU": 0.9527, "arm": "I90_avg4angle"},
    },
    "keyhole_cont": {
        "shape": "keyhole",
        "label": "keyhole, continuous-rotation co-solve, stored 448-move program",
        "source": "novel_polygon",
        "angles_deg": None,
        "program": {"kind": "stored",
                    "path": OUT_INTAKE / "keyhole_novel.json",
                    "key": "continuous"},
        "map": (OUT_INTAKE / "keyhole_maps.npz", "continuous_cont"),
        "static_map": (OUT_INTAKE / "keyhole_maps.npz", "static_cont"),
        "stored_120": {"report": "GEOMETRY_GENERALIZATION_REPORT.md Section 6.1, "
                                 "quasi-static twelve-angle average, area-fill chi",
                       "J": 8.0759, "IoU": 0.9753, "arm": "A_continuous_4bpp"},
    },
}


def base_cfg(spec: dict, n_grid: int) -> dict:
    """The configuration this shape is solved and scored in, at `n_grid`.

    The library shapes carry a calibrated configuration on disk. The keyhole is
    an IMPORTED polygon and carries none, so its configuration is rebuilt from
    the same vertex list the solve used, at the requested grid, through the same
    intake the solve used. Rebuilding rather than resampling is the point: the
    part mask and chi are regenerated by the production domain builder at the
    hold-out grid, exactly as they are for the library shapes.
    """
    if spec["source"] == "library":
        return load_cfg(shape_config(spec["shape"]))
    from adjoint2d import geometry_intake as gi
    from novel_shapes import NOVEL
    poly = NOVEL[spec["shape"]]()
    return gi.from_polygon(poly, grid=int(n_grid), name=spec["shape"]).cfg


def program_for(spec: dict, dt_s: float, n_steps: int
                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """(candidate angles, per-outer-step position index, provenance)."""
    pr = spec["program"]
    if pr["kind"] == "index":
        ang = np.asarray(spec["angles_deg"], dtype=float)
        pos = rr.index_program_positions(len(ang), pr["interval_s"], dt_s,
                                         n_steps)
        info = {"kind": "fixed-interval indexing", "interval_s": pr["interval_s"],
                "positions_deg": [float(a) for a in ang],
                "n_moves": int(np.sum(np.diff(pos) != 0)) + 1,
                "source": "CONTINUOUS_ROTATION_REPORT.md Section 7"}
        return ang, pos, info
    prog = _dwell_program(pr["path"], pr.get("key"))
    ang = np.asarray(prog["positions_deg"], dtype=float)
    pos = rr.moves_to_positions(prog["moves"], ang, dt_s, n_steps)
    info = {"kind": "stored turntable program", "source": str(pr["path"]),
            "key": pr.get("key"), "cycle_time_s": prog["cycle_time_s"],
            "control_step_s": prog["control_step_s"],
            "positions_deg": [float(a) for a in ang],
            "n_moves": int(prog["n_moves"]),
            "requested_dwell_fraction": prog["realized_dwell_fraction"]}
    return ang, pos, info


# ---------------------------------------------------------------------------
# one grid
# ---------------------------------------------------------------------------

def run_one_grid(name: str, spec: dict, n_grid: int) -> dict:
    tag = f"{name}@{n_grid}"
    t0 = time.perf_counter()
    cfg = base_cfg(spec, n_grid)

    # the outer step and the horizon fix how a wall-clock program expands, so
    # they are read from a plain case before the kernel (which is the expensive
    # object) is built.
    from adjoint2d.pins import build_case
    probe = build_case(rr.cfg_at_grid(cfg, n_grid))
    dt = float(probe.pins.dt)
    n_steps = int(probe.pins.n_steps)
    ang, pos, prog_info = program_for(spec, dt, n_steps)
    del probe

    rc = rr.build_rot_case(cfg, n_grid, ang, recalibrate=True)
    cal = rc.calibration
    print(f"[{tag}] drive {cal['voltage_v_pinned']:.1f} V gives "
          f"{cal['p_at_pinned_v_w_per_m']:.1f} W/m at this grid; recalibrated to "
          f"{cal['voltage_v']:.1f} V, verified {cal['p_verified_w_per_m']:.2f} W/m",
          flush=True)
    print(f"[{tag}] part cells {int(rc.case.part_mask.sum())}, chi area "
          f"{rc.chi_info['area_m2']:.6e} m2, raster minus area "
          f"{rc.raster_vs_area['area_rel_delta'] * 100:+.3f} %", flush=True)
    pm = rc.part_mask
    ones = np.ones(pm.shape)

    maps120 = np.load(spec["map"][0])
    s_solved = rr.transfer_map(np.asarray(maps120[spec["map"][1]], dtype=float),
                               pm, bpp=rr.BPP)
    smaps120 = np.load(spec["static_map"][0])
    s_static = rr.transfer_map(
        np.asarray(smaps120[spec["static_map"][1]], dtype=float), pm, bpp=rr.BPP)

    # The REALIZED dwell fraction of the executed program is the weight vector
    # of its own quasi-static limit, so the two evaluations below differ ONLY in
    # whether the cycle is treated as infinitely fast. That is what separates
    # "the solved arm does not survive the grid" from "the solved arm was never
    # evaluated at its own finite cycle time".
    w_real = np.array([np.sum(pos == k) for k in range(len(ang))], dtype=float)
    w_real /= w_real.sum()

    arms: dict[str, dict] = {}
    arms["ROT_uniform"] = rr.score_program(rc, ones, pos, n_steps)
    rr.log_row(tag, "ROT_uniform", arms["ROT_uniform"])
    arms["ROT_solved"] = rr.score_program(rc, s_solved, pos, n_steps)
    rr.log_row(tag, "ROT_solved", arms["ROT_solved"])
    arms["QS_uniform"] = rr.score_quasistatic(rc, ones, weights=w_real)
    rr.log_row(tag, "QS_uniform", arms["QS_uniform"])
    arms["QS_solved"] = rr.score_quasistatic(rc, s_solved, weights=w_real)
    rr.log_row(tag, "QS_solved", arms["QS_solved"])
    # The DESIGN model: equal angle weights, which is the averaged kernel every
    # one of these maps was solved against. It differs from `QS_solved` only
    # where the emitted machine program cannot realize equal dwell on the
    # control-step grid, and that gap belongs to the program, not to the grid.
    arms["QS_solved_equalW"] = rr.score_quasistatic(rc, s_solved, weights=None)
    rr.log_row(tag, "QS_solved_equalW", arms["QS_solved_equalW"])
    arms["QS_uniform_equalW"] = rr.score_quasistatic(rc, ones, weights=None)
    rr.log_row(tag, "QS_uniform_equalW", arms["QS_uniform_equalW"])
    arms["STATIC_uniform"] = rr.score_static(rc, ones)
    rr.log_row(tag, "STATIC_uniform", arms["STATIC_uniform"])
    arms["STATIC_solved"] = rr.score_static(rc, s_static)
    rr.log_row(tag, "STATIC_solved", arms["STATIC_solved"])

    fields = {}
    for k, a in arms.items():
        a.pop("J_at_first_step", None)
        ph = a.pop("_phi_at_stop", None)
        if ph is not None:
            fields[f"phi_{k}"] = np.asarray(ph, dtype=np.float32)

    return {
        "n_grid": int(n_grid),
        "n_part_cells": int(pm.sum()),
        "dt_s": dt, "n_steps": n_steps,
        "calibration": rc.calibration,
        "chi": rc.chi_info,
        "raster_vs_area": rc.raster_vs_area,
        "program": prog_info,
        "angles_deg": [float(a) for a in ang],
        "executed_position_histogram": [int(np.sum(pos == k))
                                        for k in range(len(ang))],
        "executed_dwell_fraction": [float(v) for v in w_real],
        "arms": arms,
        "maps": {"solved": {"npz": str(spec["map"][0]), "key": spec["map"][1]},
                 "static": {"npz": str(spec["static_map"][0]),
                            "key": spec["static_map"][1]}},
        "wall_s": time.perf_counter() - t0,
    }, {"part_mask": pm, "chi": rc.chi, "sat_ROT_solved": s_solved,
        "sat_STATIC_solved": s_static, "x": rc.case.x, "y": rc.case.y, **fields}


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------

def verdict(res: dict) -> dict:
    g0, g1 = res["grids"][str(SOLVED_AT)], res["grids"][str(GRIDS[-1])]
    a0, a1 = g0["arms"], g1["arms"]

    def marg(a, ref):
        return (a[ref]["J"] - a["ROT_solved"]["J"]) / max(abs(a[ref]["J"]), 1e-30)

    out = {
        "IoU_at_120": a0["ROT_solved"]["IoU"],
        "IoU_at_160": a1["ROT_solved"]["IoU"],
        "IoU_area_at_120": a0["ROT_solved"]["IoU_area"],
        "IoU_area_at_160": a1["ROT_solved"]["IoU_area"],
        "solved_class_at_120": bool(a0["ROT_solved"]["IoU"] >= SOLVED_CLASS_IOU),
        "solved_class_at_160": bool(a1["ROT_solved"]["IoU"] >= SOLVED_CLASS_IOU),
        "dJ_vs_rot_uniform_120": marg(a0, "ROT_uniform"),
        "dJ_vs_rot_uniform_160": marg(a1, "ROT_uniform"),
        "dJ_vs_static_solved_120": marg(a0, "STATIC_solved"),
        "dJ_vs_static_solved_160": marg(a1, "STATIC_solved"),
        "dJ_vs_static_uniform_120": marg(a0, "STATIC_uniform"),
        "dJ_vs_static_uniform_160": marg(a1, "STATIC_uniform"),
    }
    # The finite cycle time, isolated: same map, same grid, same drive, the only
    # difference being whether the cycle is treated as infinitely fast.
    out["quasistatic_minus_program_J_120"] = (
        (a0["QS_solved"]["J"] - a0["ROT_solved"]["J"])
        / max(abs(a0["ROT_solved"]["J"]), 1e-30))
    out["quasistatic_minus_program_J_160"] = (
        (a1["QS_solved"]["J"] - a1["ROT_solved"]["J"])
        / max(abs(a1["ROT_solved"]["J"]), 1e-30))
    out["QS_IoU_at_120"] = a0["QS_solved"]["IoU"]
    out["QS_IoU_at_160"] = a1["QS_solved"]["IoU"]
    out["QS_solved_class_at_120"] = bool(a0["QS_solved"]["IoU"] >= SOLVED_CLASS_IOU)
    out["QS_solved_class_at_160"] = bool(a1["QS_solved"]["IoU"] >= SOLVED_CLASS_IOU)
    # the program-realization gap: equal design dwell against the dwell the
    # emitted program can actually realize on the control-step grid
    out["program_realization_gap_J_120"] = (
        (a0["QS_solved"]["J"] - a0["QS_solved_equalW"]["J"])
        / max(abs(a0["QS_solved_equalW"]["J"]), 1e-30))
    out["program_realization_gap_J_160"] = (
        (a1["QS_solved"]["J"] - a1["QS_solved_equalW"]["J"])
        / max(abs(a1["QS_solved_equalW"]["J"]), 1e-30))
    out["equalW_IoU_at_120"] = a0["QS_solved_equalW"]["IoU"]
    out["equalW_IoU_at_160"] = a1["QS_solved_equalW"]["IoU"]
    out["equalW_solved_class_at_120"] = bool(
        a0["QS_solved_equalW"]["IoU"] >= SOLVED_CLASS_IOU)
    out["equalW_solved_class_at_160"] = bool(
        a1["QS_solved_equalW"]["IoU"] >= SOLVED_CLASS_IOU)

    for ref in ("ROT_uniform", "QS_solved", "QS_solved_equalW",
                "STATIC_solved", "STATIC_uniform"):
        w0 = bool(a0["ROT_solved"]["J"] < a0[ref]["J"])
        w1 = bool(a1["ROT_solved"]["J"] < a1[ref]["J"])
        i0 = bool(a0["ROT_solved"]["IoU"] > a0[ref]["IoU"])
        i1 = bool(a1["ROT_solved"]["IoU"] > a1[ref]["IoU"])
        out[f"beats_{ref}_on_J_120"] = w0
        out[f"beats_{ref}_on_J_160"] = w1
        out[f"ranking_preserved_J_vs_{ref}"] = bool(w0 == w1)
        out[f"beats_{ref}_on_IoU_120"] = i0
        out[f"beats_{ref}_on_IoU_160"] = i1
        out[f"ranking_preserved_IoU_vs_{ref}"] = bool(i0 == i1)
    out["class_changes"] = bool(out["solved_class_at_120"]
                                != out["solved_class_at_160"])
    out["all_rankings_preserved_on_J"] = bool(all(
        out[f"ranking_preserved_J_vs_{r}"]
        for r in ("ROT_uniform", "STATIC_solved", "STATIC_uniform")))
    out["reproduction_of_stored_120"] = {
        "stored": res["stored_120_reference"],
        "this_harness_J_raster_at_120": a0["ROT_solved"]["J_raster_chi"],
        "this_harness_IoU_at_raster_stop_at_120":
            a0["ROT_solved"]["IoU_at_raster_stop"],
        "this_harness_J_area_chi_at_120": a0["ROT_solved"]["J"],
        "note": "the stored number and this one are the same arm only when the "
                "stored one was produced by the same execution model; the "
                "production engine's lab-frame turntable and this part-frame "
                "march are not the same object for a part whose mask changes "
                "at a rotation event",
    }
    return out


def main(name: str) -> None:
    if name not in ARMS:
        raise SystemExit(f"unknown arm {name!r}; have {sorted(ARMS)}")
    spec = ARMS[name]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = {
        "arm": name, "label": spec["label"], "shape": spec["shape"],
        "task": "grid hold-out of a rotating / dwell arm: solve at 120, score at 160",
        "solved_at_grid": SOLVED_AT, "holdout_grid": GRIDS[-1],
        "objective": "J = sum over the whole domain of (phi - chi_area)^2; "
                     "J_raster_chi against the binary raster reported alongside",
        "stop": "argmin of J over the arm's own trajectory; horizon flagged",
        "execution": "part-frame march of the stored program; no field "
                     "interpolation, so no rotation remap error",
        "drive": "recalibrated at each grid so the uniform STATIC arm absorbs "
                 "500.0 W/m in state B; rotating arms not dose matched",
        "channel": "conductivity only, 4 bits per pixel in part",
        "resample_convention": "scipy.ndimage.zoom order 1 then clip, "
                               "rfam_eqs_coupled.py:374-380, then re-quantized "
                               "in part by printability.quantize_in_part",
        "stored_120_reference": spec["stored_120"],
        "grids": {}, "store": {},
    }
    store: dict[str, np.ndarray] = {}
    for n in GRIDS:
        g, arrs = run_one_grid(name, spec, n)
        res["grids"][str(n)] = g
        for k, v in arrs.items():
            store[f"g{n}_{k}"] = v
    res["verdict"] = verdict(res)
    res["energy_gate_violations"] = [
        f"{n}:{a}" for n in res["grids"]
        for a, m in res["grids"][n]["arms"].items()
        if not m["energy_gate"]["PASS"]]
    res["wall_s"] = time.perf_counter() - t0
    res.pop("store")

    (OUT / f"{name}.json").write_text(json.dumps(res, indent=2, default=float))
    np.savez_compressed(OUT / f"{name}_maps.npz", **store)
    v = res["verdict"]
    print(f"[{name}] VERDICT IoU {v['IoU_at_120']:.4f} at 120 -> "
          f"{v['IoU_at_160']:.4f} at 160; SOLVED class "
          f"{v['solved_class_at_120']} -> {v['solved_class_at_160']}; "
          f"rankings on J preserved {v['all_rankings_preserved_on_J']}; "
          f"wall {res['wall_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
