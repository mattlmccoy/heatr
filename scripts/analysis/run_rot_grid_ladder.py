#!/usr/bin/env python3
"""ROTATING FORWARD GRID LADDER: is the two-dimensional forward model grid
converged on the ROTATING actuator, and at what grid does the cross's rotating
arm intersection-over-union stabilize?

WHY THIS EXISTS. `ROTATING_HOLDOUT_REPORT.md` Section 6 measured that the
cross's ROTATING uniform-dopant arm, which contains no solved dopant map at all,
moves 0.0971 intersection-over-union points between grid 120 and grid 160, while
its STATIC uniform arm moves 0.0637 the OTHER way. `HOLDOUT_FOLLOWUP_REPORT.md`
Section 3.2 then showed that a dopant map SOLVED natively at grid 160 recovers
only 42.4 percent of the cross's lost fidelity, and named the failure FORWARD
NON-CONVERGENCE at that solve budget. Both reports name this ladder as the
blocking measurement. Until it exists, no absolute intersection-over-union on a
rotating cross arm is quotable at any grid.

WHAT THIS RUNS. FORWARD RUNS ONLY. Nothing is solved here and no gradient is
computed, so no finite-difference gate is re-run; the standing subgradient gate
of the previous reports applies unchanged to the one arm that carries a solved
map.

Per shape, at each grid in the ladder:

  ROT_uniform     the rotating actuator with a UNIFORM dopant map, saturation 1
                  everywhere. This is the primary arm: it contains no solved map
                  and no transfer, so anything it does across the ladder is the
                  discretization and nothing else.
  STATIC_uniform  the same uniform map with the part held still, the contrast
                  arm. Its 120 to 160 movement is already known (+0.0637
                  intersection over union on the cross) and it is included so
                  the ladder says whether the rotating and static forwards
                  converge at the same rate and in the same direction.
  ROT_transfer    OPTIONAL (`--with-map`). The four-angle map solved at grid 120
                  moved to this grid by the production resample and co-rotated,
                  so the map's transfer curve rides the same ladder. This arm
                  DOES carry a solved map and is therefore NOT evidence about
                  the forward on its own; it is here to separate the transfer
                  curve from the forward curve.
  QS_uniform      the quasi-static angle average of the uniform arm, the
                  infinitely-fast-cycle limit. Free relative to the march and it
                  says whether any grid movement is the finite cycle time.

CONVENTIONS, identical to `ROTATING_HOLDOUT_REPORT.md` and carried on every
number here.

  * J(s, t_stop) = sum over the WHOLE domain of (phi - chi)^2 with chi the
    sub-cell AREA FILL indicator (`adjoint2d.chi_area`). J is a SUM OVER CELLS
    and is therefore NOT comparable between grids. A cell-count-normalized
    J_per_part_cell is emitted alongside it so the ladder has one J-like
    quantity that can be read across grids at all; it is a normalization and not
    a convergence proof, and the report says so.
  * t_stop = argmin of J over that arm's OWN trajectory, horizon flagged.
  * The melted region for intersection over union, growth and under-melt is
    phi >= 0.5. Intersection over union is against the BINARY part mask, which
    is the reading the SOLVED class threshold of 0.95 has always been quoted on.
  * The drive is RECALIBRATED AT EACH GRID by one electro-quasi-static solve and
    one exact quadratic rescale, verified by a second solve, so that the uniform
    STATIC arm absorbs 500.0 W/m in electrical state B
    (`FROZEN_CONVENTIONS_2D` Section 3). Rotating arms are NOT dose matched.
  * Conductivity channel only, 4 bits per pixel inside the part for the
    transferred map, dopant held at the nominal 1 outside the part.
  * Execution is the PART-frame march, which interpolates no field and so
    carries none of the production engine's rotation remap error.

COST NOTE, measured and not assumed. The outer step (0.5 s) and the horizon
(1500 steps, 750.0 s) are pinned in the configuration and do NOT move with the
grid, but the explicit thermal substep count is CFL limited and scales as the
inverse square of the cell size, so the march cost scales roughly as the FOURTH
power of the grid number. The ladder therefore writes its JSON after EVERY grid,
logs the wall time of every arm, and the driver prints a projection of the
remaining grids as soon as the first grid is finished.

Run:
  ./.venv312/bin/python scripts/analysis/run_rot_grid_ladder.py cross \
      --grids 96 120 160 200 --with-map
"""
from __future__ import annotations

import argparse
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
from adjoint2d.pins import build_case, load_cfg           # noqa: E402
from rot_ladder_variants import snap_cross_cfg            # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_rot_ladder"
LOGS = REPO / "fgm_solve_campaign/logs_rot_ladder"
OUT_ROT = REPO / "fgm_solve_campaign/out_rot"
OUT_HOLDOUT = REPO / "fgm_solve_campaign/out_rot_holdout"
OUT_INTAKE = REPO / "fgm_solve_campaign/out_intake"

LADDER = (96, 120, 160, 200)
SOLVED_CLASS_IOU = 0.95

# The stored numbers this ladder must reproduce before any new grid is trusted.
# Source: ROTATING_HOLDOUT_REPORT.md Sections 4.1 and 4.4, re-read from the
# stored JSON at run time rather than transcribed, so a transcription error
# cannot pass the gate.
REPRO_SOURCE = {
    "cross": (OUT_HOLDOUT / "cross_index90.json",
              "ROTATING_HOLDOUT_REPORT.md Section 4.1"),
    "keyhole": (OUT_HOLDOUT / "keyhole_cont_fixedprog.json",
                "HOLDOUT_FOLLOWUP_REPORT.md Section 2.4"),
}
REPRO_ARMS = ("ROT_uniform", "STATIC_uniform")
REPRO_TOL_J = 5e-3        # relative
REPRO_TOL_IOU = 5e-4      # absolute


SHAPES: dict[str, dict] = {
    "cross": {
        "shape": "cross",
        "label": "cross, 90-degree indexing at 2.0 s, uniform dopant",
        "source": "library",
        "angles_deg": [0.0, 90.0, 180.0, 270.0],
        "program": {"kind": "index", "interval_s": 2.0},
        "map": (OUT_ROT / "cross_rotavg_step90_maps.npz", "sat_cont"),
    },
    "keyhole": {
        "shape": "keyhole",
        "label": "keyhole, continuous rotation, divisor-aware twelve-position "
                 "program, uniform dopant",
        "source": "novel_polygon",
        "angles_deg": None,
        "program": {"kind": "stored",
                    "path": OUT_HOLDOUT / "keyhole_program_fixed.json",
                    "keypath": ["program"]},
        "map": (OUT_INTAKE / "keyhole_maps.npz", "continuous_cont"),
    },
}


def base_cfg(spec: dict, n_grid: int) -> dict:
    """The configuration this shape is scored in, at `n_grid`.

    Identical to `run_rot_holdout.base_cfg`: library shapes carry a calibrated
    configuration on disk; the keyhole is an imported polygon whose intake is
    re-run at the requested grid so its mask and chi are regenerated by the
    production domain builder exactly as the library shapes' are.
    """
    if spec["source"] == "library":
        return load_cfg(shape_config(spec["shape"]))
    from adjoint2d import geometry_intake as gi
    from novel_shapes import NOVEL
    poly = NOVEL[spec["shape"]]()
    return gi.from_polygon(poly, grid=int(n_grid), name=spec["shape"]).cfg


def program_for(spec: dict, dt_s: float, n_steps: int):
    """(candidate angles, per-outer-step position index, provenance).

    The stored turntable program is a wall-clock object and is carried across
    every grid UNCHANGED, which is the whole point: the machine runs the same
    program whatever the simulation grid.
    """
    pr = spec["program"]
    if pr["kind"] == "index":
        ang = np.asarray(spec["angles_deg"], dtype=float)
        pos = rr.index_program_positions(len(ang), pr["interval_s"], dt_s,
                                         n_steps)
        info = {"kind": "fixed-interval indexing",
                "interval_s": pr["interval_s"],
                "positions_deg": [float(a) for a in ang],
                "n_moves": int(np.sum(np.diff(pos) != 0)) + 1}
        return ang, pos, info
    d = json.loads(Path(pr["path"]).read_text())
    for k in pr.get("keypath", []):
        d = d[k]
    if pr.get("key"):
        d = d["turntable_programs"][pr["key"]]
    ang = np.asarray(d["positions_deg"], dtype=float)
    pos = rr.moves_to_positions(d["moves"], ang, dt_s, n_steps)
    info = {"kind": "stored turntable program", "source": str(pr["path"]),
            "cycle_time_s": d["cycle_time_s"],
            "control_step_s": d["control_step_s"],
            "positions_deg": [float(a) for a in ang],
            "n_moves": int(d["n_moves"])}
    return ang, pos, info


def run_one_grid(name: str, spec: dict, n_grid: int, with_map: bool,
                 snap: bool = False) -> dict:
    tag = f"{name}@{n_grid}"
    t0 = time.perf_counter()
    cfg = base_cfg(spec, n_grid)
    snap_info = None
    if snap:
        # VARIANT A. Both cross boundaries moved to the nearest whole cell
        # multiple at THIS grid, so the raster represents them exactly and the
        # sub-cell area-fill target collapses onto the binary raster. The
        # geometry deltas are recorded because they are the price of the
        # variant: the part itself moves by up to half a cell.
        cfg, snap_info = snap_cross_cfg(cfg, int(n_grid))
        print(f"[{tag}] SNAP limb half {snap_info['limb_half_m_original'] * 1e3:.4f}"
              f" -> {snap_info['limb_half_m'] * 1e3:.4f} mm "
              f"({snap_info['limb_cells']} cells, "
              f"{snap_info['d_limb_frac_of_cell']:+.3f} cell, "
              f"{snap_info['d_limb_pct_of_dimension']:+.2f} %); arm half "
              f"{snap_info['arm_half_m_original'] * 1e3:.4f} -> "
              f"{snap_info['arm_half_m'] * 1e3:.4f} mm "
              f"({snap_info['arm_cells']} cells, "
              f"{snap_info['d_arm_frac_of_cell']:+.3f} cell, "
              f"{snap_info['d_arm_pct_of_dimension']:+.2f} %)"
              + ("  NO-OP" if snap_info["is_no_op"] else ""), flush=True)

    probe = build_case(rr.cfg_at_grid(cfg, n_grid))
    dt = float(probe.pins.dt)
    n_steps = int(probe.pins.n_steps)
    n_substeps = int(probe.pins.n_substeps)
    del probe
    ang, pos, prog_info = program_for(spec, dt, n_steps)

    rc = rr.build_rot_case(cfg, n_grid, ang, recalibrate=True)
    cal = rc.calibration
    print(f"[{tag}] drive {cal['voltage_v_pinned']:.1f} V gives "
          f"{cal['p_at_pinned_v_w_per_m']:.1f} W/m at this grid; recalibrated to "
          f"{cal['voltage_v']:.1f} V, verified {cal['p_verified_w_per_m']:.2f} W/m",
          flush=True)
    print(f"[{tag}] part cells {int(rc.case.part_mask.sum())}, chi area "
          f"{rc.chi_info['area_m2']:.6e} m2, raster minus area "
          f"{rc.raster_vs_area['area_rel_delta'] * 100:+.3f} %, "
          f"thermal substeps per outer step {n_substeps}", flush=True)

    pm = rc.part_mask
    ones = np.ones(pm.shape)
    w_real = np.array([np.sum(pos == k) for k in range(len(ang))], dtype=float)
    w_real /= w_real.sum()

    arms: dict[str, dict] = {}
    arms["ROT_uniform"] = rr.score_program(rc, ones, pos, n_steps)
    rr.log_row(tag, "ROT_uniform", arms["ROT_uniform"])
    arms["STATIC_uniform"] = rr.score_static(rc, ones)
    rr.log_row(tag, "STATIC_uniform", arms["STATIC_uniform"])
    arms["QS_uniform"] = rr.score_quasistatic(rc, ones, weights=w_real)
    rr.log_row(tag, "QS_uniform", arms["QS_uniform"])
    if with_map:
        maps = np.load(spec["map"][0])
        s = rr.transfer_map(np.asarray(maps[spec["map"][1]], dtype=float),
                            pm, bpp=rr.BPP)
        arms["ROT_transfer"] = rr.score_program(rc, s, pos, n_steps)
        rr.log_row(tag, "ROT_transfer", arms["ROT_transfer"])

    n_part = int(pm.sum())
    fields = {}
    for k, a in arms.items():
        a.pop("J_at_first_step", None)
        ph = a.pop("_phi_at_stop", None)
        if ph is not None:
            fields[f"phi_{k}"] = np.asarray(ph, dtype=np.float32)
        a["J_per_part_cell"] = float(a["J"]) / max(n_part, 1)

    return {
        "n_grid": int(n_grid),
        "snap": snap_info,
        "n_part_cells": n_part,
        "dx_m": float(rc.case.dx), "dy_m": float(rc.case.dy),
        "dt_s": dt, "n_steps": n_steps, "n_substeps": n_substeps,
        "calibration": rc.calibration,
        "chi": rc.chi_info,
        "raster_vs_area": rc.raster_vs_area,
        "program": prog_info,
        "angles_deg": [float(a) for a in ang],
        "executed_position_histogram": [int(np.sum(pos == k))
                                        for k in range(len(ang))],
        "executed_dwell_fraction": [float(v) for v in w_real],
        "arms": arms,
        "wall_s": time.perf_counter() - t0,
    }, {"part_mask": pm, "chi": rc.chi, "x": rc.case.x, "y": rc.case.y,
        **fields}


def reproduction_gate(name: str, grids: dict, snap: bool = False) -> dict:
    """Reproduce the stored 120 and 160 numbers before any new grid is trusted.

    Read from the stored JSON, not transcribed. A grid the stored file does not
    contain is reported as NOT CHECKED rather than silently passing.

    When the geometry has been DELIBERATELY changed (variant A), the stored
    numbers are numbers about a different part and the gate does not apply. It
    says so in its own status rather than failing, and rather than passing: the
    variant carries its own identity control instead, a snapped run at a grid
    where the snap is a no-op against an unsnapped run at the same grid.
    """
    if snap:
        return {"status": "NOT APPLICABLE, the geometry was deliberately "
                          "modified by the snap, so the stored numbers "
                          "describe a different part",
                "ALL_PASS": None, "n_comparisons_actually_made": 0,
                "substitute_control": "a snapped run at a grid where the snap "
                                      "is a no-op must equal an unsnapped run "
                                      "at that grid; see the grid-181 identity "
                                      "gate"}
    path, ref = REPRO_SOURCE[name]
    if not path.exists():
        return {"status": "NO STORED FILE", "path": str(path)}
    stored = json.loads(path.read_text())
    rows = []
    ok = True
    n_checked = 0
    for g in ("120", "160"):
        if g not in stored.get("grids", {}) or g not in grids:
            rows.append({"grid": int(g), "status": "NOT CHECKED"})
            continue
        for arm in REPRO_ARMS:
            sa = stored["grids"][g]["arms"].get(arm)
            na = grids[g]["arms"].get(arm)
            if sa is None or na is None:
                rows.append({"grid": int(g), "arm": arm,
                             "status": "NOT CHECKED"})
                continue
            dj = abs(na["J"] - sa["J"]) / max(abs(sa["J"]), 1e-30)
            di = abs(na["IoU"] - sa["IoU"])
            p = bool(dj <= REPRO_TOL_J and di <= REPRO_TOL_IOU)
            ok = ok and p
            n_checked += 1
            rows.append({"grid": int(g), "arm": arm,
                         "stored_J": sa["J"], "this_J": na["J"],
                         "rel_dJ": dj,
                         "stored_IoU": sa["IoU"], "this_IoU": na["IoU"],
                         "abs_dIoU": di, "PASS": p})
    # An empty check is NOT a pass. Absence of evidence must be visually and
    # semantically distinct from verified-good, or the gate is a false green.
    return {"source": str(path), "report": ref, "rows": rows,
            "n_comparisons_actually_made": n_checked,
            "ALL_PASS": (bool(ok) if n_checked else None),
            "status": ("PASS" if (n_checked and ok)
                       else "FAIL" if n_checked else "NOT CHECKED, no overlap "
                       "between this ladder and the stored grids"),
            "tolerance": {"rel_J": REPRO_TOL_J, "abs_IoU": REPRO_TOL_IOU}}


def convergence(grids: dict, arm: str) -> dict:
    """Successive-grid differences and, where the sequence supports it, a
    Richardson-style observed order and extrapolated limit.

    The ladder 96, 120, 160, 200 is NOT a constant refinement ratio, so the
    textbook three-grid Richardson formula does not apply directly. The observed
    order is fitted instead on log|IoU(n) - IoU(n_finest)| against log(n), and
    it is reported ONLY when the successive differences are monotone decreasing
    in magnitude and of one sign. Otherwise the verdict is "not in the
    asymptotic range", which is a real answer and not a failure.
    """
    ns = sorted(int(g) for g in grids if arm in grids[g]["arms"])
    if len(ns) < 2:
        return {"status": "too few grids"}
    iou = [float(grids[str(n)]["arms"][arm]["IoU"]) for n in ns]
    jpc = [float(grids[str(n)]["arms"][arm]["J_per_part_cell"]) for n in ns]
    d = [iou[i + 1] - iou[i] for i in range(len(iou) - 1)]
    steps = [{"from": ns[i], "to": ns[i + 1], "dIoU": d[i],
              "abs_dIoU": abs(d[i])} for i in range(len(d))]
    monotone = all(abs(d[i + 1]) < abs(d[i]) for i in range(len(d) - 1))
    one_sign = all(np.sign(x) == np.sign(d[0]) and x != 0.0 for x in d)
    out = {"grids": ns, "IoU": iou, "J_per_part_cell": jpc,
           "successive": steps,
           "differences_shrink_monotonically": bool(monotone),
           "differences_all_one_sign": bool(one_sign),
           "asymptotic_range": bool(monotone and one_sign)}
    if out["asymptotic_range"] and len(ns) >= 3:
        ref = iou[-1]
        e = np.array([abs(v - ref) for v in iou[:-1]])
        n = np.array(ns[:-1], dtype=float)
        good = e > 0
        if good.sum() >= 2:
            p = np.polyfit(np.log(n[good]), np.log(e[good]), 1)
            out["observed_order_p"] = float(-p[0])
            out["observed_order_note"] = (
                "fitted on the error against the FINEST grid, which biases the "
                "order low; it is an indication, not a proof")
    # last-step size, the number the verdict is actually read from
    out["last_step_abs_dIoU"] = abs(d[-1])
    out["first_step_abs_dIoU"] = abs(d[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", choices=sorted(SHAPES))
    ap.add_argument("--grids", type=int, nargs="+", default=list(LADDER))
    ap.add_argument("--with-map", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--snap-geometry", action="store_true",
                    help="VARIANT A: snap both cross boundaries to whole cell "
                         "multiples at every grid")
    a = ap.parse_args()

    name = a.shape
    spec = SHAPES[name]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or name
    t0 = time.perf_counter()

    res = {
        "arm": name, "label": spec["label"], "shape": spec["shape"],
        "task": "ROTATING FORWARD GRID LADDER: forward runs only, no solve, "
                "uniform dopant on the primary arm",
        "ladder": [int(g) for g in a.grids],
        "with_transferred_map": bool(a.with_map),
        "snap_geometry": bool(a.snap_geometry),
        "variant": ("VARIANT A, snapped geometry: both cross boundaries moved "
                    "to the nearest whole cell multiple at each grid"
                    if a.snap_geometry else
                    "original geometry, unchanged from the stored ladder"),
        "objective": "J = sum over the whole domain of (phi - chi_area)^2; "
                     "J is a sum over cells and is NOT comparable between "
                     "grids; J_per_part_cell is emitted as a normalization "
                     "and is not itself a convergence proof",
        "stop": "argmin of J over the arm's own trajectory; horizon flagged",
        "execution": "part-frame march of the stored program; no field "
                     "interpolation, so no rotation remap error",
        "drive": "recalibrated at EACH grid so the uniform STATIC arm absorbs "
                 "500.0 W/m in state B; rotating arms not dose matched",
        "channel": "conductivity only; the transferred map is 4 bits per pixel "
                   "in part, nominal 1 outside",
        "grids": {},
    }
    store: dict[str, np.ndarray] = {}
    walls: dict[int, float] = {}
    for i, n in enumerate(a.grids):
        g, arrs = run_one_grid(name, spec, int(n), a.with_map,
                               snap=a.snap_geometry)
        res["grids"][str(int(n))] = g
        walls[int(n)] = g["wall_s"]
        for k, v in arrs.items():
            store[f"g{n}_{k}"] = v
        # project the rest of the ladder from the grids measured so far, using
        # the fourth-power cost model the CFL substep count implies. MEASURED
        # after the first grid, as the run rules require.
        done = sorted(walls)
        base_n, base_w = done[-1], walls[done[-1]]
        left = [int(x) for x in a.grids[i + 1:]]
        proj = {int(x): base_w * (float(x) / base_n) ** 4 for x in left}
        print(f"[{name}] grid {n} done in {g['wall_s']:.1f} s. "
              f"remaining projection (fourth-power cost model): "
              + ", ".join(f"n={k} ~{v:.0f} s" for k, v in proj.items())
              + f"; projected total remaining ~{sum(proj.values()) / 60:.1f} min",
              flush=True)
        res["wall_by_grid_s"] = {str(k): v for k, v in walls.items()}
        res["projection_after_this_grid_s"] = {str(k): v for k, v in proj.items()}
        (OUT / f"{tag}_ladder.json").write_text(
            json.dumps(res, indent=2, default=float))

    res["reproduction_gate"] = reproduction_gate(name, res["grids"],
                                                 snap=a.snap_geometry)
    res["convergence"] = {
        k: convergence(res["grids"], k)
        for k in ("ROT_uniform", "STATIC_uniform", "QS_uniform")
        + (("ROT_transfer",) if a.with_map else ())}
    res["energy_gate_violations"] = [
        f"{n}:{arm}" for n in res["grids"]
        for arm, m in res["grids"][n]["arms"].items()
        if not m["energy_gate"]["PASS"]]
    res["wall_s"] = time.perf_counter() - t0

    (OUT / f"{tag}_ladder.json").write_text(
        json.dumps(res, indent=2, default=float))
    np.savez_compressed(OUT / f"{tag}_ladder_maps.npz", **store)

    rg = res["reproduction_gate"]
    print(f"[{name}] REPRODUCTION GATE {rg.get('status')} "
          f"({rg.get('n_comparisons_actually_made')} comparisons)", flush=True)
    for k, c in res["convergence"].items():
        if c.get("status"):
            continue
        print(f"[{name}] {k:14s} IoU " +
              " -> ".join(f"{v:.4f}" for v in c["IoU"]) +
              f"  last step {c['last_step_abs_dIoU']:.4f}"
              f"  asymptotic {c['asymptotic_range']}", flush=True)
    print(f"[{name}] energy gate violations {res['energy_gate_violations']}; "
          f"wall {res['wall_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
