#!/usr/bin/env python3
"""REAL-DATA GATE: an UNEQUAL-DWELL turntable program on the production engine.

`DWELL_SCHEDULE_REPORT.md` Section 4 could only compare the part-frame march
against the engine on EQUAL dwell over multiples of 90 degrees, because the
engine as shipped drives one fixed rotation increment at one fixed interval.
The engine now has a program mode (`rfam_eqs_coupled.parse_turntable_program`),
so the asymmetric deliverables can finally be executed by the production code.

What this script compares, per shape:

  march   `adjoint2d.dwell_march.program_forward`, the PART-frame time-resolved
          execution of the same program, no interpolation anywhere.
  engine  `rfam_eqs_coupled.run_sim` in turntable program mode, which rotates
          the part in the LAB frame and remaps temperature, relative density
          and melt fraction at every event, with the dopant map co-rotated by
          the engine's own new code path (the glue's co-rotation patch is
          switched OFF so the engine path is what is under test).

Two engine arms are run so the dielectric-ghost term can be attributed rather
than absorbed into a single number:

  eps_static   `corotate_eps_geometry: false`, the engine as shipped. The
               relative-permittivity field is rasterized once at startup and
               never rebuilt, so a part at 45 degrees sits inside a stationary
               dielectric outline of itself at 0 degrees.
  eps_corot    `corotate_eps_geometry: true`, permittivity re-rasterized at
               every event.

Both are honest engine numbers; which one is "the engine" is a modelling
choice, and the gap between them measures the ghost.

Acronyms: EQS = electro-quasi-static. IoU = intersection over union.
FGM = functionally graded material. J = the whole-domain shape objective
sum((phi - chi_part)^2).

Run:
  ./.venv312/bin/python scripts/analysis/run_engine_dwell_program_gate.py cross
  ./.venv312/bin/python scripts/analysis/run_engine_dwell_program_gate.py T_shape
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import dwell, dwell_march as dmarch                    # noqa: E402
from adjoint2d import energy_gate as eg, library_solve as lib         # noqa: E402
from adjoint2d import shape_objective as so                           # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                        # noqa: E402
from adjoint2d.pins import load_cfg                                   # noqa: E402
from scripts.analysis.turntable_glue import run_rotating              # noqa: E402

N_STEPS = 1500
DT_S = 0.5
OUT = REPO / "fgm_solve_campaign/out_dwell"


def load_program(shape: str, arm: str, key: str) -> tuple[list[dict], dict]:
    """The deliverable's ordered holds, and the deliverable document.

    `key` selects which program in the file: `moves` the literal one,
    `reduced_program` the half-turn-reduced one. The synthetic key
    `control90` is not in the file: it is an EQUAL dwell over the four
    multiples of 90 degrees at the same cycle time, executed on the same shape
    and the same dopant map. It exists as a BISECT: at multiples of 90 the
    engine's rotation remap is an exact pixel permutation, so a control90 gap
    isolates everything that is NOT the 45-degree interpolation.
    """
    doc = json.loads((OUT / f"{shape}_turntable_{arm}.json").read_text())
    if key == "control90":
        cyc = float(doc.get("cycle_time_s", 20.0))
        prog = dwell.cycle_program(np.full(4, 0.25),
                                   np.array([0.0, 90.0, 180.0, 270.0]),
                                   cycle_time_s=cyc, total_s=N_STEPS * DT_S,
                                   dt_s=DT_S)
        return [dict(m) for m in prog.moves], doc
    moves = doc["moves"] if key == "moves" else doc[key]["moves"]
    return list(moves), doc


def march_program(shape: str, moves: list[dict], sat: np.ndarray) -> dict:
    """Part-frame time-resolved execution of the same ordered program."""
    cfg = load_cfg(lib.shape_config(shape))
    angles = np.array(sorted({round(float(m["position_deg"]), 6) for m in moves}),
                      dtype=float)
    kern = DwellKernel.build(cfg, angles=angles)
    case = kern.case0
    # weights only pick which per-position heating fields get built; the
    # program decides what is actually injected at each outer step.
    w = np.full(angles.size, 1.0 / angles.size)
    kern.set_weights(w)
    kern.averaged_Q(sat)
    prog = dwell.TurntableProgram(
        moves=tuple(moves), kept_positions_deg=tuple(float(a) for a in angles),
        steps_per_position=tuple([1] * angles.size),
        realized_weights=w, requested_weights=w,
        cycle_time_s=float("nan"), total_s=N_STEPS * DT_S, dt_s=DT_S,
        n_cycles=float("nan"))
    idx = dmarch.program_step_positions(prog, angles, DT_S, N_STEPS)
    tr = dmarch.program_forward(kern, idx, N_STEPS)
    m = so.full_metrics(tr, case)
    i = int(m["t_stop_index"])
    gate = eg.gate_from_trajectory(tr, i)
    return {"J": float(m["J"]), "IoU": float(m["IoU"]),
            "t_stop_s": float(m["t_stop_s"]),
            "at_horizon": bool(m.get("t_stop_at_horizon", False)),
            "under_pct": float(m["part_under_melt_pct"]),
            "growth_pct": float(m["bed_melt_pct_of_part"]),
            "max_T_c": float(m.get("max_T_part_c", np.nan)),
            "energy_residual_rel": float(gate["rel_residual_at_index"]),
            "n_positions": int(angles.size),
            "positions_deg": [float(a) for a in angles]}


def engine_program(shape: str, moves: list[dict], sat: np.ndarray | None,
                   corotate_eps: bool) -> dict:
    cfg = copy.deepcopy(load_cfg(lib.shape_config(shape)))
    cfg["thermal"]["n_steps"] = N_STEPS
    cfg["geometry"]["part"]["rotation_deg"] = 0.0
    cfg.pop("fgm_feedback", None)
    cfg["turntable"] = {
        "enabled": True,
        "program": [{"angle_deg": float(m["position_deg"]),
                     "duration_s": float(m["dwell_s"]),
                     "move_at_s": float(m["move_at_s"])} for m in moves],
        "corotate_dopant": True,
        "corotate_eps_geometry": bool(corotate_eps),
    }
    # corotate=False: the GLUE does not touch the map, so what is measured is
    # the engine's own in-tree co-rotation.
    r = run_rotating(cfg, sat_map=sat, corotate=False, record=True, quiet=True)
    s = r.at_stop()
    return {"J": float(s["J"]), "IoU": float(s["IoU"]),
            "t_stop_s": float(s["t_stop_s"]),
            "at_horizon": bool(s["t_stop_at_horizon"]),
            "under_pct": float(s["part_under_melt_pct"]),
            "growth_pct": float(s["bed_melt_pct_of_part"]),
            "max_T_c": float(s["max_T_part_c"]),
            "energy_residual_rel": float(s["energy_residual_rel"]),
            "energy_gate_PASS": bool(s["energy_gate_PASS"]),
            "n_events": len(r.rotation_events),
            "wall_s": float(r.wall_s)}


def compare(march: dict, eng: dict) -> dict:
    return {"J_rel_gap_pct": 100.0 * (march["J"] - eng["J"]) / max(eng["J"], 1e-12),
            "IoU_gap": march["IoU"] - eng["IoU"],
            "stop_gap_s": march["t_stop_s"] - eng["t_stop_s"]}


def main(shape: str, arm: str = "deliverable", key: str = "moves") -> dict:
    t0 = time.perf_counter()

    def log(m):
        print(f"[{shape}/{arm}/{key}] {m}", flush=True)

    moves, doc = load_program(shape, arm, key)
    sat = np.asarray(np.load(OUT / doc["dopant_map_npz"])[doc["dopant_map_key"]],
                     dtype=float)
    log(f"{len(moves)} holds, map {doc['dopant_map_key']} "
        f"(max {sat.max():.4f}), stored recommended stop "
        f"{doc['recommended_stop_s']} s")

    m = march_program(shape, moves, sat)
    log(f"march      J {m['J']:8.2f} IoU {m['IoU']:.4f} stop {m['t_stop_s']:6.1f} s "
        f"Eres {100 * m['energy_residual_rel']:.2f}%  positions {m['positions_deg']}")

    rows = {}
    for tag, ce in (("eps_static", False), ("eps_corot", True)):
        e = engine_program(shape, moves, sat, corotate_eps=ce)
        c = compare(m, e)
        rows[tag] = {"engine": e, "vs_march": c}
        log(f"engine {tag:10s} J {e['J']:8.2f} IoU {e['IoU']:.4f} stop "
            f"{e['t_stop_s']:6.1f} s events {e['n_events']:4d} "
            f"Eres {100 * e['energy_residual_rel']:.2f}% | "
            f"dJ {c['J_rel_gap_pct']:+7.2f}%  dIoU {c['IoU_gap']:+.4f}  "
            f"dstop {c['stop_gap_s']:+.1f} s  wall {e['wall_s']:.0f} s")

    out = {"shape": shape, "arm": arm, "program_key": key,
           "n_holds": len(moves), "n_steps": N_STEPS, "dt_s": DT_S,
           "dopant_map_key": doc["dopant_map_key"],
           "stored_recommended_stop_s": float(doc["recommended_stop_s"]),
           "march": m, "engine_arms": rows,
           "wall_s": time.perf_counter() - t0}
    (OUT / f"{shape}_{arm}_{key}_engine_program_gate.json").write_text(
        json.dumps(out, indent=1, default=float))
    log(f"DONE wall {out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else "deliverable",
         sys.argv[3] if len(sys.argv) > 3 else "moves")
