#!/usr/bin/env python3
"""REAL-DATA GATE for the dwell campaign's time-resolved march.

The dwell solve is done in the PART frame, where the part never moves and the
heating pattern switches instead. That is a modelling choice, and it has to be
checked against the production engine, which does the opposite: it rotates the
part in the LAB frame and remaps the thermal fields at every event.

The check can only be exact on the arms the production engine can execute
WITHOUT interpolation error. `CONTINUOUS_ROTATION_REPORT.md` Section 6.1
measured that the engine's rotation-event remap loses 0.007 to 0.010 percent of
dose per event at a general angle, but that at multiples of 90 degrees on this
grid the remap is an exact pixel permutation and loses nothing. So the check
runs on the two four-fold symmetric shapes (cross, square) with an EQUAL dwell
over the four multiples of 90 degrees, which is exactly the engine's 90-degree
indexing mode.

What this gate can and cannot say. It CAN say whether the part-frame march
reproduces the production engine on a schedule both can express exactly. It
CANNOT validate an asymmetric dwell program, because the production engine as
shipped drives a FIXED rotation increment at a FIXED interval and has no way to
express unequal dwells (`rfam_eqs_coupled.py:2827-2836`); its legacy
`phases` branch, which does take arbitrary angles, raises `TypeError` on
`len(_n_evts)` at `rfam_eqs_coupled.py:2819` (an int passed to `len`) and is
unusable as shipped, and it spaces its events EQUALLY in any case. That
limit is reported rather than worked around, because working around it would
mean editing the engine.

Run:
  ./.venv312/bin/python scripts/analysis/run_dwell_engine_check.py <shape> [cycle_s]
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

from adjoint2d import dwell, dwell_march as dmarch                  # noqa: E402
from adjoint2d import energy_gate as eg, library_solve as lib       # noqa: E402
from adjoint2d import shape_objective as so                         # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                      # noqa: E402
from adjoint2d.pins import load_cfg                                 # noqa: E402
from scripts.analysis.turntable_glue import run_rotating            # noqa: E402

N_STEPS = 1500
DT_S = 0.5
ANG4 = np.array([0.0, 90.0, 180.0, 270.0])
OUT = REPO / "fgm_solve_campaign/out_dwell"


def engine_cfg(shape: str, interval_s: float) -> dict:
    cfg = copy.deepcopy(load_cfg(lib.shape_config(shape)))
    cfg["thermal"]["n_steps"] = N_STEPS
    cfg["geometry"]["part"]["rotation_deg"] = 0.0
    cfg.pop("fgm_feedback", None)
    steps = max(1, round(float(interval_s) / DT_S))
    cfg["turntable"] = {"enabled": True, "rotation_deg": 90.0,
                        "total_rotations": int(np.ceil(N_STEPS / steps)) + 2,
                        "rotation_interval_s": float(steps * DT_S)}
    return cfg


def main(shape: str, cycle_s: float = 20.0) -> dict:
    t0 = time.perf_counter()

    def log(m):
        print(f"[{shape}/enginecheck] {m}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    interval_s = float(cycle_s) / 4.0
    cfg = load_cfg(lib.shape_config(shape))
    kern = DwellKernel.build(cfg, angles=ANG4)
    case = kern.case0
    pm = case.part_mask

    maps = {"uniform": np.ones(pm.shape)}
    npz = OUT / f"{shape}_dwell_maps.npz"
    if npz.exists():
        d = np.load(npz)
        for key, tag in (("sat_D_map_equal_4bpp", "equal_map_4bpp"),
                         ("sat_D_joint_4bpp", "joint_map_4bpp")):
            if key in d:
                maps[tag] = np.asarray(d[key], dtype=float)

    rows = []
    for tag, sat in maps.items():
        # part-frame time-resolved march, equal dwell over the four positions
        prog = dwell.cycle_program(np.full(4, 0.25), ANG4, cycle_time_s=cycle_s,
                                   total_s=N_STEPS * DT_S, dt_s=DT_S)
        kern.set_weights(np.full(4, 0.25))
        kern.averaged_Q(sat)
        idx = dmarch.program_step_positions(prog, ANG4, DT_S, N_STEPS)
        tr = dmarch.program_forward(kern, idx, N_STEPS)
        m = so.full_metrics(tr, case)
        i = int(m["t_stop_index"])
        m["energy_gate"] = eg.gate_from_trajectory(tr, i)

        # production engine, 90-degree indexing at the same event rate
        e = run_rotating(engine_cfg(shape, interval_s),
                         sat_map=(None if tag == "uniform" else sat),
                         corotate=True, record=True, quiet=True)
        em = e.at_stop()

        row = {"map": tag, "cycle_time_s": float(cycle_s),
               "event_interval_s": interval_s,
               "partframe_J": float(m["J"]), "partframe_IoU": float(m["IoU"]),
               "partframe_stop_s": float(m["t_stop_s"]),
               "partframe_under_pct": float(m["part_under_melt_pct"]),
               "partframe_growth_pct": float(m["bed_melt_pct_of_part"]),
               "partframe_energy_residual_rel": float(m["energy_gate"]["rel_residual_at_index"]),
               "engine_J": float(em["J"]), "engine_IoU": float(em["IoU"]),
               "engine_stop_s": float(em["t_stop_s"]),
               "engine_under_pct": float(em["part_under_melt_pct"]),
               "engine_growth_pct": float(em["bed_melt_pct_of_part"]),
               "engine_events": len(e.rotation_events),
               "engine_energy_residual_rel": float(em["energy_residual_rel"]),
               "engine_energy_gate_PASS": bool(em["energy_gate_PASS"]),
               "J_rel_gap_pct": 100.0 * (float(m["J"]) - float(em["J"]))
                                 / max(float(em["J"]), 1e-12),
               "IoU_gap": float(m["IoU"]) - float(em["IoU"]),
               "stop_gap_s": float(m["t_stop_s"]) - float(em["t_stop_s"])}
        rows.append(row)
        log(f"  {tag:16s} part-frame J {row['partframe_J']:8.2f} IoU "
            f"{row['partframe_IoU']:.4f} stop {row['partframe_stop_s']:6.1f} s | "
            f"engine J {row['engine_J']:8.2f} IoU {row['engine_IoU']:.4f} stop "
            f"{row['engine_stop_s']:6.1f} s | gap {row['J_rel_gap_pct']:+6.2f}% "
            f"IoU {row['IoU_gap']:+.4f}  events {row['engine_events']}  "
            f"Eresid {100 * row['engine_energy_residual_rel']:.2f}%")

    out = {"shape": shape, "cycle_time_s": float(cycle_s),
           "positions_deg": [float(a) for a in ANG4],
           "dwell": "EQUAL, the only schedule the production engine can execute",
           "engine": "rfam_eqs_coupled.run_sim through its own turntable block",
           "glue": "scripts.analysis.turntable_glue (monkeypatch, no engine edit)",
           "n_steps": N_STEPS, "dt_s": DT_S, "grid": 120,
           "rows": rows, "wall_s": time.perf_counter() - t0}
    (OUT / f"{shape}_engine_check.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"DONE wall {out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 20.0)
