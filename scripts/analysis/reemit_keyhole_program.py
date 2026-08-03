#!/usr/bin/env python3
"""Re-emit the keyhole's continuous-rotation turntable program, divisor aware.

WHY. `ROTATING_HOLDOUT_REPORT.md` Section 5 measured a program-emission bug
with no physics in it. The emitted program allocates 40 control steps per 20 s
cycle across 12 turntable positions. Forty does not divide by twelve, the old
emitter applied largest remainder INSIDE one cycle and then repeated that same
allocation every cycle, and the per-cycle rounding therefore became a permanent
bias: realized dwell fractions 0.100 on four positions and 0.075 on eight,
against the design 1/12 = 0.08333 the map was solved against. At grid 120, in
the quasi-static limit, that cost 73.2 percent of J.

WHAT THIS DOES. Re-runs the SAME emission call with the SAME inputs (the twelve
half-turn-distinct positions, equal design weights, the 20 s cycle, the 0.5 s
control step and the arm's own 747.5 s stop) through the fixed
`dwell.cycle_program`, whose leftover control steps now rotate from cycle to
cycle. Nothing else about the arm changes: same shape, same solved map, same
drive, same grid.

WHAT IT DOES NOT DO. It does not overwrite `out_intake/keyhole_novel.json`.
`ROTATING_HOLDOUT_REPORT.md` cites that file, so the old program stays on disk
and the corrected one is written alongside.

Run:
  ./.venv312/bin/python scripts/analysis/reemit_keyhole_program.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import dwell                                    # noqa: E402

SRC = REPO / "fgm_solve_campaign/out_intake/keyhole_novel.json"
OUT = REPO / "fgm_solve_campaign/out_rot_holdout/keyhole_program_fixed.json"
CYCLE_TIME_S = 20.0


def main() -> None:
    old = json.loads(SRC.read_text())["turntable_programs"]["continuous"]
    ang = tuple(float(a) for a in old["positions_deg"])
    dt = float(old["control_step_s"])
    total = float(old["total_exposure_s"])
    w = np.full(len(ang), 1.0 / len(ang))

    prog = dwell.cycle_program(w, ang, cycle_time_s=CYCLE_TIME_S,
                               total_s=total, dt_s=dt)
    new = prog.as_json()
    new["gauge_note"] = old["gauge_note"]

    n_total = int(round(total / dt))
    r_old = np.asarray(old["realized_dwell_fraction"], dtype=float)
    r_new = np.asarray(new["realized_dwell_fraction"], dtype=float)
    design = 1.0 / len(ang)
    rec = {
        "task": "divisor-aware re-emission of the keyhole continuous-rotation "
                "turntable program",
        "source_program": str(SRC),
        "source_key": "turntable_programs/continuous",
        "unchanged": {
            "positions_deg": list(ang), "cycle_time_s": CYCLE_TIME_S,
            "control_step_s": dt, "total_exposure_s": total,
            "design_dwell_fraction": [design] * len(ang),
        },
        "control_steps_per_cycle": int(round(CYCLE_TIME_S / dt)),
        "n_positions": len(ang),
        "divides_evenly": bool(int(round(CYCLE_TIME_S / dt)) % len(ang) == 0),
        "n_control_steps_over_exposure": n_total,
        "one_control_step_as_a_fraction": 1.0 / n_total,
        "old": {
            "realized_dwell_fraction": [float(x) for x in r_old],
            "max_abs_error_against_design": float(np.max(np.abs(r_old - design))),
            "n_moves": int(old["n_moves"]),
            "steps_per_position_per_cycle": list(
                old["steps_per_position_per_cycle"]),
        },
        "new": {
            "realized_dwell_fraction": [float(x) for x in r_new],
            "max_abs_error_against_design": float(np.max(np.abs(r_new - design))),
            "n_moves": int(new["n_moves"]),
            "steps_per_position_first_cycle": list(
                new["steps_per_position_per_cycle"]),
            "steps_per_position_over_exposure": list(
                new["steps_per_position_over_exposure"]),
        },
        "program": new,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2, default=float))

    print(f"positions {len(ang)}, control steps per cycle "
          f"{rec['control_steps_per_cycle']}, divides {rec['divides_evenly']}")
    print(f"old realized {np.round(r_old, 5)}")
    print(f"    max error against design {rec['old']['max_abs_error_against_design']:.5f} "
          f"({rec['old']['max_abs_error_against_design'] / design * 100:.1f} percent), "
          f"{old['n_moves']} moves")
    print(f"new realized {np.round(r_new, 5)}")
    print(f"    max error against design {rec['new']['max_abs_error_against_design']:.5f} "
          f"({rec['new']['max_abs_error_against_design'] / design * 100:.1f} percent), "
          f"{new['n_moves']} moves")
    print(f"one control step over the exposure is {1.0 / n_total:.6f} of it")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
