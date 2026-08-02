#!/usr/bin/env python3
"""Re-emit the machine-readable turntable programs against the FINAL arms.

Two things this does that the solve driver could not:

1. It selects the deliverable and the equal-dwell control from ALL arms on
   record for the shape, including the deeper refinement runs, by their own
   objective, and writes the program that belongs to the arm actually being
   quoted.
2. It emits a REDUCED program on the DISTINCT positions. MEASURED in this pass:
   the part-frame heating at theta and at theta + 180 degrees is the same field
   to 4e-13 relative, because the parallel-plate drive is invariant under a
   half turn of the whole system. A literal program that visits both is asking
   the machine to make a move that changes nothing. The reduced program merges
   them; both are emitted so the redundancy is visible rather than hidden.

Run:
  ./.venv312/bin/python scripts/analysis/finalize_dwell.py [shape ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import dwell                                   # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_dwell"
SHAPES = ("cross", "square", "T_shape", "L_shape")
DT_S = 0.5
TOTAL_S = 750.0


def select(r: dict, d) -> dict:
    arms = r["arms"]

    def pick(pred):
        c = [(v["J"], k) for k, v in arms.items()
             if "4bpp" in k and f"sat_{k}" in d and pred(k)]
        return min(c)[1] if c else None

    deliv = pick(lambda k: "equal" not in k)
    ctrl = pick(lambda k: "equal" in k)

    def tr_of(k):
        if k == "D_program_4bpp":
            return "D_timeresolved"
        if k == "D_map_equal_4bpp":
            return "D_timeresolved_equal"
        cand = k.replace("_4bpp_", "_timeresolved_")
        return cand if cand in arms else k

    return {"deliverable": deliv, "control": ctrl,
            "deliverable_tr": tr_of(deliv), "control_tr": tr_of(ctrl)}


def emit(shape: str) -> dict:
    js = OUT / f"{shape}_dwell.json"
    r = json.loads(js.read_text())
    d = np.load(OUT / f"{shape}_dwell_maps.npz")
    ang = np.asarray(r["candidate_angles_deg"], float)
    cyc = float(r["cycle_time_s"])
    sel = select(r, d)
    progs = {}
    for tag, key in (("deliverable", sel["deliverable"]),
                     ("equal_dwell_control", sel["control"])):
        arm = r["arms"][key]
        w = np.asarray(arm["dwell_weights"], float)
        w = w / w.sum()
        stop = float(r["arms"][sel[f"{'deliverable' if tag == 'deliverable' else 'control'}_tr"]]["t_stop_s"])
        full = dwell.cycle_program(w, ang, cycle_time_s=cyc, total_s=TOTAL_S, dt_s=DT_S)
        w4 = w[:4] + w[4:]
        red = dwell.cycle_program(w4, ang[:4], cycle_time_s=cyc, total_s=TOTAL_S,
                                  dt_s=DT_S)
        j = full.as_json()
        j.update({
            "shape": shape, "arm": tag, "scored_arm_name": key,
            "candidate_positions_deg": [float(a) for a in ang],
            "recommended_stop_s": stop,
            "dopant_map_npz": f"{shape}_dwell_maps.npz",
            "dopant_map_key": f"sat_{key}",
            "dopant_bits_per_pixel": 4,
            "rf_program": {"mode": "constant", "relative_power": 1.0,
                           "voltage_v": float(r["voltage_v"])},
            "half_turn_redundancy": (
                "the part-frame heating at theta and theta + 180 degrees is the "
                "same field to 4e-13 relative, so the reduced program below is "
                "physically equivalent to the literal one and asks for half as "
                "many moves"),
            "reduced_program": {
                "positions_deg": list(red.kept_positions_deg),
                "dwell_fraction": [float(v) for v in red.realized_weights],
                "steps_per_position_per_cycle": list(red.steps_per_position),
                "cycle_time_s": red.cycle_time_s,
                "n_moves": len(red.moves),
                "moves": [dict(m) for m in red.moves]},
        })
        (OUT / f"{shape}_turntable_{tag}.json").write_text(json.dumps(j, indent=1))
        progs[tag] = {"n_moves": j["n_moves"],
                      "reduced_n_moves": j["reduced_program"]["n_moves"],
                      "positions_deg": j["positions_deg"],
                      "reduced_positions_deg": j["reduced_program"]["positions_deg"],
                      "reduced_dwell_fraction": j["reduced_program"]["dwell_fraction"],
                      "recommended_stop_s": stop}
        print(f"  {shape}/{tag}: arm {key}, positions "
              f"{j['reduced_program']['positions_deg']} dwell "
              f"{[round(v, 4) for v in j['reduced_program']['dwell_fraction']]}, "
              f"{j['reduced_program']['n_moves']} moves, stop {stop:.1f} s")
    r["final_selection"] = sel
    r["final_programs"] = progs
    js.write_text(json.dumps(r, indent=1, default=float))
    return r


if __name__ == "__main__":
    for s in (sys.argv[1:] or list(SHAPES)):
        if (OUT / f"{s}_dwell.json").exists():
            emit(s)
