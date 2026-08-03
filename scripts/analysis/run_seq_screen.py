#!/usr/bin/env python3
"""SEQUENTIAL DWELL, step 1: the UNIFORM-MAP two-segment screen.

This is the arm that tests the sub-hypothesis on its own: does a uniform dopant
map plus a sequential hold schedule already beat the best static arm? No dopant
grading is solved anywhere in this script.

Prefix sharing (`seq_screen.py`) makes the (first position) x (switch time) x
(second position) grid cost one full march per first position plus the tails.

Run:
  ./.venv312/bin/python scripts/analysis/run_seq_screen.py <shape>
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

from adjoint2d import library_solve as lib                       # noqa: E402
from adjoint2d import shape_objective as so                      # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                   # noqa: E402
from adjoint2d.pins import load_cfg                              # noqa: E402
import seq_screen as scr                                          # noqa: E402
from run_seq_probe import ANGLES, DT_S, N_STEPS, limb_masks, limb_report  # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_seq"

# The screen grid, chosen from the measured limb response of
# `out_seq/<shape>_probe.json` and stated per shape rather than swept blind.
PLAN = {
    # L_shape: 0 degrees melts the narrow limb (94.3 percent) and 105 degrees
    # the wide limb (91.5 percent); 90 and 120 are the neighbours of the wide
    # optimum and 135 is the static winner, carried as the collapse control.
    "L_shape": [
        {"a1_deg": 0.0, "a2_deg": [90.0, 105.0, 120.0],
         "switch_s": [75.0, 125.0, 175.0, 225.0, 275.0]},
        {"a1_deg": 105.0, "a2_deg": [0.0, 150.0],
         "switch_s": [150.0, 250.0, 350.0, 450.0]},
        {"a1_deg": 90.0, "a2_deg": [0.0],
         "switch_s": [150.0, 250.0, 350.0, 450.0]},
    ],
}


def main(shape: str) -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = PLAN[shape]
    cfg = load_cfg(lib.shape_config(shape))
    kern = DwellKernel.build(cfg, angles=ANGLES)
    case = kern.case0
    pm = case.part_mask
    wide, narrow = limb_masks(pm)
    s = np.ones(pm.shape)
    kern.set_weights(np.full(ANGLES.size, 1.0 / ANGLES.size))
    kern.averaged_Q(s)
    idx_of = {float(a): j for j, a in enumerate(ANGLES)}
    print(f"[{shape}/screen] uniform map, {len(plan)} first positions", flush=True)

    rows = []
    for blk in plan:
        a1 = idx_of[blk["a1_deg"]]
        print(f"  first position {blk['a1_deg']:.0f} deg", flush=True)
        got = scr.screen_two_segment(
            kern, a1, [idx_of[a] for a in blk["a2_deg"]],
            [int(round(t / DT_S)) for t in blk["switch_s"]], N_STEPS)
        for r in got:
            T = r.pop("T_at_stop")
            curve = r.pop("J_curve")
            r["a1_deg"] = float(ANGLES[r["a1"]])
            r["a2_deg"] = float(ANGLES[r["a2"]])
            r.update(so.region_metrics(T, case))
            r.update(limb_report(T, case, wide, narrow))
            r["max_T_at_stop_c"] = float(np.max(T))
            r["over_ceiling_250c"] = bool(r["max_T_at_stop_c"] > 250.0)
            r["J_curve_min_after_switch"] = float(np.min(curve[r["switch_step"]:]))
            rows.append(r)
            print(f"      IoU {r['IoU']:.4f}  wide {r['wide_melted_pct']:6.2f}%  "
                  f"narrow {r['narrow_melted_pct']:6.2f}%  "
                  f"grow {r['bed_melt_pct_of_part']:5.2f}%  "
                  f"maxT {r['max_T_at_stop_c']:6.1f} C"
                  f"{'  CEILING' if r['over_ceiling_250c'] else ''}", flush=True)

    rows.sort(key=lambda r: r["J"])
    out = {"shape": shape, "map": "uniform saturation 1.0", "plan": plan,
           "n_steps": N_STEPS, "dt_s": DT_S, "early_stop": "DISABLED",
           "rows": rows, "wall_s": time.perf_counter() - t0}
    (OUT / f"{shape}_screen.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[{shape}/screen] best five by J:")
    for r in rows[:5]:
        print(f"   {r['a1_deg']:5.0f} -> {r['a2_deg']:5.0f} deg switch "
              f"{r['switch_s']:6.1f} s : J {r['J']:8.2f} IoU {r['IoU']:.4f} "
              f"stop {r['stop_s']:6.1f} s")
    print(f"[{shape}/screen] {out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "L_shape")
