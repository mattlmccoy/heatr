#!/usr/bin/env python3
"""SEQUENTIAL DWELL, step 2: extend the switch grid and add a third segment.

The first screen (`run_seq_screen.py`) was still IMPROVING at the longest
switch time it sampled on the L_shape, so the grid is extended here rather than
the optimum being declared at the edge of a grid. A three-segment block is run
on top of the two-segment winner, sharing both the first and the second phase.
Uniform dopant map throughout: this is still the sub-hypothesis arm.

Run:
  ./.venv312/bin/python scripts/analysis/run_seq_screen2.py <shape>
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
from run_seq_probe import DT_S, N_STEPS, limb_masks, limb_report  # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_seq"

PLAN = {
    "L_shape": {
        "angles_deg": [0.0, 90.0, 105.0, 135.0],
        "two": [{"a1_deg": 90.0, "a2_deg": [0.0],
                 "switch_s": [500.0, 550.0, 600.0, 650.0, 700.0]},
                {"a1_deg": 105.0, "a2_deg": [0.0],
                 "switch_s": [500.0, 550.0, 600.0]}],
        "three": [{"a1_deg": 90.0, "sw1_s": [350.0, 450.0],
                   "a2_deg": 0.0, "sw2_s": [500.0, 575.0, 650.0],
                   "a3_deg": [135.0, 105.0]}],
    },
    "T_shape": {
        # The limb response of the T is already on record: at 0 degrees the
        # stem melts 94.7 percent and the crossbar 4.5 percent, at 90 degrees
        # the crossbar 98.3 percent and the stem 8.5 percent
        # (`CONTINUOUS_ROTATION_REPORT.md` Section 3). No probe is re-run.
        "angles_deg": [0.0, 45.0, 90.0, 135.0],
        "two": [{"a1_deg": 90.0, "a2_deg": [0.0],
                 "switch_s": [250.0, 350.0, 450.0, 550.0, 650.0]},
                {"a1_deg": 0.0, "a2_deg": [90.0],
                 "switch_s": [200.0, 300.0, 400.0, 500.0]}],
        "three": [],
    },
}


def main(shape: str) -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = PLAN[shape]
    angles = np.asarray(plan["angles_deg"], dtype=float)
    cfg = load_cfg(lib.shape_config(shape))
    kern = DwellKernel.build(cfg, angles=angles)
    case = kern.case0
    pm = case.part_mask
    wide, narrow = limb_masks(pm)
    kern.set_weights(np.full(angles.size, 1.0 / angles.size))
    kern.averaged_Q(np.ones(pm.shape))
    idx_of = {float(a): j for j, a in enumerate(angles)}
    print(f"[{shape}/screen2] uniform map, angles {list(angles)}", flush=True)

    def finish(r):
        T = r.pop("T_at_stop")
        curve = r.pop("J_curve")
        for key in ("a1", "a2", "a3"):
            if key in r:
                r[f"{key}_deg"] = float(angles[r[key]])
        r.update(so.region_metrics(T, case))
        r.update(limb_report(T, case, wide, narrow))
        r["max_T_at_stop_c"] = float(np.max(T))
        r["over_ceiling_250c"] = bool(r["max_T_at_stop_c"] > 250.0)
        r["n_curve"] = int(curve.size)
        print(f"      IoU {r['IoU']:.4f}  wide {r['wide_melted_pct']:6.2f}%  "
              f"narrow {r['narrow_melted_pct']:6.2f}%  "
              f"grow {r['bed_melt_pct_of_part']:5.2f}%  "
              f"under {r['part_under_melt_pct']:5.2f}%  "
              f"maxT {r['max_T_at_stop_c']:6.1f} C"
              f"{'  CEILING' if r['over_ceiling_250c'] else ''}", flush=True)
        return r

    rows2, rows3 = [], []
    for blk in plan["two"]:
        print(f"  two segments, first position {blk['a1_deg']:.0f} deg", flush=True)
        for r in scr.screen_two_segment(
                kern, idx_of[blk["a1_deg"]], [idx_of[a] for a in blk["a2_deg"]],
                [int(round(t / DT_S)) for t in blk["switch_s"]], N_STEPS):
            rows2.append(finish(r))
    for blk in plan["three"]:
        print(f"  three segments, {blk['a1_deg']:.0f} -> {blk['a2_deg']:.0f} -> "
              f"{blk['a3_deg']}", flush=True)
        for r in scr.screen_three_segment(
                kern, idx_of[blk["a1_deg"]],
                [int(round(t / DT_S)) for t in blk["sw1_s"]],
                idx_of[blk["a2_deg"]],
                [int(round(t / DT_S)) for t in blk["sw2_s"]],
                [idx_of[a] for a in blk["a3_deg"]], N_STEPS):
            rows3.append(finish(r))

    rows2.sort(key=lambda r: r["J"])
    rows3.sort(key=lambda r: r["J"])
    out = {"shape": shape, "map": "uniform saturation 1.0", "plan": plan,
           "n_steps": N_STEPS, "dt_s": DT_S, "early_stop": "DISABLED",
           "two_segment": rows2, "three_segment": rows3,
           "wall_s": time.perf_counter() - t0}
    (OUT / f"{shape}_screen2.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[{shape}/screen2] best two-segment: "
          f"{rows2[0]['a1_deg']:.0f} -> {rows2[0]['a2_deg']:.0f} at "
          f"{rows2[0]['switch_s']:.1f} s, J {rows2[0]['J']:.2f}, "
          f"IoU {rows2[0]['IoU']:.4f}" if rows2 else "no two-segment rows")
    if rows3:
        print(f"[{shape}/screen2] best three-segment: J {rows3[0]['J']:.2f}, "
              f"IoU {rows3[0]['IoU']:.4f}")
    print(f"[{shape}/screen2] {out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "L_shape")
