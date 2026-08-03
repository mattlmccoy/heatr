#!/usr/bin/env python3
"""SEQUENTIAL DWELL: a fine one-dimensional scan of the switch time.

WHY. On the L_shape the gradient-driven refinement moved the switch time by
0.01 s and did not improve the objective, which has two possible readings: the
screen grid point is already a local optimum, or the optimizer stalled. A
direct scan at 5 s resolution over a 100 s window settles it by measurement
instead of by argument. Prefix sharing makes it cost about two forward
equivalents.

Run:
  ./.venv312/bin/python scripts/analysis/run_seq_finescan.py <shape> <lo_s> <hi_s> <step_s>
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
from run_seq_arms import CFG                                      # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_seq"


def main(shape: str, lo: float, hi: float, step: float) -> dict:
    t0 = time.perf_counter()
    r = json.loads((OUT / f"{shape}_arms.json").read_text())
    a = r["arms"]["S_seq_uniform_two_refined"]
    seg_deg = a["segments_deg"]
    angles = np.asarray(CFG[shape]["angles_deg"], dtype=float)
    kern = DwellKernel.build(load_cfg(lib.shape_config(shape)), angles=angles)
    case = kern.case0
    pm = case.part_mask
    wide, narrow = limb_masks(pm)
    kern.set_weights(np.full(angles.size, 1.0 / angles.size))
    kern.averaged_Q(np.ones(pm.shape))
    idx_of = {float(x): j for j, x in enumerate(angles)}
    sw = [int(round(t / DT_S)) for t in np.arange(lo, hi + 1e-9, step)]
    print(f"[{shape}/finescan] {seg_deg[0]:.0f} -> {seg_deg[1]:.0f} deg, "
          f"switch {lo} to {hi} s every {step} s ({len(sw)} points)", flush=True)
    rows = scr.screen_two_segment(kern, idx_of[seg_deg[0]], [idx_of[seg_deg[1]]],
                                  sw, N_STEPS)
    for row in rows:
        T = row.pop("T_at_stop")
        row.pop("J_curve")
        row.update(so.region_metrics(T, case))
        row.update(limb_report(T, case, wide, narrow))
        row["max_T_at_stop_c"] = float(np.max(T))
    best = min(rows, key=lambda x: x["J"])
    out = {"shape": shape, "segments_deg": seg_deg, "map": "uniform",
           "rows": rows, "best": best,
           "refined_switch_s": float(a["durations_s"][0]),
           "refined_J": float(a["J"]),
           "wall_s": time.perf_counter() - t0}
    (OUT / f"{shape}_finescan.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[{shape}/finescan] scan minimum J {best['J']:.2f} at switch "
          f"{best['switch_s']:.1f} s, IoU {best['IoU']:.4f}; the gradient-refined "
          f"arm is J {a['J']:.2f} at {a['durations_s'][0]:.2f} s "
          f"({100.0 * (a['J'] - best['J']) / best['J']:+.3f} % against the scan)")
    print(f"[{shape}/finescan] {out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
