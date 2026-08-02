#!/usr/bin/env python3
"""SEQUENTIAL DWELL: why the L_shape switch-time refinement stalled.

MEASUREMENT, not argument. The screen grid put the starting switch time at
450.0 s, which is EXACTLY 900 control steps, that is exactly on the one kink
the overlap-matrix parameterization has. This script measures, at that point
and at an interior point 0.25 s away:

  * the analytic dJ/d(switch),
  * the one-sided differences on each side,
  * the central difference,

so the stall can be attributed to the kink or ruled out. It then re-runs the
switch-time refinement from an INTERIOR start and re-scores.

Run:
  ./.venv312/bin/python scripts/analysis/run_seq_kink_probe.py <shape>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from adjoint2d import gradops, library_solve as lib                  # noqa: E402
from adjoint2d import printability as pq                             # noqa: E402
from adjoint2d import seq_dwell as sq, seq_dwell_march as sqm        # noqa: E402
from adjoint2d import shape_objective as so                          # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                       # noqa: E402
from adjoint2d.pins import load_cfg                                  # noqa: E402
from run_seq_arms import CFG, log_row, score                         # noqa: E402
from run_seq_probe import DT_S, N_STEPS, limb_masks                  # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_seq"


def main(shape: str) -> dict:
    t0 = time.perf_counter()
    r = json.loads((OUT / f"{shape}_arms.json").read_text())
    fs = json.loads((OUT / f"{shape}_finescan.json").read_text())
    a_ref = r["arms"]["S_seq_uniform_two_refined"]
    seg_deg = a_ref["segments_deg"]
    angles = np.asarray(CFG[shape]["angles_deg"], dtype=float)
    kern = DwellKernel.build(load_cfg(lib.shape_config(shape)), angles=angles)
    case = kern.case0
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    wide, narrow = limb_masks(pm)
    kern.set_weights(np.full(angles.size, 1.0 / angles.size))
    idx_of = {float(x): j for j, x in enumerate(angles)}
    seg = [idx_of[x] for x in seg_deg]
    s = np.ones(pm.shape)
    total = N_STEPS * DT_S

    def J_of(d0):
        kern.averaged_Q(s)
        tr = sqm.sequential_forward(kern, seg, [d0, total - d0], N_STEPS)
        return so.optimal_stop(tr, case).J

    def grad_of(d0):
        kern.averaged_Q(s)
        tr = sqm.sequential_forward(kern, seg, [d0, total - d0], N_STEPS,
                                    keep_checkpoints=True)
        st = so.optimal_stop(tr, case)
        _J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
        _gs, gd = sqm.sequential_gradients(kern, s, tr, {st.index: seed},
                                           grad_ops=ops)
        return float(gd[0]), st.index, float(st.J)

    probes = {}
    for tag, d0 in (("on_step_edge", float(a_ref["durations_s"][0])),
                    ("interior", float(a_ref["durations_s"][0]) + 0.25)):
        g, i_stop, J0 = grad_of(d0)
        eps = 0.05
        jp, jm = J_of(d0 + eps), J_of(d0 - eps)
        probes[tag] = {
            "switch_s": d0,
            "distance_to_step_edge_s": float(min(np.mod(d0, DT_S),
                                                 DT_S - np.mod(d0, DT_S))),
            "J": J0, "stop_index": int(i_stop),
            "analytic_dJ_dswitch": g,
            "forward_difference": (jp - J0) / eps,
            "backward_difference": (J0 - jm) / eps,
            "central_difference": (jp - jm) / (2 * eps),
            "rel_err_vs_forward": abs((jp - J0) / eps - g) / max(abs(g), 1e-30),
            "rel_err_vs_central": abs((jp - jm) / (2 * eps) - g) / max(abs(g), 1e-30),
        }
        p = probes[tag]
        print(f"[{shape}/kink] {tag:14s} switch {d0:8.3f} s (edge distance "
              f"{p['distance_to_step_edge_s']:.4f} s): analytic {g:+.6f}, "
              f"forward {p['forward_difference']:+.6f}, backward "
              f"{p['backward_difference']:+.6f}, central "
              f"{p['central_difference']:+.6f}", flush=True)

    # re-refine from an INTERIOR start near the scan minimum
    d_start = float(fs["best"]["switch_s"]) + 0.25
    print(f"[{shape}/kink] re-refining from an interior start {d_start:.2f} s",
          flush=True)
    rows = []

    def fun(x):
        if len(rows) >= 6:
            raise StopIteration
        d0 = float(x[0])
        g, i_stop, J0 = grad_of(d0)
        rows.append({"switch_s": d0, "J": J0, "grad": g, "stop_index": i_stop})
        return J0, np.array([g])

    try:
        minimize(fun, np.array([d_start]), jac=True, method="L-BFGS-B",
                 bounds=[(1.0, total - 1.0)],
                 options={"maxiter": 100, "maxfun": 100, "ftol": 1e-16,
                          "gtol": 1e-16})
    except StopIteration:
        pass
    for row in rows:
        print(f"    J {row['J']:10.4f}  switch {row['switch_s']:9.4f} s  "
              f"gradient {row['grad']:+.6f}", flush=True)
    b = min(rows, key=lambda x: x["J"])
    d_best = b["switch_s"]

    out = {"shape": shape, "segments_deg": seg_deg, "probes": probes,
           "refine_from_interior": rows, "best_switch_s": d_best,
           "best_J": b["J"],
           "previous_refined_switch_s": float(a_ref["durations_s"][0]),
           "previous_refined_J": float(a_ref["J"]),
           "fine_scan_min_J": float(fs["best"]["J"]),
           "fine_scan_min_switch_s": float(fs["best"]["switch_s"])}

    # score the two arms at the improved switch time
    arms = {}
    kern.averaged_Q(s)
    tr = sqm.sequential_forward(kern, seg, [d_best, total - d_best], N_STEPS)
    m, _p, _j = score(tr, case, s, wide, narrow,
                      {"note": "UNIFORM map, switch time refined from an "
                               "INTERIOR start, 6 gradient evaluations",
                       "schedule": "sequential", "segments_deg": seg_deg,
                       "durations_s": [d_best, total - d_best]})
    m["arm"] = "S_seq_uniform_two_refined_interior"
    log_row(shape, m)
    arms[m["arm"]] = m
    del tr

    d_map = np.load(OUT / f"{shape}_seq_maps.npz")
    s_q = np.where(pm, np.clip(np.asarray(d_map["sat_S_seq_cosolved_4bpp"],
                                          dtype=float), 0.0, 1.0), 1.0)
    kern.averaged_Q(s_q)
    tr = sqm.sequential_forward(kern, seg, [d_best, total - d_best], N_STEPS)
    m, _p, _j = score(tr, case, s_q, wide, narrow,
                      {"note": "the co-solved 4 bits-per-pixel map, run at the "
                               "interior-refined switch time. The map was "
                               "solved at the OLD switch time and is NOT "
                               "re-solved here",
                       "schedule": "sequential", "segments_deg": seg_deg,
                       "durations_s": [d_best, total - d_best]})
    m["arm"] = "S_seq_cosolved_4bpp_interior_switch"
    log_row(shape, m)
    arms[m["arm"]] = m
    out["arms"] = arms
    out["wall_s"] = time.perf_counter() - t0
    (OUT / f"{shape}_kink.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[{shape}/kink] wrote {OUT / f'{shape}_kink.json'} in "
          f"{out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "L_shape")
