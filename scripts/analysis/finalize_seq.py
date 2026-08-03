#!/usr/bin/env python3
"""SEQUENTIAL DWELL: merge the final arms, re-snapshot, and emit the programs.

The switch time of the deliverable is the one the SCAN selected, not the one
the switch-time gradient selected, because that gradient was measured to be a
microscopic slope at this read state (see the report, and
`out_seq/<shape>_kink.json`). This script makes the stored arms, the melt
snapshots and the machine-readable programs all refer to that same schedule.

Run:
  ./.venv312/bin/python scripts/analysis/finalize_seq.py <shape>
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

from adjoint2d import library_solve as lib                          # noqa: E402
from adjoint2d import seq_dwell as sq, seq_dwell_march as sqm       # noqa: E402
from adjoint2d import shape_objective as so                         # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                      # noqa: E402
from adjoint2d.pins import load_cfg                                 # noqa: E402
from run_seq_arms import CFG, log_row, score                        # noqa: E402
from run_seq_probe import DT_S, N_STEPS, limb_masks                 # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_seq"
DELIVERABLE = "S_seq_cosolved_4bpp_interior_switch"


def main(shape: str) -> dict:
    t0 = time.perf_counter()
    r = json.loads((OUT / f"{shape}_arms.json").read_text())
    k = json.loads((OUT / f"{shape}_kink.json").read_text())
    r["arms"].update(k["arms"])
    r["kink_probe"] = {kk: vv for kk, vv in k.items() if kk != "arms"}
    r["fine_scan"] = json.loads((OUT / f"{shape}_finescan.json").read_text())["best"]
    r["prior_best_on_record"] = json.loads(
        (OUT / "prior_baseline.json").read_text())[shape]

    angles = np.asarray(CFG[shape]["angles_deg"], dtype=float)
    kern = DwellKernel.build(load_cfg(lib.shape_config(shape)), angles=angles)
    case = kern.case0
    pm = case.part_mask
    wide, narrow = limb_masks(pm)
    kern.set_weights(np.full(angles.size, 1.0 / angles.size))
    idx_of = {float(x): j for j, x in enumerate(angles)}
    a = r["arms"][DELIVERABLE]
    seg = [idx_of[x] for x in a["segments_deg"]]
    dur = np.asarray(a["durations_s"], dtype=float)
    fields = dict(np.load(OUT / f"{shape}_seq_maps.npz"))
    s_q = np.where(pm, np.clip(np.asarray(fields["sat_S_seq_cosolved_4bpp"],
                                          dtype=float), 0.0, 1.0), 1.0)
    fields[f"sat_{DELIVERABLE}"] = s_q.astype(np.float32)

    # snapshots at the DELIVERABLE schedule, and at the static baseline
    j_stat = idx_of[CFG[shape]["static_best_deg"]]
    snap_times = {}
    seq_picks: list[int] = []
    for tag, (sg, dr, ss) in {
            "seq": (seg, dur, s_q),
            "static": ([j_stat], [N_STEPS * DT_S], np.ones(pm.shape))}.items():
        kern.averaged_Q(ss)
        tr = sqm.sequential_forward(kern, sg, dr, N_STEPS)
        i_stop = int(so.optimal_stop(tr, case).index)
        sw = float(np.cumsum(np.asarray(dr, dtype=float))[0])
        i_sw = int(round(sw / DT_S))
        if tag == "seq":
            picks = sorted({int(0.5 * i_sw), i_sw - 1,
                            i_sw + int(0.34 * max(i_stop - i_sw, 0)),
                            i_sw + int(0.67 * max(i_stop - i_sw, 0)), i_stop})
            picks = [p for p in picks if 0 <= p <= i_stop]
            seq_picks = list(picks)
        else:
            # the SAME wall-clock times as the sequential row, so row (c) is a
            # like-for-like comparison and not a single end-state thumbnail
            picks = [min(p, tr.n_outer - 1) for p in seq_picks]
        for p_i in picks:
            phi_p, _ = so.phi_field(tr.T_at_end(p_i), case)
            fields[f"snap_{tag}_{p_i}"] = phi_p.astype(np.float32)
        if tag == "seq":
            m, phi, jc = score(tr, case, ss, wide, narrow, dict(a))
            m["arm"] = DELIVERABLE
            r["arms"][DELIVERABLE] = m
            fields[f"phi_{DELIVERABLE}"] = phi.astype(np.float32)
            fields[f"J_curve_{DELIVERABLE}"] = jc.astype(np.float32)
            log_row(shape, m)
        snap_times[tag] = {"steps": picks,
                           "times_s": [(p + 1) * DT_S for p in picks],
                           "switch_s": sw, "stop_step": i_stop,
                           "stop_s": (i_stop + 1) * DT_S,
                           "segments_deg": [float(angles[x]) for x in sg]}
        del tr
    # the uniform-map arm at the same schedule, for the melt-vs-nominal row
    for name in ("S_seq_uniform_two_refined_interior",):
        if name not in r["arms"]:
            continue
        aa = r["arms"][name]
        kern.averaged_Q(np.ones(pm.shape))
        tr = sqm.sequential_forward(
            kern, [idx_of[x] for x in aa["segments_deg"]],
            np.asarray(aa["durations_s"], dtype=float), N_STEPS)
        _m, phi, jc = score(tr, case, np.ones(pm.shape), wide, narrow)
        fields[f"phi_{name}"] = phi.astype(np.float32)
        fields[f"J_curve_{name}"] = jc.astype(np.float32)
        fields[f"sat_{name}"] = np.ones(pm.shape, dtype=np.float32)
        del tr
    # the PREVIOUS pass's best arm on record, so the comparison row in the
    # figure is against the real baseline and not a re-derived one
    from adjoint2d import dwell, dwell_march as dmarch                # noqa: E402
    from run_seq_arms import CYCLE_ANGLES, CYCLE_TIME_S               # noqa: E402
    pb = r["prior_best_on_record"]
    kc = DwellKernel.build(load_cfg(lib.shape_config(shape)), angles=CYCLE_ANGLES)
    w = np.asarray(pb["dwell_weights"], dtype=float)
    sat = np.load(REPO / "fgm_solve_campaign/out_dwell"
                  / f"{shape}_dwell_maps.npz")[f"sat_{pb['source_arm']}"]
    s_pb = np.where(pm, np.clip(np.asarray(sat, dtype=float), 0.0, 1.0), 1.0)
    kc.set_weights(w)
    kc.averaged_Q(s_pb)
    prog_c = dwell.cycle_program(w, CYCLE_ANGLES, cycle_time_s=CYCLE_TIME_S,
                                 total_s=N_STEPS * DT_S, dt_s=DT_S)
    tr = dmarch.program_forward(
        kc, dmarch.program_step_positions(prog_c, CYCLE_ANGLES, DT_S, N_STEPS),
        N_STEPS)
    m, phi, jc = score(tr, kc.case0, s_pb, wide, narrow, dict(pb))
    m["arm"] = "S_prior_best_on_record"
    r["arms"]["S_prior_best_on_record"] = m
    fields["phi_S_prior_best_on_record"] = phi.astype(np.float32)
    fields["J_curve_S_prior_best_on_record"] = jc.astype(np.float32)
    fields["sat_S_prior_best_on_record"] = s_pb.astype(np.float32)
    log_row(shape, m)
    del kc, tr

    r["snapshots"] = snap_times

    p = sq.sequential_program(a["segments_deg"], list(range(len(a["segments_deg"]))),
                              a["durations_s"], DT_S, N_STEPS * DT_S, snap=True)
    prog = p.as_json()
    prog.update({
        "shape": shape, "arm": DELIVERABLE, "grid": 120,
        "recommended_stop_s": r["arms"][DELIVERABLE]["t_stop_s"],
        "J_phi_at_stop": r["arms"][DELIVERABLE]["J"],
        "IoU_at_stop": r["arms"][DELIVERABLE]["IoU"],
        "peak_temperature_up_to_stop_c": r["arms"][DELIVERABLE]["max_T_upto_stop_c"],
        "dopant_map": f"{shape}_seq_maps.npz::sat_{DELIVERABLE}",
        "radio_frequency_program": "constant, calibrated drive voltage",
        "switch_time_chosen_by": ("a 5 s scan, NOT the switch-time gradient; "
                                  "see out_seq/<shape>_kink.json"),
        "caveat": ("grid 120 only; not verified on the production engine, "
                   "which cannot express an unequal or sequential dwell")})
    (OUT / f"{shape}_turntable_DELIVERABLE.json").write_text(
        json.dumps(prog, indent=2, default=float))
    r["deliverable_arm"] = DELIVERABLE
    r["finalize_wall_s"] = time.perf_counter() - t0
    (OUT / f"{shape}_arms.json").write_text(json.dumps(r, indent=2, default=float))
    np.savez_compressed(OUT / f"{shape}_seq_maps.npz", **fields)
    print(f"[{shape}] finalized in {r['finalize_wall_s']:.0f} s; program "
          f"{OUT / f'{shape}_turntable_DELIVERABLE.json'}")
    return r


if __name__ == "__main__":
    for sh in (sys.argv[1:] or ["L_shape", "T_shape"]):
        main(sh)
