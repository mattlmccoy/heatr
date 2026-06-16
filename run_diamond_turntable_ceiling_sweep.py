#!/usr/bin/env python3
"""
run_diamond_turntable_ceiling_sweep.py
======================================
Operator diagnostic (NOT a dissertation file).  Two jobs:

  (1) TRAJECTORY: for the diamond turntable strategies at the nominal drive,
      pull the FULL coupled transient trajectory (time_s, max_T_part_c,
      mean_phi_part) from `hist`, and report the time at which max part T first
      reaches the 250 C ceiling and phi-bar at that instant.  Confirms maxT never
      EXCEEDS 250 C anywhere along the run (the prior ceiling script only saved a
      single snapshot, not the trace).

  (2) DRIVE SWEEP: re-drive turntable_noFGM and composite_of_masks at a ladder of
      absorbed powers (analogous to L-shape Table 8.6: 500/250/150/100/50 W
      generator -> 10/5/3/2/1 W absorbed at eff 0.02) and report phi-bar / maxT /
      sigma_T / mean_rho / exposure at the ceiling-respecting evaluation point.
      Find the drive where the diamond reaches good melt while keeping maxT
      comfortably UNDER 250 C (diffusion-limited regime).

Reuses the EXACT strategy construction + transient runner from the archived
run_turntable_composite_transient.py and the ceiling evaluator from
run_turntable_composite_ceiling.py.  No physics changed.

Run:
  ./.venv-heatr3d/bin/python run_diamond_turntable_ceiling_sweep.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# the transient module (strategy construction) was archived 2026-06-12
ARCHIVE = HERE / "archive" / "2026-06-12-deprecated-prewarp-epe-and-transient-turntable" / "transient-turntable"
sys.path.insert(0, str(ARCHIVE))

import rfam_eqs_coupled as R  # noqa: E402
from run_perlayer_fgm_pilot import fgm_sat_from_proxy, thermal_proxy  # noqa: E402
from run_turntable_composite_transient import (  # noqa: E402
    build_base_kw,
    part_frame_qrf_stack,
)
from run_turntable_composite_ceiling import (  # noqa: E402
    CEILING_C,
    evaluate_ceiling_respecting,
    run_full_transient,
)

SHAPE = "diamond"
CFG_NAME = "shape_diamond_6min.yaml"
ANGLES = [0.0, 90.0, 180.0, 270.0]
DIFF_SIGMA = 3.0
MAG = 1.0
# absorbed power ladder (W, 3D) = generator_W * 0.02 efficiency.
# 500/250/150/100/50 W generator -> 10/5/3/2/1 W absorbed.
POWER_LADDER_W = [10.0, 5.0, 3.0, 2.0, 1.0]
GEN_FOR_ABS = {10.0: 500, 5.0: 250, 3.0: 150, 2.0: 100, 1.0: 50}

OUT_DIR = HERE / "outputs_eqs" / "runs" / SHAPE / "turntable_composite" / "experimental"
WORKDIR = HERE / "_tt_ceiling_diamond_sweep"


def build_strategy_qrfs(cfg, base_kw):
    qrf_noFGM, pm = part_frame_qrf_stack(cfg, base_kw, ANGLES, None)
    qrf_avg_noFGM = qrf_noFGM.mean(axis=0)
    Tproxy = np.array([thermal_proxy(q, pm, DIFF_SIGMA) for q in qrf_noFGM])

    per_masks = np.array([fgm_sat_from_proxy(Tproxy[k], pm, magnitude=MAG)
                          for k in range(len(ANGLES))])
    sat_composite = per_masks.mean(axis=0)
    sat_composite[~pm] = 0.0

    qrf_composite = part_frame_qrf_stack(cfg, base_kw, ANGLES, sat_composite)[0].mean(axis=0)
    return {
        "baseline": qrf_noFGM[0],
        "turntable_noFGM": qrf_avg_noFGM,
        "composite_of_masks": qrf_composite,
    }, pm


def time_to_ceiling(hist, ceiling=CEILING_C):
    """First time max_T_part_c reaches `ceiling`, with phi-bar there, plus peak maxT."""
    t = np.asarray(hist.get("time_s", []), dtype=float)
    mt = np.asarray(hist.get("max_T_part_c", []), dtype=float)
    ph = np.asarray(hist.get("mean_phi_part", []), dtype=float)
    peak = float(mt.max()) if mt.size else float("nan")
    idx = np.where(mt >= ceiling)[0]
    if idx.size == 0:
        return {"reaches_ceiling": False, "t_to_ceiling_s": None,
                "phi_at_ceiling": None, "peak_maxT_c": peak,
                "exceeds_ceiling": bool(peak > ceiling + 1e-6)}
    i = int(idx[0])
    return {"reaches_ceiling": True, "t_to_ceiling_s": float(t[i]),
            "phi_at_ceiling": float(ph[i]), "maxT_at_ceiling_c": float(mt[i]),
            "peak_maxT_c": peak, "exceeds_ceiling": bool(peak > ceiling + 1e-6)}


def main():
    cfg = R.load_config(HERE / "configs" / CFG_NAME)
    base_kw, x, y, dx, dy = build_base_kw(cfg)
    strategies, pm = build_strategy_qrfs(cfg, base_kw)
    _x2, _y2, _p, pmask_run, _dmask, *_ = R.make_domain(cfg)
    dt_s = float(cfg["thermal"]["dt_s"])

    # ---- (1) TRAJECTORIES at nominal drive (10 W absorbed) ----
    traj_out = {}
    nominal_abs_w = float(cfg["electric"]["generator_power_w"]) * float(
        cfg["electric"]["generator_transfer_efficiency"])
    print(f"\n##### TRAJECTORIES @ nominal {nominal_abs_w:.1f} W absorbed #####")
    for tag, qrf in strategies.items():
        state, summary, hist, opt_data, npy = run_full_transient(
            cfg, qrf, x, y, f"traj_{tag}", WORKDIR, target_power_w=None)
        tc = time_to_ceiling(hist)
        st = evaluate_ceiling_respecting(state, summary, hist, opt_data, pmask_run)
        traj_out[tag] = {
            "trajectory": {
                "time_s": list(map(float, hist.get("time_s", []))),
                "max_T_part_c": list(map(float, hist.get("max_T_part_c", []))),
                "mean_phi_part": list(map(float, hist.get("mean_phi_part", []))),
            },
            "ceiling_crossing": tc,
            "eval_point": {k: st[k] for k in (
                "eval_mode", "sigma_T_c", "max_T_c", "mean_phi", "mean_rho",
                "ceiling_respected") if k in st},
            "snapshot_time_s": float(st.get("snapshot_time_s", -1.0)),
        }
        if tc["reaches_ceiling"]:
            print(f"  {tag:<20} maxT hits 250 at t={tc['t_to_ceiling_s']:.1f}s  "
                  f"phi-bar={tc['phi_at_ceiling']:.3f}  peakMaxT={tc['peak_maxT_c']:.1f}C  "
                  f"exceeds250={tc['exceeds_ceiling']}")
        else:
            print(f"  {tag:<20} NEVER reaches 250 (peakMaxT={tc['peak_maxT_c']:.1f}C)  "
                  f"final phi-bar={hist['mean_phi_part'][-1]:.3f}")

    # ---- (2) DRIVE SWEEP for turntable_noFGM and composite_of_masks ----
    sweep_out = {}
    sweep_tags = ["turntable_noFGM", "composite_of_masks"]
    print(f"\n##### DRIVE SWEEP #####")
    for tag in sweep_tags:
        qrf = strategies[tag]
        rows = []
        for pw in POWER_LADDER_W:
            state, summary, hist, opt_data, npy = run_full_transient(
                cfg, qrf, x, y, f"sweep_{tag}_{int(pw*10)}", WORKDIR, target_power_w=pw)
            st = evaluate_ceiling_respecting(state, summary, hist, opt_data, pmask_run)
            tc = time_to_ceiling(hist)
            # exposure used at eval point: snapshot time if ceiling crossed, else full
            if st["eval_mode"] == "max_density_snapshot":
                exposure_s = float(st.get("snapshot_time_s", float("nan")))
            else:
                exposure_s = float(summary.get("time_final_s", 0.0))
            row = {
                "abs_power_w": pw,
                "generator_w": GEN_FOR_ABS.get(pw),
                "mean_phi": float(st["mean_phi"]),
                "max_T_c": float(st["max_T_c"]),
                "sigma_T_c": float(st["sigma_T_c"]),
                "mean_rho": float(st["mean_rho"]),
                "exposure_s": exposure_s,
                "eval_mode": st["eval_mode"],
                "peak_maxT_c": tc["peak_maxT_c"],
                "exceeds_ceiling": tc["exceeds_ceiling"],
                "ceiling_respected": bool(st["ceiling_respected"]),
                "t_to_ceiling_s": tc.get("t_to_ceiling_s"),
                "phi_at_ceiling": tc.get("phi_at_ceiling"),
                "frac_cells_dT_clipped_mean": float(summary.get("frac_cells_dT_clipped_mean", 0.0)),
                "energy_residual_J_per_m": float(summary.get("energy_balance_residual_final_J_per_m", 0.0)),
            }
            rows.append(row)
            print(f"  {tag:<20} {pw:4.1f}W abs ({GEN_FOR_ABS.get(pw)}W gen)  "
                  f"phi={row['mean_phi']:.3f}  maxT={row['max_T_c']:.1f}  "
                  f"sigT={row['sigma_T_c']:.2f}  rho={row['mean_rho']:.3f}  "
                  f"exp={row['exposure_s']:.1f}s  peakMaxT={row['peak_maxT_c']:.1f}  "
                  f"[{st['eval_mode']}]")
        sweep_out[tag] = rows

    payload = {
        "shape": SHAPE, "config": CFG_NAME, "angles_deg": ANGLES,
        "magnitude": MAG, "diff_sigma": DIFF_SIGMA, "ceiling_c": CEILING_C,
        "dt_s": dt_s, "nominal_abs_power_w": nominal_abs_w,
        "power_ladder_abs_w": POWER_LADDER_W,
        "note": ("MEASURED this run. Trajectories at nominal drive + drive sweep. "
                 "Reuses archived strategy construction; no physics changed."),
        "trajectories": traj_out,
        "drive_sweep": sweep_out,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "diamond_turntable_ceiling_sweep.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
