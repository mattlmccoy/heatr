#!/usr/bin/env python3
"""
run_lshape_lowpower_sweep.py
============================
Test Matt's hypothesis: the reentrant L-shape's "thermally infeasible" verdict from the
prior 500 W ceiling re-run (turntable_composite_ceiling_L_shape.json) is an ARTIFACT of
the high (fast-heating) operating point, NOT a fundamental limit.

PHYSICS UNDER TEST
------------------
At 500 W / 2 % drive (=10 W absorbed 3D) the doped lobe races to 250 C in ~55-67 s --
faster than conduction can carry heat into the shadowed reentrant corner -- so the run
stops at a low mean melt fraction (phi-bar ~0.10-0.65) with a big sigma_T.

Lower the drive and lengthen the exposure: the part heats slowly, becomes
DIFFUSION-LIMITED, conduction equalizes the field, and mean melt fraction rises while the
peak stays under 250 C. Faster turntable -> closer to the rotation-average field, so the
averaged-field (static composite Qrf) model used here is the FAST-SPIN LIMIT (stated
honestly; it is an approximation of a physically spun part).

WHAT THIS DOES
--------------
For the L-shape, sweep total 3D absorbed power DOWN (10 -> 5 -> 3 -> 2 -> 1 -> 0.5 W,
i.e. roughly 500 -> 250 -> 150 -> 100 -> 50 -> 25 W generator at 2 % eff) and the time
horizon UP so each run reaches its own ceiling-limited (or phi-bar=0.90) stop. Two masks:
  - turntable_noFGM   : rotation-averaged plain Qrf (no dopant grading)  [the operating pt]
  - avgfield_composite: rotation-averaged Qrf with the averaged-field composite FGM grading
Each is evaluated at the CEILING-RESPECTING point (max_density snapshot if 250 C crossed,
else the final field if the part stays sub-ceiling for the whole exposure).

Reuses the exact transient machinery and FGM transform of
run_turntable_composite_ceiling.py -- physics unchanged.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rfam_eqs_coupled as R  # noqa: E402
from run_perlayer_fgm_pilot import fgm_sat_from_proxy, thermal_proxy  # noqa: E402
from run_turntable_composite_transient import (  # noqa: E402
    CONFIG_FOR_SHAPE,
    build_base_kw,
    part_frame_qrf_stack,
)
from run_turntable_composite_ceiling import (  # noqa: E402
    CEILING_C,
    evaluate_ceiling_respecting,
)

SHAPE = "L_shape"
ANGLES = [0.0, 90.0, 180.0, 270.0]
MAG = 1.0
DIFF_SIGMA = 3.0
DT_S = 0.5

# (label, total 3D absorbed power [W], n_steps horizon)
# Nominal drive = 500 W gen * 0.02 eff = 10 W absorbed. Lower absorbed W <-> lower gen W.
# Horizon lengthened ~ inversely with power so each reaches its ceiling-limited stop.
SWEEP = [
    ("gen500W", 10.0, 800),     # 400 s  (reproduces the prior fast-heat operating point)
    ("gen250W", 5.0, 1600),     # 800 s
    ("gen150W", 3.0, 2600),     # 1300 s
    ("gen100W", 2.0, 4000),     # 2000 s
    ("gen50W", 1.0, 8000),      # 4000 s
    ("gen25W", 0.5, 16000),     # 8000 s
]


def run_one(cfg, qrf_field, x, y, tag, workdir, target_power_w, n_steps):
    workdir.mkdir(parents=True, exist_ok=True)
    npy_path = workdir / f"qrf_{tag}.npy"
    np.save(npy_path, qrf_field.astype(np.float64))
    meta = {
        "grid_nx": int(len(x)), "grid_ny": int(len(y)),
        "x_min": float(x.min()), "x_max": float(x.max()),
        "y_min": float(y.min()), "y_max": float(y.max()),
    }
    (npy_path.parent / (npy_path.stem + "_meta.json")).write_text(json.dumps(meta))

    run_cfg = copy.deepcopy(cfg)
    run_cfg["thermal"]["n_steps"] = int(n_steps)
    run_cfg["electric"]["qrf_file_npy"] = str(npy_path)
    run_cfg["electric"]["target_power_w"] = float(target_power_w)
    run_cfg.setdefault("depth_correction", {})
    run_cfg["depth_correction"]["part_depth_m"] = float(
        cfg["geometry"]["part"].get("height", 0.02))
    run_cfg.setdefault("optimizer", {})
    run_cfg["optimizer"]["enabled"] = True
    run_cfg["optimizer"]["temp_ceiling_c"] = CEILING_C
    # phi snapshots incl 0.90 so we can detect a sub-ceiling full-melt stop
    run_cfg["optimizer"]["phi_snapshots"] = [0.5, 0.75, 0.9, 0.95]

    state, summary, hist, tt_steps, opt_data = R.run_sim(run_cfg)
    return state, summary, hist, opt_data, npy_path


def main():
    cfg_name = CONFIG_FOR_SHAPE.get(SHAPE, "shape_L_shape_6min.yaml")
    cfg = R.load_config(HERE / "configs" / cfg_name)
    cfg["thermal"]["dt_s"] = DT_S

    base_kw, x, y, dx, dy = build_base_kw(cfg)

    # --- build the two part-frame averaged Qrf fields (fast-spin limit) ---
    qrf_noFGM, pm = part_frame_qrf_stack(cfg, base_kw, ANGLES, None)
    qrf_avg_noFGM = qrf_noFGM.mean(axis=0)              # turntable_noFGM operating point
    Tproxy_avg = thermal_proxy(qrf_avg_noFGM, pm, DIFF_SIGMA)
    sat_avgfield = fgm_sat_from_proxy(Tproxy_avg, pm, magnitude=MAG)
    qrf_avgfield = part_frame_qrf_stack(cfg, base_kw, ANGLES, sat_avgfield)[0].mean(axis=0)

    masks = {
        "turntable_noFGM": qrf_avg_noFGM,
        "avgfield_composite": qrf_avgfield,
    }

    workdir = HERE / f"_lowpower_sweep_{SHAPE}"
    x2, y2, _p, pmask_run, dmask_run, *_ = R.make_domain(cfg)

    results = {}
    for mask_tag, qrf in masks.items():
        results[mask_tag] = {}
        for label, pw, nst in SWEEP:
            tag = f"{mask_tag}_{label}"
            print(f"\n========== L_shape / {mask_tag} / {label} "
                  f"(P3D={pw} W, n_steps={nst}, t_max={nst*DT_S:.0f}s) ==========")
            state, summary, hist, opt_data, npy = run_one(
                cfg, qrf, x, y, tag, workdir, target_power_w=pw, n_steps=nst)
            st = evaluate_ceiling_respecting(state, summary, hist, opt_data, pmask_run)

            # Also capture the phi=0.90 snapshot time if reached under ceiling.
            snaps = opt_data.get("snapshots", {})
            t_phi90 = None
            if 0.9 in snaps:
                t_phi90 = float(snaps[0.9][2])

            maxT_trace = np.asarray(hist.get("max_T_part_c", []), dtype=float)
            er = float(summary.get("energy_balance_residual_final_J_per_m", 0.0))
            ein = float(hist.get("energy_doped_J_per_m", [0.0])[-1])
            st.update({
                "gen_label": label,
                "target_power_w_3D": pw,
                "gen_power_w_equiv": pw / 0.02,
                "n_steps": int(nst),
                "horizon_s": float(nst * DT_S),
                "stop_time_s": float(st.get("snapshot_time_s", nst * DT_S)),
                "t_phi90_under_ceiling_s": t_phi90,
                "energy_residual_J_per_m": er,
                "energy_doped_J_per_m": ein,
                "energy_residual_pct": float(abs(er) / max(abs(ein), 1.0) * 100.0),
                "frac_cells_dT_clipped_mean": float(summary.get("frac_cells_dT_clipped_mean", 0.0)),
                "frac_cells_dT_clipped_final": float(summary.get("frac_cells_dT_clipped_final", 0.0)),
                "peak_maxT_full_horizon_c": float(maxT_trace.max()) if maxT_trace.size else float("nan"),
                "mean_T_final_full_horizon_c": float(np.asarray(hist.get("mean_T_part_c", [0]))[-1]),
                "qrf_file": str(npy),
            })
            results[mask_tag][label] = st
            print(f"  -> [{st['eval_mode']}] phi-bar={st['mean_phi']:.3f}  "
                  f"maxT={st['max_T_c']:.1f}C  sigma_T={st['sigma_T_c']:.2f}C  "
                  f"meanRho={st['mean_rho']:.3f}  stop_t={st['stop_time_s']:.0f}s  "
                  f"ceiling_ok={st['ceiling_respected']}  "
                  f"dTclip={st['frac_cells_dT_clipped_mean']:.1e}  "
                  f"Eres={st['energy_residual_pct']:.3f}%")

    out = HERE / f"lowpower_sweep_{SHAPE}.json"
    payload = {
        "shape": SHAPE, "angles_deg": ANGLES, "magnitude": MAG,
        "diff_sigma": DIFF_SIGMA, "config": cfg_name, "dt_s": DT_S,
        "ceiling_c": CEILING_C,
        "approximation_note": (
            "Averaged-field (static rotation-averaged Qrf) model = FAST-SPIN LIMIT of a "
            "physically rotated part. Faster turntable -> closer to this average."),
        "sweep_def": [{"label": l, "P3D_w": p, "n_steps": n} for (l, p, n) in SWEEP],
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n=== L_shape LOW-POWER SWEEP: phi-bar vs drive (ceiling-respecting) ===")
    hdr = f"  {'mask':<20}{'drive':<9}{'P3D_W':>6}{'phi-bar':>9}{'maxT':>8}{'sigT':>7}{'rho':>7}{'stop_s':>9}{'mode':>22}"
    print(hdr)
    for mask_tag, rr in results.items():
        for label, r in rr.items():
            print(f"  {mask_tag:<20}{label:<9}{r['target_power_w_3D']:>6.1f}"
                  f"{r['mean_phi']:>9.3f}{r['max_T_c']:>8.1f}{r['sigma_T_c']:>7.1f}"
                  f"{r['mean_rho']:>7.3f}{r['stop_time_s']:>9.0f}{r['eval_mode']:>22}")
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
