#!/usr/bin/env python3
"""
run_turntable_composite_ceiling.py
==================================
Dissertation note 31 -- CEILING-RESPECTING re-run of the turntable composite-mask
study (Matt's correction).

WHY THIS EXISTS
---------------
The prior run (run_turntable_composite_transient.py -> turntable_composite_transient_*.json)
evaluated every strategy at T_phi90 (the time the part reaches 90 %% mean melt).  At a
fixed 500 W / 2 %% generator drive that pushed PEAK part T to 320-505 C -- far over the
250 C material ceiling.  That is an over-driven / invalid operating point.

This script does NOT change the physics or the dopant strategies.  It re-evaluates each
strategy at the CEILING-RESPECTING operating point: the best melt state the part reaches
while max part T stays under 250 C.  The coupled engine already tracks this -- it saves a
"max_density" snapshot = the full (T, phi, rho) field at the LAST step before max part T
crosses opt.temp_ceiling_c (=250 C).  We read that snapshot and report sigma_T, max T,
mean_phi and mean_rho there.

DECISION RULE per strategy
--------------------------
  - run the full coupled transient at the nominal drive;
  - if max(max_T_part_c over the whole run) < 250 C  -> the part is ceiling-SAFE at full
    exposure: evaluate at T_phi90 (full melt is reached under the ceiling). Report the
    peak-over-time max T as the ceiling check.
  - else (ceiling violated mid-run) -> evaluate at the "max_density" snapshot (the state
    just before the ceiling crossing). This is the best achievable melt state under the
    ceiling. Report sigma_T / max T / mean_phi there.

This makes the comparison ceiling-respecting WITHOUT forcing an over-ceiling run and
WITHOUT manufacturing a melt that cannot exist under the constraint.  If the L-shape
cannot reach full melt under 250 C at any of its angles (the honest expected outcome),
that is reported as the finding: best sigma_T and melt fraction reached at the ceiling,
and whether the composite mask still helps relative to turntable-alone under it.

GRADIENTS
---------
None.  The FGM grading is a closed-form algebraic transform of Qrf (the same
fgm_sat_from_proxy rule as the prior pilot).  No finite-difference gate applies (stated
honestly).  Numerical health is verified instead: energy-balance residual, dT-clip
fraction, and phi reached are reported per strategy.

Run under the heatr3d venv:
  ./.venv-heatr3d/bin/python run_turntable_composite_ceiling.py --shape circle
  ./.venv-heatr3d/bin/python run_turntable_composite_ceiling.py --shape L_shape
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rfam_eqs_coupled as R  # noqa: E402
from run_perlayer_fgm_pilot import (  # noqa: E402
    fgm_sat_from_proxy,
    rotate_field,
    solve_orientation,
    thermal_proxy,
)
from run_turntable_composite_transient import (  # noqa: E402
    CONFIG_FOR_SHAPE,
    build_base_kw,
    part_frame_qrf_stack,
)

CEILING_C = 250.0


def run_full_transient(cfg, qrf_field, x, y, tag, workdir, target_power_w=None):
    """Inject a static part-frame Qrf and run the full coupled transient.

    Returns (state, summary, hist, opt_data, npy_path).
    target_power_w (3D absorbed W) overrides the default 500 W * eff drive when given,
    so the same exposure-length run can be re-driven to respect the ceiling.
    """
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
    run_cfg["electric"]["qrf_file_npy"] = str(npy_path)
    if target_power_w is None:
        gen_w = float(cfg["electric"].get("generator_power_w", 500.0))
        eff = float(cfg["electric"].get("generator_transfer_efficiency", 0.02))
        target_power_w = gen_w * eff  # 10 W absorbed (3D), the nominal drive
    run_cfg["electric"]["target_power_w"] = float(target_power_w)
    run_cfg.setdefault("depth_correction", {})
    run_cfg["depth_correction"]["part_depth_m"] = float(
        cfg["geometry"]["part"].get("height", 0.02)
    )
    run_cfg.setdefault("optimizer", {})
    run_cfg["optimizer"]["enabled"] = True            # track snapshots + max_density
    run_cfg["optimizer"]["temp_ceiling_c"] = CEILING_C

    state, summary, hist, tt_steps, opt_data = R.run_sim(run_cfg)
    return state, summary, hist, opt_data, npy_path


def _interior_stats(T, phi, rho, pmask):
    tin = T[pmask]
    return {
        "sigma_T_c": float(tin.std()),
        "mean_T_c": float(tin.mean()),
        "max_T_c": float(tin.max()),
        "p95_T_c": float(np.percentile(tin, 95.0)),
        "mean_phi": float(phi[pmask].mean()),
        "frac_phi_gt_0p5": float((phi[pmask] >= 0.5).mean()),
        "mean_rho": float(rho[pmask].mean()),
    }


def evaluate_ceiling_respecting(state, summary, hist, opt_data, pmask):
    """Pick the ceiling-respecting evaluation point and compute interior stats.

    Returns a dict with the evaluation mode, the stats at that point, and the
    peak-over-time max T (the ceiling check).
    """
    maxT_trace = np.asarray(hist.get("max_T_part_c", []), dtype=float)
    peak_maxT = float(maxT_trace.max()) if maxT_trace.size else float("nan")
    snaps = opt_data.get("snapshots", {})

    if peak_maxT < CEILING_C:
        # Part stays under the ceiling for the WHOLE exposure -> full melt is reached
        # under the ceiling. Evaluate at T_phi90 (its field), with phi/rho final.
        Tfield = np.asarray(opt_data.get("T_phi90"))
        phi_f = np.asarray(state.phi)
        rho_f = np.asarray(state.rho_rel)
        st = _interior_stats(Tfield, phi_f, rho_f, pmask)
        st["eval_mode"] = "phi90_under_ceiling"
        st["peak_maxT_over_run_c"] = peak_maxT
        st["ceiling_respected"] = True
        return st

    # Ceiling crossed mid-run -> use the max_density snapshot (state just before crossing).
    if "max_density" in snaps:
        snap_state, snap_hist, snap_t = snaps["max_density"]
        Tfield = np.asarray(snap_state.T)
        phi_f = np.asarray(snap_state.phi)
        rho_f = np.asarray(snap_state.rho_rel)
        st = _interior_stats(Tfield, phi_f, rho_f, pmask)
        st["eval_mode"] = "max_density_snapshot"
        st["snapshot_time_s"] = float(snap_t)
        st["peak_maxT_over_run_c"] = peak_maxT
        # The snapshot is the last step BEFORE crossing, so its own max T < ceiling.
        st["ceiling_respected"] = bool(st["max_T_c"] < CEILING_C)
        return st

    # Ceiling crossed on the very first tracked step (no prior snapshot): the part
    # is over-driven from t=0. Report the earliest field we have, flagged.
    Tfield = np.asarray(opt_data.get("T_phi90"))
    st = _interior_stats(Tfield, np.asarray(state.phi), np.asarray(state.rho_rel), pmask)
    st["eval_mode"] = "no_subceiling_state"
    st["peak_maxT_over_run_c"] = peak_maxT
    st["ceiling_respected"] = False
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="circle")
    ap.add_argument("--angles", default="0,90,180,270")
    ap.add_argument("--magnitude", type=float, default=1.0)
    ap.add_argument("--diff-sigma", type=float, default=3.0)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--target-power-w", type=float, default=None,
                    help="override 3D absorbed power (W) for all strategies")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    angles = [float(a) for a in args.angles.split(",")]
    cfg_name = CONFIG_FOR_SHAPE.get(args.shape, "shape_diamond_6min.yaml")
    cfg = R.load_config(HERE / "configs" / cfg_name)
    if args.shape == "square":
        cfg["geometry"]["part"]["shape"] = "square"
    if args.n_steps is not None:
        cfg["thermal"]["n_steps"] = int(args.n_steps)

    base_kw, x, y, dx, dy = build_base_kw(cfg)
    diff_sigma = args.diff_sigma
    mag = args.magnitude

    qrf_noFGM, pm = part_frame_qrf_stack(cfg, base_kw, angles, None)
    qrf_avg_noFGM = qrf_noFGM.mean(axis=0)
    Tproxy = np.array([thermal_proxy(q, pm, diff_sigma) for q in qrf_noFGM])
    Tproxy_avg = thermal_proxy(qrf_avg_noFGM, pm, diff_sigma)

    sat_avgfield = fgm_sat_from_proxy(Tproxy_avg, pm, magnitude=mag)
    per_masks = np.array([fgm_sat_from_proxy(Tproxy[k], pm, magnitude=mag)
                          for k in range(len(angles))])
    sat_composite = per_masks.mean(axis=0)
    sat_composite[~pm] = 0.0
    sat_single = fgm_sat_from_proxy(Tproxy[0], pm, magnitude=mag)

    qrf_single = part_frame_qrf_stack(cfg, base_kw, angles, sat_single)[0].mean(axis=0)
    qrf_avgfield = part_frame_qrf_stack(cfg, base_kw, angles, sat_avgfield)[0].mean(axis=0)
    qrf_composite = part_frame_qrf_stack(cfg, base_kw, angles, sat_composite)[0].mean(axis=0)
    qrf_baseline = qrf_noFGM[0]

    strategies = {
        "baseline": qrf_baseline,
        "turntable_noFGM": qrf_avg_noFGM,
        "single_static": qrf_single,
        "avgfield_composite": qrf_avgfield,
        "composite_of_masks": qrf_composite,
    }

    workdir = HERE / f"_tt_ceiling_{args.shape}"
    results = {}
    for tag, qrf in strategies.items():
        print(f"\n========== CEILING TRANSIENT: {args.shape} / {tag} ==========")
        state, summary, hist, opt_data, npy = run_full_transient(
            cfg, qrf, x, y, tag, workdir, target_power_w=args.target_power_w)
        x2, y2, _p, pmask_run, dmask_run, *_ = R.make_domain(cfg)
        st = evaluate_ceiling_respecting(state, summary, hist, opt_data, pmask_run)
        st.update({
            "energy_residual_J_per_m": float(summary.get("energy_balance_residual_final_J_per_m", 0.0)),
            "energy_doped_J_per_m": float(hist.get("energy_doped_J_per_m", [0.0])[-1]),
            "frac_cells_dT_clipped_mean": float(summary.get("frac_cells_dT_clipped_mean", 0.0)),
            "frac_cells_dT_clipped_final": float(summary.get("frac_cells_dT_clipped_final", 0.0)),
            "time_final_s": float(summary.get("time_final_s", 0.0)),
            "phi90_reached_full_exposure": bool(opt_data.get("T_phi90_reached", False)),
            "max_T_part_final_c": float(summary.get("max_T_part_final_c", 0.0)),
            "qrf_file": str(npy),
        })
        results[tag] = st
        er = st["energy_residual_J_per_m"]
        ein = max(abs(st["energy_doped_J_per_m"]), 1.0)
        print(f"  -> [{st['eval_mode']}] sigma_T={st['sigma_T_c']:.2f} C  "
              f"maxT={st['max_T_c']:.1f} C  meanT={st['mean_T_c']:.1f} C  "
              f"mean_phi={st['mean_phi']:.3f}  peakMaxT_run={st['peak_maxT_over_run_c']:.1f} C  "
              f"ceiling_ok={st['ceiling_respected']}  "
              f"dTclip={st['frac_cells_dT_clipped_mean']:.2e}  "
              f"Eres={er:.3e} ({abs(er)/ein*100:.3f}%)")

    out = Path(args.out) if args.out else HERE / f"turntable_composite_ceiling_{args.shape}.json"
    payload = {
        "shape": args.shape, "angles_deg": angles, "magnitude": mag,
        "diff_sigma": diff_sigma, "config": cfg_name,
        "n_steps": int(cfg["thermal"]["n_steps"]),
        "ceiling_c": CEILING_C,
        "target_power_w_override": args.target_power_w,
        "metric": ("sigma_T = std of T over part interior (deg C) at the CEILING-RESPECTING "
                   "evaluation point (max_density snapshot if ceiling crossed, else T_phi90), "
                   "FULL coupled transient"),
        "strategies": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        out.with_suffix(".npz"),
        part_mask=pm, sat_single=sat_single, sat_avgfield=sat_avgfield,
        sat_composite=sat_composite, qrf_avg_noFGM=qrf_avg_noFGM,
        qrf_avgfield=qrf_avgfield, qrf_composite=qrf_composite, x=x, y=y,
    )

    print(f"\n=== CEILING-RESPECTING SUMMARY ({args.shape}) — sigma_T, maxT (deg C) ===")
    print(f"  {'strategy':<22} {'sigma_T':>8} {'maxT':>8} {'mean_phi':>9} {'mean_rho':>9}  mode")
    for tag, r in results.items():
        print(f"  {tag:<22} {r['sigma_T_c']:8.2f} {r['max_T_c']:8.1f} "
              f"{r['mean_phi']:9.3f} {r['mean_rho']:9.3f}  {r['eval_mode']} "
              f"{'OK' if r['ceiling_respected'] else 'OVER'}")
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
