"""solve3d Stage B4: lower-drive, headroom-margin augmented Lagrangian.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_stage_b4.py -x -q

WHY B4 (verify_stage_b3_square_heatr3d.json): B3's augmented-Lagrangian shaped map
at 0.40x is is_shippable on dolfinx (true peak 249.90 <= 250) but is_sendable=
FALSE cross-engine -- the Studio's heatr3d puts the SAME map at 257.0 C (7 C over
250). Root cause: the AL drives MY (dolfinx) shaped peak TO the ceiling by
construction, so targeting 250 lands heatr3d ~7 C over (engine offset + shaped-map
relocation). B4 makes the shaped map pass BOTH engines' ceiling gates with two
coupled changes and NO new physics (it reuses ALL of B3's FD-gated machinery):

  1. EFFECTIVE CEILING. Run the AL targeting T_eff = 250 - Delta_headroom,
     Delta_headroom = 15 C (max-over-engines Delta_dopant from
     reloc_phase2_square_heatr3d.json: 11 dolfinx / 13 heatr3d, rounded up). So the
     restoration shift drives MY shaped peak to 235, and heatr3d (offset + shaped
     relocation ~+7) lands ~242 <= 250. The restoration shift is keyed to the
     dolfinx arbiter as in B3, just against T_eff=235 instead of 250.
  2. LOWER DRIVE. The uniform-map dolfinx peak must be a few C UNDER T_eff=235 so
     the AL has a feasible region + room to shape UP to 235. At 0.40x uniform is
     239.99 (too hot for a 235 target). drive_probe measures the uniform peak at
     candidate drives and picks the backed-off drive giving uniform ~228-232 C.

NO FALSE-GREEN: the T_eff=235 is the AL SOFT target only. is_shippable and the
cross-engine is_sendable gate BOTH read the TRUE peak against the REAL 250 ceiling
(b4_targets split). The cross-engine is_sendable verdict is a SEPARATE heatr3d
verify step the main session runs after this solve, exactly as with B3.

The combined AL gradient is UNCHANGED from B3 (same envelope adjoint, same
FD-gated dks_peak_ds, same max(0, lambda+mu*g) factor); fd_gate_reconfirm_b4 only
re-confirms the numbers still agree at the backed-off absolute drive.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from solve3d import density_adjoint as da
from solve3d import stage_a
from solve3d import stage_a_phase2 as p2
from solve3d import stage_b
from solve3d import stage_b3 as b3

RESULTS = Path(__file__).resolve().parent / "results"

# max-over-engines Delta_dopant (reloc: 11 dolfinx / 13 heatr3d, round up for safety)
DELTA_HEADROOM_C = 15.0
DRIVE_CANDIDATES = (0.34, 0.36, 0.38)          # 0.40x uniform is 239.99, too hot for T_eff=235
UNIFORM_TARGET_BAND_C = (228.0, 232.0)         # backed-off uniform peak band (a few C < T_eff)


# --------------------------------------------------------------------------- #
# Pure-logic target split + drive mapping (no physics)
# --------------------------------------------------------------------------- #
def t_ceiling_eff(ceiling: float, delta_headroom: float) -> float:
    """The EFFECTIVE ceiling the AL restoration shift targets: T_eff = ceiling -
    Delta_headroom. A SOFT target only -- is_shippable / is_sendable are judged
    against the REAL `ceiling` (see b4_targets). At Delta_headroom=15, T_eff=235;
    the AL drives MY dolfinx peak to 235 so heatr3d (offset + shaped relocation
    ~+7) lands ~242 <= 250."""
    return float(ceiling) - float(delta_headroom)


def b4_targets(ceiling: float, delta_headroom: float) -> dict:
    """Split the AL soft target from the real acceptance ceiling. NO FALSE-GREEN:
    the restoration shift targets `t_ceiling_eff_c` (235), but is_shippable and the
    cross-engine is_sendable gate BOTH read the TRUE peak against `real_ceiling_c`
    (250). Conflating the two would certify a map against the soft 235 target --
    exactly the false-green this split prevents."""
    return {
        "real_ceiling_c": float(ceiling),
        "delta_headroom_c": float(delta_headroom),
        "t_ceiling_eff_c": t_ceiling_eff(ceiling, delta_headroom),
        "rule": "restoration shift targets t_ceiling_eff; is_shippable/is_sendable "
                "judge the TRUE peak vs real_ceiling",
    }


def power_density_for_drive_a(a: float) -> float:
    """Absolute power density (W/m^3) for drive multiplier `a` = a * baseline,
    reusing the Stage A baseline (recommended_power_settings). The B4 drive-backoff
    is a lower `a` than the 0.40x chosen drive; nothing else about the forward
    changes."""
    return float(stage_a.recommended_power_settings(float(a))
                 ["power_density_w_per_m3"])


def pick_backed_off_drive(peaks: dict, band: tuple = UNIFORM_TARGET_BAND_C) -> dict:
    """Select the backed-off drive from measured uniform peaks.

    `peaks` maps drive_a -> uniform_dolfinx_peak_c. Prefer the HIGHEST drive whose
    uniform peak lands in `band` (~228-232 C): the highest such drive keeps the
    most part throughput while leaving the AL room to shape UP toward T_eff=235 and
    still stay feasible. If none land in band, fall back to the drive whose peak is
    closest to the band centre (reported honestly, not silently)."""
    lo, hi = float(band[0]), float(band[1])
    in_band = {a: p for a, p in peaks.items() if lo <= float(p) <= hi}
    if in_band:
        a = max(in_band)                       # highest drive in band
        return {"drive_a": float(a), "uniform_peak_c": float(peaks[a]),
                "in_band": True, "band_c": [lo, hi]}
    ctr = 0.5 * (lo + hi)
    a = min(peaks, key=lambda k: abs(float(peaks[k]) - ctr))
    return {"drive_a": float(a), "uniform_peak_c": float(peaks[a]),
            "in_band": False, "band_c": [lo, hi]}


# --------------------------------------------------------------------------- #
# Task A: FD-gate RE-CONFIRM at the backed-off drive (gradient UNCHANGED)
# --------------------------------------------------------------------------- #
def fd_gate_reconfirm(power_density: float, lam: float = b3.GATE_LAMBDA,
                      mu: float = b3.GATE_MU, h: float = 1e-4,
                      tol: float = 1e-6) -> dict:
    """Re-confirm the combined AL gradient FD gate at the B4 backed-off drive.

    The gradient composition is UNCHANGED from B3 (same envelope adjoint, same
    FD-gated dks_peak_ds, same max(0, lambda+mu*g) factor); only the absolute drive
    differs, so this re-confirms the numbers still agree, it is not a new
    derivation. The hinge is kept ACTIVE by setting t_target just BELOW the coarse
    KS peak MEASURED at this drive (a lower drive => a lower coarse peak, so the
    fixed 204 C B3 gate constant would go inactive). Central difference on the
    top-|g| probe indices; PASS when worst_rel_err <= tol (expect ~1e-8)."""
    probe = b3.build_al_coarse_case(lam=lam, mu=mu, t_target=0.0,
                                    power_density=power_density)
    v = probe.design_point()
    ks_coarse = float(da.ks_peak_forward(probe.da_case, v))
    t_target = ks_coarse - 1.0                 # just below the peak: hinge active
    case = b3.build_al_coarse_case(lam=lam, mu=mu, t_target=t_target,
                                   power_density=power_density)
    J, g = b3.al_objective_and_grad(case, v)
    assert np.all(np.isfinite(g)), "non-finite AL gradient at backed-off drive"
    worst = 0.0
    probes = case.probe_indices()
    for i in probes:
        vp = v.copy(); vp[i] += h
        vm = v.copy(); vm[i] -= h
        Jp, _ = b3.al_objective_and_grad(case, vp)
        Jm, _ = b3.al_objective_and_grad(case, vm)
        fd = (Jp - Jm) / (2 * h)
        rel = abs(fd - g[i]) / max(1.0, abs(fd))
        worst = max(worst, rel)
    _J0, g_drop = b3.al_objective_and_grad(case, v, _drop_al_term=True)
    i0 = probes[0]
    mutation_bites = abs(g[i0] - g_drop[i0]) > 1e-3 * max(1.0, abs(g[i0]))
    return {"power_density_w_per_m3": float(power_density),
            "ks_coarse_c": ks_coarse, "gate_t_target_c": float(t_target),
            "worst_rel_err": float(worst), "tol": float(tol),
            "launch_ok": bool(worst <= tol), "mutation_bites": bool(mutation_bites),
            "hinge_active": bool(b3.al_gradient_factor(lam, mu, ks_coarse - t_target)
                                 > 0.0)}


# --------------------------------------------------------------------------- #
# Task B: the drive-backoff probe (uniform dolfinx peak at each candidate)
# --------------------------------------------------------------------------- #
def drive_probe(candidates: tuple = DRIVE_CANDIDATES,
                band: tuple = UNIFORM_TARGET_BAND_C,
                delta_headroom: float = DELTA_HEADROOM_C) -> dict:
    """Measure the UNIFORM (s=1) dolfinx hold-out peak at each candidate drive and
    pick the backed-off drive whose uniform peak lands a few C UNDER T_eff.

    Reuses stage_b.uniform_holdout_peak (the SAME arbiter the AL restoration shift
    reads), so the probe peak and the AL feasibility test are like-for-like. A
    handful of densify forwards per drive, not a heavy solve. Writes
    stage_b4_drive_probe.json (no false-green: peaks are MEASURED)."""
    ceiling_c = float(stage_a.thermal_config()["T_ceiling_C"])
    t_eff = t_ceiling_eff(ceiling_c, delta_headroom)
    peaks = {}
    records = []
    for a in candidates:
        pw = power_density_for_drive_a(a)
        rec = stage_b.uniform_holdout_peak(power_density=pw, drive_a=a)
        tp = float(rec["true_peak_c"])
        peaks[float(a)] = tp
        records.append({"drive_a": float(a), "power_density_w_per_m3": float(pw),
                        "uniform_true_peak_c": tp,
                        "under_t_eff": bool(tp < t_eff),
                        "margin_to_t_eff_c": float(t_eff - tp)})
        print(f"[B4 probe drive={a:.2f}x pw={pw:.1f}] uniform peak={tp:.2f}C "
              f"(T_eff={t_eff:.1f}, margin={t_eff - tp:+.2f}C)", flush=True)
    pick = pick_backed_off_drive(peaks, band=band)
    doc = {
        "what": "Stage B4 drive-backoff probe: uniform (s=1) dolfinx hold-out peak "
                "at candidate drives, to pick the backed-off drive whose uniform "
                "peak sits a few C UNDER T_eff so the AL has a feasible region to "
                "shape up into. Peaks are MEASURED via stage_b.uniform_holdout_peak "
                "(the same arbiter the AL restoration shift reads).",
        "stage": "B4_drive_backoff_probe",
        "part": "square",
        "ceiling_c": ceiling_c,
        "delta_headroom_c": float(delta_headroom),
        "t_ceiling_eff_c": t_eff,
        "target_band_c": list(band),
        "candidates": records,
        "pick": pick,
        "rule": "pick the HIGHEST drive whose uniform peak is in-band (~228-232), "
                "leaving AL room to shape UP toward T_eff=235 while staying feasible",
    }
    stage_b._write_json(RESULTS / "stage_b4_drive_probe.json", doc)
    print(f"[B4 probe] PICK drive={pick['drive_a']:.2f}x "
          f"uniform={pick['uniform_peak_c']:.2f}C in_band={pick['in_band']}",
          flush=True)
    return doc


# --------------------------------------------------------------------------- #
# Task C: the outer AL solve at the backed-off drive + T_eff (heavy -- STOP)
# --------------------------------------------------------------------------- #
def run_solve_al_b4(drive_a: float, delta_headroom: float = DELTA_HEADROOM_C,
                    outer_max: int = b3.OUTER_MAX,
                    inner_budget: int = b3.INNER_BUDGET,
                    holdout_nodes: int = stage_b.SOLVE_HOLDOUT_NODES,
                    holdout_lc0: float = stage_b.SOLVE_HOLDOUT_LC0_M) -> dict:
    """The outer augmented-Lagrangian solve at the BACKED-OFF drive with the
    EFFECTIVE ceiling T_eff = ceiling - delta_headroom as the restoration-shift
    target. Two coupled B4 changes vs b3.run_solve_al, NO new physics:
      1. drive = power_density_for_drive_a(drive_a) (< 0.40x), threaded through the
         AL case, the KS forward, and the arbiter gate.
      2. the restoration shift targets T_eff (235), so the AL drives MY dolfinx
         shaped peak to 235; heatr3d (offset + shaped relocation ~+7) should land
         ~242 <= 250.
    is_shippable AND honest_null STILL read the TRUE peak vs the REAL 250 ceiling
    (b4_targets split -- no false-green vs the soft 235 target). The cross-engine
    is_sendable verdict is a SEPARATE heatr3d verify step (the main session runs it
    after this solve, exactly as with B3).

    REFUSES to start unless B1 launch_ok AND the B2 _march fidelity check are both
    green (stage_b._preconditions), same guard as b3.run_solve_al."""
    pre = stage_b._preconditions()
    if not (pre["launch_ok"] and pre["fidelity_agree"]):
        raise RuntimeError(
            f"run_solve_al_b4 refused: launch_ok={pre['launch_ok']} "
            f"fidelity_agree={pre['fidelity_agree']}. Both must be green before the "
            "heavy B4 AL solve.")
    from solve3d.phase_e import checkpoint as ck

    t0 = time.perf_counter()
    ceiling_c = float(stage_a.thermal_config()["T_ceiling_C"])
    rho_t = float(stage_a.thermal_config()["rho_target"]["practical_ideal"])
    targets = b4_targets(ceiling_c, delta_headroom)
    t_eff = targets["t_ceiling_eff_c"]
    pw = power_density_for_drive_a(drive_a)

    # uniform feasibility at the backed-off drive (NOT the cached 0.40x artifact)
    uniform_rec = stage_b.uniform_holdout_peak(power_density=pw, drive_a=drive_a)
    uniform_tp = float(uniform_rec["true_peak_c"])

    lam, mu, delta = b3.LAM0, b3.MU0, 0.0
    t_target = t_eff                          # shift targets T_eff, not 250
    case = b3.build_al_solve_case(lam, mu, t_target, power_density=pw)
    dcase, chain = case.da_case, case.da_case.chain
    n = chain.n_design
    v = np.ones(n)

    status = RESULTS / "stage_b4_square_status.json"
    stage_b._write_json(status, {
        "state": "running", "pid": os.getpid(), "n_design": int(n),
        "drive_a": float(drive_a), "power_density_w_per_m3": float(pw),
        "t_ceiling_eff_c": float(t_eff), "real_ceiling_c": float(ceiling_c),
        "outer_max": int(outer_max), "inner_budget": int(inner_budget),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    outers = []
    true_peak = float("nan")
    ks_solve = float("nan")
    viol_prev = float("inf")
    converged = False
    for k in range(int(outer_max)):
        case.lam, case.mu, case.t_target = float(lam), float(mu), float(t_target)

        def fg(vv):
            J, g = b3.al_objective_and_grad(case, vv)
            fg.last_wall = time.perf_counter() - t0
            return J, g
        fg.last_wall = 0.0

        def on_eval(vv, J, g, _k=k):
            ks = da.ks_peak_forward(dcase, vv)
            print(f"  [B4 outer={_k} lam={case.lam:.3e} mu={case.mu:.0e} "
                  f"t*={case.t_target:.2f}] eval J={J:.6e} KS={ks:.2f}C "
                  f"|g|={np.linalg.norm(g):.3e} wall={fg.last_wall:.0f}s", flush=True)
            stage_b._write_json(status, {
                "state": "running", "pid": os.getpid(), "outer": _k,
                "lam": float(case.lam), "mu": float(case.mu),
                "t_target_c": float(case.t_target), "last_J": float(J),
                "last_ks_peak_c": float(ks),
                "wall_s": round(time.perf_counter() - t0, 1),
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            return {"ks_peak_c": float(ks)}

        ckpt = RESULTS / f"ckpt_stage_b4_square_outer{k}.npz"
        res = ck.run_with_checkpoint(fg, v, int(inner_budget), ckpt,
                                     bounds=(0.0, 1.0), scale_first_step=True,
                                     on_eval=on_eval)
        v = np.asarray(res["best_v"], float)

        # arbiter: TRUE end-state peak on the mesh hold-out at the backed-off drive
        s_map = chain.design_to_map(v, 0.0)
        gate = p2.ceiling_end_state_gate(
            s_map, chain.centroids, holdout_nodes=int(holdout_nodes),
            holdout_lc0=float(holdout_lc0), rho_target=rho_t, power_density=pw)
        true_peak = float(gate["true_peak_c"])
        ks_solve = float(da.ks_peak_forward(dcase, v))

        # multiplier update + restoration shift keyed to T_eff (not 250)
        g_con = ks_solve - t_target
        lam = b3.multiplier_update(lam=lam, mu=mu, g=g_con)
        shift = b3.restoration_shift(ceiling=t_eff, true_peak=true_peak,
                                     ks_peak=ks_solve, prev_delta=delta,
                                     ema=b3.DELTA_EMA)
        delta, t_target = shift["delta"], shift["t_target"]
        viol_now = abs(true_peak - t_eff)         # converge dolfinx peak to T_eff
        mu = b3.mu_escalation(mu=mu, viol_prev=viol_prev, viol_now=viol_now,
                              factor=b3.MU_FACTOR, shrink=b3.MU_SHRINK)

        outers.append({
            "outer": k, "inner_evals": int(res["evals_used"]),
            "best_J": float(res["best_J"]), "status": res.get("status"),
            "lam_after": float(lam), "mu_after": float(mu),
            "delta": float(delta), "t_target_next_c": float(t_target),
            "ks_solve_c": ks_solve, "true_peak_c": true_peak,
            "viol_vs_t_eff_c": float(viol_now),
            "arbiter_feasible_vs_real_ceiling": bool(true_peak <= ceiling_c),
            "wall_s": round(time.perf_counter() - t0, 1)})
        print(f"[B4 outer={k}] true_peak={true_peak:.2f}C KS={ks_solve:.2f}C "
              f"Delta={delta:.2f} t_target->{t_target:.2f} (T_eff={t_eff:.1f}) "
              f"lam->{lam:.3e} mu->{mu:.0e} viol_vs_Teff={viol_now:.2f}C", flush=True)

        if viol_now < b3.CONVERGE_TOL_C and true_peak <= t_eff + b3.CONVERGE_TOL_C:
            converged = True
            break
        viol_prev = viol_now

    v_best = v
    s_best = chain.design_to_map(v_best, 0.0)
    np.savez_compressed(RESULTS / "map_stage_b4_square.npz", v_raw=v_best,
                        s_map=s_best, centroids=chain.centroids,
                        volumes=chain.volumes)

    fd_and_fidelity = bool(pre["launch_ok"] and pre["fidelity_agree"])
    # is_shippable / honest_null read the REAL 250 ceiling (NOT the soft T_eff)
    verdict = stage_b.shippable_verdict(
        true_peak_c=true_peak, ks_peak_c=ks_solve, ceiling_c=ceiling_c,
        fd_gate_passed=fd_and_fidelity)
    nullv = b3.honest_null_verdict(shaped_true_peak=true_peak,
                                   uniform_true_peak=uniform_tp, ceiling=ceiling_c)

    doc = {
        "what": "Stage B4: lower-drive, headroom-margin augmented-Lagrangian dopant "
                "SHAPE-solve. Restoration shift targets T_eff = 250 - "
                f"{delta_headroom:.0f} = {t_eff:.0f} C so the dolfinx shaped peak "
                "lands under T_eff and heatr3d (offset + shaped relocation) lands "
                "<= 250; is_shippable/honest_null judge the TRUE peak vs the REAL "
                "250 ceiling. Cross-engine is_sendable is a SEPARATE heatr3d verify.",
        "stage": "B4_lower_drive_headroom_margin_al",
        "part": "square",
        "drive_a": float(drive_a),
        "power_density_w_per_m3": float(pw),
        "ceiling_c": ceiling_c,
        "targets": targets,
        "preconditions": pre,
        "outer_loop": {"outer_max": int(outer_max), "inner_budget": int(inner_budget),
                       "lam0": b3.LAM0, "mu0": b3.MU0, "mu_factor": b3.MU_FACTOR,
                       "mu_shrink": b3.MU_SHRINK, "delta_ema": b3.DELTA_EMA,
                       "converge_tol_c": b3.CONVERGE_TOL_C,
                       "converged": bool(converged)},
        "solve_mesh": {"target_nodes": stage_b.SOLVE_TARGET_NODES,
                       "lc0_m": stage_b.SOLVE_LC0_M, "dt_s": stage_b.SOLVE_DT_S,
                       "n_steps": stage_b.SOLVE_N_STEPS, "n_design": int(n)},
        "outer_iterations": outers,
        "solved_map_ks_peak_c": ks_solve,
        "acceptance": {
            "fd_gate_passed": bool(pre["launch_ok"]),
            "fidelity_check_passed": bool(pre["fidelity_agree"]),
            "true_holdout_peak_c": true_peak,
            "shippable_verdict": verdict,
            "is_shippable": bool(verdict["is_shippable"]),
            "honest_null": nullv,
            "uniform_true_peak_c": uniform_tp,
            "rule": "is_shippable = FD-gated gradient AND B2 _march fidelity AND "
                    "true hold-out peak <= REAL 250 ceiling; the T_eff=235 target is "
                    "the AL soft target only. Cross-engine is_sendable pending.",
        },
        "recommended_power_settings":
            stage_a.recommended_power_settings(float(drive_a)),
        "wall_total_s": round(time.perf_counter() - t0, 1),
    }
    stage_b._write_json(RESULTS / "stage_b4_square.json", doc)
    stage_b._write_json(status, {
        "state": "done", "pid": os.getpid(),
        "is_shippable": bool(verdict["is_shippable"]),
        "true_holdout_peak_c": true_peak, "converged": bool(converged),
        "drive_a": float(drive_a), "t_ceiling_eff_c": float(t_eff),
        "wall_total_s": doc["wall_total_s"],
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({"is_shippable": verdict["is_shippable"],
                      "true_holdout_peak_c": true_peak,
                      "t_ceiling_eff_c": t_eff, "converged": converged,
                      "reason": verdict["reason"],
                      "honest_null": nullv["verdict"],
                      "note": "cross-engine is_sendable pending heatr3d verify",
                      "wall_total_s": doc["wall_total_s"]}, indent=1), flush=True)
    return doc


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="B4 drive-backoff probe (uniform peak at candidate drives)")
    ap.add_argument("--solve", action="store_true",
                    help="B4 heavy AL solve at the backed-off drive + T_eff")
    ap.add_argument("--drive-a", type=float, default=None,
                    help="B4 backed-off drive multiplier (from the probe)")
    ap.add_argument("--outer-max", type=int, default=b3.OUTER_MAX)
    ap.add_argument("--inner-budget", type=int, default=b3.INNER_BUDGET)
    a = ap.parse_args()
    if a.probe:
        drive_probe()
    elif a.solve:
        if a.drive_a is None:
            raise SystemExit("--solve requires --drive-a X (from the probe)")
        run_solve_al_b4(drive_a=a.drive_a, outer_max=a.outer_max,
                        inner_budget=a.inner_budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
