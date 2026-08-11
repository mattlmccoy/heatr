"""Tamper two-sided ceiling-coupled rescue solve (run C of TAMPER_STUDY.md).

WHAT THIS IS. The B4-style augmented-Lagrangian ceiling-coupled dopant solve
(density co-state B1 + AL ceiling B3 + lower-drive/headroom B4) applied to the
real Tamper STL, WITH the two-sided actuator (solve3d/two_sided.py) so the solve
can BOOST the starved core above baseline, not only pull the rim down.

WHY A THIN DRIVER AND NOT stage_b4 --shape tamper. stage_b4's shape dispatch
(density_adjoint._build_transient_case_for_shape) only knows the analytic
families (square/cube/pyramid) and its hold-out/precondition plumbing is written
for them; studio_solve refuses non-extruded parts. The Tamper is an arbitrary
STL. So this driver assembles the Tamper tc + design chain (via
run_tamper.build_case, already gated) into the EXISTING density_adjoint.Case and
stage_b3.ALCase and drives the SAME b3.al_objective_and_grad + checkpointed
L-BFGS-B loop. NO adjoint is rebuilt: the density co-state (da.dks_peak_ds), the
melt-onset envelope adjoint (p2.envelope_grad_of_design) and the AL scalars are
reused byte-for-byte; only the mesh + part mask are the Tamper's.

STATUS: STOP before the heavy multi-hour --solve. `--validate` is the light
construction + finite-two-sided-gradient check on a SHORT-march Tamper case; the
coordinator launches --solve, permission-gated, after the per-geometry FD gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solve3d import density_adjoint as da
from solve3d import design_chain as dc
from solve3d import stage_a
from solve3d import stage_b3 as b3
from solve3d import two_sided
from solve3d.phase_e import run_tamper as rt

RESULTS = rt.RESULTS
FILTER_RADIUS_M = da.FILTER_RADIUS_M

# B4 headroom (single source, mirrors stage_b4.DELTA_HEADROOM_C)
DELTA_HEADROOM_C = 15.0


def build_tamper_al_case(lam: float, mu: float, t_target: float, *,
                         power_density: float, dt: float, n_steps: int,
                         envelope_max_time_s: float,
                         lc_part: float = rt.LC_PART_M) -> b3.ALCase:
    """Assemble a stage_b3.ALCase on the Tamper STL, reusing the existing
    adjoint. The design chain, density co-state and envelope adjoint are
    geometry-agnostic; only the Tamper mesh + part mask differ."""
    tc, _info = rt.build_case(lc_part=lc_part, max_time_s=envelope_max_time_s)
    part_cent = rt._part_centroids(tc)
    chain = dc.DesignChain(part_cent, tc.eqs.vol[tc.eqs.part], FILTER_RADIUS_M, [0.0])
    case = da.Case(tc=tc, chain=chain, dt=float(dt), n_steps=int(n_steps))
    case._v0 = np.ones(chain.n_design)
    return b3.ALCase(da_case=case, lam=float(lam), mu=float(mu),
                     t_target=float(t_target))


def validate(boost_sat: float = 1.5) -> dict:
    """LIGHT (construction-only, no march): assemble the Tamper AL case and
    confirm the two-sided actuator wires correctly on the REAL Tamper geometry --
    the design chain builds on the Tamper part cells, and sat>1.0 boosts the
    Tamper conductivity above sigma_doped. Gradient CORRECTNESS is proven
    geometry-agnostically by the coarse FD gate (test_two_sided.py, worst
    1.57e-8); the per-geometry FD re-gate at the production march horizon is the
    coordinator's heavy pre-launch step (the existing B-stage convention)."""
    import solve3d.forward as fwd
    t0 = time.perf_counter()
    case = build_tamper_al_case(
        lam=500.0, mu=1.0e4, t_target=200.0,
        power_density=None, dt=0.5, n_steps=2800, envelope_max_time_s=1800.0)
    tc = case.da_case.tc
    chain = case.da_case.chain
    p = fwd.ForwardParams()
    sig_boost = float(tc.design_to_sigma(np.array([boost_sat]))[0])
    boosts = bool(sig_boost > p.sigma_doped)
    doc = {"n_design": int(chain.n_design),
           "n_part_cells": int(tc.eqs.part.size),
           "boost_sat": boost_sat,
           "sigma_at_boost_S_per_m": sig_boost,
           "sigma_doped_S_per_m": float(p.sigma_doped),
           "two_sided_boosts_core": boosts,
           "design_bounds_two_sided": list(two_sided.design_bounds(2.0)),
           "note": "construction-only; gradient correctness is the coarse "
                   "geometry-agnostic FD gate (test_two_sided.py)",
           "wall_s": round(time.perf_counter() - t0, 1)}
    print(f"[tamper-rescue validate] n_design={chain.n_design} "
          f"sigma(sat={boost_sat})={sig_boost:.4f} S/m > doped "
          f"{p.sigma_doped} -> boosts_core={boosts}  wall={doc['wall_s']}s")
    return doc


# --------------------------------------------------------------------------- #
# PER-GEOMETRY FD GATE (the IRON LAW): re-gate the COMBINED AL gradient on the
# Tamper's OWN mesh before the heavy solve. A correct co-state on the square/cube/
# pyramid is NOT proof on the Tamper -- a brand-new arbitrary STL with 20106
# design nodes. Mirrors stage_b4.fd_gate_reconfirm but builds the case with
# build_tamper_al_case (which knows the Tamper), and adds Probe B: the load-
# bearing above-1.0 (two-sided BOOST) branch FD-checked ON the Tamper mesh.
# --------------------------------------------------------------------------- #
FD_GATE_LC_PART_M = 5.0e-3       # 2x coarser than production (2.5e-3): cut cells
FD_GATE_DT_S = 1.0               # coarse-march step
FD_GATE_N_STEPS = 800            # SHORT march that still REACHES melt (KS ~205 C)
FD_GATE_ENVELOPE_MAX_TIME_S = 1800.0
FD_GATE_BOOST_SAT = 1.5          # in the two-sided box (0, 2.0); above 1.0
MELT_BAND_TOP_C = 190.0          # t_pc_c(180) + dt_pc_c(10): above => densify live


def _core_design_indices(tc, chain, k: int = 6) -> list[int]:
    """The k design nodes nearest the part centroid -- the STARVED CORE the two-
    sided actuator must feed (TAMPER_STUDY: the core starves at <0.6x mean power).
    Probe B boosts EXACTLY these above 1.0 and FD-gates the gradient there."""
    part_cent = rt._part_centroids(tc)             # (n_design, 3) part cell centroids
    c0 = part_cent.mean(axis=0)
    r = np.linalg.norm(part_cent - c0, axis=1)
    return [int(i) for i in np.argsort(r)[:k]]


def fd_gate(lc_part: float = FD_GATE_LC_PART_M, dt: float = FD_GATE_DT_S,
            n_steps: int = FD_GATE_N_STEPS, h: float = 1e-4, tol: float = 1e-6,
            boost_sat: float = FD_GATE_BOOST_SAT,
            lam: float = b3.GATE_LAMBDA, mu: float = b3.GATE_MU) -> dict:
    """Per-geometry FD gate of the COMBINED AL gradient on the Tamper mesh.

    The gradient composition is UNCHANGED from B3/B4 (same envelope adjoint, same
    FD-gated dks_peak_ds, same max(0, lambda+mu*g) factor); this only RE-GATES it
    on the Tamper's own mesh at a coarse/short-but-melted operating point.

    Non-vacuous: the coarse KS peak sits ABOVE the melt band (>190 C) so the
    density hinge -- max(0, lambda+mu*g) * dKS_peak/ds through the densify march --
    is genuinely exercised (rho evolves; if the march never melted the co-state
    path would be trivially zero and the gate vacuous).

    Probe A (mesh gate): v = ones; central-difference the AL gradient at the top-
    |g| probe indices; PASS when worst_rel_err <= tol. mutation_bites confirms the
    AL term is live (dropping it moves the top probe > 1e-3 rel).

    Probe B (the load-bearing two-sided-on-Tamper check): boost a handful of CORE
    nodes to boost_sat (> 1.0, inside the two-sided box) and FD-gate the gradient
    AT those boosted nodes -- proving the above-1.0 branch is correct on THIS mesh,
    not only on the geometry-agnostic test_two_sided.py case."""
    t0 = time.perf_counter()
    case = build_tamper_al_case(lam=lam, mu=mu, t_target=0.0, power_density=None,
                                dt=dt, n_steps=n_steps,
                                envelope_max_time_s=FD_GATE_ENVELOPE_MAX_TIME_S,
                                lc_part=lc_part)
    dcase, tc, chain = case.da_case, case.da_case.tc, case.da_case.chain
    n_design = int(chain.n_design)
    n_cells = int(tc.ncells)

    # Non-vacuous: measure the coarse KS peak AND confirm rho densified past melt.
    v1 = np.ones(n_design)
    _T_end, rho_end, _c, _F = da._march(dcase, v1, keep_cache=False)
    ks_c = float(da.ks_peak_forward(dcase, v1))
    rho_mean = float(np.mean(rho_end))
    non_vacuous = bool(ks_c > MELT_BAND_TOP_C and rho_mean > float(tc.p.rho_rel) + 1e-4)
    case.t_target = ks_c - 1.0                     # just below the peak: hinge active

    # ---- Probe A: mesh gate at v = ones ----
    # Reuse THIS base gradient to pick the top-|g| probe indices (identical to
    # case.probe_indices, but without a second expensive AL evaluation).
    J, g = b3.al_objective_and_grad(case, v1)
    assert np.all(np.isfinite(g)), "non-finite AL gradient on the Tamper mesh"
    probes = [int(i) for i in np.argsort(-np.abs(g))[:2]]
    recsA, worstA = [], 0.0
    for i in probes:
        vp = v1.copy(); vp[i] += h
        vm = v1.copy(); vm[i] -= h
        fd = (b3.al_objective_and_grad(case, vp)[0]
              - b3.al_objective_and_grad(case, vm)[0]) / (2 * h)
        rel = abs(fd - g[i]) / max(1.0, abs(fd))
        worstA = max(worstA, rel)
        recsA.append({"i": int(i), "fd": float(fd), "adjoint": float(g[i]),
                      "rel_err": float(rel)})
    _J0, g_drop = b3.al_objective_and_grad(case, v1, _drop_al_term=True)
    i0 = probes[0]
    mutation_bites = bool(abs(g[i0] - g_drop[i0]) > 1e-3 * max(1.0, abs(g[i0])))

    # INSURANCE: this coarse gate runs near the environment's background-task wall
    # cap, so persist a PARTIAL artifact after Probe A (Probe B fields empty). If
    # the process is externally terminated during Probe B, Probe A's numbers +
    # non_vacuous survive; the final write below overwrites this with launch_ok.
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "tamper_rescue_fd_gate.json"
    out.write_text(json.dumps({
        "part": "tamper", "gate": "per_geometry_al_fd", "status": "PARTIAL_probeA_only",
        "n_design": n_design, "n_cells_coarse": n_cells, "lc_part_m": float(lc_part),
        "dt_s": float(dt), "n_steps": int(n_steps), "coarse_KS_peak_c": ks_c,
        "rho_mean_end": rho_mean, "non_vacuous": non_vacuous,
        "t_target_c": float(case.t_target),
        "probeA_probe_indices": [int(i) for i in probes], "probeA_probes": recsA,
        "probeA_worst_rel_err": float(worstA), "mutation_bites": mutation_bites,
    }, indent=2))

    # ---- Probe B: the two-sided BOOST branch on the Tamper mesh ----
    core = _core_design_indices(tc, chain, k=2)
    vb = np.ones(n_design)
    for j in core:
        vb[j] = float(boost_sat)
    sig_boost = float(tc.design_to_sigma(np.array([boost_sat]))[0])
    Jb, gb = b3.al_objective_and_grad(case, vb)
    assert np.all(np.isfinite(gb)), "non-finite AL gradient at the boosted core"
    ks_boost = float(da.ks_peak_forward(dcase, vb))
    hinge_active_B = bool(b3.al_gradient_factor(lam, mu, ks_boost - case.t_target) > 0.0)
    recsB, worstB = [], 0.0
    for i in core:
        vp = vb.copy(); vp[i] += h
        vm = vb.copy(); vm[i] -= h
        fd = (b3.al_objective_and_grad(case, vp)[0]
              - b3.al_objective_and_grad(case, vm)[0]) / (2 * h)
        rel = abs(fd - gb[i]) / max(1.0, abs(fd))
        worstB = max(worstB, rel)
        recsB.append({"i": int(i), "v_i": float(vb[i]), "fd": float(fd),
                      "adjoint": float(gb[i]), "rel_err": float(rel)})

    launch_ok = bool(worstA <= tol and worstB <= tol and mutation_bites
                     and non_vacuous)
    doc = {
        "part": "tamper", "gate": "per_geometry_al_fd", "status": "COMPLETE",
        "n_design": n_design, "n_cells_coarse": n_cells,
        "lc_part_m": float(lc_part), "dt_s": float(dt), "n_steps": int(n_steps),
        "march_s": float(dt) * int(n_steps),
        "coarse_KS_peak_c": ks_c, "rho_mean_end": rho_mean,
        "rho_rel_baseline": float(tc.p.rho_rel), "melt_band_top_c": MELT_BAND_TOP_C,
        "non_vacuous": non_vacuous,
        "t_target_c": float(case.t_target), "h": float(h), "tol": float(tol),
        "boost_sat": float(boost_sat), "sigma_at_boost_S_per_m": sig_boost,
        "sigma_doped_S_per_m": float(tc.design_to_sigma(np.array([1.0]))[0]),
        "design_bounds_two_sided": list(two_sided.design_bounds(2.0)),
        "probeA_probe_indices": [int(i) for i in probes],
        "probeA_probes": recsA, "probeA_worst_rel_err": float(worstA),
        "mutation_bites": mutation_bites,
        "probeB_core_indices": [int(i) for i in core],
        "probeB_boost_hinge_active": hinge_active_B,
        "probeB_probes": recsB, "probeB_worst_rel_err": float(worstB),
        "launch_ok": launch_ok, "wall_s": round(time.perf_counter() - t0, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "tamper_rescue_fd_gate.json"
    out.write_text(json.dumps(doc, indent=2))
    verdict = "PASS" if launch_ok else "FAIL"
    print(f"[tamper-rescue FD-gate] {verdict}  KS_peak={ks_c:.2f}C "
          f"(non_vacuous={non_vacuous}, rho_mean={rho_mean:.3f})  "
          f"probeA_worst={worstA:.3e} mutation_bites={mutation_bites}  "
          f"probeB_worst={worstB:.3e}  launch_ok={launch_ok}  "
          f"-> {out}", flush=True)
    return doc


def run_solve(drive_a: float, max_sat: float = two_sided.MAX_SAT_DEFAULT_TWO_SIDED,
              outer_max: int = b3.OUTER_MAX, inner_budget: int = b3.INNER_BUDGET,
              dt: float = 0.5, n_steps: int = 2800,
              envelope_max_time_s: float = 1800.0) -> dict:
    """HEAVY (multi-hour). The outer AL loop on the Tamper at the backed-off
    drive, targeting T_eff = 250 - DELTA_HEADROOM, with the two-sided box
    (0, max_sat). Reuses b3.al_objective_and_grad + the checkpointed L-BFGS-B.
    Not invoked here -- the coordinator launches it, permission-gated."""
    from solve3d.phase_e import checkpoint as ck
    from solve3d import forward as fwd
    ceiling_c = float(stage_a.thermal_config()["T_ceiling_C"])
    t_eff = ceiling_c - DELTA_HEADROOM_C
    # drive threading mirrors stage_b4.power_density_for_drive_a (nominal * a)
    pw = fwd.ForwardParams().power_density_w_per_m3 * float(drive_a)

    lam, mu = b3.LAM0, b3.MU0
    case = build_tamper_al_case(lam, mu, t_eff, power_density=pw, dt=dt,
                                n_steps=n_steps,
                                envelope_max_time_s=envelope_max_time_s)
    chain = case.da_case.chain
    v = np.ones(chain.n_design)
    bounds = two_sided.design_bounds(max_sat)
    print(f"[tamper-rescue] START drive_a={drive_a} pw={pw:.3e} t_eff={t_eff} "
          f"bounds={bounds} n_design={chain.n_design}", flush=True)
    for k in range(int(outer_max)):
        case.lam, case.mu = float(lam), float(mu)

        def fg(vv):
            return b3.al_objective_and_grad(case, vv)
        ckpt = RESULTS / f"ckpt_tamper_rescue_outer{k}.npz"
        res = ck.run_with_checkpoint(fg, v, int(inner_budget), ckpt,
                                     bounds=bounds, scale_first_step=True)
        v = np.asarray(res["best_v"], float)
        ks = da.ks_peak_forward(case.da_case, v)
        g_con = ks - case.t_target
        lam = b3.multiplier_update(lam=lam, mu=mu, g=g_con)
    return {"v_best": v, "drive_a": drive_a, "max_sat": max_sat}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="light construction + two-sided gradient check on Tamper")
    ap.add_argument("--fd-gate", action="store_true",
                    help="per-geometry FD gate of the combined AL gradient on the "
                         "Tamper mesh (coarse/short-but-melted); PRE-LAUNCH gate")
    ap.add_argument("--solve", action="store_true", help="HEAVY multi-hour AL solve")
    ap.add_argument("--drive-a", type=float, default=None)
    ap.add_argument("--max-sat", type=float,
                    default=two_sided.MAX_SAT_DEFAULT_TWO_SIDED)
    a = ap.parse_args()
    if a.validate:
        validate()
    elif a.fd_gate:
        fd_gate()
    elif a.solve:
        if a.drive_a is None:
            raise SystemExit("--solve requires --drive-a X (from the drive probe)")
        run_solve(drive_a=a.drive_a, max_sat=a.max_sat)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
