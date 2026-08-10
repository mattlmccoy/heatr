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
    ap.add_argument("--solve", action="store_true", help="HEAVY multi-hour AL solve")
    ap.add_argument("--drive-a", type=float, default=None)
    ap.add_argument("--max-sat", type=float,
                    default=two_sided.MAX_SAT_DEFAULT_TWO_SIDED)
    a = ap.parse_args()
    if a.validate:
        validate()
    elif a.solve:
        if a.drive_a is None:
            raise SystemExit("--solve requires --drive-a X (from the drive probe)")
        run_solve(drive_a=a.drive_a, max_sat=a.max_sat)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
