"""solve3d Stage B3: augmented Lagrangian with true-peak restoration.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_stage_b3.py -x -q

WHY THIS EXISTS (spec 2026-08-07-stage-b3-augmented-lagrangian-design.md):
B2 (the ceiling-coupled penalty solve, solve3d.stage_b) converged at fixed 0.40x
but landed is_shippable=FALSE by 0.69 C -- true hold-out peak 250.69 C while the
KS aggregate it penalized was 248.45 C. The root cause is intrinsic: the KS
mean-log-sum-exp aggregate is a LOWER bound on the true peak, so constraining
KS <= ceiling leaves the true peak ~2 C above. B3 enforces the ceiling on the
quantity that gates is_shippable -- the TRUE arbiter peak -- by construction.

Two additions, nothing re-derived:
  1. Augmented Lagrangian. Swap B2's penalty term mu*(KS-ceil)_+^2 for the
     standard inequality AL term (1/2mu)[max(0, lambda+mu*g)^2 - lambda^2] with
     g = KS_peak - T_target. Its design gradient is dJ_shape/ds +
     max(0, lambda+mu*g) * dKS_peak/ds. dJ_shape/ds is the phase-2 melt-onset
     envelope adjoint; dKS_peak/ds is the B1 rho+T density co-state, ALREADY
     FD-gated to 3.0e-9 (solve3d.density_adjoint.dks_peak_ds) -- B3 adds only the
     scalar factor max(0, lambda+mu*g). Both reads are the SAME assembly B2 uses
     (stage_b.penalty_objective_and_grad); nothing in the gated B1/B2 code is
     touched.
  2. The restoration shift (the crux). KS is a lower bound, so g on KS alone does
     not bound the true peak. Each OUTER iteration re-estimates the gap on the
     arbiter mesh, Delta_k = true_peak(arbiter, s_k) - KS_peak(solve, s_k), EMA-
     damped, and targets KS <= T_ceiling - Delta_k. Delta_k folds in BOTH the
     KS-vs-true gap and the solve-vs-arbiter mesh gap; as s_k converges Delta_k
     stabilizes and the TRUE arbiter peak -> T_ceiling. is_shippable reads the
     TRUE arbiter peak ONLY, never KS.

The AL combined gradient is FD-gated (test_al_combined_grad_matches_fd) at a
fixed (lambda, mu, t_target) with the hinge active BEFORE any solve: the density
co-state is already gated, so this confirms only the scalar AL composition.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solve3d import (density_adjoint as da, stage_a, stage_a_phase2 as p2,
                     stage_b)

RESULTS = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------- #
# Task 1: AL outer-loop pure logic (no physics)
# --------------------------------------------------------------------------- #
def multiplier_update(lam: float, mu: float, g: float) -> float:
    """KKT multiplier update for the inequality constraint g <= 0:

        lambda_new = max(0, lambda + mu * g).

    Stays >= 0; grows while violated (g > 0) and decays toward 0 while satisfied
    (g < 0). Identical scalar to al_gradient_factor -- the multiplier update and
    the gradient factor are the same max(0, lambda+mu*g), evaluated at the inner
    optimum (update) vs the current design (factor)."""
    return max(0.0, float(lam) + float(mu) * float(g))


def al_gradient_factor(lam: float, mu: float, g: float) -> float:
    """The active-branch AL design-gradient factor: max(0, lambda + mu*g).

    dL/ds = dJ_shape/ds + al_gradient_factor(lambda, mu, g) * dKS_peak/ds, with
    g = KS_peak - T_target. Zero when lambda + mu*g <= 0 (the inactive branch),
    so the density co-state contributes nothing when the constraint is slack."""
    return max(0.0, float(lam) + float(mu) * float(g))


def restoration_shift(ceiling: float, true_peak: float, ks_peak: float,
                      prev_delta: float, ema: float) -> dict:
    """Re-estimate the true-vs-KS gap on the arbiter and shift the KS target.

        Delta_raw = true_peak(arbiter, s_k) - KS_peak(solve, s_k)
        Delta_k   = ema * Delta_raw + (1 - ema) * prev_delta      (EMA damping)
        T_target  = ceiling - Delta_k

    Delta_k folds in BOTH the KS-vs-true-peak gap and the solve-vs-arbiter mesh
    gap. Constraining KS <= T_target drives the TRUE arbiter peak to the ceiling
    as s_k converges. The EMA (0 < ema <= 1) damps oscillation; ema=1 is the
    undamped shift, prev_delta is last outer iteration's Delta_k."""
    raw = float(true_peak) - float(ks_peak)
    e = float(ema)
    delta = e * raw + (1.0 - e) * float(prev_delta)
    return {"delta": float(delta), "delta_raw": float(raw),
            "t_target": float(ceiling) - float(delta)}


def mu_escalation(mu: float, viol_prev: float, viol_now: float,
                  factor: float, shrink: float) -> float:
    """Grow the penalty weight mu when the constraint violation stalls.

    If the violation did not drop by at least `shrink` fraction between outer
    iterations (viol_now > (1 - shrink) * viol_prev), multiply mu by `factor`;
    otherwise hold. A non-positive prior violation (nothing to improve on, e.g.
    the first outer iteration or an already-feasible step) holds mu."""
    vp, vn = float(viol_prev), float(viol_now)
    if vp <= 0.0:
        return float(mu)
    if vn > (1.0 - float(shrink)) * vp:
        return float(mu) * float(factor)
    return float(mu)


def honest_null_verdict(shaped_true_peak: float, uniform_true_peak: float,
                        ceiling: float) -> dict:
    """The honest-null verdict, FIXED vs B2's mislabel.

    B2's null_verdict fired "no_feasible_dopant_at_this_drive" whenever the
    SHAPED map's true peak was over the ceiling -- but a feasible dopant is known
    (the uniform map). Honest-null must fire ONLY if even the peak-minimizing /
    known-uniform map is over the ceiling. At 0.40x the uniform map is 239.99 C
    (feasible), so B3 must NOT emit no_feasible_dopant; the shaped endpoint being
    momentarily over ceiling is a solver-progress fact, not an infeasibility
    proof."""
    uni = float(uniform_true_peak)
    ceil = float(ceiling)
    uniform_over = bool(uni > ceil)
    return {
        "verdict": "no_feasible_dopant_at_this_drive" if uniform_over
        else "feasible_dopant_exists",
        "shaped_true_peak_c": float(shaped_true_peak),
        "uniform_true_peak_c": uni,
        "ceiling_c": ceil,
        "uniform_margin_c": ceil - uni,
        "rule": "honest-null fires ONLY if the uniform (known-feasible-candidate) "
                "true peak is over the ceiling; a shaped endpoint over ceiling is "
                "solver progress, not infeasibility",
    }


# --------------------------------------------------------------------------- #
# Task 2: the combined AL objective + gradient (reuses B1 dks_peak_ds, B2 shape)
# --------------------------------------------------------------------------- #
# A gate-case KS target BELOW the coarse KS peak (~204.9 C at 0.40x) so the AL
# hinge is ACTIVE (max(0, lambda+mu*g) > 0) and the factor*dKS_peak/ds path is
# exercised by the FD gate -- exactly why stage_b sets GATE_CEILING_C=200.0 below
# the same coarse peak. The REAL solve reads t_target = 250 C ceiling minus the
# restoration shift; this gate constant is a device only, documented so it cannot
# be mistaken for the physical target.
GATE_LAMBDA = 500.0
GATE_MU = 1.0e4
GATE_T_TARGET_C = 204.0
ENVELOPE_MAX_TIME_S = 1800.0     # J_shape melt-onset envelope horizon (argmin interior)


@dataclass
class ALCase:
    """Wraps a density_adjoint.Case with the AL multiplier lambda, weight mu, and
    the SHIFTED KS target t_target. J_shape (melt-onset envelope) reads a
    non-densify forward via the phase-2 envelope adjoint; KS_peak reads the
    densify end-state via the B1 density co-state. Both share the SAME mesh,
    drive and design chain, so the two reads compose on one design vector -- the
    identical assembly stage_b.PenaltyCase uses, with the penalty term swapped
    for the AL term."""
    da_case: da.Case
    lam: float
    mu: float
    t_target: float

    def design_point(self) -> np.ndarray:
        return self.da_case.design_point()

    def probe_indices(self, k: int = 4) -> list[int]:
        v = (self.da_case._v0 if self.da_case._v0 is not None
             else self.design_point())
        _J, g = al_objective_and_grad(self, v)
        return [int(i) for i in np.argsort(-np.abs(g))[:k]]


def build_al_coarse_case(lam: float = GATE_LAMBDA, mu: float = GATE_MU,
                         t_target: float = GATE_T_TARGET_C,
                         n_steps: int = da.COARSE_N_STEPS,
                         envelope_max_time_s: float = ENVELOPE_MAX_TIME_S
                         ) -> ALCase:
    """The coarse AL gate case (mirrors stage_b.build_penalty_coarse_case): the
    density march uses case.n_steps; the melt-onset envelope uses tc.max_time_s,
    set larger so the argmin sits interior."""
    case = da.build_coarse_case(n_steps=int(n_steps))
    case.tc.max_time_s = float(envelope_max_time_s)
    return ALCase(da_case=case, lam=float(lam), mu=float(mu),
                  t_target=float(t_target))


def al_objective_and_grad(case: ALCase, v: np.ndarray,
                          _drop_al_term: bool = False):
    """(L, dL/dv) for the augmented Lagrangian

        L = J_shape + (1/(2 mu)) [ max(0, lambda + mu*g)^2 - lambda^2 ],
        g = KS_peak(v) - t_target,

    design gradient dL/dv = dJ_shape/dv + max(0, lambda + mu*g) * dKS_peak/dv.

    dJ_shape/dv is the phase-2 melt-onset envelope adjoint (the SAME assembly
    stage_b.penalty_objective_and_grad uses); dKS_peak/dv is the B1 rho+T density
    co-state da.dks_peak_ds, ALREADY FD-gated to 3.0e-9 -- B3 adds only the scalar
    factor al_gradient_factor(lambda, mu, g). `_drop_al_term` ABLATES the AL
    contribution (both value and gradient), the mutation that must change the
    gradient when the hinge is active."""
    dcase, tc, chain = case.da_case, case.da_case.tc, case.da_case.chain
    v = np.asarray(v, float)
    J_shape, g_shape, _info = p2.envelope_grad_of_design(tc, chain, v, beta=0.0)
    g = np.array(g_shape, dtype=float)
    if _drop_al_term:
        return float(J_shape), g
    ks = da.ks_peak_forward(dcase, v)
    g_con = ks - case.t_target
    factor = al_gradient_factor(case.lam, case.mu, g_con)
    al_term = (1.0 / (2.0 * case.mu)) * (factor * factor - case.lam * case.lam)
    J = float(J_shape) + float(al_term)
    if factor > 0.0:
        g = g + factor * da.dks_peak_ds(dcase, v)
    return float(J), g


# --------------------------------------------------------------------------- #
# Task 3: the outer augmented-Lagrangian solve (heavy -- STOP before running)
# --------------------------------------------------------------------------- #
# The heavy AL solve reuses B2's solve mesh EXACTLY (stage_b.SOLVE_* constants),
# so the KS objective, the density march and the fidelity check the arbiter reads
# are byte-consistent with the FD-gated B1/B2 machinery. B3 changes only the
# OUTER-loop scalars around the same inner solve.
OUTER_MAX = 6                    # outer AL iterations (multiplier + shift + mu)
INNER_BUDGET = 12                # gradient evals per inner L-BFGS solve
LAM0 = 0.0                       # initial multiplier
MU0 = 1.0e3                      # initial penalty weight
MU_FACTOR = 5.0                  # mu growth on a stalled outer violation
MU_SHRINK = 0.5                  # required violation drop fraction to hold mu
DELTA_EMA = 0.5                  # restoration-shift EMA damping
CONVERGE_TOL_C = 0.25            # |true arbiter peak - ceiling| convergence band


def build_al_solve_case(lam: float, mu: float, t_target: float) -> ALCase:
    """The heavy AL case on B2's solve mesh (stage_b.SOLVE_* constants), wrapped
    with the AL scalars. Mirrors stage_b.build_penalty_solve_case, penalty term
    swapped for the AL term."""
    case = da.build_coarse_case(
        target_nodes=stage_b.SOLVE_TARGET_NODES, lc0=stage_b.SOLVE_LC0_M,
        dt=stage_b.SOLVE_DT_S, n_steps=stage_b.SOLVE_N_STEPS)
    case.tc.max_time_s = stage_b.SOLVE_ENVELOPE_MAX_TIME_S
    return ALCase(da_case=case, lam=float(lam), mu=float(mu),
                  t_target=float(t_target))


def _uniform_true_peak() -> float:
    """The uniform (s=1) true hold-out peak at 0.40x -- the known-feasible
    candidate for honest_null_verdict. Read the cached B2 artifact if present
    (239.99 C on the fine arbiter mesh), else measure it (a few densify forwards,
    not a heavy solve)."""
    cached = RESULTS / "stage_b_uniform_holdout.json"
    if cached.exists():
        doc = json.loads(cached.read_text())
        tp = doc.get("true_peak_c")
        if tp is not None:
            return float(tp)
    return float(stage_b.uniform_holdout_peak()["true_peak_c"])


def run_solve_al(outer_max: int = OUTER_MAX, inner_budget: int = INNER_BUDGET,
                 holdout_nodes: int = stage_b.SOLVE_HOLDOUT_NODES,
                 holdout_lc0: float = stage_b.SOLVE_HOLDOUT_LC0_M) -> dict:
    """The outer augmented-Lagrangian solve at the fixed 0.40x drive with
    true-peak restoration.

    Per outer iteration k:
      1. inner solve: L-BFGS-B min of al_objective_and_grad at fixed
         (lambda, mu, t_target), frozen conventions (1/|g0| rescale via the
         checkpoint's scale_first_step; filter + tanh projection via the design
         chain, beta 0 exactly as B1/B2), per-eval checkpoint (resumable).
      2. arbiter: measure the TRUE end-state peak on the mesh HOLD-OUT via
         p2.ceiling_end_state_gate (never the KS aggregate).
      3. multiplier update: lambda <- max(0, lambda + mu*(KS_solve - t_target)).
      4. restoration shift: Delta_k = EMA(true_peak - KS_solve), t_target =
         ceiling - Delta_k -- so the TRUE arbiter peak converges to the ceiling.
      5. mu escalation if |true_peak - ceiling| did not shrink by MU_SHRINK.
    Stop when |true_peak - ceiling| < CONVERGE_TOL_C or outer_max reached.

    is_shippable reads the TRUE arbiter peak ONLY (stage_b.shippable_verdict);
    honest_null_verdict uses the uniform 239.99 C feasibility, so a shaped
    endpoint momentarily over ceiling is NOT reported as no_feasible_dopant.

    REFUSES to start unless B1 launch_ok AND the B2 _march fidelity check are both
    green (stage_b._preconditions)."""
    pre = stage_b._preconditions()
    if not (pre["launch_ok"] and pre["fidelity_agree"]):
        raise RuntimeError(
            f"run_solve_al refused: launch_ok={pre['launch_ok']} "
            f"fidelity_agree={pre['fidelity_agree']}. Both must be green (B1 FD "
            "gate + the B2 _march-vs-production fidelity check) before the heavy "
            "AL solve.")
    from solve3d.phase_e import checkpoint as ck

    t0 = time.perf_counter()
    ceiling_c = float(stage_a.thermal_config()["T_ceiling_C"])
    rho_t = float(stage_a.thermal_config()["rho_target"]["practical_ideal"])
    uniform_tp = _uniform_true_peak()

    lam, mu, delta, t_target = LAM0, MU0, 0.0, ceiling_c
    case = build_al_solve_case(lam, mu, t_target)
    dcase, chain = case.da_case, case.da_case.chain
    n = chain.n_design
    v = np.ones(n)

    status = RESULTS / "stage_b3_square_status.json"
    stage_b._write_json(status, {
        "state": "running", "pid": os.getpid(), "n_design": int(n),
        "outer_max": int(outer_max), "inner_budget": int(inner_budget),
        "drive_a": p2.chosen_drive_a(),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    outers = []
    true_peak = float("nan")
    ks_solve = float("nan")
    viol_prev = float("inf")
    converged = False
    for k in range(int(outer_max)):
        case.lam, case.mu, case.t_target = float(lam), float(mu), float(t_target)

        def fg(vv):
            J, g = al_objective_and_grad(case, vv)
            fg.last_wall = time.perf_counter() - t0
            return J, g
        fg.last_wall = 0.0

        def on_eval(vv, J, g, _k=k):
            ks = da.ks_peak_forward(dcase, vv)
            print(f"  [B3 outer={_k} lam={case.lam:.3e} mu={case.mu:.0e} "
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

        ckpt = RESULTS / f"ckpt_stage_b3_square_outer{k}.npz"
        res = ck.run_with_checkpoint(fg, v, int(inner_budget), ckpt,
                                     bounds=(0.0, 1.0), scale_first_step=True,
                                     on_eval=on_eval)
        v = np.asarray(res["best_v"], float)

        # arbiter: TRUE end-state peak on the mesh hold-out (never KS)
        s_map = chain.design_to_map(v, 0.0)
        gate = p2.ceiling_end_state_gate(
            s_map, chain.centroids, holdout_nodes=int(holdout_nodes),
            holdout_lc0=float(holdout_lc0), rho_target=rho_t)
        true_peak = float(gate["true_peak_c"])
        ks_solve = float(da.ks_peak_forward(dcase, v))

        # multiplier update at the current target, restoration shift, mu escalation
        g_con = ks_solve - t_target
        lam = multiplier_update(lam=lam, mu=mu, g=g_con)
        shift = restoration_shift(ceiling=ceiling_c, true_peak=true_peak,
                                  ks_peak=ks_solve, prev_delta=delta, ema=DELTA_EMA)
        delta, t_target = shift["delta"], shift["t_target"]
        viol_now = abs(true_peak - ceiling_c)
        mu = mu_escalation(mu=mu, viol_prev=viol_prev, viol_now=viol_now,
                           factor=MU_FACTOR, shrink=MU_SHRINK)

        outers.append({
            "outer": k, "inner_evals": int(res["evals_used"]),
            "best_J": float(res["best_J"]), "status": res.get("status"),
            "lam_after": float(lam), "mu_after": float(mu),
            "delta": float(delta), "t_target_next_c": float(t_target),
            "ks_solve_c": ks_solve, "true_peak_c": true_peak,
            "viol_c": float(viol_now), "arbiter_feasible": bool(gate["feasible"]),
            "wall_s": round(time.perf_counter() - t0, 1)})
        print(f"[B3 outer={k}] true_peak={true_peak:.2f}C KS={ks_solve:.2f}C "
              f"Delta={delta:.2f} t_target->{t_target:.2f} lam->{lam:.3e} "
              f"mu->{mu:.0e} viol={viol_now:.2f}C", flush=True)

        if viol_now < CONVERGE_TOL_C and true_peak <= ceiling_c + CONVERGE_TOL_C:
            converged = True
            break
        viol_prev = viol_now

    v_best = v
    s_best = chain.design_to_map(v_best, 0.0)
    np.savez_compressed(RESULTS / "map_stage_b3_square.npz", v_raw=v_best,
                        s_map=s_best, centroids=chain.centroids,
                        volumes=chain.volumes)

    fd_and_fidelity = bool(pre["launch_ok"] and pre["fidelity_agree"])
    verdict = stage_b.shippable_verdict(
        true_peak_c=true_peak, ks_peak_c=ks_solve, ceiling_c=ceiling_c,
        fd_gate_passed=fd_and_fidelity)
    nullv = honest_null_verdict(shaped_true_peak=true_peak,
                                uniform_true_peak=uniform_tp, ceiling=ceiling_c)

    doc = {
        "what": "Stage B B3: augmented-Lagrangian dopant SHAPE-solve at fixed "
                "0.40x with true-peak restoration. L = J_shape + "
                "(1/2mu)[max(0,lambda+mu*g)^2 - lambda^2], g = KS_peak - t_target; "
                "t_target = ceiling - Delta re-estimated on the arbiter each outer "
                "iteration so the TRUE hold-out peak converges to the ceiling. "
                "is_shippable reads the TRUE arbiter peak only (never KS).",
        "stage": "B3_augmented_lagrangian_true_peak_restoration",
        "part": "square",
        "drive_a": p2.chosen_drive_a(),
        "power_density_w_per_m3": p2.chosen_drive_power_density(),
        "ceiling_c": ceiling_c,
        "preconditions": pre,
        "outer_loop": {"outer_max": int(outer_max), "inner_budget": int(inner_budget),
                       "lam0": LAM0, "mu0": MU0, "mu_factor": MU_FACTOR,
                       "mu_shrink": MU_SHRINK, "delta_ema": DELTA_EMA,
                       "converge_tol_c": CONVERGE_TOL_C, "converged": bool(converged)},
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
                    "true hold-out peak <= ceiling (by construction via the shift)",
        },
        "recommended_power_settings":
            stage_a.recommended_power_settings(p2.chosen_drive_a()),
        "wall_total_s": round(time.perf_counter() - t0, 1),
    }
    stage_b._write_json(RESULTS / "stage_b3_square.json", doc)
    stage_b._write_json(status, {
        "state": "done", "pid": os.getpid(),
        "is_shippable": bool(verdict["is_shippable"]),
        "true_holdout_peak_c": true_peak, "converged": bool(converged),
        "wall_total_s": doc["wall_total_s"],
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({"is_shippable": verdict["is_shippable"],
                      "true_holdout_peak_c": true_peak,
                      "converged": converged, "reason": verdict["reason"],
                      "honest_null": nullv["verdict"],
                      "wall_total_s": doc["wall_total_s"]}, indent=1), flush=True)
    return doc


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--outer-max", type=int, default=OUTER_MAX)
    ap.add_argument("--inner-budget", type=int, default=INNER_BUDGET)
    a = ap.parse_args()
    if a.solve:
        run_solve_al(outer_max=a.outer_max, inner_budget=a.inner_budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
