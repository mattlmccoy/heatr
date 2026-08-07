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
