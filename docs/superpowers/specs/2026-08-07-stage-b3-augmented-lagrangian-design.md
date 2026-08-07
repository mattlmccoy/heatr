# Stage B3: Augmented Lagrangian with True-Peak Restoration

Date: 2026-08-07
Status: APPROVED direction by Matt 2026-08-07 ("B3 augmented Lagrangian (principled)").
This design increment specifies the formulation; awaiting spec review before the plan.
Parent: docs/superpowers/specs/2026-08-07-stage-b-ceiling-coupled-dopant-design.md
(B3 is the "later" stage named there). Reuses the B1 density co-state (FD-gated).
Owner: Matt McCoy

## Why (the B2 result that triggered it)

B2 (solve3d/results/stage_b_square.json): the penalty solve at fixed 0.40x
converged with the ceiling-coupled density gradient controlling the peak, but
landed is_shippable=FALSE by 0.69 C -- true hold-out peak 250.69 C while the KS
aggregate it penalized was 248.45 C. Root cause is intrinsic: the KS aggregate is a
LOWER bound on the true peak (mean-log-sum-exp approaches the max from below), so
constraining KS <= ceiling leaves the true peak ~2 C above. A hand-tuned penalty
cannot fix this class; the constraint must be enforced on the quantity that gates
is_shippable (the TRUE arbiter peak), by construction.

## Formulation

Inequality constraint: g(s) = KS_peak(s) - T_target <= 0, where T_target is a
SHIFTED ceiling (below). Augmented Lagrangian (standard inequality form):

    L(s; lambda, mu) = J_shape(s) + (1/(2 mu)) * [ max(0, lambda + mu*g(s))^2 - lambda^2 ]

Design gradient (active branch): dL/ds = dJ_shape/ds + max(0, lambda + mu*g) *
dKS_peak/ds. The dKS_peak/ds is the B1 rho+T density co-state, ALREADY built and
FD-gated to 3.0e-9 -- B3 adds only the scalar multiplier factor. J_shape is the
melt-onset envelope objective (Stage A phase-2), unchanged.

Outer augmented-Lagrangian loop:
1. Inner solve: minimize L over s at fixed (lambda, mu) with L-BFGS-B (frozen
   conventions: 1/|g0| rescale, filter + tanh projection).
2. Multiplier update: lambda <- max(0, lambda + mu * g(s*))  (KKT for inequality).
3. If |constraint violation| did not shrink by a set factor, increase mu.
4. Repeat until the TRUE arbiter peak is within tolerance of the ceiling.

## The restoration shift (the crux -- makes the TRUE peak, not KS, hit the ceiling)

KS is a lower bound, so g on KS alone does not bound the true peak. Each OUTER
iteration re-estimates the gap on the arbiter and shifts the target:

    Delta_k = true_peak(arbiter mesh, s_k) - KS_peak(solve mesh, s_k)
    T_target = T_ceiling - Delta_k

Delta_k folds in BOTH the KS-vs-true gap and the solve-vs-arbiter mesh gap (in B2,
Delta ~= 2.24 C: solve-mesh KS 248.45 vs arbiter true 250.69). As s_k converges
Delta_k stabilizes and the true arbiter peak -> T_ceiling. This is the "peak <=
ceiling by construction" guarantee, enforced on the arbiter quantity that decides
is_shippable, not on the KS proxy. Damp Delta_k (e.g. EMA) so the shift does not
oscillate; log every Delta_k.

Honest limit named: the restoration uses the arbiter mesh as truth. If the arbiter
mesh itself under-resolves the peak, the shift is only as good as that mesh; the
cross-engine Studio is_sendable gate remains the final independent check, and the
Delta_dopant ~15 C max-over-engines headroom (parent spec) still applies if a
cross-engine margin is wanted.

## Gates / acceptance (pre-registered)

- The combined AL gradient dL/ds FD-gated against central differences at a fixed
  (lambda, mu) with the hinge active -- light, since dKS_peak/ds is already gated;
  the new factor is the scalar max(0, lambda + mu*g). Mutation: dropping the
  multiplier term must change the gradient.
- Outer-loop logic (multiplier update, shift update, mu escalation) unit-tested as
  pure functions (no physics): given (lambda, mu, g, Delta) produce the right next
  (lambda, mu, T_target).
- Acceptance: the converged map's TRUE hold-out peak <= 250 C (by construction via
  the shift) on the arbiter mesh; is_shippable reads the true peak only (never KS);
  best shape under that constraint; then route (shaped map, 0.40x, chamber tag,
  rho_target) to the Studio heatr3d is_sendable gate.
- Fix the B2 honest_null MISLABEL as part of this: null_verdict must NOT claim
  "no_feasible_dopant_at_this_drive" when a feasible dopant is known (uniform 239.99
  C < ceiling). Honest-null fires only if the PEAK-MINIMIZING map (or the known
  uniform) is over ceiling. At 0.40x it is not, so B3 must converge feasible, not null.

## Scope

In: the outer AL loop, the restoration shift, the multiplier updates, the
combined-gradient FD gate, the honest_null fix, one solve at fixed 0.40x, the
hold-out arbiter, cross-engine routing. Reuses B1 dKS_peak/ds and B2
penalty_objective machinery.

Out (named): joint drive (B4), exposure (parent Stage B), the schedule adjoint
(Stage C), eps_r (Phase D), in-march EQS re-solve coupling (still OFF, consistent
with the arbiter), premix (separate session).

## Files (anticipated; the plan pins exact paths)

- solve3d/stage_b3.py: the outer AL loop, restoration shift, multiplier/mu updates,
  run_solve_al.
- solve3d/stage_b.py: reuse penalty_objective_and_grad's gradient assembly (swap the
  penalty term for the AL term); density_adjoint.dks_peak_ds unchanged.
- solve3d/tests/test_stage_b3.py: AL-gradient FD gate + pure-logic tests for the
  multiplier/shift/mu updates + the honest_null-not-spurious test.
