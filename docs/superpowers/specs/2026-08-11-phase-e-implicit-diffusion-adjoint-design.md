# Phase-E implicit-diffusion forward + adjoint - design

**Status:** design for review (brainstorming output). No code yet.
**Goal:** make the Phase-E differentiable densify forward + adjoint numerically
stable and feasible on FINE meshes (the Tamper and arbitrary fine uploads), so a
ceiling-coupled shaped solve can run on them without the reverse co-state
overflowing to NaN.

---

## 1. Problem (diagnosed, not hypothesized)

`solve3d/density_adjoint.py` `_march`/`_substep_forward` integrate the thermal
diffusion with an EXPLICIT Euler step at `n_sub=1`:
`H2 = H + dt*(-K*T_in + F - C*T_in)/vol`. Explicit diffusion is CFL-limited:
`dt < dt_stable ~ cell_size^2*rhoc_p/k`. Fine on the coarse square/cube (CFL-stable);
on the fine Tamper mesh (54495 cells, min-cell `dt_stable~3.3e-3 s`) `dt=0.5` is
~150x above CFL and **54% of cells are explicit-unstable**. The forward survives
via clamps (temp cap, `nan_to_num`); the linearized REVERSE co-state has no
clamps, amplifies ~1.36x/step, overflows to Inf by reverse-step ~1300, then
`Infx0` (a zeroed subgradient mask) -> NaN. Drive-independent (0.7x and 1.2x
blow up bit-identically). See memory `phase-e-adjoint-cfl-instability.md`,
evidence `tamper_rescue_solve_FAILED_drive1.2_nan.log`.

## 2. Why the mechanical fix (explicit CFL substepping) is REJECTED

Mirroring production `forward.march_enthalpy`'s CFL substepping
(`n_sub=ceil(dt/(0.9*dt_stable))`) would STABILIZE it, but on the Tamper
`n_sub~168` -> ~470k substeps/march. Two blockers:
- **Memory:** the reverse sweep stores a cache per substep (`keep_cache=True`,
  `for c in reversed(caches)`); 470k caches x ~20 nodal arrays x 20k nodes ~
  hundreds of TB. Would need checkpointing (recompute-during-reverse).
- **Compute:** 168x more substeps -> the ~15 h Tamper solve becomes months.

Explicit substepping is numerically correct but **not viable**. The feasible fix
changes the DISCRETIZATION so no CFL substepping is needed.

## 3. Chosen direction: implicit (unconditionally stable) diffusion

Make the diffusion + convection update IMPLICIT (backward-Euler) so it is
unconditionally stable -> `n_sub=1` at `dt=0.5` on any mesh -> feasible memory
(no checkpointing) and feasible compute (one linear solve/step vs 168 explicit
steps). Matt approved this direction (2026-08-11) over forward-only-for-fine-parts.

**Recommended scheme (finalized during the FD-gated build): semi-implicit,
linearly-implicit backward-Euler.** Per step, lag the nonlinear coefficients at
`T_in` (conductivity `k(T_in)`, `rhoc_p(T_in)`, phase state) so `T_new` appears
LINEARLY, and solve ONE sparse linear system:
`(M/dt + K(T_in) + C)*T_new = (M/dt)*T_in + F`,
with latent heat carried by an apparent heat capacity or an enthalpy-consistent
source so the phase-change energy balance is preserved. Density update stays
EXPLICIT (it is finite and not the unstable term). Requirements the build must
satisfy (FD-gate-verified), scheme swappable if one fails:
- unconditionally stable (Tamper forward finite at dt=0.5, no NaN);
- one linear solve per step, reusing a factorization (the operator is
  time-varying only through lagged coefficients - refactor per step, or freeze
  and Newton-correct if fidelity demands);
- differentiable: the per-step VJP back-props through the linear solve as ONE
  TRANSPOSE solve (reuse the factorization). The reverse-sweep STRUCTURE is
  unchanged (one VJP per step); at `n_sub=1` the cache count is `n_steps` (~2800)
  -> feasible, NO checkpointing;
- consistency: reduces to the current explicit result as `dt->0` (a convergence
  check), and preserves melt onset / densification physics.

## 4. THE load-bearing open decision: production consistency

The Phase-E gate forward exists to give the OPTIMIZER a gradient; the B2 arbiter
(`ceiling_end_state_gate` -> `march_densify` -> `forward.march_enthalpy`) is
EXPLICIT and is what certifies the result. `march_fidelity_check` guards that the
two forwards agree (else the optimizer drives a peak the arbiter never sees). If
the gate forward goes implicit and production stays explicit, they DIVERGE on
fine meshes by construction. Two options:

- **(4a) Change BOTH** the gate `_march` and production `march_enthalpy` to the
  implicit scheme, keep them bit-matched, re-run `march_fidelity_check`. HONEST
  and consistent, but production `march_enthalpy` underpins the whole trust
  ladder + the Studio arbiter + dissertation physics - changing its
  discretization RE-BASELINES numbers everywhere (every B-stage result, the cube
  milestone) and must be re-validated against heatr3d. LARGE blast radius.
- **(4b) Gate-only implicit**, production explicit, and REDEFINE the fidelity
  contract for fine meshes: accept that on a fine mesh the arbiter's explicit
  forward is itself CFL-unstable (so it can't be the fine-mesh truth either), and
  make heatr3d (voxel FD, stable) the fine-mesh arbiter instead of
  `march_enthalpy`. Smaller code blast radius, but shifts the fine-mesh trust
  anchor from `march_enthalpy` to heatr3d and needs that re-argued.

DECISION (Matt, 2026-08-11): **(4b) chosen.** Gate-only implicit; the validated
explicit production `march_enthalpy` is UNTOUCHED (keeps every coarse-part result
+ the trust ladder + dissertation physics as-is); heatr3d (voxel FD, stable)
becomes the FINE-mesh arbiter. `march_fidelity_check` keeps asserting bit-identity
on COARSE meshes (where both forwards are CFL-stable and must still agree), and is
explicitly scoped to NOT apply on fine meshes (where the explicit arbiter is itself
CFL-unstable). (4a) is deferred unless a coarse part ever needs implicit.

## 5. Testing / verification (non-negotiable gates)

- **FD-gate the new adjoint on the Tamper** (the whole point): `dks_peak_ds` vs
  central differences, worst rel-err <= 1e-6, mutation-bites. This is the gate the
  explicit adjoint could never reach (it NaN'd).
- **Re-confirm coarse cases** (square/cube): the implicit scheme must still
  FD-gate there, and its shaped result must match (or be re-baselined against)
  the explicit B4 result - a stability fix must not silently move the coarse
  physics.
- **Stability test:** the Tamper forward runs to completion finite at dt=0.5.
- **Convergence test:** implicit -> explicit as dt->0 on a coarse case.
- `test_two_sided.py` stays green; the two-sided actuator is orthogonal.

## 6. Scope / phases (for the plan)

1. Implicit forward step (`_substep_forward` implicit variant) + stability test on
   the Tamper. 2. Its per-step VJP (transpose solve) + FD-gate on a coarse case.
3. Wire into `_march`/`dks_peak_ds`, FD-gate on the Tamper. 4. Fidelity decision
   (Sec 4) wired + `march_fidelity_check` re-baselined. 5. Re-run a coarse B-stage to
   confirm/ re-baseline. 6. Relaunch the Tamper two-sided rescue.

EXECUTION: this is squarely the computational-solver-engineer's domain
(adjoint/differentiable physics, FD-gates every gradient). After spec approval ->
writing-plans -> subagent-driven build with FD-gates between phases.
