# Phase-E implicit-diffusion forward + adjoint - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development
> with the computational-solver-engineer agent. Each phase's FD-gate is the
> correctness oracle - do NOT trust a gradient that has not passed it. Steps use
> checkbox (- [ ]) syntax. ASCII only in all files (this repo renders non-ASCII as
> mojibake).

**Goal:** replace the Phase-E densify forward's EXPLICIT diffusion step with an
unconditionally-stable IMPLICIT step, and re-derive its per-step adjoint, so the
differentiable ceiling solve runs finite on fine meshes (the Tamper) at n_sub=1.

**Architecture:** decision 4b (spec 2026-08-11): the GATE forward
(solve3d/density_adjoint.py _substep_forward/_march + dks_peak_ds) goes implicit;
production forward.march_enthalpy is UNTOUCHED; heatr3d is the fine-mesh arbiter.
The implicit step is a single linear solve per step with lagged (frozen-at-T_in)
coefficients; its VJP is one transpose solve. At n_sub=1 the reverse sweep keeps
its current structure (one VJP per step, ~2800 caches) - no checkpointing.

**Tech Stack:** Python, numpy, dolfinx/PETSc (existing FEM assembly in forward.py),
scipy sparse (existing), pytest. Env: heatr3d_d1_spike/env/bin/python.

**Reference the spec:** docs/superpowers/specs/2026-08-11-phase-e-implicit-diffusion-adjoint-design.md

---

## Ground rules for the executor

- Every gradient is FD-gated (central difference, worst rel-err <= 1e-6) BEFORE it
  is trusted. This is the iron law here; the explicit adjoint NaN'd precisely
  because a stability assumption went unchecked on a new mesh.
- Keep test_two_sided.py green after every phase (the two-sided actuator is
  orthogonal and must not regress).
- Commit per phase, linearly on feat/pernode-twosided-tuning (shared tree; do NOT
  create/switch branches - GH #57048).
- ASCII only.

---

## Phase 0: Characterize the current explicit step + freeze a regression baseline

**Files:** Read solve3d/density_adjoint.py (_substep_forward 231-304, _substep_vjp
358-438, _march 307-322, dks_peak_ds 440-466, march_fidelity_check 469+),
solve3d/forward.py (march_enthalpy 744-833, _stability_dt, the n_sub formula).

- [ ] **Step 1: Capture the explicit gate's FD-gate baseline on a COARSE case.**
  Run the existing `density_adjoint.fd_gate()` on the coarse SQUARE case; record
  worst_rel_err and the gradient vector to a scratch npz. This is the number the
  implicit scheme must reproduce (the coarse physics must not move).
  Run: `heatr3d_d1_spike/env/bin/python -c "from solve3d import density_adjoint as da; import json; print(json.dumps(da.fd_gate(), default=float))"`
  Expected: PASS (worst_rel_err ~1e-8) - record it.
- [ ] **Step 2: Reproduce the Tamper forward blow-up** (the thing we are fixing):
  build the Tamper case, run the explicit `_march` forward-only, confirm the
  temperature field goes non-finite / the co-state overflows (per
  tamper_rescue_solve_FAILED_drive1.2_nan.log). Record the reverse-step at which
  |gT| overflows. This is the regression target: after the fix, the Tamper adjoint
  is finite.
- [ ] **Step 3: Commit the baseline artifacts** (scratch npz + a short
  PHASE_E_IMPLICIT_BASELINE.md noting the coarse FD number and the Tamper blow-up
  step). `git commit -m "test(phase-e): baseline explicit gate FD + Tamper blow-up before implicit"`

## Phase 1: Implicit forward step (stability), coarse-equivalent

**Files:** Modify solve3d/density_adjoint.py (_substep_forward). Test:
solve3d/tests/test_implicit_step.py (new).

The executor derives the implicit step. REQUIRED interface + gates:
- `_substep_forward(case, T_in, rho_in, F, keep_cache)` keeps its signature and
  return `(T_out, rho_out, cache)`; internally the T update becomes an implicit
  solve `A @ T_new = b` with A, b assembled from T_in-lagged coefficients
  (k(T_in), rho_cp(T_in), phase state), latent heat preserved (apparent-cp or
  enthalpy-consistent source). Density update stays explicit.
- [ ] **Step 1 (RED): forward ACCURACY on the Tamper (NOT mere finiteness).**
  Phase 0 found the explicit n_sub=1 forward is ALREADY clamped-finite (T_end
  244.8 C) - it survives via the temp/nan_to_num clamps, so `isfinite` does NOT
  discriminate; the instability is in the REVERSE. So use an ACCURACY test:
  compute a TRUSTED reference forward once - the explicit forward CFL-SUBSTEPPED at
  n_sub from `forward._stability_dt` (stable, slow) on a REDUCED Tamper case (small
  n_steps to keep the reference affordable; lc_part <= 2.5e-3 REQUIRED - Phase 0
  found lc_part=5e-3 is CFL-stable and hides the bug). Write
  `test_implicit_forward_matches_substepped_reference_on_tamper`: the IMPLICIT
  forward end-state peak matches the substepped reference to a stated tol AND
  activates the temp/dt clamps on far fewer cells than the explicit n_sub=1 forward
  (report both counts). Run against current explicit n_sub=1 code; expected FAIL
  (n_sub=1 diverges from the reference and clamps ~54% of cells). This is the
  valid RED.
- [ ] **Step 2 (RED): coarse-equivalence test.** Write
  `test_implicit_matches_explicit_on_coarse_within_tol`: on the coarse square
  (CFL-stable, so explicit is valid), the implicit end-state peak must match the
  explicit end-state peak to a stated tol (e.g. <= 1 C, tightened as the scheme is
  finalized). Run against current code; expected FAIL (function not implicit yet).
- [ ] **Step 3 (GREEN): implement the implicit step.** Executor derives + implements
  A@T_new=b. Keep the explicit path available behind a case flag
  (`case.implicit: bool`, default True) so march_fidelity_check can still exercise
  explicit on coarse meshes.
- [ ] **Step 4: run both tests + test_two_sided.py.** Expected: Tamper finite PASS,
  coarse-equivalence PASS, two-sided still 6/6.
- [ ] **Step 5: convergence check** (not a unit test, a recorded probe): on the
  coarse square, implicit end-state peak -> explicit end-state peak as dt is halved
  twice; record the sequence in PHASE_E_IMPLICIT_BASELINE.md. Confirms the implicit
  scheme is consistent (same PDE), not a different model.
- [ ] **Step 6: Commit.** `git commit -m "feat(phase-e): implicit diffusion step - stable on the Tamper forward, matches explicit on coarse"`

## Phase 2: Per-step implicit VJP (the transpose solve), FD-gated on coarse

**Files:** Modify solve3d/density_adjoint.py (_substep_vjp, _dT_from_enthalpy if
the enthalpy inverse changes). Test: solve3d/tests/test_implicit_step.py.

- [ ] **Step 1 (RED): single-step VJP FD-gate.** Write
  `test_implicit_substep_vjp_matches_fd`: on a SMALL coarse case, take one
  `_substep_forward` step; FD-check the full `_substep_vjp` (perturb a few T_in and
  rho_in entries, central difference the scalar `sum(w*T_out)` and `sum(w*rho_out)`,
  compare to the VJP-propagated cotangents). worst rel-err <= 1e-6. Run against
  current (explicit) VJP applied to the implicit forward; expected FAIL (mismatch -
  the VJP no longer matches the changed forward).
- [ ] **Step 2 (GREEN): re-derive the implicit VJP.** Executor derives the adjoint
  of `A@T_new=b`: `gT_in += A^T \ (dstuff)`, plus the cotangents through the lagged
  coefficient assembly and the enthalpy/clip chain (the existing clip subgradients
  are reused; only the diffusion-solve block changes). Reuse the forward
  factorization of A for A^T (store it in the cache).
- [ ] **Step 3: run the VJP FD-gate.** Expected PASS (<= 1e-6).
- [ ] **Step 4: Commit.** `git commit -m "feat(phase-e): implicit per-step VJP (transpose solve), FD-gated on coarse"`

## Phase 3: Whole-march adjoint FD-gate on the COARSE case (no-regression)

**Files:** solve3d/density_adjoint.py (dks_peak_ds - should need no change if
Phases 1-2 kept the interfaces; verify). Test: reuse density_adjoint.fd_gate.

- [ ] **Step 1 (RED then GREEN): full dks_peak_ds FD-gate on the coarse square.**
  Run `density_adjoint.fd_gate()` (square). It marches implicitly now. Assert
  worst_rel_err <= 1e-6 AND the gradient matches the Phase-0 explicit baseline
  gradient to a stated tol (the coarse physics must not have moved). If it fails,
  the derivation in Phase 2 is wrong - fix there, do not loosen the gate.
- [ ] **Step 2: mutation-bites check** stays true (drop-density-costate breaks the
  gate) - confirm.
- [ ] **Step 3: Commit.** `git commit -m "test(phase-e): full implicit adjoint FD-gate green on coarse, matches explicit baseline"`

## Phase 4: The payoff - FD-gate the adjoint on the TAMPER

**Files:** Test: solve3d/phase_e/tests or a new test using the Tamper case at a
COARSE-ENOUGH-to-FD-gate resolution that is STILL fine enough to have been
explicit-unstable (so the gate proves the implicit fix on the real failure mode).

- [ ] **Step 1: Tamper adjoint FD-gate.** Build the Tamper case (a resolution where
  explicit NaN'd - confirm via Phase 0 Step 2), run `dks_peak_ds`, FD-check the
  top-|g| probe indices (central difference on the scalar KS peak). Assert:
  gradient FINITE everywhere (the headline - the explicit adjoint could not reach
  here) AND worst rel-err <= 1e-6. Budget the FD probes to keep wall time sane
  (2-4 probes; each eval is one implicit march).
- [ ] **Step 2: record the result** in PHASE_E_IMPLICIT_BASELINE.md (Tamper adjoint
  finite + FD number) - this is the artifact that unblocks the Tamper rescue.
- [ ] **Step 3: Commit.** `git commit -m "feat(phase-e): Tamper adjoint FD-gate GREEN - fine-mesh gradient finite (was NaN)"`

## Phase 5: Fidelity contract (decision 4b) + march_fidelity_check scoping

**Files:** Modify solve3d/density_adjoint.py (march_fidelity_check). Docs:
update the fidelity docstring + a note in the spec's realized-scope.

- [ ] **Step 1 (RED): coarse fidelity still bit-matches.** march_fidelity_check on
  a COARSE case runs BOTH forwards with enforce_cfl matched (both stable) and
  asserts bit-identity (or the current tol). This must still PASS - the implicit
  gate on a coarse case must reduce to / agree with the explicit production forward
  where both are valid. If the implicit-vs-explicit coarse agreement is not
  bit-identical, redefine the coarse fidelity contract to the achieved tol and
  STATE it (do not silently widen).
- [ ] **Step 2 (GREEN): scope fidelity out on fine meshes, explicitly.** Add a
  guard: on a mesh whose min-cell CFL makes explicit unstable at dt, the fidelity
  check RETURNS a stated "fine-mesh: explicit arbiter is CFL-unstable, heatr3d is
  the fine-mesh arbiter" record instead of asserting bit-identity against an
  unstable explicit run. This encodes decision 4b in the code, not just the doc.
- [ ] **Step 3: run march_fidelity_check on coarse (PASS bit-match) + fine (returns
  the 4b record, no false assertion).**
- [ ] **Step 4: Commit.** `git commit -m "feat(phase-e): fidelity contract per decision 4b - coarse bit-match, heatr3d as fine-mesh arbiter"`

## Phase 6: Re-baseline a coarse B-stage + relaunch the Tamper rescue

**Files:** none new; runs the existing stage_b4 / run_tamper_rescue drivers.

- [ ] **Step 1: re-confirm a coarse B-stage.** Re-run the CUBE (or square) B4 solve
  with the implicit gate; confirm the shaped result matches the recorded explicit
  B4 (235/247-class) within tol, OR record a deliberate re-baseline with the reason.
  A stability fix must not silently change a shipped coarse result.
- [ ] **Step 2: relaunch the Tamper two-sided rescue.** Now that the adjoint is
  finite on the Tamper: `run_tamper_rescue --solve --drive-a <probed> --max-sat 2.0`,
  detached + first-iteration descent watched. Pin the drive via a forward probe
  (the forward is finite at 0.7-1.2x). This is the original goal, now unblocked.
- [ ] **Step 3: symmetry-gate the resulting Tamper map** (per-part group; the
  Tamper is near-axisymmetric about its build axis) before citing/shipping.

---

## Self-review notes

- Spec coverage: Phases 1-2 cover Sec 3 (implicit scheme + VJP); Phase 3-4 cover Sec
  5 (FD-gates coarse + Tamper); Phase 5 covers Sec 4 decision 4b; Phase 6 covers Sec
  5 coarse-no-regression + Sec 6 relaunch. All spec sections mapped.
- Deliberate non-placeholder: the scheme/adjoint *formulas* are the executor's
  derivation, verified by the named FD-gates. This is not a "TODO" - the FD-gate IS
  the acceptance spec, and for a novel adjoint that is the only honest way to
  specify correctness. Every test/gate above is concrete and runnable.
- Interfaces held constant: `_substep_forward` and `_substep_vjp` keep their
  signatures across phases; `dks_peak_ds` and `fd_gate` are reused unchanged.
