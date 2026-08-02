# solve3d Phase C: First 3-D Solve (Cylinder Null Demonstration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The first gated 3-D dopant solve: on the extruded circle (the cylinder-null case, where the inversion heuristic yields +0.3% = nothing), run the full solve stack (filter + chi + objective + L-BFGS-B on the Phase B gradient) and answer the port's decisive question: does the direct solve find improvement the inversion rule cannot? A map earns the SOLVED label only by surviving the mesh hold-out and smoothing-robustness acceptance gates.

**Architecture:** `solve3d/objective.py` (asymmetric shape-fidelity J on phi vs volume-fill chi, envelope stop time - the Phase B B4 machinery), `solve3d/design_chain.py` (filter with 1.0 mm PHYSICAL radius + optional smoothed-Heaviside projection, transposes gated), `solve3d/solve.py` (L-BFGS-B, box [0,1], single full-depth start, budget accounting in measured forward-equivalents). Everything sits on the unchanged Phase A forward + Phase B adjoint.

**Tech Stack:** spike env + jit_fix, scipy L-BFGS-B. `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

**Authorship note:** same as Phases A/B - the executing agent (Phase A/B author) writes code against cited anchors; every task has red-first tests and numeric gates.

---

## Ground rules

- FROZEN 2-D conventions apply (FROZEN_CONVENTIONS_2D.md): filter radius 1.0 mm physical; eta=0.5, beta continuation 1/2/4/8/16 for the projection arm; conductivity-only actuator, outside-part saturation 1.0; L-BFGS-B box [0,1]; 40 forward-equivalents per arm, adjoints counted at the MEASURED 1.774 fwd-equiv (phase_b_cost.json); single full-depth start.
- Drive: the Phase A forward's fixed-power renormalization convention (drive reconciliation is Phase D; state this in the report).
- chi: grid-independent sub-cell VOLUME fill from the analytic geometry (the 3-D analog of the 2-D area fill; include the analytic circle test - solid fraction error -> 0 with refinement). Never a binary raster on the solve mesh.
- Filter/projection transposes gated exactly (adjoint-vs-transpose, 1e-13-class, per the frozen checklist); the composed design-chain gradient re-gated with the Phase B FD protocol before any solve runs.
- PRE-REGISTRATION before any solve: budgets, baselines, both objective weightings, and the acceptance bands - all in JSON, committed first.
- No threshold widening, JSON-quoted reporting, atomic solve3d/-only commits, no push, dissertation_materials READ-ONLY.

## File map

- Create: `solve3d/objective.py`, `solve3d/design_chain.py`, `solve3d/solve.py`, `solve3d/tests/test_objective.py`, `solve3d/tests/test_design_chain.py`, `solve3d/tests/test_solve_gates.py`, `solve3d/results/phase_c_*.json`
- Reference: `solve3d/{forward,adjoint,gate_fd}.py`, FROZEN_CONVENTIONS_2D.md, FGM_BENEFIT_RERUN.md (the cylinder-null baseline numbers and the corrected-design inversion map artifacts), docs/superpowers/specs/2026-07-31-solve-port-3d-design.md (objective refinement + Phase C acceptance)

---

### Task 0: Pre-registration

- [x] `solve3d/results/phase_c_preregistration.json`, committed BEFORE any objective/solve code: case = extruded circle (Phase A anchor geometry); budget 40 fwd-equiv/arm with the measured adjoint cost; arms = {uniform baseline, corrected-design inversion map (locate the artifact from the FGM benefit re-run - cite the exact file; if it cannot be resampled honestly onto the solve mesh, record that and drop the arm rather than fake it), solve/filter-only, solve/projection with beta continuation}; objective weighting arms (see Task 1); acceptance bands: mesh hold-out on the Phase A fine mesh (solve on mid, score on fine; band = the measured mid-vs-fine self-spread of J and IoU from phase_a_shape_gate.json self-spreads, rule stated before computing) and smoothing robustness (J change under a sub-filter-radius perturbation, 0.5 mm class, band pre-stated); the SOLVED label rule. Commit.

### Task 1: Objective (asymmetric, per Matt's recorded refinement)

- [x] Failing tests: closed-form J on synthetic phi/chi fields for BOTH weightings: (a) SYMMETRIC control J = integral (phi - chi)^2 (cross-lane comparability), (b) ASYMMETRIC J per the spec's objective refinement - out-of-bounds melt penalized hard, in-bounds under-density soft below a floor (hinge at phi_floor = 0.85, the middle of Matt's 80-90% band; the out/in weight ratio is a PRE-REGISTERED choice justified in Task 0, with ONE alternative ratio run as a sensitivity arm, not tuned).
- [x] Implement both + the envelope stop-time read (Phase B B4 machinery). Green. Commit. **9 passed; the default selector is bit-identical to the Phase B functional (J and seed both exactly 0.0 difference).**

### Task 2: Design chain (filter + projection) + gradient re-gate

- [ ] Failing tests: filter transpose vs adjoint exact (1e-13 class); projection chain rule vs FD at beta 1 and 16; area/volume-fill chi analytic circle test; composed dJ/d(raw design) FD gate (Phase B protocol, all four probes) through filter (+ projection at beta=1) on the small Phase B case.
- [ ] Implement `design_chain.py` (1.0 mm physical radius on the mesh; document the discrete kernel). Green. Commit.

### Task 3: The solves

- [ ] Run the pre-registered arms on the extruded circle at the Phase A mid mesh. Log per-iteration J, budget spent, envelope t_stop. Emit `solve3d/results/phase_c_solves.json` with every arm's trajectory and final map artifact (npz).
- [ ] Score ALL arms on the shared Phase A read grid: J (both weightings), IoU at phi>=0.8/0.9, front SSD, out-of-part melt fraction, and sigma_T as a reported diagnostic. Commit.

### Task 4: Acceptance gates

- [ ] Mesh hold-out: re-run the FORWARD (not the solve) for the winning solved map on the fine mesh; score there; pass iff within the pre-registered band.
- [ ] Smoothing robustness: perturb the solved map below the filter radius; J change within band.
- [ ] Apply the SOLVED label rule from Task 0. Emit `solve3d/results/phase_c_gate.json`. Commit.

### Task 5: The decisive comparison + report

- [ ] `solve3d/PHASE_C_REPORT.md`: the cylinder-null answer (solve vs uniform vs inversion, with the 2-D lane's +0.3% inversion null as context), both objective weightings, both regularization arms, acceptance gate table, cost table, honest deviations and not-covered (one shape; no eps channel; no drive reconciliation; no densification in the forward; the t90 offset still S3's). If the solve FINDS NOTHING beyond uniform, report that as the finding - on an already-uniform corrected field that is a physically meaningful null, not a failure of the port. Commit.

## Out of scope

Multi-shape library campaign (Phase E), eps_r channel + drive reconciliation (Phase D), Studio integration, densification coupling in the forward, any dissertation edit.
