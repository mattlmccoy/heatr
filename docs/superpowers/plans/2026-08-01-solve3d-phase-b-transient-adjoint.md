# solve3d Phase B: Transient Adjoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A gated gradient dJ/d(design) through the FULL Phase A coupled forward (steady EQS + enthalpy thermal-phase march + in-march re-solves + fixed-power renormalization), validated by the frozen 2-D FD/subgradient protocol, with Griewank-style checkpointing, at a cost of <= ~2 forward-equivalents.

**Architecture:** `solve3d/adjoint.py` lifts the steady EQS adjoint from `heatr3d_d1_spike/adjoint_core.py` (D1 Task 5, already gated: worst per-dof FD err 7.4e-5 over 105k dofs, mutation-tested) and adds the transient reverse march: adjoint of the enthalpy update (piecewise-linear H(T) inversion VJPs, melt-window clip VJPs), coupled to EQS re-solve events (adjoint restarts at re-solve boundaries, sigma-coupling VJP from `apply_sigma_coupling` semantics) and through the renormalization (D1 proved freezing it is wrong by up to 150% per dof). Store-everything first on a small case, checkpointing second, equivalence-gated.

**Tech Stack:** dolfinx 0.11 complex build in `heatr3d_d1_spike/env` (import `jit_fix` first), the Phase A `solve3d/forward.py` unchanged. `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

**Authorship note (same deviation as Phase A):** the executing agent is the Phase A author (`computational-solver-engineer` lineage) with full codebase context; tasks specify files, the failing test FIRST, commands, and numeric gates rather than inlined dolfinx code.

---

## Ground rules

- The FD/subgradient protocol is the FROZEN 2-D checklist, ported verbatim (FROZEN_CONVENTIONS_2D.md, their b04e356): L0 bit-identity vs production forward to 0.0 before any gradient; central-difference gates at the max-sensitivity cell + a random cell + a random direction; the 1e-5 subgradient standard (melt-front pinning) with the MEASURED evaluation floor reported alongside; every linear-operator transpose checked against its adjoint exactly (their filter transpose achieved 4.7e-14 - same order expected); flag-off bit-identity regression for every new code path.
- Gate thresholds are the 2-D acceptance thresholds; if a 3-D gate needs a different threshold, MEASURE the reason and record it - never widen to pass.
- Mutation tests are mandatory at the end: the frozen-renormalization mutant and the dropped-adjoint-term mutant must both FAIL the FD gate (D1 pattern). A gate that mutants pass is not a gate.
- Every gate emits JSON under `solve3d/results/`; report quotes JSONs only.
- `forward.py` may gain hooks (state recording, t_start segmentation) ONLY behind flags proven bit-identical when off.
- Atomic commits, only solve3d/ files + this plan's checkboxes. dissertation_materials/ READ-ONLY. Marches within Phase A mesh sizes; the FD-gate case must be SMALL (minutes per forward, else central differences are unaffordable - pick the case and justify its representativeness: it must cross the melt window and trigger >= 2 EQS re-solves).

## File map

- Create: `solve3d/adjoint.py`, `solve3d/tests/test_adjoint_steady.py`, `solve3d/tests/test_adjoint_transient.py`, `solve3d/tests/test_checkpointing.py`
- Modify (flag-gated only): `solve3d/forward.py`
- Reference: `heatr3d_d1_spike/adjoint_core.py`, `FROZEN_CONVENTIONS_2D.md`, the 2-D lane's `fgm_solve_campaign/adjoint2d/` (read-only - their transient adjoint is the semantic template: coupled T,rho reverse march, clip VJPs, envelope stop-time)

---

### Task 0: Pre-register the gate protocol and the FD case

- [x] Write `solve3d/results/phase_b_protocol.json` BEFORE any adjoint code: the ported checklist items, the 2-D thresholds verbatim with citations, the chosen small FD case (mesh, drive, horizon, re-solve interval, why it crosses melt and re-solves >= 2 times), and the forward-equivalent accounting rule. Commit.

### Task 1: Steady EQS adjoint lift + re-gate on Phase A meshes

- [ ] Failing test: `test_adjoint_steady.py::test_dj_dsigma_fd_gate` - dJ/dsigma (J = a simple quadratic of Q_rf) on the Phase A circle mesh, central-difference at max-sensitivity dof + random dof + random direction, thresholds from the protocol JSON. Run: fails (module missing).
- [ ] Lift from `adjoint_core.py` (complex-symmetric A^H = conj(A) reuse; THROUGH the fixed-power renormalization).
- [ ] Green + the two mutation tests (frozen-renorm, dropped-term) failing the gate as required. Commit.

### Task 2: Transient adjoint, store-everything, small case

- [ ] Failing test: `test_adjoint_transient.py::test_reverse_march_fd_gate` - dJ/dsigma through the FULL small-case coupled march (enthalpy VJPs incl. melt-window clips, sigma-coupling VJP at each re-solve event, renorm VJP), J = the melt-state functional sum((phi - chi)^2) at FIXED read time (envelope handling is Task 4). Central-difference + subgradient labeling per protocol; report the measured evaluation floor.
- [ ] Implement the reverse march storing all states.
- [ ] Green + flag-off bit-identity of any forward.py hooks. Commit.

### Task 3: Design-field composition and cost accounting

- [ ] Failing test: gradient w.r.t. the per-cell design field (the sat/dopant channel feeding sigma, matching the 2-D actuator convention: conductivity only, outside-part saturation 1.0) on the small case; FD gate at protocol thresholds.
- [ ] Implement; measure gradient cost in forward-equivalents (target <= ~2; 2-D achieved 1.4-1.7). Record in `solve3d/results/phase_b_cost.json`. Commit.

### Task 4: Envelope stop-time exactness

- [ ] Failing test: the S1-vs-S2 exact-agreement gate from the 2-D lane, ported: J at t_stop = argmin over the stored trajectory, gradient WITHOUT a dt*/ds term must agree exactly (0.0 rel, their verified result) with the fixed-time gradient evaluated at the argmin. Run red, implement the envelope read, green. Commit.

### Task 5: Checkpointing

- [ ] Failing test: `test_checkpointing.py::test_gradient_identical` - interval-checkpoint/recompute reverse march reproduces the store-everything gradient to rel ~1e-12 on the small case; plus a memory measurement (peak RSS or stored-state bytes) showing the reduction on the Phase A circle mesh horizon.
- [ ] Implement (interval checkpointing is sufficient; full binomial Griewank only if the interval scheme misses the cost target - justify either way).
- [ ] Green; re-measure forward-equivalent cost WITH checkpointing (this is the number that must be <= ~2). Commit.

### Task 6: Gate report

- [ ] `solve3d/PHASE_B_REPORT.md`: protocol table with every gate number quoted from JSONs, mutation-test outcomes, cost table (store-everything vs checkpointed), honest deviations, not-covered list (no optimizer loop, no regularization chain, no eps channel, no rho-densification adjoint - the Phase A forward holds rho fixed, so the adjoint matches; the rho-coupled VJP layer arrives when densify ports into solve3d). Commit.

## Out of scope

Optimizer loop, filter/projection chain and its transposes in 3-D (Phase C - though the transpose-check discipline is already in the protocol for when it arrives), eps_r channel (Phase D), Studio, dissertation.
