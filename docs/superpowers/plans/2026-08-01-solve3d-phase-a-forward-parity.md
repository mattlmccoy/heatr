# solve3d Phase A: Forward Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dolfinx forward model (complex EQS + enthalpy thermal-phase march) in `solve3d/` that reproduces heatr3d's coupled forward on the extrusion-anchor cases and passes the S1 analytic benchmarks, with every gate recorded as JSON.

**Architecture:** `solve3d/forward.py` lifts the spike's EQS (corrected-Q only) from `heatr3d_d1_spike/eqs_common.py` and adds a transient enthalpy march on the FEM mesh whose semantics are ported from `heatr3d.py` `phase_update="enthalpy"` (see `enthalpy_from_T` / `T_from_enthalpy`). The in-march EQS re-solve semantics port from `heatr3d.py` commit `699ed79` (`eqs_update_interval_s`, `apply_sigma_coupling`, fixed-power renormalization on re-solve). Parity targets the COUPLED forward per the approved spec (docs/superpowers/specs/2026-07-31-solve-port-3d-design.md §7.4).

**Tech Stack:** dolfinx 0.11 complex build in `heatr3d_d1_spike/env` (MUST import `jit_fix` before dolfinx — Dropbox-path shim), scipy, numpy. `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for every run.

**Authorship note (deviation from the writing-plans template):** full dolfinx code blocks are not inlined here; the executing agent is `computational-solver-engineer`, which authors the code against the cited source anchors. Every task still specifies exact files, the failing test to write FIRST, the command, and the numeric gate. Red-green order is mandatory per task.

---

## Ground rules (apply to every task)

- TDD: write the failing test, run it, see it fail for the right reason, then implement.
- No invented tolerances: parity tolerances are MEASURED in Task 1 (heatr3d's own self-discretization spread), not asserted.
- Every gate emits JSON under `solve3d/results/`; reports quote `results.json` only (no transcription from logs).
- Never compare against legacy-Q heatr3d fields; corrected/masked-gradient Q only. Building on legacy Q is a construction-time error.
- Atomic commits: `git add <your files> && git commit` in one shot; only files this plan owns.
- `dissertation_materials/` is READ-ONLY.
- Reproducibility gates use relative tolerances (~1e-12), never `array_equal` (BLAS noise finding from S1).

## File map

- Create: `solve3d/__init__.py`, `solve3d/forward.py`, `solve3d/gates.py`, `solve3d/cases.py` (anchor-case definitions), `solve3d/results/` (JSON artifacts)
- Create: `solve3d/tests/test_forward_analytic.py`, `solve3d/tests/test_forward_parity.py`
- Reference (read, do not modify): `heatr3d.py`, `heatr3d_d1_spike/{jit_fix,eqs_common,adjoint_core,metrics}.py`, `test_heatr3d_s1.py`
- Modify: none of the existing solver files.

---

### Task 0: Scaffold + environment smoke test

- [x] Write `solve3d/tests/test_forward_analytic.py::test_env_imports` asserting `solve3d.forward` imports dolfinx (via jit_fix) and exposes `ForwardParams` / `run_forward`. Run: fails (module missing).
- [x] Create the scaffold with a `ForwardParams` frozen dataclass mirroring the relevant `heatr3d.Params` fields (material props, drive, `eqs_update_interval_s`, `sigma_temp_coeff_per_K`, `sigma_density_coeff`, `sigma_ref_temp_c`, `eqs_resolve_drift_rtol`, `phase_update` fixed to enthalpy).
- [x] Run test: passes. Commit.

### Task 1: Pre-register parity tolerances from heatr3d self-spread

- [x] Script `solve3d/cases.py::measure_self_spread()`: run heatr3d (enthalpy, coupled defaults OFF) on the extruded circle anchor at n=64 and n=96; record the relative spread of t90, part-mean heating curve (rel-L2), and std(T) at heating-peak.
- [x] Write the tolerances into `solve3d/results/parity_tolerances.json` as 1.5x the measured spread, with the raw numbers alongside. These are the FROZEN Phase A gates; later tasks read this file.
- [x] Commit with the JSON.

### Task 2: EQS forward lift

- [x] Failing test: `test_forward_parity.py::test_eqs_qrf_pattern` — dolfinx Q_rf on the extruded circle vs heatr3d masked-gradient Q_rf (resampled to the voxel grid): pattern rel-L2 < 0.05 (the D1 Task-4 gate, already proven achievable) and total power matched to 1e-9 after renormalization.
  - **GATE CORRECTED (escalated, see PHASE_A_REPORT.md §3):** the 0.05 is D1's *dolfinx-vs-dolfinx* self-convergence gate (`run_scale_test.py` l.144), not a dolfinx-vs-heatr3d one; the measured cross-engine values are 0.1765/0.1079 (`results.json` `task2.gate.maskgrad_all_points`). Replaced with an exact-reproduction gate — PASSES to 3.8e-13. Power renorm PASSES (8.3e-15).
- [x] Implement by lifting `eqs_common` conventions into `solve3d/forward.py` (corrected-Q only).
- [x] Run: pass. Commit.

### Task 3: Enthalpy march — analytic benchmarks (S1 ports)

- [x] Failing tests (three, from `test_heatr3d_s1.py` analogs): (a) latent plateau — uniform heating pins the melt-range plateau at the correct enthalpy budget; (b) Fourier decay — zero-source decay rate matches the analytic eigenvalue within 2%; (c) energy audit — |residual_frac| < 1e-2 on a heat+melt case.
- [x] Implement the transient enthalpy march (piecewise-linear H(T) inversion ported from `heatr3d.enthalpy_from_T`/`T_from_enthalpy`; implicit or CFL-guarded explicit stepping — agent's choice, justified in the report).
- [x] Run: pass. Commit.

### Task 4: Coupled-forward parity on the anchor cases

- [x] Failing test: `test_forward_parity.py::test_anchor_parity` — for extruded circle AND extruded square: t90, part-mean heating curve rel-L2, and heating-peak std(T) within the Task-1 frozen tolerances, run with `eqs_update_interval_s > 0` and a nonzero `sigma_temp_coeff_per_K` (use −0.002 /K, the S4 re-score's best-behaved value, |a| well inside the 0.0044 validity bound) so re-solve semantics are exercised, plus the defaults-off case.
- [x] Also assert `n_eqs_solves` parity semantics: same re-solve count as heatr3d for the same interval and horizon.
- [x] Implement the re-solve loop (port `apply_sigma_coupling` semantics + fixed-power renormalization).
- [x] Run: **FAILS as specified** (`verdict.gate_ok = false`). `n_eqs_solves` parity PASSES exactly on all four arms (1/1 off, 6/6 coupled); t90 passes on both square arms; curve and sigma_T fail on all four. Tolerances NOT widened — see PHASE_A_REPORT.md §5 for the localized cause and the three options for a cross-family tolerance. `solve3d/results/phase_a_gate.json` emitted with every number. Commit.

### Task 5: Gate report

- [x] `solve3d/PHASE_A_REPORT.md`: gates table quoted from the JSONs, cost table (wall time per solve/march vs heatr3d), honest deviations, what is NOT covered (no adjoint, no objective, no regularization — those are Phases B+). Commit.

## Out of scope for Phase A

Adjoint (Phase B), objective/chi/regularization (Phase C prep), eps_r channel and drive reconciliation (Phase D), any Studio integration, any dissertation edit.
