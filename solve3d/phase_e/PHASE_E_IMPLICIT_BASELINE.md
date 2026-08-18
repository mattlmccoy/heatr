# Phase-E implicit-diffusion adjoint - Phase 0 baseline (FROZEN)

Regression baseline captured BEFORE the explicit densify diffusion step is
replaced by an implicit one (plan
docs/superpowers/plans/2026-08-11-phase-e-implicit-diffusion-adjoint.md,
spec .../specs/2026-08-11-phase-e-implicit-diffusion-adjoint-design.md).

Two numbers the later phases must hit:
1. The coarse-square FD-gate worst_rel_err + gradient vector -> the implicit
   scheme must REPRODUCE these (Phase 3): the coarse physics must not move.
2. The Tamper reverse co-state NaN reverse-step -> after the fix (Phase 4) the
   Tamper adjoint must be FINITE where the explicit one went NaN.

All runs: `heatr3d_d1_spike/env/bin/python` (the .venv312 lacks dolfinx),
PYTHONPATH=repo-root, OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1. Production code
UNCHANGED (Phase 0 only reads + probes + writes artifacts).

---

## 1. Coarse-square FD gate (the "must-not-move" number)

`density_adjoint.fd_gate()` on the default coarse SQUARE case (explicit path):

- case: square, target_nodes=120 -> n_design=320, n_part_nodes=132,
  lc0=6.0e-3 m, dt=1.0 s, n_steps=900, drive=0.40x (fixed), coupling OFF.
- **worst_rel_err = 3.017960919551819e-09**  (PASS, frozen gate 1e-6, no widening)
- mutation (drop lambda_rho) worst_rel_err = 1.4155881965446126e-02  (bites; the
  density co-state is load-bearing)
- gradient: |g| = 7.443346403673966, max|g| = 2.053091599003726
- probe indices (top-|g|): [217, 212, 184, 235]
- per-probe rel_err: 217 -> 8.75e-10, 212 -> 1.43e-09, 184 -> 6.86e-10,
  235 -> 3.02e-09

Gradient vector + design point saved (float64) to:

    solve3d/phase_e/results/phase_e_explicit_baseline_coarse_grad.npz

npz keys: grad (320,), v (320,), probe_indices (4,), worst_rel_err,
target_nodes, lc0_m, dt_s, n_steps.  (npz is force-added: the results/.gitignore
ignores *.npz, matching the already-tracked sibling field_*.npz.)

Phase 3 gate: implicit `dks_peak_ds` on this same coarse square must give
worst_rel_err <= 1e-6 AND a gradient matching `grad` above to a stated tol.

---

## 2. Tamper reverse co-state blow-up (the regression target)

Reproduced the explicit-adjoint instability from memory
`phase-e-adjoint-cfl-instability.md` / evidence
`results/tamper_rescue_solve_FAILED_drive1.2_nan.log`, on the EXACT production
Tamper config.

Config (the one that must go NaN -> finite after Phase 4):

- STL: Grade-and-Print job feb850ec "Part Studio 1 - Tamper.stl"
- lc_part = 2.5e-3 m (production; DOES NOT coarsen - see the warning below),
  adaptive chamber (L=None), dt = 0.5 s, n_steps = 2800.
- design v = ones (uniform), drive = nominal (power_density=None). (The memory's
  failed run used drive_a=1.2 + a two-sided start; the mechanism is identical,
  only the exact overflow step shifts - see below.)
- mesh: n_cells = 54495, n_part_cells = 20106, n_nodes = 10221,
  n_design = 20106  (matches the memory's mesh exactly).

Result:

- **FORWARD is FINITE** (survives via clamps: +-max_dt_step cap, temp clip,
  nan_to_num): T_end_max = 244.8 C, mean_rho = 0.580. The instability is NOT in
  the forward.
- REVERSE co-state (same seed + same `_substep_vjp` as `dks_peak_ds`, uniform v):
  seed max|gT| = 0.565, then amplifies GEOMETRICALLY at ~1.32x per reverse-step.
- **gT goes NaN (Inf*0) at reverse-step 2424** (of 2800). An entry exceeds the
  float64 range (~1.8e308) and is multiplied by a zeroed subgradient mask
  (m_tmp / m_cap / drho_cap_mask) in the same `_substep_vjp` call -> Inf*0 = NaN
  (the RuntimeWarnings at density_adjoint.py:377,378,412,413,433 in the failure
  log are downstream victims of exactly this).

Reverse-step -> max finite |gT| trace (uniform v, nominal drive):

    step    50   ->  1.5e-01
    step   100   ->  1.2e+03
    step   200   ->  1.1e+15
    step   400   ->  4.9e+41
    step   800   ->  2.2e+92
    step  1200   ->  5.2e+144
    step  1600   ->  2.9e+194
    step  2000   ->  3.2e+246
    step  2400   ->  4.0e+294
    step  2424   ->  NaN   (Inf*0)

Cross-check at a SHORTER forward horizon (n_steps=1600, part only 225.6 C):
same ~1.30x/step amplification, reached 2.4e+184 at step 1600 without yet
overflowing - i.e. a shorter/cooler march is stable-looking only because it has
fewer reverse steps to amplify through; the mechanism is the same. This confirms
the memory's "even a sub-melt run blows up identically."

Phase 4 gate: on this Tamper config the implicit `dks_peak_ds` must be FINITE
everywhere (the headline the explicit adjoint could never reach) AND FD-gate to
worst rel-err <= 1e-6 at a few top-|g| probes.

---

## 3. Warnings for the Phase-1..4 implementer (things that will surprise you)

1. **lc_part does NOT coarsen the instability.** lc_part = 5.0e-3 (2x coarser
   than production) yields a CFL-STABLE mesh (min cell h 0.69 mm) whose forward
   reaches only 166 C and whose reverse is tame - the coarsening smooths OUT the
   fine tessellation feature that creates the sliver tets. You must keep
   lc_part <= 2.5e-3 (20106 design nodes) to preserve the blow-up. So the Phase 4
   "coarse-enough-to-FD-gate but still explicit-unstable" Tamper resolution is
   NOT achieved by coarsening the mesh - use FEWER n_steps + only 2-4 FD probes
   to keep wall time sane instead.

2. **A cell-diameter CFL proxy LIES here.** dolfinx `msh.h` (cell diameter)
   reports this mesh as CFL-stable (dt/dt_stable_min ~ 0.14x) because the
   explicit-diffusion limit is set by the sliver-tet INRADIUS (tiny), not the
   circumdiameter. Do not trust a diameter-based CFL check to decide "fine mesh"
   in the Phase 5 fidelity guard - the empirical reverse blow-up is the
   authority. (forward.py's own `_stability_dt` / n_sub formula is the right
   proxy; the memory quotes min-cell dt_stable ~3.3e-3 s from it.)

3. **Memory/compute at n_sub=1 is fine.** The full 2800-step keep_cache reverse
   on this mesh is ~7 GB and ~45 s end to end - no checkpointing needed. (The
   51 GB SIGKILL noted in run_tamper.py was the n_sub=36 PRODUCTION forward
   storing every substep, a different path.) This is exactly the spec's argument
   that implicit @ n_sub=1 keeps the reverse-sweep structure feasible.

4. **The overflow step is drive/design-dependent, the mechanism is not.** Memory
   quoted "~1300"; uniform v + nominal drive here gives 2424. Both are the same
   geometric reverse amplification -> Inf -> Inf*0 -> NaN. Report the fix as
   "finite" vs "NaN", not against a specific step count.

---

## 4. Reproduce

    cd <repo-root>
    PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      heatr3d_d1_spike/env/bin/python <scratch>/phase0_step1_coarse_fd.py
    PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      heatr3d_d1_spike/env/bin/python <scratch>/phase0_step2_reverse.py 2.5e-3 0.5 2800

(The two probe scripts live in the session scratchpad; their logic is the coarse
`density_adjoint.fd_gate()` and a straight mirror of `dks_peak_ds`'s reverse loop
with per-step max|gT| instrumentation. Nothing in production code was changed.)
