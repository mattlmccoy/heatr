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

---

## 5. Phase 1 result (implicit FORWARD step landed; adjoint still Phase 2)

The explicit enthalpy Euler in `density_adjoint._substep_forward` is replaced,
behind `Case.implicit` (default True), by an unconditionally-stable IMPLICIT
backward-Euler step `_implicit_T_step`. Density update stays explicit.

**Scheme (linearly-implicit / apparent-cp backward-Euler).** Per step, transport
is LAGGED at `T_in` (conductivity `k(T_in)` already on `tc.k_fn`), and the latent
heat is carried by an APPARENT heat capacity `Capp = dH/dT|_{T_in} =
rho_cp + rho_L*m_frac/dt_pc` (the exact slope of `enthalpy_from_T`, so the scheme
reduces to the explicit enthalpy Euler as `dt -> 0`). One SPD solve per step:

    (M/dt + K(T_in) + A_conv) @ T_new = (M/dt) @ T_in + F + A_conv @ (preheat 1)

`M` is the SAME lumped nodal mass (`vol_safe`) the explicit step divides by.
Solved MATRIX-FREE by CG, reusing the ALREADY-COMPILED linear forms
(`diffG_form` -> `K@x`, `convG_form` -> `A_conv@x`) with a Jacobi(mass)
preconditioner, `rtol=1e-11`. Matrix-free is FORCED by the environment: the
Dropbox path has a space that breaks the FFCX JIT, so no NEW UFL form (no bilinear
`a(u,v)`) can be compiled; only cached forms may be reused. `A` is symmetric, so
Phase 2's transpose solve is the SAME operator/factorization.

CHOSEN apparent-cp (over an enthalpy-consistent source): it reproduces the coarse
baseline within 0.05 C, its `dH/dT` slope is byte-identical to the existing
enthalpy VJP chain (so Phase 2 reuses that machinery), and it stays a single
linear solve. No tradeoff surfaced at the Phase-1 gates.

**PHASE-1 SCOPE / pins.** Only the FORWARD changed. The per-step VJP is still the
EXPLICIT adjoint, so the adjoint-gate entry points (`ks_peak_forward`,
`dks_peak_ds`, `diagnose_case`, `march_fidelity_check`) pin `implicit=False` and
`_substep_forward` REFUSES `implicit + keep_cache` (the reverse sweep needs the
explicit `num`/`H2` caches). Result: every existing FD gate + `test_two_sided`
stay byte-identical/green. The implicit forward is exercised only forward-only.

### Gate 1 -- coarse-equivalence (square, dt=1.0, 900 steps; both CFL-stable)

- explicit ks_peak = 204.8804 C, implicit ks_peak = 204.9272 C,
  |diff| = 0.0468 C  (frozen band 1.0 C -> PASS). Non-zero diff confirms the
  implicit path is genuinely exercised (not a passthrough false-green).

### Gate 2 -- Tamper stability (lc_part=2.5e-3, 54495 cells, 10221 nodes, 30 steps)

CFL `n_sub` from `_stability_dt` = 458 (dt_stable ~1.21e-3 s) -> deeply unstable.
- reference (explicit substepped n_sub=458) peak = 61.229 C
- implicit (n_sub=1) peak = 61.034 C  -> **0.194 C** from reference (PASS, tol 1.0)
- explicit (n_sub=1) peak = 60.855 C  -> 0.374 C from reference
- **dt-cap clamp cells over the march: implicit = 0 (0.0%) vs explicit n_sub=1 =
  5379 (52.6%).** The 52.6% matches Phase 0's "54% explicit-unstable" and is the
  RED evidence: the un-substepped explicit step is CFL-broken here; the implicit
  step is clamp-free. (The PEAK alone does not discriminate -- the explicit clamp
  masks the instability, exactly the Phase-0 warning -- so the CLAMP FRACTION is
  the load-bearing test.)

### Step 5 -- convergence probe (coarse square, halve dt, fixed horizon)

implicit -> explicit end-state ks_peak as dt -> 0 (first-order, ~linear shrink):

    dt=1.000  n_steps=900   explicit=204.8804  implicit=204.9272  |diff|=0.04677 C
    dt=0.500  n_steps=1800  explicit=204.8655  implicit=204.8773  |diff|=0.01177 C
    dt=0.250  n_steps=3600  explicit=204.8581  implicit=204.8533  |diff|=0.00484 C

Confirms the implicit scheme is a CONSISTENT discretization of the SAME PDE
(not a different model).

### Regression

`test_two_sided` 6/6 (AL-grad FD gate worst_rel_err 1.572e-8, unchanged);
`test_density_adjoint`/`test_densify_forward` 11/11; `test_gate_fd`/`test_stage_b3`
14/14 -- the explicit adjoint is byte-preserved by the pins.

Tests: `solve3d/tests/test_implicit_step.py`
(`test_implicit_matches_explicit_on_coarse_within_tol`,
`test_implicit_forward_matches_substepped_reference_on_tamper` [slow]).

### Next layer (Phase 2)

Re-derive the per-step IMPLICIT VJP: back-prop through `A@T_new=b` as one
TRANSPOSE solve (A symmetric -> reuse the CG operator), plus the cotangents
through the lagged-coefficient assembly and the apparent-cp/enthalpy chain. Then
FD-gate on the coarse case (worst rel-err <= 1e-6, match the Phase-0 gradient) and
remove the `implicit=False` pins so the adjoint honors `Case.implicit`.

---

## 6. Phase 2 result (implicit per-step VJP -- the transpose solve)

`_substep_vjp` now branches on `cache.implicit`. For the implicit step
`T_new = A^{-1} b` (A = diag(mdt) + K(k_cells) + A_conv, SPD), the adjoint of the
linear solve is ONE transpose solve; A symmetric -> `lam = A^{-1} g_Tnew` via the
SAME matrix-free operator the forward built (`_implicit_operator`, assemble-once
consistency). The cotangents:

    g_F     = lam                       (b = mdt*T_in + F + A_conv(preheat 1))
    gT_in  += lam * mdt                 (T_in inside b)
    g_mdt   = lam * (T_in - T_solve)    (mdt in b, and in the A diagonal)
    g_k_c   = -int_c grad(lam).grad(T_new)   (K coefficient; reuses gk_form)
    g_Capp  = g_mdt * vol_safe / dt ;  g_rho_cp = g_Capp ;
    g_rho_L = g_Capp * m_frac / dt_pc

There is NO stiffness/convection transpose ON gT_in (unlike the explicit branch):
in the implicit operator K multiplies T_new, not T_in, so T_in enters K only via
the lagged coefficient k_cells (-> g_k_cells -> shared bottom). `T_solve`
(= pre-clamp T_new) is cached; the shared downstream (clip subgradients, densify,
k_cells / rho_cp / phi_in chains) is unchanged.

**CG preconditioner upgraded** to the FULL Jacobi diagonal
`diag(A) = mdt + diag(K) + lumped(A_conv)` (the mass-only diagonal was a poor
preconditioner where a cell's conductivity dominates its mass -- the two-sided
BOOSTED node -- degrading both the forward ks and the adjoint lam). `diag(K)` is
assembled from the SAME UFL as `forward._stability_dt` (cache hit; the space-in-
path build forbids compiling a new form). This affects convergence only, not the
solved (consistent-convection) operator.

### Phase-2 gate -- single-step VJP FD (coarse, melting state)

`test_implicit_substep_vjp_matches_fd`: one implicit step at a melting/densifying
coarse state (7 part nodes IN the melt window so the latent apparent-cp term is
live; rho densified; clips inactive). Central-difference `sum(w_T*T_out)+
sum(w_R*rho_out)` vs the VJP cotangents on T_in and rho_in probes (incl. all
melt-window nodes): **worst_rel_err = 4.9e-9** (frozen 1e-6). g_F separately
FD-verified to 6.7e-10.

Pins removed: `dks_peak_ds`, `ks_peak_forward`, `diagnose_case` now honor
`Case.implicit`. `march_fidelity_check` stays explicit (decision 4b coarse
contract). `build_coarse_case` gains `implicit=False` DEFAULT (see sec 7).

---

## 7. Phase 3 result (whole-march implicit adjoint FD-gate on the coarse square)

`test_implicit_full_march_adjoint_fd_gate_coarse`: `density_adjoint.fd_gate` on a
coarse square built with `implicit=True`.

- **worst_rel_err = 1.72e-8** (implicit adjoint self-consistent with the implicit
  forward's central FD; frozen 1e-6, PASS -- the correctness oracle).
- mutation (drop lambda_rho) worst_rel_err = 2.6e-2 (bites; density co-state
  load-bearing).

### The apparent-cp latent tradeoff (measured, NOT a VJP bug)

The implicit apparent-cp GRADIENT does NOT match the Phase-0 explicit
(exact-enthalpy) baseline gradient bit-for-bit:

- below the melt onset (n_steps=150, part < t_pc): implicit-vs-explicit gradient
  rel diff = **6.7e-3** (the transport / EQS / design-chain paths agree).
- full melting march (n_steps=900): rel diff = **1.445e-01 (14%)**, uniform across
  the high-|g| melt-window-adjacent probes, and it does NOT shrink as dt halves
  (0.1445 -> 0.1437 -> 0.1424 at dt 1.0/0.5/0.25) even though the PEAK converges
  (0.047 -> 0.012 -> 0.005 C).

Diagnosis: this is the **apparent-cp vs exact-enthalpy latent-heat treatment**
(the spec's flagged Sec-3 tradeoff), NOT a VJP error. Three independent proofs the
implicit adjoint is correct: (a) the single-step VJP FD-gate (4.9e-9), (b) the
whole-march self-consistent fd_gate (1.72e-8), (c) an FD-step sweep at a boosted
node shows the implicit central FD CONVERGES to the implicit adjoint as h->0
(3.8e-6 @ h=1e-3 -> 1.5e-7 @ h=3e-6). The apparent-cp forward tracks the substepped
truth on the Tamper to 0.19 C (sec 5), so its gradient is a correct descent
direction for an accurate forward; it simply propagates latent-heat SENSITIVITY
differently from the exact-enthalpy explicit scheme through the melt window.

### Consequence: the coarse B-stage stays EXPLICIT

Because the coarse gradient genuinely moves 14% under implicit AND the frozen
two-sided AL gate (h=1e-4, penalty factor ~3e4) amplifies the implicit scheme's
larger melt-window FD-truncation past 1e-6, `build_coarse_case` DEFAULTS
`implicit=False`. This honors decision 4b (coarse is CFL-stable + certified by the
explicit production forward -> must not move) and keeps every B1/B2/B3/B4 +
two-sided gate byte-stable and green. `Case.implicit` stays True as the fine-mesh
(Tamper) default; the implicit adjoint is gated on coarse via `implicit=True` and
is reserved for the fine-mesh solve (Phase 4).

### Regression

`test_two_sided` 6/6 (1.572e-8), `test_stage_b3` (1.016e-8), `test_density_adjoint`
/`test_densify_forward`/`test_gate_fd`/`test_stage_b`/`test_stage_b4` all green;
the Phase-1 Tamper forward test still PASS. Explicit coarse path byte-preserved.

### Next layer (Phase 4 -- coordinator-checkpointed compute)

FD-gate the implicit adjoint on the TAMPER (a resolution where the explicit adjoint
NaN'd): `dks_peak_ds` FINITE everywhere + worst rel-err <= 1e-6 at a few top-|g|
probes. If the 14% apparent-cp latent difference is judged to matter for the
fine-mesh solve, an enthalpy-consistent-source variant is the fallback (a separate
FD-gated layer).

---

## 8. Phase 4 result -- THE PAYOFF: the Tamper implicit adjoint is FINITE (was NaN)

On the FINE Tamper mesh -- the EXACT resolution where the explicit adjoint NaN'd
(sec 2): STL feb850ec, **lc_part = 2.5e-3** (REQUIRED; 5e-3 is CFL-stable and hides
the bug), n_cells = 54495, n_nodes = 10221, n_design = 20106, dt = 0.5, design
v = ones (uniform), nominal drive. The FD probes use a REDUCED horizon
(n_steps = 300; the MESH, not the step count, is the explicit-instability source),
which heats the part to ~176.6 C (near the melt onset).

FINITE vs NaN (reported as finite/NaN, NOT vs a step count -- the overflow step is
drive/design-dependent, sec 3.4):

- **IMPLICIT `dks_peak_ds` (Case.implicit=True): FINITE EVERYWHERE.** max|g| = 0.79,
  no Inf/NaN. Backward-Euler at n_sub=1 is unconditionally stable, so the reverse
  co-state cannot geometrically overflow -- this is the gradient the explicit
  adjoint could never reach. (dks wall ~146 s at n_steps=300.)
- **EXPLICIT `dks_peak_ds` at the Phase-0 config (n_steps = 2800): NaN**
  (has_nan = True), reproduced LIVE here on the same mesh (the reverse co-state
  overflows to Inf then Inf*0 -> NaN, sec 2).

FD-gate (self-consistency of the implicit adjoint with the implicit forward on the
FINE mesh -- the same gate that passed 1.72e-8 on coarse):

- top-|g| design probes: index 11872 rel_err = 1.60e-8, index 11859 = 3.26e-9.
- **worst_rel_err = 1.60e-8 (frozen 1e-6, PASS).**

Total wall ~577 s (~9.6 min): build 5 s, implicit dks 146 s, 2 FD probes ~309 s,
explicit-NaN reference 39 s. No checkpointing (n_sub=1, ~sub-10 GB).

Durable gate: `test_tamper_implicit_adjoint_finite_and_fd_gated`
(`solve3d/tests/test_implicit_step.py`, @slow) -- asserts FINITE + worst
rel-err <= 1e-6 on the Tamper.

**This is the artifact that unblocks the Tamper two-sided rescue** (Phase 6): the
fine-mesh ceiling-coupled gradient is now finite and FD-verified. Remaining before
the heavy rescue: Phase 5 (fidelity contract per decision 4b -- coarse bit-match,
heatr3d the fine-mesh arbiter) and Phase 6 (relaunch `run_tamper_rescue --solve`).
The apparent-cp latent tradeoff (sec 7, 14% coarse gradient vs exact-enthalpy) is
accepted for the fine-mesh solve (heatr3d is the arbiter, 4b); the
enthalpy-consistent-source variant remains the fallback if that tradeoff is later
judged to matter.

---

## 9. Phase 5 result -- decision 4b encoded in march_fidelity_check

`march_fidelity_check` now branches on the EXPLICIT-forward CFL substep count
`n_sub` (new helper `_explicit_n_sub`: `forward._stability_dt` at the worst-case
conductivity + the `ceil(dt/(CFL_SAFETY*dt_stable))` formula, mirroring
`march_enthalpy`). This is the inradius-driven lumped-mass limit (the assembled
stiffness diagonal), NOT a cell-diameter proxy -- Phase 0 sec 3.2 proved the
diameter proxy LIES here.

- **COARSE (n_sub == 1):** runs BOTH forwards byte-matched (SAME mesh, drive Q, dt,
  n_sub=1, horizon) and asserts bit-identity, unchanged. The gate `_march` is
  EXPLICIT on coarse, so this is the explicit-gate-vs-explicit-production match.
  MEASURED on the coarse square: `fine_mesh=False`, `n_sub_explicit=1`,
  **rel_T = 0.0, rel_mean_rho = 0.0, agree = True (exact bit-match)** -- the coarse
  certified path did NOT move under the implicit work (this is the no-op bit-match
  the coordinator asked to confirm). The doc gains `fine_mesh`, `n_sub_explicit`,
  `dt_stable_s`.
- **FINE (n_sub > 1):** the explicit production forward is itself CFL-unstable at
  dt, so it is not a valid fine-mesh reference. Instead of asserting bit-identity
  against an unstable run, `march_fidelity_check` RETURNS
  `{"fine_mesh": True, "arbiter": "heatr3d", "reason": "explicit forward is
  CFL-unstable at this dt (n_sub>1); heatr3d (voxel FD) is the fine-mesh arbiter
  per decision 4b", "n_sub_explicit": ..., "dt_stable_s": ..., "agree": None}`.
  MEASURED on the fine Tamper (lc_part=2.5e-3): `fine_mesh=True`, `arbiter=heatr3d`,
  **`n_sub_explicit = 323`**, `dt_stable = 1.72e-3 s`. Detection only -- NO heavy
  march / EQS solve is run (one `_stability_dt` assemble on top of the mesh build).

Decision 4b is now encoded in CODE, not just the doc: the explicit arbiter is the
truth only where it is CFL-stable (coarse); on fine meshes heatr3d is the arbiter.

Tests (`solve3d/tests/test_implicit_step.py`):
`test_march_fidelity_coarse_bit_matches` (bit-match, `fine_mesh=False`, n_sub=1)
and `test_march_fidelity_fine_mesh_returns_4b_record` (@slow; the Tamper returns
the 4b record, n_sub_explicit>1, no false assertion, no heavy march).

Regression: `test_two_sided` 6/6, `test_density_adjoint` 6/6 (incl.
`test_march_matches_production_densify` still agree=True). The next layer is
Phase 6 (relaunch the Tamper two-sided rescue) -- Matt-checkpointed heavy compute.
