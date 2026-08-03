# Gate S1 report: heatr3d numerical integrity

Date: 2026-07-30
Author: computational-solver-engineer pass over
`docs/superpowers/plans/2026-07-30-heatr3d-s1-numerical-integrity.md`
Spec: `docs/superpowers/specs/2026-07-30-heatr3d-graduation-design.md`, Gate S1
Evidence file (every number below is reproduced there with its command):
`docs/superpowers/plans/s1-findings.md`
Interpreter: `./.venv312/bin/python` (Python 3.12.7, numpy 1.26.4, scipy 1.13.1)
Machine: Apple Silicon, 12 cores, 34.36 GB RAM

---

## 0. Verdict

> **AMENDED 2026-07-31 after the S1b pass (commits 22122da, 06bd87d and the
> n=200 thermal evidence run).** The original 2026-07-30 verdict and its table
> are preserved verbatim below; the amended verdict supersedes them. Nothing in
> the original numbers changed -- what changed is that both blockers were worked.
>
> **Amended verdict: S1 PARTIALLY PASSED -- the thermal/phase core passes all
> criteria including full-scale; overall S1 remains open solely on the large-N
> EQS path (D1).**
>
> | S1 pass criterion | Amended status |
> |---|---|
> | Instability reproduced and explained | **MET** (both mechanisms, sections 2 and 3) |
> | Fixed, with a regression test | **MET for the thermal/phase core**: mechanism A fixed by the enthalpy update; mechanism B (THM-03) fixed by the CFL guard + auto-substepping (commit 06bd87d), closed-form-verified, default-inert for n <= 169 |
> | Analytic benchmarks within stated tolerances | **MET** (section 5) |
> | Standing conservation gate wired into all entry points | **MET** (section 6), and it stays exact under substepping |
> | "Avoidance by grid cap is explicitly not a pass" | **MET for the thermal march** (n=200 runs clean to melt onset, section 4.2); **NOT MET for the EQS solve** -- n=200 EQS is infeasible and now fails loudly instead of segfaulting (EQS-01), with the scalable path deferred to the D1 dolfinx spike |
>
> Blocker status after S1b:
> - **THM-03: FIXED.** `Params.enforce_cfl` (default True) substeps the thermal
>   update so `dt_sub <= 0.9 h^2/(6 alpha_max)`. Verified against the closed-form
>   checkerboard amplification, and evidenced at full scale by the n=200 thermal
>   march of section 4.2. Default-inert; note the corrected no-change ceiling is
>   **n <= 169**, not the n <= 178 stability threshold quoted in section 3 -- the
>   0.9 safety factor moves the substepping trigger below the marginal-stability
>   grid.
> - **EQS-01: CONVERTED, NOT SOLVED.** The process-killing SIGSEGV is gone;
>   `solve_eqs_3d` now raises an informative `MemoryError` above
>   `EQS_DIRECT_MAX_UNKNOWNS = 2e6` naming N, the grid, the ~29.8 GB LU estimate
>   and the measured ceilings (**n=96 full pipeline, n=128 EQS-only**). A
>   scalable large-N EQS path is explicitly **deferred to D1 (dolfinx FEM
>   spike)**; it is the sole remaining S1 blocker.

**[ORIGINAL 2026-07-30 VERDICT, superseded]** **Gate S1 does NOT pass yet.**
Three of the four S1 pass criteria are met; the
fourth (the previously blowing-up case runs clean) is blocked by two defects
that this pass MEASURED for the first time and that no work in Tasks 1-7
addresses.

| S1 pass criterion (spec section "Gate S1") | Status |
|---|---|
| Instability reproduced and explained | **MET** (two mechanisms, sections 2 and 3) |
| Fixed, with a regression test | **PARTIAL**: the latent-skip mechanism is fixed and gated; the grid-threshold mechanism is diagnosed but unfixed |
| Analytic benchmarks within stated tolerances | **MET** (section 5; 3 benchmarks, tolerances in-test) |
| Standing conservation gate wired into all entry points | **MET** (section 6) |
| "Avoidance by grid cap is explicitly not a pass" | **NOT MET**: n <= ~96 is currently a de-facto cap (sections 3, 4) |

Recommended decision: **do not declare S1 passed**; approve the canonical sync
of the Tasks 1-7 work (section 9, it is strictly additive and default-inert),
and authorise a short S1b pass for the two blockers before S2 convergence work
starts. Rationale: the S2 convergence study is a grid-refinement study, and both
blockers are grid-refinement defects -- S2 cannot be run past n ~ 96 until they
are cleared. (This S1b pass was authorised and has now been executed; see the
amendment above.)

---

## 1. What was built

| Artifact | Path | Content |
|---|---|---|
| S1 test suite | `test_heatr3d_s1.py` | 9 fast tests + 1 slow full-scale regression |
| Solver changes | `heatr3d.py` | energy audit, enthalpy phase update, `T_final`, `T0_override`, empty-part guard, standing gate print |
| Diagnostics | `scripts/analysis/s1_diagnose_instability.py` | CFL / source-dT / latent-barrier arithmetic |
| Diagnostics | `scripts/analysis/s1_cfl_powder_mode.py` | THM-03 checkerboard growth experiment |
| Pytest config | `pytest.ini` | registers `slow`, `addopts = -m "not slow"` (repo had no pytest config before) |
| Evidence | `docs/superpowers/plans/s1-findings.md` | sections 1-13, every measurement |

Default fast suite: `9 passed, 1 deselected in 304.37 s`.

---

## 2. Mechanism A (pinned and fixed): latent-heat skip on a window-crossing step

**Diagnosis.** The legacy phase treatment computes latent heat pointwise as
`cp_eff = cp + latent * dphi(T_now)`. A voxel that starts a step OUTSIDE the
melt window has `dphi(T_now) = 0`, so a single explicit Euler step of size
>= `dt_pc_c` carries it across the whole window paying none of the latent
barrier. Observed directly (module-level trace, no solver edit, spike_mult=400,
n=32): the spiked voxel (16, 11, 13) jumps

```
phi 0.0000 -> 0.8000 in ONE step, T 173.00 -> 183.00 C, dphi/dT at step start = 0.0000 1/C
```

The skipped barrier is large: `latent/dt_pc = 9670 J/(kg C)` vs
`cp_solid = 2500 J/(kg C)`, i.e. the in-window apparent heat capacity is 4.9x
the sensible value. `max_dt_step_c = 10.0` exactly equals `dt_pc_c = 10.0`, so
even a fully clamped step is one whole window wide: the limiter cannot prevent
the crossing, it only hides it.

**Fix.** `Params.phase_update`: `"apparent_cp"` (default, legacy, bit-for-bit)
or `"enthalpy"` -- an exact piecewise-linear volumetric-enthalpy inversion
(`enthalpy_from_T` / `T_from_enthalpy`) that deposits `num*dt` into H and
inverts, so the latent plateau can never be stepped over.

**Evidence it works** (`bulk_crossing_case`, mult=130, n=32, no limiter binding,
identical RF input in both arms):

| t_end | scheme | in (J) | stored (J) | residual_frac |
|---|---|---|---|---|
| 1.0 s (mid-window, phi_mean 0.669) | apparent_cp | 851.044 | 919.320 | **-0.0802** |
| 1.0 s | enthalpy | 851.044 | 851.044 | **+2e-16** |
| 3.0 s (fully molten) | apparent_cp | 2553.131 | 2553.246 | -4.5e-05 |
| 3.0 s | enthalpy | 2553.131 | 2553.131 | -1.5e-16 |

The enthalpy arm is exact by construction (the update deposits into the same
H(T) the audit books, and `phase_fraction`'s phi IS the enthalpy ramp fraction);
1e-16 is the honest reading, not a tuned tolerance.

**Regression tests:** `test_legacy_phase_update_skips_latent_on_window_crossing`
(pins the legacy signature: clamp_bound True, residual +0.3017) and
`test_enthalpy_update_conserves_energy_on_window_crossing` (the fix, with a
differential legacy control on the identical case).

**Honest scope limit.** Mechanism A is real and directly measured, but it is NOT
the documented grid >= 200 trigger. See mechanism B.

---

## 3. Mechanism B (NEW; diagnosed here, FIXED in S1b): explicit conduction CFL
violation in the powder bed -- this is the grid >= 200 trigger

> **STATUS AMENDED 2026-07-31: THM-03 is FIXED** (commit 06bd87d), by fix
> option (a)+(b) of the list at the end of this section: `run()` computes
> `dt_stable = h^2/(6 alpha_max)` over all materials present and, with the new
> `Params.enforce_cfl` (default **True**), auto-substeps the thermal update
> `n_sub = ceil(dt_s / (0.9 dt_stable))` times per `dt_s`. `Result` gains
> `n_substeps_used` and `cfl_violated`. Everything measured below stands
> unchanged; it is the diagnosis the fix is built on.
>
> - Closed-form verification: at CFL ratio 1.25 (the n=200 value, reproduced
>   cheaply at n=32 by raising dt_s) the legacy single step grows the
>   checkerboard at **1.5000/step vs the predicted 1.5003**, while the enforced
>   `n_sub = 2` decays it to 2.6e-06 C with the domain mean conserved to 0.0.
> - The standing energy gate stays exact under substepping (the audit banks per
>   substep with `dt_sub`): RED with the audit reverted to `p.dt_s` gives
>   residual **+0.5000**, GREEN gives **-1.09e-15**.
> - **Default inertness / corrected ceiling:** `n_sub = 1` for every
>   configuration with **n <= 169** at dt_s = 0.05 s, L = 0.060 m -- NOT n <= 178.
>   The 178.9 figure below is the marginal-stability grid; the 0.9 safety factor
>   pulls the substepping trigger down to 170. Corrected here explicitly because
>   an earlier statement of this report's finding quoted n <= 178 as the
>   no-change ceiling. Verified out of band against the pre-change module
>   (22122da), n=32 sphere, both phase schemes x densify on/off: **4/4
>   BIT-IDENTICAL** (max|dT| = 0.0 on T_final / T_phi90 / phi_final / rho_final,
>   identical sigma_T, energy_in_j, t90, phi_hist), so every historical heatr3d
>   number is preserved.
> - **Full-scale evidence at n=200:** section 4.2 below.

The Task-3 note concluded "conduction CFL is innocent, ratio 0.262 at n=200".
That used `alpha_max = k_liquid/(rho_liquid*cp_liquid)`. **The fastest medium is
the powder bed, which is ~98 % of the domain:**

| medium | k | rho | cp | alpha [m^2/s] |
|---|---|---|---|---|
| powder | 0.197 | 490.0 | 1072.0 | **3.7504e-07** |
| solid | 0.10 | 460.0 | 2500.0 | 8.6957e-08 |
| liquid | 0.26 | 1010.0 | 3279.0 | 7.8507e-08 |

With alpha_powder (4.78x the value used), `dt < h^2/(6 alpha)` fails for
**n > 178.9** at dt_s = 0.05 s, L = 0.060 m: ratio 0.512 at n=128, 0.800 at
n=160, **1.250 at n=200**. The threshold grid 178.9 IS the documented
"blow-up at grid >= 200" boundary.

Direct experiment (`scripts/analysis/s1_cfl_powder_mode.py`): all-powder domain,
no source, no convection, 1e-3 C 3-D checkerboard initial condition (fastest
discrete mode), 60 steps. Predicted per-step amplification
`|1 - 12 alpha dt/h^2|` vs measured:

| n | CFL ratio | predicted | measured | clamp_bound |
|---|---|---|---|---|
| 176 | 0.968 | 0.9362 | **0.9362** (decays) | False |
| 184 | 1.058 | 1.1162 | **1.1162** (GROWS) | False |
| 200 | 1.250 | 1.5003 | 1.1558 (GROWS) | **True** |

Four-decimal agreement on both sides of the threshold, and the sign flips
exactly where predicted. At n=200 the measured rate is lower than predicted only
because the +-10 C THM-01 limiter binds and truncates the growth -- the clamp is
hiding a divergence, which is precisely what the spec refuses to accept.

Consequences:
- `assert res.clamp_bound is False` at n=200 cannot hold at dt_s = 0.05 s
  regardless of the phase scheme. Mechanism A's fix is necessary, not sufficient.
- The energy-residual gate does NOT catch mechanism B (these runs still book
  small residuals); `clamp_bound` is the flag that catches it.

Fix options, increasing cost: (a) enforce `dt_s <= 0.9 h^2/(6 alpha_max)` as an
asserted stability criterion in `run()` -- spec-legal ("a fix OR a principled
stability criterion"), costs ~1.4x steps at n=200; (b) conduction substepping;
(c) implicit/IMEX conduction. Recommendation: (a) now (cheap, honest, unblocks
S2), (c) evaluated in the D1 dolfinx spike.

---

## 4. Full-scale regression: written, run once, BLOCKED on EQS-01 only
(originally "currently BLOCKED", by both defects)

> **STATUS AMENDED 2026-07-31.** All numbers in this section stand. What changed:
> `test_full_scale_n200_melt_onset_clean_with_enthalpy` no longer SIGSEGVs -- it
> now fails with an informative `MemoryError` from `solve_eqs_3d` (EQS-01,
> commit 22122da), which names N = 8.0e6, the grid, the ~29.8 GB direct-LU
> estimate and the measured supported grids (**n=96 full pipeline, n=128
> EQS-only**). The ceiling itself is unchanged and a scalable large-N EQS path is
> **deferred to D1 (dolfinx FEM spike)**; that is now the test's SOLE remaining
> blocker. The second blocker it carried (THM-03) is fixed -- see section 3 --
> and the thermal/phase half of this regression has been run at full scale and
> passes; see the new section 4.2.

`test_full_scale_n200_melt_onset_clean_with_enthalpy`, `@pytest.mark.slow`,
assertions exactly as planned (`reached is True`, `|residual_frac| < 0.05`,
`clamp_bound is False`). **Nothing was weakened.**

**Runtime estimate made before launching** (as required): thermal loop at n=200
measured at **0.370 s/step** (10 steps 4.1 s, 40 steps 15.2 s, peak RSS
2.35 GB) -> **1.85 h** for max_time_s=900 (18 000 steps), 3.08 h for 1500 s.
Under the 6 h stop line -- so the run was launched.

**Actual outcome: SIGSEGV after 17 s.**

```
Fatal Python error: Segmentation fault
  File ".../scipy/sparse/linalg/_dsolve/linsolve.py", line 293 in spsolve
  File ".../geo-prewarp/heatr3d.py", line 265 in solve_eqs_3d
  File ".../geo-prewarp/heatr3d.py", line 597 in run
PYTEST_EXIT=139
```

**NEW FINDING EQS-01.** At N = 8.0e6 complex unknowns
`spla.spilu(A, drop_tol=1e-4, fill_factor=12)` cannot allocate
(`malloc fails for local dworkptr[]`, SuperLU zgstrf); `solve_eqs_3d` catches
that with a bare `except Exception: V = None` and falls through to
`spla.spsolve` -- a direct complex LU of an 8-million-unknown 3-D Laplacian,
hopeless at any memory size, and here it segfaults rather than raising.
Reproduced 3/3. Two defects: an armed-but-unusable direct fallback, and a
process-killing failure mode a caller cannot handle.

Measured EQS cost/feasibility: n=64 (N=2.6e5) 97.7 s / 0.81 GB; n=96 (N=8.8e5)
322.1 s / 2.21 GB; n=128 did not finish in 30 min; n=200 crashes.
**The largest grid at which the full pipeline runs today is n ~ 96.**

Third, smaller issue found while sizing the test: `max_time_s=900` is marginal
for `phi_target=0.90`. At n=32 mean phi reaches only 0.8385 (legacy) / 0.8787
(enthalpy) at 900 s (`run()`'s own default is 1500 s). It is grid dependent and
moves the right way -- the n=96 full-physics run reaches melt onset at
**t90 = 802.6 s** -- so 900 s is probably adequate at n=200. Noted, not changed.

### 4.1 Best available substitute: the largest full-physics run that completes (n=96)

Full pipeline (EQS + thermal, no injected Qrf), n=96 sphere d=20 mm, default
Params with `phase_update="enthalpy"`, `max_time_s=1500`, phi_target=0.90:

```
  [s1-energy] in=5381.5 J stored=5289.0 J loss=92.5 J residual_frac=+0.0000
FULL n=96 enthalpy: wall=1049s reached=True t90=802.6 sigma_T=19.278 Tmax=251.2
                    clamp=False resid=+1.3320e-13 final_phi=0.9000
```

A real production-configuration melt-onset run at 8.85e5 voxels with the S1 fix
engaged: **energy residual 1.33e-13, no clamp bound, melt onset reached.** CFL
ratio at n=96 is 0.288, so mechanism B is dormant there exactly as predicted.
This is evidence that the fix and the gate work at scale up to the feasible
grid; it is NOT a substitute for the n=200 regression the spec asks for.

### 4.2 THM-03 full-scale evidence at n=200 (2026-07-31): the THERMAL march is clean

`test_n200_thermal_march_clean_with_qrf_override`, `@pytest.mark.slow`.
Grid(n=200, L=0.060) = 8.0e6 voxels, sphere d=20 mm, `phase_update="enthalpy"`,
`enforce_cfl` at its default True, RF drive = uniform `qrf_override` at
`p.power_density_w_per_m3` over the part, `run()` defaults to the phi=0.90 stop.

Runtime estimated before launching: 0.758 s/step from a 20-step probe, ~12800
steps (t90 is nearly grid-independent under the uniform drive: 645.55 / 651.15 /
639.85 s at n=32/64/96) -> **~2.70 h**, under the 3 h stop line. **Actual:
PASSED in 10303.51 s = 2:51:43.**

```
  [s1-energy] in=4257.3 J stored=4219.8 J loss=37.5 J residual_frac=-0.0000
N200-THERMAL: reached=True nsub=2 t90=638.975 sigma_T=22.843 Tmax=260.6
              clamp=False cfl_violated=False resid=-3.2933e-13 phi_mean=0.9000
```

| assertion | value |
|---|---|
| `reached is True` | True, t90 = 638.975 s |
| `n_substeps_used == 2` | 2 (dt_sub = 0.025 s) |
| `clamp_bound is False` | False, over 12780 steps |
| `cfl_violated is False` | False |
| `abs(energy_residual_frac) < 0.05` | -3.2933e-13 |

This is the previously blowing-up grid, run at that grid -- not avoided by a
cap. **Scope: thermal/phase only.** The uniform `qrf_override` deliberately
bypasses the EQS solve, which at n=200 is still infeasible (EQS-01 -> D1), so
this does NOT constitute a full-physics n=200 pass and section 4's test stays
blocked. Full record, including the grid-trend cross-check
(sigma_T 22.843 at n=200 vs 22.950 / 22.963 / 23.010 at n=96/64/32 on the same
uniform drive), in `docs/superpowers/plans/s1-findings.md` section 13.8.

---

## 5. Analytic benchmarks (all PASS, tolerances asserted in-test)

| # | Benchmark | Configuration | Result | Tolerance |
|---|---|---|---|---|
| 1 | Adiabatic uniform heating with latent plateau | n=16, whole domain = part, q=2.0e6 W/m^3, 100 s, lands mid-plateau at T_exact = 178.482866 C (phi ~ 0.35), enthalpy scheme | err **1.71e-01 C** (default property blending) | < 0.5 C |
| 1b | same, constant properties (cp_liquid=cp_solid, k_liquid=k_solid, rho_liquid=rho_s) = the analytic solution's own assumptions | | err **3.13e-13 C**, residual < 1e-9, clamp False | < 0.05 C |
| 2 | Fourier-mode conduction decay | n=24, all powder, no source, no convection, lowest Neumann cosine mode (5 C), 400 s = 8000 steps, alpha = 3.7504e-07 | continuous ref **-1.566e-03**; semi-discrete **-1.054e-05**; fully discrete **+2.741e-13** | 5e-2 / 1e-4 / 1e-9 |
| 2b | spatial convergence (documented, not asserted) | n=24 -> n=48 | -1.566e-03 -> -3.992e-04, ratio **3.92 ~ 4** = 2nd order in h | - |
| 3 | EQS parallel-plate field, direct branch | n=24, N=13824 <= 50000 -> spsolve | max abs(V - linear) **1.25e-11 V** (1.4e-14 rel), transverse std 9.46e-13 V, max abs(Im V) 8.3e-17 V | 1e-6 / 1e-9 rel |
| 3b | EQS parallel-plate field, iterative branch | n=40, N=64000 > 50000 -> ILU-BiCGSTAB (the branch every production grid n >= 37 takes) | max abs(V - linear) **3.14e-06 V** (3.7e-09 rel), transverse std 4.84e-06 V | 1e-6 / 1e-7 rel |

Benchmark 1's 0.171 C gap is a property-model difference, not a wiring error:
once phi > 0 the solver blends rho -> rho_liquid and cp -> cp_liquid while the
closed-form H(T) assumes the fixed solid slope. With the properties the analytic
solution actually assumes, agreement is machine precision (1b). Benchmark 2's
`amp0 = 5 cos(pi/2n)` factor is the cell-centred grid's sampling deficit: the
grid never samples the mode's peak, so the observed (max-min)/2 starts at
4.98929, not 5.

Deviation from spec wording, documented not hidden: the spec lists a 1-D Stefan
problem. It is covered here by the latent-plateau benchmark (analytic moving
front in the uniform-state limit), not by a classical two-phase Stefan front
position. If reviewers want the classical Stefan test it slots into S2.

---

## 6. Standing conservation gate (spec requirement, now on every solve)

`run()` prints, unconditionally, one line per solve:

```
  [s1-energy] in=5891.8 J stored=5763.9 J loss=127.9 J residual_frac=+0.0000
  [s1-energy] in=1287.9 J stored=899.3 J loss=0.0 J residual_frac=+0.3017  CLAMP-BOUND
```

and `Result` carries `energy_in_j`, `energy_stored_j`, `energy_loss_j`,
`energy_residual_frac` for programmatic gating. Audit v2 banks sensible and
latent energy PER STEP with the same `rho`, `cp`, `rho_s_eff` maps the update
used, so the gate is meaningful at melt (v1 was not; see finding C).

---

## 7. Four further findings (documented, each with numbers)

**A. Clamp saturation is a real conservation leak, and the audit shows it.**
At spike_mult=400 / 120 s the +0.3017 residual is NOT mostly skipped latent: the
skipped latent for one voxel is ~0.3 J, while ~336 J of the ~389 J surplus is
the spiked voxel pinned at `temp_max_c = 600 C` while RF keeps depositing
~4.2 W into it. Deliberate implementation choice (deviation from the plan
snippet): audit v2 banks the ACTUAL applied change `T - T_prev`, i.e. AFTER the
temp clamp, so clamp-destroyed energy stays visible instead of being credited.

**B. heatr3d is not bit-reproducible run-to-run, in BOTH the thermal loop and
the EQS solve.** Repeating an identical n=32 / 20 s solve in one process,
run 3 of 7 differed by max abs(dT) = 7.1e-15 C. A control comparing the HEAD
module against ITSELF (two module loads, same source text) reproduced the
identical signature `T_phi90 7.105e-15, Qrf 9.313e-10` in 1 of 4 runs, so this
is BLAS/SuperLU reduction-order non-determinism, not an effect of the S1 diff.
**Correction to a claim in the Task-4 test docstring**: the EQS solve is NOT
exactly reproducible either (2 of 5 runs showed `Qrf max abs diff = 9.313e-10`).
Practical rule: quote the drift RELATIVE (Qrf is O(1e7) W/m^3, so 9.3e-10 abs =
**8.5e-17 rel**, one to two ulp). An absolute 1e-12 gate on Qrf is badly scaled
and will flap; the drift is bimodal (exactly 0 or exactly 9.313e-10), the
signature of two alternative reduction orders, not accumulating chaos.

**C. Audit v1 drifted to +0.3795 on a HEALTHY molten run** (both schemes,
`bulk_crossing_case`, t=3.0 s) because it booked stored energy with
initial-state solid properties while the solver blends to cp_liquid=3279 /
rho_liquid=1010. It was useless as a gate exactly where the instability lives.
Audit v2 (per-step property banking) brings the same case to -4.5e-05 (legacy)
and -1.5e-16 (enthalpy) WITHOUT masking the scheme defect (the mid-window
comparison in section 2 still discriminates -0.0802 vs +2e-16). Related
sub-finding: the legacy residual CANCELS to ~1e-4 once the part is fully molten
(skipped latent offset by paying the resolved latent at the blended liquid
density instead of the enthalpy basis rho_s_eff = 473.5), so a fully-molten end
state is NOT a discriminating test of the phase scheme -- only the partially
molten, window-crossing state is.

**D. The effective plate gap is (n-1)h, not L** (EQS discretization, documented,
NOT corrected). The Dirichlet rows sit at the CELL CENTRES of the first/last y
layers, so the uniform field is `abs(E_y) = v_lo/((n-1)h) = 14956.521739 V/m` at
n=24 (matched to 13 digits, ptp 2.6e-09), exceeding the nominal
`v_lo/L = 14333.33 V/m` by `L/(L-h)` = **4.3 % at n=24, 0.5 % at n=200**. It is
a grid-dependent bias on any quoted absolute V/m and on E-field grid-convergence
claims, but it is a uniform scale factor and `compute_qrf_3d` renormalizes Q to
a fixed total absorbed power, so it cancels out of Qrf on a uniform medium.
Moving the electrodes onto the domain faces is a half-cell BC change and is an
**S2 discretization decision**, not an S1 fix.

Plus, out of scope and untouched: `test_heatr3d_outputs.py` and
`test_fgm_pipeline.py` produce 25 collection errors, all
`fixture 'run_dir' not found`; there is no `conftest.py` anywhere in the repo.
Pre-existing, unrelated to heatr3d.py. `pytest.ini` was deliberately written so
those errors stay visible.

---

## 8. What is proven vs simplified vs assumed

**Proven (measured, reproducible):** mechanism A and its fix; mechanism B and
its threshold; the three analytic benchmarks; the audit's exactness on the
enthalpy arm; legacy bit-for-bit inertness at the default (6/6 exact on
T_phi90/Qrf/phi_final/sigma_T across the S1 diff, modulo finding B's ulp-level
non-determinism); EQS-01.

**Simplified:** the phase model is a linear-in-T melt ramp over a 10 C window
with property blending, not a true Stefan front; the powder bed is a static
conductive medium (no gas/vapour transport, no sintering-dependent k except
through rho_rel); the RF drive is a fixed total absorbed power renormalized onto
the Q_rf shape; densification is off in every S1 test.

**Assumed, not verified here:** material properties (taken from the COMSOL
Tuned_Sigma reference); that grid >= 200 is the historically reported trigger
(this pass reproduced the mechanism and its threshold at 178.9, not the original
run); that the documented "residual ~1e5 x dose" figure came from the same
configuration.

---

## 9. Canonical-sync decision package

**DO NOT auto-sync.** The canonical file
`research/dissertation_materials/analysis-3dfgm/heatr3d.py` was NOT touched by
this work. It currently hashes to
`5243529f819e302de2cd063dac1646a181914ef96d5561a72493461cb1985ea7`, which equals
`SYNCED_FROM_SHA256` in `geo-prewarp/heatr3d.py` -- i.e. no drift exists today
except the S1 changes in the working copy.

**Proposed diff (geo-prewarp/heatr3d.py vs the canonical body): +185 / -33 lines
in 8 hunks** (a 9th and 10th hunk are the pre-existing synced-copy preamble and
`_check_canonical_drift()` scaffolding, which must NOT be copied back):

| # | Location | Change | Default-path effect |
|---|---|---|---|
| 1 | `Params` | `phase_update: str = "apparent_cp"` | none (new field, legacy default) |
| 2 | after `phase_fraction` | `enthalpy_from_T`, `T_from_enthalpy` | none (new module functions) |
| 3 | `Result` | `energy_in_j`, `energy_stored_j`, `energy_loss_j`, `energy_residual_frac`, `T_final` | none (new fields, defaults) |
| 4 | `run()` signature | `T0_override: np.ndarray \| None = None` | none (None == old init) |
| 5 | `run()` init | empty-part-mask guard `_has_part` | none for non-empty masks; a part-free domain previously CRASHED in a zero-size reduction |
| 6 | `run()` loop | per-step energy audit accumulation (read-only w.r.t. the solve) | none (reads state) |
| 7 | `run()` update | `if p.phase_update == "enthalpy": ... else: <verbatim legacy block>` | none (legacy branch is the original code, unmodified) |
| 8 | `run()` return | standing `[s1-energy]` print + new Result fields | one extra stdout line per solve |

**Verified inertness:** default `Params()` (`phase_update="apparent_cp"`), n=32
sphere, 30 s, pre-S1 module vs post-S1 module in one process:
`np.array_equal True` on T_phi90, Qrf and phi_final, identical
`sigma_T = 21.19205274948017`, identical audit
(`in=196.3947 stored=196.3947 resid=+0.000000`) -- 6/6 exact, subject to finding
B's ulp-level run-to-run non-determinism which is present with or without this
diff.

**The default stays `"apparent_cp"`.** All historical heatr3d numbers remain
bit-for-bit reproducible. The S1 campaign and any future trusted run must opt in
with `dataclasses.replace(p, phase_update="enthalpy")` (`Params` is frozen).

**Sync procedure when approved:** copy hunks 1-8 into the canonical file, re-hash
it, update `SYNCED_FROM_SHA256` in `geo-prewarp/heatr3d.py` to the new canonical
hash, and re-run `./.venv312/bin/python -m pytest test_heatr3d_s1.py -v`
(expect 9 passed, 1 deselected) plus the inertness check above.

**Open question for the same decision:** whether the S1b blocker fixes (EQS-01
robustness, THM-03 stability criterion) should land BEFORE the canonical sync,
so the canonical file changes once instead of twice. Recommendation: yes, sync
once, after S1b -- unless another workstream needs `T_final`/`T0_override` in the
canonical file sooner.

---

## 10. Sign-off

**AMENDED 2026-07-31 (post-S1b).** Gate S1 is presented as
**S1 PARTIALLY PASSED -- the thermal/phase core passes all criteria including
full-scale; overall S1 remains open solely on the large-N EQS path (D1).**
Delivered: the mechanism work, both fixes (enthalpy phase update; THM-03 CFL
guard + auto-substepping), the benchmarks, the standing gate, the n=200
thermal/phase full-scale evidence run (section 4.2), and EQS-01 converted from a
process-killing SIGSEGV into an informative `MemoryError` with measured size
ceilings (n=96 full physics, n=128 EQS-only). Outstanding: a scalable large-N
EQS solver, deferred to **D1 (dolfinx FEM spike)**.

- [ ] **Matt McCoy** — I have read this report and I approve / do not approve:
      (a) the amended S1 verdict (S1 PARTIALLY PASSED, open on D1),
      (b) proceeding to D1 as the sole remaining S1 blocker, and
      (c) the canonical sync plan in section 9 (and its timing, which section 9
      recommends be after S1b -- i.e. now).

**[ORIGINAL 2026-07-30 sign-off, superseded]** Gate S1 is presented as
**NOT PASSED**, with the mechanism work, the fix, the
benchmarks and the standing gate delivered, and two newly measured blockers
(EQS-01, THM-03) outstanding.

- [ ] **Matt McCoy** — I have read this report and I approve / do not approve:
      (a) the S1 verdict above, (b) the S1b scope (EQS-01 + THM-03) before S2,
      and (c) the canonical sync plan in section 9 (and its timing).

Signature: ______________________  Date: ____________


---
SIGNED OFF: Matt McCoy, 2026-07-31 (via session): S1 canonical-sync package approved.
Execution note: the physical canonical sync of heatr3d.py to dissertation_materials/analysis-3dfgm/ is deliberately deferred until the EQS-02 fix lands, so the canonical file is synced once with the complete S1+EQS-02 change set and one SYNCED_FROM_SHA256 update.


---

# S1 CLOSE-OUT CRITERION (added 2026-07-31, per Matt's direction)

S1 is defined CLOSED when all four of the following hold. Each is stated
with its evidence; none requires new solves.

## C1. Thermal/phase core passes all S1 criteria including full scale
MET. Enthalpy phase update (energy-exact through melt, discriminating
tests), THM-03 CFL guard + auto-substepping (closed-form verified), n=200
thermal-only march clean (t90 638.975 s, residual -3.2933e-13, no clamps,
2 substeps), analytic benchmarks passed (latent plateau, Fourier decay,
EQS parallel plate). Sections 3-4.2 of this report.

## C2. A formalized EQS validity domain, enforced in code
MET. heatr3d native EQS is certified for n <= 96 full physics and
n <= 128 EQS-only (measured ceilings; n=128 completed at 1684 s / 3.72 GB).
Above the domain the solver raises an informative MemoryError (EQS-01
guard) rather than attempting an unsafe solve. The guard is regression-
tested (test_heatr3d_s1.py).

## C3. Large-N delegation with measured cross-engine agreement
MET, via decision D1(a) (signed off by Matt 2026-07-31). Above the native
domain, large-N EQS is delegated to dolfinx. The agreement evidence
(heatr3d_d1_spike/results.json, EQS-02-corrected fields):
- dolfinx vs heatr3d, extruded circle, in-part mid-plane normalized Q_rf:
  10.8% whole-part / 2.6% interior at the n=96-matched mesh, improving
  under refinement (task2).
- dolfinx at the n=200-equivalent resolution agrees with its own fine
  reference to 1.19% all / 0.16% interior (task4), completing in 29.6 s /
  2.40 GB where the native direct solve is infeasible (~29.8 GB estimate).
The heatr3d/dolfinx residual is localized to the part-surface band and is
the documented harmonic-vs-DG0 boundary-layer discretization difference,
shrinking under refinement.

## C4. The un-closed remainder is explicitly scoped OUT of S1
The single-engine full-physics n=200 march remains impossible in native
heatr3d and is NOT claimed. The @pytest.mark.slow full-physics test stays
in the suite as the standing marker; it passes only if a native large-N
EQS path is ever built (not planned; D1 delegates instead). This is a
validity-domain statement, not a deficiency: every consumer (the Studio
badge system, S2, the solve port) operates within the certified domain or
on the delegated engine.

## Verdict

With C1-C4 met, **Gate S1 is CLOSED: PASSED WITHIN THE STATED VALIDITY
DOMAIN** (native n <= 96 full physics; delegated large-N per D1 with
measured agreement). The prior "partially passed" verdict is superseded by
this criterion. Signed off implicitly by Matt's 2026-07-31 direction to
write this criterion to complete S1; any objection reopens the gate.
