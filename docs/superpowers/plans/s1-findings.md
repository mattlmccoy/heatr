# S1 instability findings (Task 3 of the heatr3d S1 numerical-integrity plan)

Date: 2026-07-30. Interpreter: `./.venv312/bin/python` (Python 3.12.7).
Plan: `docs/superpowers/plans/2026-07-30-heatr3d-s1-numerical-integrity.md`.
Spec gate: S1 in `docs/superpowers/specs/2026-07-30-heatr3d-graduation-design.md`.
No solver file was modified for this diagnosis: `heatr3d.py` is untouched by
Tasks 2-3, and `dissertation_materials/analysis-3dfgm/heatr3d.py` is untouched.

## 1. Diagnostic output (verbatim)

`./.venv312/bin/python scripts/analysis/s1_diagnose_instability.py`

```
h=1.875 mm  dt_s=0.05  dt_CFL=7.463 s  CFL ratio=0.007 (<1 means conduction-stable)
  n=48: dt_CFL=3.317 s ratio=0.015
  n=96: dt_CFL=0.829 s ratio=0.060
  n=200: dt_CFL=0.191 s ratio=0.262
  spike x    1: raw source dT/step =    0.069 C (melt window = 10.0 C, clamp = 10.0 C)
  spike x   10: raw source dT/step =    0.692 C (melt window = 10.0 C, clamp = 10.0 C)
  spike x  100: raw source dT/step =    6.920 C (melt window = 10.0 C, clamp = 10.0 C)
  spike x  400: raw source dT/step =   27.679 C (melt window = 10.0 C, clamp = 10.0 C)
  spike x 1000: raw source dT/step =   69.198 C (melt window = 10.0 C, clamp = 10.0 C)
latent per C inside window: 9670 J/(kg C) vs cp_solid 2500.0 J/(kg C) -> apparent cp in-window ~x4.9
```

## 2. Task 2 repro measurements (spike_mult calibration)

`test_heatr3d_s1.py::test_legacy_phase_update_skips_latent_on_window_crossing`,
`spike_case(n=32, spike_mult=400.0)`, `max_time_s=120.0`, legacy
(`apparent_cp`) phase update. Result fields from the Task 1 energy audit:

```
mult=    1.0 clamp_bound=False in=7.856e+02 stored=7.856e+02 loss=2.281e-02 resid=+0.0000 phi_max=0.0000
mult=   50.0 clamp_bound=False in=8.473e+02 stored=8.429e+02 loss=2.281e-02 resid=+0.0051 phi_max=1.0000
mult=  100.0 clamp_bound=True  in=9.102e+02 stored=8.908e+02 loss=2.282e-02 resid=+0.0213 phi_max=1.0000
mult=  200.0 clamp_bound=True  in=1.036e+03 stored=8.980e+02 loss=2.282e-02 resid=+0.1333 phi_max=1.0000
mult=  400.0 clamp_bound=True  in=1.288e+03 stored=8.993e+02 loss=2.282e-02 resid=+0.3017 phi_max=1.0000
mult= 1000.0 clamp_bound=True  in=2.043e+03 stored=8.995e+02 loss=2.282e-02 resid=+0.5598 phi_max=1.0000
```

**Calibrated value: `spike_mult = 400.0`** (the plan's default already latches
the required signature: `clamp_bound is True` AND `energy_residual_frac > 0.10`;
the threshold is first crossed between mult=100 and mult=200).

Direct mechanism measurement (module-level monkeypatch of `phase_fraction` to
trace the spiked voxel; no solver edit), `spike_mult=400.0`:

```
spiked voxel index (16, 11, 13)
max single-call phi jump = 0.8000  (phi 0.0000 -> 0.8000, T 173.00 -> 183.00 C,
                                    dphi/dT at start = 0.0000 1/C)
  next call: phi 0.8000 -> 1.0000  T 183.00 -> 185.53  dphi_start=0.1000
```

The voxel sits at T=173.00 C, i.e. BELOW the window start
(`t_pc_c - dt_pc_c/2 = 175 C`), so `phase_fraction` returns `dphi/dT = 0` and
`cp_eff = cp` with **no latent term at all**. The step (clamped to
`max_dt_step_c = 10.0 C`; raw source dT = 27.68 C) lands at 183.00 C, three
quarters of the way through the window, and phi snaps 0.0 -> 0.8 having paid
zero latent heat. This is the hypothesized mechanism, observed directly.

## 3. Required conclusions

**(1) Conduction CFL is innocent, including at n=200.** With
`alpha_max = k_liquid/(rho_liquid*cp_liquid)`, the explicit 6-neighbour stability
bound `dt = h^2/(6 alpha_max)` gives `dt_CFL = 0.191 s` at n=200 versus
`dt_s = 0.05 s`: **ratio 0.262 < 1**. At the repro grid n=32 the ratio is 0.007.
The secondary hypothesis (explicit conduction instability) is ruled out over the
whole grid range of interest. Caveat on scope: this is the pure-conduction bound
using bulk liquid properties; it does not cover the apparent-cp stiffening
(which *increases* the effective heat capacity and is therefore stabilising for
conduction) nor any k boost from `heatsink_kgain`.

**(2) One source step exceeds the melt window at spike multiplier ~145.**
The raw source-only per-step rise is `dT = q*dt_s/(rho_solid*cp_solid)`:
6.920 C at x100 and 27.679 C at x400, so it crosses the `dt_pc_c = 10.0 C`
window at `mult = 10.0/0.06920 = 145`. The repro test is run at x400
(dT/step = 27.7 C, 2.8 window widths) for margin. Note `max_dt_step_c = 10.0`
equals `dt_pc_c = 10.0`: even the *clamped* step is exactly one full window
wide, so the limiter cannot prevent a window-crossing step, it only hides it.

**(3) Mechanism pinned by the Task 2 test.** *The legacy `apparent_cp` phase
update evaluates the latent term pointwise as `cp_eff = cp + latent*dphi(T_now)`;
when a voxel starts a step outside the melt window, `dphi(T_now) = 0`, so a
single Euler step of size >= `dt_pc_c` carries it across the entire window while
paying none of the latent barrier — measured here as phi 0.000 -> 0.800 in one
step at T 173 -> 183 C with `dphi/dT = 0` at the step start.* The barrier that
is skipped is large: `latent/dt_pc = 9670 J/(kg C)` against
`cp_solid = 2500 J/(kg C)`, i.e. the in-window apparent heat capacity is ~4.9x
the sensible value, so a skipped crossing under-resolves the local energy
balance by roughly a factor of five over the window.

**(4) Grid dependence.** The trigger is a per-voxel source magnitude, not a
global one. Refining the grid concentrates the EQS corner/edge field enhancement
into fewer, smaller voxels: the corner E-field is singular for a re-entrant
dielectric wedge, so peak `Q_rf ~ |E|^2` grows without bound as h -> 0 while the
voxel volume shrinks as h^3. The per-voxel source rise `dT = Q_rf*dt_s/(rho*cp)`
therefore grows monotonically with n at fixed `dt_s`, and past the point where
peak `dT/step >= dt_pc_c = 10 C` the hottest corner voxels start taking
window-crossing steps and skipping latent heat. That is why the documented
blow-up (phi 0.019 -> 0.953 in one step, residual ~1e5 x dose) appears only at
grid >= 200 in production runs and has to be forced with a synthetic spike at
n=32, and it is also why the conduction CFL ratio (0.262 at n=200, still
stable) does not explain the grid threshold.

## 4. Did the primary hypothesis survive? Partially — with one correction

**Survives:** the latent-skip mechanism is real and directly observed
(section 2). `dphi(T_now) = 0` at the step start, one step crosses the window,
phi snaps, no latent heat is paid.

**Correction (do not overclaim):** at `spike_mult=400` the *energy residual*
(+0.3017, i.e. 1287.9 J in vs 899.3 J stored) is **not** mostly latent skip.
The skipped latent for one voxel is only `rho_s*L*dV ~ 500*96700*6.59e-9 ~ 0.3 J`.
The ~389 J surplus is dominated by the spiked voxel saturating at
`temp_max_c = 600 C` (THM-02 clamp) while RF keeps depositing
`400*power_density*dV ~ 4.2 W` into it for the remaining ~80 s (~336 J), plus a
few J from the audit's deliberately first-order property model (it books stored
energy with `cp_solid` and initial densities while the solver blends cp toward
`cp_liquid` once phi > 0).

Consequences for the plan:
- The Task 2 assertion `energy_residual_frac > 0.10` pins "this run went
  non-physical", which is the right regression tripwire, but it is **not** a
  measurement of skipped latent heat. The phi-jump trace in section 2 is what
  pins the mechanism. Both are recorded in the test docstring.
- Task 4's `test_enthalpy_update_conserves_energy_on_window_crossing` asserts
  `abs(residual) < 0.05` on the same spike case. **Expect it to fail on the
  temp-clamp saturation alone**, because the enthalpy inversion fixes the latent
  accounting but does nothing about a voxel pinned at `temp_max_c = 600 C` that
  still receives 4.2 W. Before implementing Task 4, either (a) reduce the spike
  case's `max_time_s` / spike so the voxel never saturates at `temp_max_c`, or
  (b) split the assertion: energy conservation checked on a non-saturating case,
  and the latent-payment checked via the phi-jump trace. This is a plan-level
  decision for Matt, not something to silently weaken at implementation time.

## 5. Reproduction commands

```bash
cd geo-prewarp
./.venv312/bin/python scripts/analysis/s1_diagnose_instability.py
./.venv312/bin/python -m pytest test_heatr3d_s1.py -v   # 2 passed in 59.6 s
```

---

# Tasks 4b-7 findings (2026-07-30)

Added after the Task 4 fix landed. Same interpreter, same repo root.
`dissertation_materials/analysis-3dfgm/heatr3d.py` still untouched.

## 6. Audit v2: the standing gate is now meaningful at melt (Task 4b)

The Task-1 audit booked the whole run with initial-state solid properties, so
on a HEALTHY, limiter-free molten run (`bulk_crossing_case`, mult=130, n=32,
`phi_target=2.0`) it drifted to **+0.3795 for BOTH schemes** at t=3.0 s --
useless as a gate exactly where the instability lives. Audit v2 banks sensible
and latent energy per step with the same `rho`, `cp`, `rho_s_eff` maps the
update used:

| t_end | scheme      | in (J)   | stored (J) | residual   |
|-------|-------------|----------|------------|------------|
| 1.0 s | apparent_cp |  851.044 |    919.320 | **-0.0802**|
| 1.0 s | enthalpy    |  851.044 |    851.044 |  +2e-16    |
| 3.0 s | apparent_cp | 2553.131 |   2553.246 |  -4.5e-05  |
| 3.0 s | enthalpy    | 2553.131 |   2553.131 |  -1.5e-16  |

The enthalpy arm is **exact by construction**: its update deposits `num*dt`
into the same `H(T)` the audit books (`rho*cp` sensible slope + `rho_s_eff*L`
ramped across the window), and `phase_fraction`'s phi IS the enthalpy ramp
fraction. Residual ~1e-16 is the honest reading, not a tuned tolerance.

Two things the upgrade deliberately does NOT do:

1. **It does not mask the legacy defect.** Mid-window (t=1.0 s, phi_mean 0.669)
   the legacy arm still books -0.0802 against the enthalpy arm's +2e-16.
2. **It does not hide clamp-destroyed energy.** DEVIATION from the plan snippet
   (`e_stored_acc += (rho*cp*dT).sum()*dV`): the implementation banks the
   ACTUAL applied change `T - T_prev`, i.e. after the temp_min/temp_max clamp.
   Banking the pre-clamp `dT` would credit energy into a voxel pinned at
   `temp_max_c` and erase the saturation surplus that the Task-2 legacy pin
   test asserts (`resid > 0.10`).

New numerical finding: the legacy residual **cancels to ~1e-4 once the part is
FULLY molten**. Its skipped latent is offset by paying the resolved part of the
latent at the blended liquid density (`rho` -> 1010) instead of the enthalpy
basis `rho_s_eff` = 473.5. So a fully-molten end state is NOT a discriminating
test of the phase scheme; the window-crossing (partially molten) state is.

## 7. Analytic benchmark 1: adiabatic uniform heating with latent plateau (Task 5)

Whole domain = part, uniform q, `conv_h=0`, `phase_update="enthalpy"`, n=16.

Plan sizing correction: q=2.0e5 W/m^3 for 200 s deposits 4.0e7 J/m^3 and lands
at **56.8 C**, entirely below the melt window -- it would never touch the latent
plateau it is named for. Used instead q=2.0e6 W/m^3 for 100.0 s = 2.0e8 J/m^3,
landing MID-PLATEAU at T_exact = **178.482866 C** (phi ~ 0.35).

| arm | T_num (C) | error | notes |
|-----|-----------|-------|-------|
| default property blending | 178.312174 | **1.71e-01 C** | vs 0.5 C bound |
| constant properties (escape hatch) | 178.482866 | **3.13e-13 C** | vs 0.05 C bound |

Both arms: `clamp_bound False`, `T_final.std() = 0`, energy residual ~1e-14.
The 0.171 C gap is not a wiring error: once phi > 0 the solver blends
rho -> rho_liquid and cp -> cp_liquid while the closed-form H(T) assumes the
fixed solid slope. With `cp_liquid=cp_solid, k_liquid=k_solid, rho_liquid=rho_s`
(the plan's step-3 escape hatch = exactly the analytic solution's own
assumptions) the agreement is machine precision. Both arms are asserted.

## 8. Analytic benchmark 2: Fourier-mode conduction decay (Task 6)

No source, no convection, part-free (all-powder) domain, lowest Neumann cosine
mode of amplitude 5 C, n=24, t_end=400 s (8000 explicit steps),
alpha = k_powder/(rho_powder*cp_powder) = 3.7504e-07 m^2/s.

| reference | form | rel. error |
|-----------|------|-----------|
| continuous | `5 exp(-alpha k^2 t)` | -1.566e-03 |
| semi-discrete | `amp0 exp(-lambda_d t)` | -1.054e-05 |
| fully discrete | `amp0 (1 - lambda_d dt)^nsteps` | **+2.741e-13** |

with `lambda_d = alpha (2/h^2)(1 - cos(k h))` (the plan's documented discrete
rate) and `amp0 = 5 cos(pi/2n)`. The `amp0` factor is the second finding here:
the cell-centred grid never samples the mode's true peak, so the observed
`(max-min)/2` starts at 4.98929, not 5 (-2.14e-03). That is why the raw
discrete-rate comparison looks WORSE than the continuous one -- the sampling
deficit and the discrete-rate excess partially cancel in the continuous form.
Accounting for both, the solver reproduces the mode to 2.7e-13: discrete
Laplacian eigenvalue, zero-flux wall treatment, harmonic face averaging on
uniform k, property maps and the explicit integrator are all exactly as
intended. Mean temperature is conserved to <1e-9 C.

Spatial convergence (documented, not asserted): continuous-reference error
-1.566e-03 at n=24 and -3.992e-04 at n=48, ratio **3.92 ~ 4** = second order in
h, as expected of the scheme.

Solver change required: `run()` gained a `T0_override` hook (default None,
bit-for-bit inert) AND an empty-part-mask guard. A part-free domain previously
CRASHED in `T_max_c=float(T_phi90[part].max())` (zero-size reduction). The
guard is inert for any non-empty mask; part-free runs now report
`mean_phi=0.0` and `sigma_T = T_max_c = nan`.

## 9. Analytic benchmark 3: EQS parallel-plate field (Task 7)

Characterization PASSED -- no S1 defect found in the EQS solve. Uniform virgin
bed, default Params.

| n | branch | max\|V - linear\| | transverse std | max\|Im V\| |
|---|--------|------------------|----------------|------------|
| 24 | direct spsolve (N=13824) | 1.25e-11 V (1.4e-14 rel) | 9.46e-13 V | 8.3e-17 V |
| 40 | ILU-BiCGSTAB (N=64000) | 3.14e-06 V (3.7e-09 rel) | 4.84e-06 V | 2.0e-16 V |

Both branches are asserted because every production grid n >= 37 takes the
iterative one; its error floor is set by the solver's own `rtol=1e-8`.

**FINDING (documented, NOT corrected): the effective plate gap is (n-1)h, not
L.** The Dirichlet rows sit at the CELL CENTRES of the first/last y layers, so
the measured uniform field is `|E_y| = v_lo/((n-1)h) = 14956.521739 V/m` at
n=24 (matched to 13 digits, ptp 2.6e-09), which exceeds the nominal
`v_lo/L = 14333.33 V/m` by `L/(L-h)` = **4.3% at n=24** (0.5% at n=200). It is
a grid-dependent bias on the absolute field magnitude, so it matters to any
quoted V/m and to grid-convergence claims about E. It does NOT propagate to
Qrf on a uniform medium: it is a uniform scale factor and `compute_qrf_3d`
renormalizes Q to a fixed total absorbed power, which divides it out. Whether
to move the electrodes onto the domain faces (a half-cell BC change) is an S2
discretization decision, not an S1 fix.

## 10. Legacy inertness re-verification (Tasks 4b-7)

Default `Params()` (`phase_update="apparent_cp"`), n=32 sphere, 30 s, HEAD
`84a6a8d` module vs the working tree after all four tasks, two isolated module
loads in one process:

```
T_phi90    max|diff| = 0.000e+00  array_equal=True
Qrf        max|diff| = 0.000e+00  array_equal=True
phi_final  max|diff| = 0.000e+00  array_equal=True
sigma_T    ref=21.19205274948017 cur=21.19205274948017 diff=0.000e+00
AUDIT ref: in=196.3947 stored=196.3947 resid=+0.000000
AUDIT cur: in=196.3947 stored=196.3947 resid=+0.000000
```

Gate was `< 1e-12` RELATIVE; the observed difference was exactly 0 in 6/6 of
the final sample.

**Correction to a previously documented claim.** The Task-4 note in
`test_legacy_default_is_unchanged`'s docstring says "the EQS solve is exactly
reproducible (6/6 np.array_equal on V), so the drift is inside the thermal
loop". That is wrong. Running this ref-vs-cur comparison repeatedly, 2 of 5
runs showed `Qrf max|diff| = 9.313e-10` (absolute) alongside
`T_phi90 max|diff| = 7.105e-15`, i.e. the EQS-derived field drifts too.

A CONTROL settles the attribution: comparing the HEAD module against ITSELF
(same source text, two separate module loads in one process) reproduces the
identical signature -- `T_phi90 7.105e-15, Qrf 9.313e-10` in 1 of 4 runs. So
this is inherent run-to-run non-determinism (BLAS/SuperLU threading), NOT an
effect of the S1 diff.

Two practical consequences:
- Quote the drift in RELATIVE terms: `Qrf` is O(1e7) W/m^3, so 9.3e-10
  absolute is **8.5e-17 relative** -- one or two ulp, not a physics change.
  An absolute 1e-12 gate on `Qrf` is a badly scaled gate and will flap;
  use relative.
- The drift is bimodal (exactly 0 or exactly the same 9.313e-10), which is the
  signature of two alternative reduction orders being selected at runtime,
  not of accumulating chaos.

## 12. Unrelated pre-existing breakage seen while running the suites

`test_heatr3d_outputs.py` and `test_fgm_pipeline.py` collect but produce 25
errors, ALL of them `fixture 'run_dir' not found`. There is no `conftest.py`
anywhere in the repo (`git ls-files | grep conftest` is empty), so the fixture
those files expect was never committed. This is independent of heatr3d.py
(a collection-time failure) and predates this work; 10 tests in those files
still pass. Flagged, not fixed here -- it is out of the S1 plan's scope.

## 11. Reproduction commands (Tasks 4b-7)

```bash
cd geo-prewarp
./.venv312/bin/python -m pytest test_heatr3d_s1.py -v   # 9 passed in 115.3 s
```

---

# Task 8 findings (2026-07-30): the full-scale regression does NOT pass

Same interpreter, same repo root. `dissertation_materials/analysis-3dfgm/heatr3d.py`
still untouched. Machine: Apple Silicon, 12 cores, 34.36 GB RAM.

## 13. Task 8 landed the wiring; the n=200 run itself is blocked twice over

> **STATUS RECONCILED 2026-07-31 (post-S1b).** Section 13 was written when both
> blockers were raw and outstanding. Neither is raw any more; every measurement
> below still stands, only the STATUS changed. Amendments are inline in 13.2,
> 13.3 and 13.6, and the new evidence is in 13.8.
>
> | blocker | status now | where |
> |---|---|---|
> | **THM-03** (explicit conduction CFL, n > 178.9) | **FIXED** -- CFL guard + auto-substepping, commit 06bd87d; closed-form-verified; default-inert for **n <= 169**; evidenced at n=200 | 13.3 amendment, 13.8 |
> | **EQS-01** (n=200 EQS SIGSEGV) | **CONVERTED, NOT SOLVED** -- informative MemoryError instead of a segfault, commit 22122da; measured ceilings n=96 full physics / n=128 EQS-only; scalable path **deferred to D1** | 13.2 amendment |
>
> So the "blocked twice over" of this heading is now "blocked once over", and
> the remaining block is a solver-capability item owned by D1, not an unfixed
> defect.

Landed and verified:
- `test_full_scale_n200_melt_onset_clean_with_enthalpy`, `@pytest.mark.slow`,
  assertions exactly as the plan specifies (`reached is True`,
  `|residual_frac| < 0.05`, `clamp_bound is False`). NOT weakened.
- `pytest.ini` (the repo had NO pytest config before: no pytest.ini, setup.cfg,
  tox.ini or pyproject.toml anywhere). Registers the `slow` marker and
  `addopts = -m "not slow"`. It sets no testpaths / no --ignore, so the 25
  pre-existing `fixture 'run_dir' not found` collection errors of section 12
  stay visible; only the slow marker is filtered.
  Default suite after the change: `9 passed, 1 deselected in 304.37 s`.
- The always-on standing gate line in `run()`, printed on every solve:
  `  [s1-energy] in=... J stored=... J loss=... J residual_frac=+... [CLAMP-BOUND]`.

### 13.1 Runtime estimate (measured before launching, as instructed)

Thermal loop at n=200 with an injected Qrf (EQS bypassed), clean machine:
10 steps = 4.1 s, 40 steps = 15.2 s -> **0.370 s/step**, peak RSS 2.35 GB.

| max_time_s | steps | loop-only estimate |
|-----------|-------|--------------------|
| 900 s (the test) | 18 000 | **1.85 h** |
| 1500 s (run() default) | 30 000 | 3.08 h |

EQS solve cost, measured standalone (uniform bed, default Params):

| n | N = n^3 | wall | peak RSS | branch |
|---|---------|------|----------|--------|
| 64 | 2.62e5 | 97.7 s | 0.81 GB | ILU-BiCGSTAB |
| 96 | 8.85e5 | 322.1 s | 2.21 GB | ILU-BiCGSTAB |
| 128 | 2.10e6 | did not finish in 30 min | - | ILU-BiCGSTAB |
| 200 | 8.00e6 | **crashes (see 13.2)** | - | ILU fails -> direct |

So the loop alone is under the 6 h stop line, but the run cannot start at all.

### 13.2 BLOCKER 1 (new finding, EQS-01): the n=200 EQS solve segfaults

`./.venv312/bin/python -m pytest test_heatr3d_s1.py -m slow -v`

```
test_heatr3d_s1.py::test_full_scale_n200_melt_onset_clean_with_enthalpy
Fatal Python error: Segmentation fault
Current thread 0x00000001efea5d80 (most recent call first):
  File ".../scipy/sparse/linalg/_dsolve/linsolve.py", line 293 in spsolve
  File ".../geo-prewarp/heatr3d.py", line 265 in solve_eqs_3d
  File ".../geo-prewarp/heatr3d.py", line 597 in run
  File ".../geo-prewarp/test_heatr3d_s1.py", line 382 in
        test_full_scale_n200_melt_onset_clean_with_enthalpy
PYTEST_EXIT=139        # SIGSEGV
```
Wall clock to the crash: **17 s**. Reproduced 3/3 (twice standalone, once under
pytest); the two standalone attempts printed
`malloc fails for local dworkptr[]. ... zgstrf info 2082443264` and were then
SIGKILLed by the OS.

Mechanism: at N = 8.0e6 complex unknowns `spla.spilu(A, drop_tol=1e-4,
fill_factor=12)` cannot allocate; `solve_eqs_3d` catches that with a bare
`except Exception: V = None` and falls through to `spla.spsolve(A, b)` -- a
DIRECT complex LU of an 8-million-unknown 3-D Laplacian, which is hopeless at
any memory size and here dies inside SuperLU. Two defects, not one:
1. the direct fallback is unreachable-by-design at large N but is still armed;
2. the failure is a segfault, not an exception, so a caller cannot handle it and
   a batch campaign loses the whole process.
This is a genuine S1 (numerical integrity) defect of the EQS core and it is NOT
addressed by anything in Tasks 1-7. It also caps the practical grid: the largest
grid at which the full pipeline runs today is n<=96 (section 13.5).

> **STATUS AMENDED 2026-07-31: EQS-01 defect (2) is FIXED; defect (1) is closed
> by design; the underlying CAPABILITY gap is DEFERRED TO D1.** Commit 22122da:
> the direct fallback is armed only below `EQS_DIRECT_MAX_UNKNOWNS = 2e6`; above
> it `solve_eqs_3d` raises an informative `MemoryError` naming N, the grid, the
> order-of-magnitude LU memory estimate (**~29.8 GB at n=200**), the underlying
> iterative-path failure, and the measured supported grids. Small systems keep
> the existing direct path unchanged (control test:
> `test_eqs_small_system_still_uses_direct_fallback_when_ilu_fails`, N=512 still
> solved to the parallel-plate field).
>
> What that does and does not buy:
> - **Gone:** the process-killing SIGSEGV. A batch campaign now sees a catchable
>   exception that says what to do, instead of losing the interpreter.
> - **Not gone:** the size ceiling itself. Measured, and unchanged:
>   **n = 96 for the full pipeline (EQS + thermal), n = 128 for EQS-only.**
>   n = 200 EQS is still infeasible on this machine.
> - **Owner of the remaining work: D1 (dolfinx FEM spike).** A scalable large-N
>   EQS path (matrix-free / multigrid / FEM with a proper preconditioner) is
>   explicitly out of S1b scope and deferred there. EQS-01 is therefore the SOLE
>   remaining S1 blocker, and it is a capability item, not an unfixed defect.
>
> This is why the n=200 evidence run of 13.8 uses `qrf_override`: it is the only
> way to exercise the thermal/phase core at that grid today, and it is
> explicitly NOT a full-physics n=200 pass.

### 13.3 BLOCKER 2 (new finding, THM-03): the n>=200 trigger is a conduction
CFL violation in the POWDER -- section 3 conclusion (1) is CORRECTED

Section 3 concluded "conduction CFL is innocent, including at n=200 (ratio
0.262)". That used `alpha_max = k_liquid/(rho_liquid*cp_liquid) = 7.85e-8`.
**The liquid is not the fastest medium in the domain; the powder bed is**, and
the bed is ~98 % of the voxels:

| medium | k | rho | cp | alpha [m^2/s] |
|--------|---|-----|----|---------------|
| powder | 0.197 | 490.0 | 1072.0 | **3.7504e-07** |
| solid | 0.10 | 460.0 | 2500.0 | 8.6957e-08 |
| liquid | 0.26 | 1010.0 | 3279.0 | 7.8507e-08 |

With the correct `alpha_max = 3.7504e-07` (4.78x the value used) the explicit
6-neighbour bound `dt < h^2/(6 alpha)` at L = 0.060 m, dt_s = 0.05 s gives:

| n | h | dt_CFL | ratio dt_s/dt_CFL |
|---|---|--------|-------------------|
| 32 | 1.8750 mm | 1.5623 s | 0.032 |
| 96 | 0.6250 mm | 0.1736 s | 0.288 |
| 128 | 0.4688 mm | 0.0976 s | 0.512 |
| 160 | 0.3750 mm | 0.0625 s | 0.800 |
| **178.9** | 0.3354 mm | 0.0500 s | **1.000 (threshold)** |
| 200 | 0.3000 mm | 0.0400 s | **1.250 (UNSTABLE)** |

The crossing grid is **n = 178.9**, which is exactly the documented
"blow-up at grid >= 200" trigger.

Direct experiment (`scripts/analysis/s1_cfl_powder_mode.py`): all-powder domain,
no part, no source, `conv_h=0`, initial condition = a 1e-3 C 3-D checkerboard
(the fastest-growing discrete mode), 60 steps, via the Task-6 `T0_override`
hook. Predicted per-step amplification `|1 - 12 alpha dt/h^2|` vs measured
`(amp_60/amp_0)^(1/60)`:

| n | CFL ratio | predicted | measured | clamp_bound |
|---|-----------|-----------|----------|-------------|
| 128 | 0.512 | 0.0241 | 0.9012 (decays) | False |
| 160 | 0.800 | 0.6002 | 0.8910 (decays) | False |
| 176 | 0.968 | 0.9362 | **0.9362** (decays) | False |
| 184 | 1.058 | 1.1162 | **1.1162** (GROWS) | False |
| 200 | 1.250 | 1.5003 | 1.1558 (GROWS) | **True** |

At n=176 and n=184 the measured growth matches the closed-form amplification to
4 decimals and the sign of the instability flips exactly across the predicted
threshold. (At n=128/160 the measured rate is set by the slowest surviving mode
after the checkerboard has already decayed away, not by the checkerboard rate --
the checkerboard is gone within a few steps there; the qualitative reading,
decay vs growth, is what matters.) At n=200 the measured rate is BELOW the
prediction only because the +-10 C THM-01 limiter binds and truncates the true
growth -- the clamp is hiding the divergence, the same pattern the S1 spec
refuses to accept as a pass.

Consequences:
- `assert res.clamp_bound is False` at n=200 CANNOT hold with dt_s = 0.05 s no
  matter what the phase scheme does. The enthalpy fix (Task 4) is necessary and
  correct but it is not sufficient for the documented trigger.
- The latent-skip mechanism of sections 2-4 remains real and independently
  pinned (phi 0.000 -> 0.800 in one step with dphi/dT = 0), but it is NOT the
  n>=200 trigger. Both mechanisms are melt-onset-adjacent; only THM-03 is
  grid-threshold-shaped, and its threshold (178.9) matches the observation.
- Fix options for S2/next-S1-pass, in increasing order of cost: (a) enforce
  `dt_s <= 0.9 h^2/(6 alpha_max)` as a hard, ASSERTED stability criterion in
  `run()` (spec-legal: "a fix OR a principled stability criterion"), which at
  n=200 means dt_s <= 0.036 s and ~1.4x more steps; (b) conduction substepping;
  (c) implicit / IMEX conduction. Note the audit already reports these runs as
  energy-clean-ish, so the residual gate alone does NOT catch THM-03 -- the
  clamp_bound flag does.

> **STATUS AMENDED 2026-07-31: THM-03 is FIXED** (commit 06bd87d), by fix
> options (a)+(b) of the list immediately above. Everything measured in 13.3
> stands unchanged -- it is the diagnosis the fix is built on. What changed:
>
> - `run()` computes `dt_stable = h^2/(6 alpha_max)` over all materials present
>   (`alpha_max_thermal`, `dt_stable_thermal`, `cfl_substeps`, `CFL_SAFETY=0.9`)
>   and, with the new `Params.enforce_cfl` (default **True**), SUBSTEPS the
>   thermal update `n_sub = ceil(dt_s / (0.9 dt_stable))` times per `dt_s`. Qrf
>   is held fixed across the substeps of a step (exact: Qrf is computed once per
>   run). `Result` gains `n_substeps_used` and `cfl_violated`.
>   `enforce_cfl=False` keeps the legacy single step, latches `cfl_violated` and
>   logs the closed-form growth rate.
> - Closed-form verification at CFL ratio 1.25 (the n=200 value, reproduced
>   cheaply at n=32 by raising dt_s): legacy grows the checkerboard at
>   **1.5000/step vs the predicted 1.5003**, to 3.33 C in 20 steps; enforced
>   `n_sub = 2` DECAYS it to 2.6e-06 C, domain mean conserved to 0.0.
> - The standing energy gate stays exact under substepping (the audit banks per
>   substep with `dt_sub`): RED with the audit reverted to `p.dt_s` gives
>   residual **+0.5000**; GREEN gives **-1.09e-15**.
> - **CORRECTION to the no-change ceiling.** The default-inert ceiling is
>   **n <= 169** at dt_s = 0.05 s, L = 0.060 m -- NOT the n <= 178 that the
>   marginal-stability threshold (178.9) in the table above would suggest. The
>   0.9 safety factor pulls the substepping trigger down to n = 170. Recorded
>   explicitly because earlier write-ups of this finding quoted n <= 178 as the
>   ceiling. Verified out of band against the pre-change module (22122da), n=32
>   sphere, fixed Qrf, both phase schemes x densify on/off: **4/4 BIT-IDENTICAL**
>   (max|dT| = 0.0 on T_final / T_phi90 / phi_final / rho_final; identical
>   sigma_T, energy_in_j, t90, phi_hist). Every historical heatr3d number is
>   preserved.
> - Regression tests: `test_cfl_stability_criterion_and_substep_count`
>   (n=32/96/160/169 -> 1; n=170/184/200 -> 2),
>   `test_cfl_substepping_bounds_the_checkerboard_mode_legacy_grows`,
>   `test_energy_gate_stays_exact_under_substepping`,
>   `test_cfl_guard_is_inert_at_the_default_working_grid`.
> - **Full-scale evidence at the trigger grid: section 13.8** (n=200, 12780
>   steps, no clamp bound, residual -3.29e-13, melt onset reached).

### 13.4 `max_time_s=900` is marginal for phi_target=0.90, and grid dependent

Measured at n=32, default Params, phi_target never triggered
(`phi_target=2.0` sweep), 18 000 steps:

```
n=32 apparent_cp: reached=False final_phi=0.8385  phi>=0.5 at 529.7 s, phi>=0.8 at 829.8 s
n=32 enthalpy   : reached=False final_phi=0.8787  phi>=0.5 at 493.4 s, phi>=0.8 at 767.7 s,
                                                  phi>=0.85 at 854.3 s
  [s1-energy] in=5891.8 J stored=5635.7 J loss=126.7 J residual_frac=+0.0220   (apparent_cp)
  [s1-energy] in=5891.8 J stored=5763.9 J loss=127.9 J residual_frac=+0.0000   (enthalpy, 1.86e-14)
```
At n=32, then, `assert res.reached is True` would fail for a pure run-length
reason (`run()`'s own default is `max_time_s=1500.0`, not 900). But the melt
time is grid dependent and moves the RIGHT way: the n=96 full-physics run
(13.5) reaches phi_target at **t90 = 802.6 s**, comfortably inside the 900 s
window. So 900 s is probably adequate at n=200 and this is a marginal-margin
note, not a fourth blocker. Nothing changed in the test.

Incidental value of the n=32 pair above: identical RF input (5891.8 J in both
arms), legacy books +2.2e-02 residual, enthalpy +1.9e-14 -- the S1 fix visible
on a plain benign production case, not only on the engineered spike.

### 13.5 The largest full-physics run that completes today (n=96): CLEAN

Full pipeline (EQS + thermal, no qrf_override), n=96 sphere d=20 mm, default
Params with `phase_update="enthalpy"`, `max_time_s=1500`, phi_target=0.90,
`scripts` copy at `/private/tmp/.../full_n96.py`:

```
  [s1-energy] in=5381.5 J stored=5289.0 J loss=92.5 J residual_frac=+0.0000
FULL n=96 enthalpy: wall=1049s reached=True t90=802.6 sigma_T=19.278 Tmax=251.2
                    clamp=False resid=+1.3320e-13 in=5381.5 stored=5289.0
                    loss=92.5 final_phi=0.9000
```

This is the best available substitute for the blocked n=200 regression: a real,
production-configuration, full-physics melt-onset run at 8.85e5 voxels with the
S1 fix engaged. **Energy residual 1.33e-13, no clamp bound, melt onset reached.**
CFL ratio at n=96 is 0.288 (well below 1), so THM-03 is dormant there, exactly
as predicted. Wall clock 1049 s (322 s of it the EQS solve).

### 13.6 Verdict

> **VERDICT AMENDED 2026-07-31 (post-S1b). The 2026-07-30 verdict below is kept
> verbatim; this supersedes it.**
>
> **S1 PARTIALLY PASSED -- the thermal/phase core passes all criteria including
> full-scale; overall S1 remains open solely on the large-N EQS path (D1).**
>
> Of the two blockers listed below:
> 1. **EQS-01: converted, not solved.** No more segfault -- an informative
>    MemoryError with the measured ceilings (n=96 full physics, n=128 EQS-only).
>    The scalable large-N EQS path is **deferred to D1**. This is the sole
>    remaining S1 blocker.
> 2. **THM-03: FIXED** (CFL guard + auto-substepping, closed-form-verified,
>    default-inert for n <= 169) **and evidenced at full scale** by the n=200
>    thermal march of 13.8: reached=True, n_sub=2, clamp_bound=False,
>    cfl_violated=False, residual -3.2933e-13, t90=638.975 s, sigma_T=22.843,
>    wall 2:51:43.
>
> On the spec's "avoidance by grid cap is explicitly not a pass": the thermal
> march is NOT capped -- it runs AT n=200, the documented trigger grid. The EQS
> solve IS still capped (n<=96 full physics), and that is exactly what keeps S1
> from a full pass.

**[ORIGINAL 2026-07-30 verdict, superseded]**
**Gate S1 is NOT passed.** Delivered: the mechanism diagnosis (sections 1-4,
corrected by 13.3), the enthalpy fix with its RED test, the audit v2 standing
gate wired into every solve, three analytic benchmarks (sections 7-9), and the
full-scale regression test itself. Outstanding, both newly measured here and
both required by the spec's own wording ("Avoidance by grid cap is explicitly
not a pass"):
1. EQS-01: n=200 EQS segfault (13.2).
2. THM-03: explicit conduction CFL violation for n > 178.9 (13.3).

### 13.7 Reproduction commands (Task 8)

```bash
cd geo-prewarp
./.venv312/bin/python -m pytest test_heatr3d_s1.py -v        # 9 passed, 1 deselected
./.venv312/bin/python -m pytest test_heatr3d_s1.py -m slow -v # SIGSEGV in 17 s (13.2)
./.venv312/bin/python scripts/analysis/s1_cfl_powder_mode.py  # the 13.3 table
```

### 13.8 THM-03 full-scale evidence (2026-07-31): the n=200 THERMAL march runs clean

Test: `test_n200_thermal_march_clean_with_qrf_override`, `@pytest.mark.slow`.
Grid(n=200, L=0.060) = 8.0e6 voxels, h = 0.300 mm; sphere d = 0.020 m;
`phase_update="enthalpy"`; `enforce_cfl` at its default True; RF drive =
`qrf_override` uniform at `p.power_density_w_per_m3 = 1591549.43 W/m^3` over the
part; `run()` defaults for the stop (`max_time_s=1500`, `phi_target=0.90`).

**SCOPE (stated up front, also in the test docstring): this is THERMAL / PHASE
evidence only.** The uniform `qrf_override` deliberately BYPASSES the EQS solve,
which at n=200 remains INFEASIBLE (EQS-01, 13.2 -- N = 8.0e6 complex unknowns,
~29.8 GB direct LU; `solve_eqs_3d` now raises MemoryError instead of
segfaulting, commit 22122da). The scalable large-N EQS path is deferred to the
**D1 dolfinx spike**. **This run does NOT constitute a full-physics n=200 pass**,
and `test_full_scale_n200_melt_onset_clean_with_enthalpy` stays blocked on
exactly that.

**Runtime estimate made BEFORE launching** (20-step probe at n=200, n_sub=2,
this machine): 5 steps 4.25 s, 20 steps 15.16 s -> **0.758 s/step**, RSS
2.27 GB. Under the uniform `qrf_override` the melt-onset time is nearly
grid-independent -- t90 = 645.55 s at n=32, 651.15 s at n=64, 639.85 s at n=96
(same config, wall 23.1 / 167.0 / 574.5 s) -- so ~640 s / 0.05 s = ~12 800 steps
-> **~2.70 h estimated**, under the 3 h stop line, so the run was launched.

**Actual: PASSED in 10 303.51 s = 2:51:43** (2.86 h; the realised rate was
0.806 s/step). Verbatim tail:

```
  t= 630.0s  Tmax= 259.9  phi=0.891  rho=0.550
  [s1-energy] in=4257.3 J stored=4219.8 J loss=37.5 J residual_frac=-0.0000

N200-THERMAL: reached=True nsub=2 t90=638.975 sigma_T=22.843 Tmax=260.6
              clamp=False cfl_violated=False resid=-3.2933e-13 phi_mean=0.9000
PASSED
================ 1 passed, 16 deselected in 10303.51s (2:51:43) ================
```

Assertion values:

| assertion | value | verdict |
|---|---|---|
| `res.reached is True` | True (t90 = **638.975 s**, sub-step resolved) | PASS |
| `res.n_substeps_used == 2` | 2 (dt_sub = 0.025 s; CFL ratio 1.250 -> 0.625) | PASS |
| `res.clamp_bound is False` | False | PASS |
| `res.cfl_violated is False` | False | PASS |
| `abs(res.energy_residual_frac) < 0.05` | **-3.2933e-13** | PASS (11 orders of margin) |

Other measured values: sigma_T = **22.843 C**, T_max = **260.6 C**,
final mean phi = 0.9000, energy in = 4257.3 J, stored = 4219.8 J, loss = 37.5 J.
Wall-clock start-to-finish 2:51:43; peak RSS ~2.3 GB.

What this closes and what it does not:
- **Closes** the THM-03 half of the S1 "the previously blowing-up case runs
  clean" criterion. This is exactly the grid where the legacy explicit
  conduction update is CFL-unstable (ratio 1.250, checkerboard amplification
  1.5003/step, 13.3) and where the old code could only stay bounded by letting
  the +-10 C THM-01 clamp truncate a divergence. With the guard on: n_sub = 2,
  **no clamp bound anywhere in 12 780 steps**, melt onset reached, and the
  standing energy gate exact to 3.3e-13. The fix is not a grid cap -- the run is
  AT the previously failing grid.
- **Does not close** the EQS half. No EQS solve was performed. Nothing here says
  anything about the accuracy or feasibility of the n=200 field solve; that is
  EQS-01 -> D1.
- Consistency cross-check: sigma_T = 22.843 at n=200 vs 22.950 / 22.963 / 23.010
  at n=96 / 64 / 32 on the identical uniform-drive configuration (a 0.7 % spread
  over a 6x grid refinement), and t90 638.975 vs 639.85 / 651.15 / 645.55 s. The
  substepped n=200 march sits on the same trend as the unsubstepped coarse runs,
  i.e. the guard did not distort the solution. (These are uniform-Qrf numbers
  and are NOT comparable to the full-physics n=96 numbers of 13.5, where the EQS
  field concentrates the drive: there sigma_T = 19.278 and t90 = 802.6 s.)

Reproduce (2.9 h):

```bash
cd geo-prewarp
./.venv312/bin/python -m pytest test_heatr3d_s1.py -m slow \
    -k n200_thermal_march -v -s
```
