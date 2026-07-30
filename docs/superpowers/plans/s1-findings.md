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
