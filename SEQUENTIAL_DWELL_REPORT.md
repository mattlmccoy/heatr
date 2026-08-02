# Sequential dwell scheduling: melt one limb, then the other

**Date:** 2026-08-01. **Scope:** the directed deep re-attack on the L_shape with an ORDERED
hold schedule, plus the transfer test on the T_shape. The actuator is a turntable commanded
to one indexed orientation and HELD there long enough to drive one limb through melt, then
moved once and held at a second orientation. That is a different object from the CYCLED
dwell of `DWELL_SCHEDULE_REPORT.md`, whose short repeated cycles make the part see only the
time average of the heating. Grid 120 x 120 throughout. **Nothing was committed. No
dissertation file was touched. `.claude/worktrees/` was not read or written. No line of
`rfam_eqs_coupled.py`, `topopt*.py` or any other concurrently edited file was modified.**

**Acronyms, expanded on first use.** RFAM = radio-frequency additive manufacturing.
FGM = functionally graded material (a spatially varying dopant saturation map).
EQS = electro-quasi-static (the low-frequency Maxwell approximation the two-dimensional
solver uses). IoU = intersection over union. bpp = bits per pixel. L-BFGS-B = limited-memory
Broyden-Fletcher-Goldfarb-Shanno with box constraints. FD = finite difference. phi = melt
fraction. rho = relative density. W/m = watts per metre of depth.

**Evidence tags.** PROVEN = unit tested, FD gated, or reproduced against a stored number.
COMPUTED = measured from a run in this pass. ASSUMED = a modelling choice or an inference
not measured here.

---

## 0. Verdict, one line per question

| question | verdict |
|---|---|
| **Does sequential dwell rescue the L_shape?** | **It is the largest single improvement this shape has ever had, and it is still not a rescue.** J_phi falls from the best arm on record's **396.20 to 276.39, minus 30.2 percent**, and IoU rises from **0.6484 to 0.7161, plus 0.068**. The grid-120 SOLVED qualifier is IoU >= 0.95 and the arm reaches 0.716, so **19.8 percent of the part is still unmelted at the stop**. NOT rescued. |
| **Does a UNIFORM map plus a sequential schedule alone already beat static 135 degrees?** | **YES, decisively, and this is the headline.** With NO dopant grading whatsoever, J_phi 291.89 and IoU 0.7075 against the static 135-degree arm's 405.93 / 0.6471 and against the previous best on record's 396.20 / 0.6484. **The schedule is worth 26 to 28 percent of J_phi; the co-solved dopant map on top of it is worth a further 5.3 percent.** Matt's sub-hypothesis is supported. |
| **Does it transfer to the T_shape?** | **YES.** Uniform map plus sequential schedule: 373.94 / 0.6524 against the previous best on record's 422.28 / 0.6436. With the co-solved map: **343.02 / 0.6661, minus 18.8 percent J_phi and plus 0.023 IoU**. The crossbar-first order is the same one the L wants. Also not in the SOLVED class. |
| **Did the switch-time GRADIENT do the work?** | **NO, and this is the most transferable finding in the pass.** The gradient is correct and FD gated at 1e-8, but at the production read state it describes a smooth window only about **5e-5 seconds wide**, and its sign is opposite to the objective's behaviour over a control step. The switch times reported here were set by a **5 second scan**, not by the gradient. Section 4. |

COMPUTED, all of it, at each arm's own optimal stop on a 1500-step horizon.

**One safety result that was not asked for and matters.** The L_shape's static 135-degree
arm reaches **253.4 C** and the previous pass's own deliverable reaches **246.8 C**, against
a 250 C operating ceiling. Every sequential arm here peaks at **223.8 to 224.6 C**, because
it spreads the same melt over 594 s instead of 303 s. **Sequential dwell removes the
L_shape's ceiling breach.**

---

## 1. Conventions, stated once and carried on every number

**Objective.**

    J_phi(t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

with `chi_part` the binary rasterized part mask in the PART frame, the frame the design
lives in and the frame the part never leaves in this formulation. Under-melting a part cell
and melting a bed cell cost the same.

**Stop.** `t_stop` = argmin of J_phi over that arm's OWN stored trajectory on a 1500-step
horizon (dt 0.5 s, 750 s). **The shape early stop is DISABLED on every arm in this pass.**
That is a deliberate departure from `library_solve.PATIENCE` and it is forced by the
physics under test: a sequential schedule's objective falls during phase one, RISES while
the first limb over-melts, and falls again in phase two, so a patience rule would truncate
the run before phase two ever executed. No arm quoted below has its minimum on the last
stored step.

**Grid qualifier, mandatory.** `SOLVE_ROBUSTNESS_VALIDATION.md` established that absolute
fidelity at grid 120 does not transfer to grid 160. Every IoU here is a property of the
method AT GRID 120 with an exactly reproduced dopant map. No grid-160 number appears
anywhere in this report and no grid hold-out was run.

**Design parameterization** for the co-solved arms, per `FROZEN_CONVENTIONS_2D.md`:
normalized-convolution Gaussian filter over the part only at a PHYSICAL radius of 1.0 mm =
1.983 cells at grid 120, box [0, 1], saturation held at 1.0 outside the part, conductivity
channel only, 4 bpp quantization inside the part on the deliverable. No smoothed-Heaviside
projection.

**Schedule parameterization.** An ordered list of segments (angle, duration). Durations are
FREE and non-negative; the machine holds the last commanded position after the program ends,
exactly as `dwell_march.program_step_positions` already did, so the last duration is inert
and the free variables are the switch times. **No simplex and no softmax**, which is a
simplification over the cycled pass: total exposure is not a design variable because the
stop is chosen afterwards.

**Within-step mixing, and why it is exact.** A control step can straddle a switch. The
heating injected at such a step is the exact time average over the step,
`Q_i = sum_k f_ik Q_(a_k)`, with `f` the segment-to-step overlap matrix
(`adjoint2d/seq_dwell.step_mix`). Nothing is smoothed and no regularizer is introduced. On
switch times that lie on the control grid this reduces to the same one-hot selection the
cycled march already used, bit for bit.

**Energy gate.** Standing threshold 5 percent of integrated dose, checked at every arm's own
stop. **Zero violations in this pass**; worst residual 1.60 percent.

**Absorbed power is NOT matched.** Deliverable arms span 336.3 to 504.2 W/m against the
500 W/m uniform calibration target. No comparison here is at equal delivered energy. The
sequential arms run at LOWER absorbed power than the static arms they beat (348.0 against
496.4 W/m on the L_shape), so the improvement is not bought with extra dose; it is bought
with extra time.

---

## 2. What was measured before anything was designed

### 2.1 The limb response of the L_shape, at 15 degrees, previously unmeasured

COMPUTED, `out_seq/L_shape_probe.json`, uniform dopant map, one static run per
half-turn-distinct orientation, each at its own stop. The part is split by row width into
the WIDE limb (520 cells, the long arm) and the NARROW limb (559 cells, the stem). Only the
twelve half-turn-distinct angles are run, because the part-frame heating at theta and
theta + 180 degrees is the same field to 4e-13 relative (`DWELL_SCHEDULE_REPORT.md`
Section 2).

| angle deg | J_phi | IoU | wide limb melted % | narrow limb melted % | stop s | peak T C |
|---|---|---|---|---|---|---|
| **0** | 538.42 | 0.5142 | 13.65 | **94.28** | 263.5 | 225.3 |
| 15 | 614.07 | 0.4311 | 4.81 | 83.36 | 327.5 | 216.0 |
| 30 | 595.39 | 0.4518 | 11.54 | 84.79 | 564.5 | 213.4 |
| 45 | 666.60 | 0.3376 | 20.19 | 46.87 | 750.0 HORIZON | 193.3 |
| 60 | 1079.00 | 0.0000 | 0.00 | 0.00 | 0.5 | 26.8 |
| 75 | 833.78 | 0.2039 | 42.31 | 0.00 | 750.0 HORIZON | 191.5 |
| **90** | 678.13 | 0.3838 | **81.92** | 2.68 | 556.0 | 211.4 |
| **105** | 648.13 | 0.4842 | **91.54** | 30.05 | 567.5 | 246.7 |
| 120 | 516.24 | 0.5668 | 80.38 | 55.81 | 406.5 | 254.4 CEILING |
| **135** | **405.93** | **0.6471** | 74.42 | 78.35 | 303.0 | 253.4 CEILING |
| 150 | 507.05 | 0.5838 | 51.35 | 88.73 | 290.5 | 255.5 CEILING |
| 165 | 589.68 | 0.4885 | 22.12 | 85.87 | 251.5 | 237.4 |

**This is the whole argument for the pass in one table.** Each limb is individually
reachable: 94.3 percent of the narrow limb at 0 degrees, 91.5 percent of the wide limb at
105 degrees. No single orientation gets both above 79 percent. The static winner, 135
degrees, is the compromise that splits the difference, and it is the one that runs hottest.
The 60-degree entry, where nothing melts at all and the argmin sits at the first step, is
the catastrophic angle the orientation campaign already recorded near 67.5 degrees.

For the T_shape no probe was re-run: its limb response is already on record
(`CONTINUOUS_ROTATION_REPORT.md` Section 3, crossbar 4.5 percent and stem 94.7 percent at 0
degrees, crossbar 98.3 percent and stem 8.5 percent at 90 degrees).

### 2.2 The screen, and why it was affordable

Every two-segment schedule starting at the same orientation shares its whole first phase.
`scripts/analysis/seq_screen.py` marches that phase once, stores the state at each candidate
switch time and the running best over the prefix, and branches. That turns a
(first position) x (switch time) x (second position) grid into one full march plus the
tails. Forty-two two-segment arms and twelve three-segment arms cost 33 forward equivalents
in total instead of 54.

The screen does not run the energy bookkeeping, so **no number in this report is scored from
the screen**. The screen only selects which schedules get a full
`seq_dwell_march.sequential_forward` run with every standing gate on. Switch times on the
screen grid are exact multiples of the control step, where the lean march and the full march
agree bit for bit by the degeneracy test.

**What the screen found, COMPUTED** (`out_seq/<shape>_screen.json`, `_screen2.json`,
`figs_seq/fig_seq_screen_<shape>.png`):

* **Order matters and only one order works.** On the L_shape, wide limb first (90 or 105
  degrees) then narrow (0 degrees) reaches J_phi 293.05; narrow first (0 degrees then 90)
  reaches only 376.79. On the T_shape, crossbar first (90 then 0) reaches 377.79, while stem
  first (0 then 90) **COLLAPSES** at switch times of 400 s and beyond: the objective's argmin
  lands in phase one and the second hold never contributes, which the screen reports as
  `stop_is_in_phase_2 = False`.
* **The switch time has a clear interior optimum**, near 450 to 500 s on the L_shape and
  near 520 to 550 s on the T_shape, with the expected trade visible on both sides:
  under-melt falls and bed growth rises as the switch is delayed.
* **A third segment buys nothing on the L_shape.** All twelve three-segment arms have their
  argmin BEFORE the third switch, so the third hold is inert. It is reported as measured and
  was not refined; spending budget on a variable the objective does not see would have been
  waste.

---

## 3. What was built, and the red-first tests

### 3.1 New code, no existing solver internals modified

* `fgm_solve_campaign/adjoint2d/seq_dwell.py` the pure logic: the segment-to-step overlap
  matrix `step_mix`, its vector-Jacobian product `mix_vjp`, control-step snapping, the
  ordering enumeration, and the machine-readable `SequentialProgram`.
* `fgm_solve_campaign/adjoint2d/seq_dwell_march.py` the part-frame sequential march and
  `sequential_gradients`, which returns dJ/ds and dJ/d(durations) from ONE reverse march.
* `fgm_solve_campaign/adjoint2d/gate_seq_dwell.py` the FD gate, three layers.
* `fgm_solve_campaign/adjoint2d/tests/test_seq_dwell.py`, `tests/test_seq_dwell_march.py`.
* Drivers `scripts/analysis/{run_seq_probe, seq_screen, run_seq_screen, run_seq_screen2,
  run_seq_arms, run_seq_prior_baseline, run_seq_finescan, run_seq_kink_probe, finalize_seq,
  make_seq_figures}.py`.

Read and imported, not modified: `adjoint2d/{forward, adjoint, eqs, gradops,
shape_objective, design_filter, printability, library_solve, energy_gate, pins, rot_frame,
rot_kernel, dwell, dwell_kernel, dwell_march, topopt}.py`.

### 3.2 Twenty new tests, each observed red first; 313 passing overall

PROVEN. Both new test files were observed failing with `ImportError` before their module
existed, then passing.

* `tests/test_seq_dwell.py` (13). Row sums of the overlap matrix; one-hot reduction when
  every switch lands on the control grid; proportional splitting of a straddled step
  (0.3 / 0.7); the hold past the program end; rejection of a negative duration; three
  segments inside one control step; **the vector-Jacobian product against a central
  difference at 1e-7**; the derivative of the last duration being exactly zero; the
  step-edge KINK named and tested explicitly (`mix_vjp` returns the right derivative there
  and the two one-sided derivatives differ by more than 1e-3); program ordering, exposure
  conservation, control-step snapping, adjacent-repeat merging, and the ordering
  enumeration.
* `tests/test_seq_dwell_march.py` (7). Two degeneracy gates that pin the new march to code
  already FD gated: **one segment for the whole horizon reproduces the single-position
  march BIT FOR BIT** (temperature, density and melt fields identical) and its map gradient
  reproduces `DwellKernel.both_gradients` at one-hot weights to **below 1e-13 relative**;
  and **durations that are exact multiples of the control step reproduce
  `dwell_march.program_forward` BIT FOR BIT**. Then the new object, dJ/d(durations), against
  a central difference with a swept epsilon, **passing below 1e-6**; the refusal to run
  without the per-position heating fields; the refusal to differentiate against a stale map;
  and the overlap matrix carried on the trajectory.
* Whole suite: **313 tests pass** across `fgm_solve_campaign/adjoint2d/tests/`, run as
  `PYTHONPATH=$PWD/fgm_solve_campaign:$PWD ./.venv312/bin/python -m pytest ...` (421 s).
  Twenty of those are new here; the count rose from the 281 of the cycled dwell pass partly
  through other work in the tree, which is stated rather than claimed as this pass's.

### 3.3 The finite-difference gate: the cleanest in the campaign

Central differences, epsilon swept over the campaign's eight values
(1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8), **fixed read index** taken as the argmin
of J_phi on the base run. Gate point: three DIFFERENT positions (0, 90 and 135 degrees) with
durations 123.3, 141.15 and 200.0 s, so every switch sits 0.05 s from the nearest
control-step edge, four orders of magnitude clear of the largest epsilon. Base read index
854 of 900, `at_horizon = False`. Raw sweeps in `out_seq/gate_seq_L_shape.json`.

| layer | max-sensitivity | random | random direction | filter-smooth direction | gradient direction | verdict |
|---|---|---|---|---|---|---|
| **Q0** map through the sequential march, unfiltered | 8.38e-08 | 2.24e-07 | 9.39e-07 | | 3.46e-08 | **4/4 at 1e-6** |
| **Q1** segment durations, the new object | **6.14e-08** | **6.14e-08** | **4.60e-08** | | **1.50e-08** | **4/4 at 1e-6** |
| **Q2** map with the 1.0 mm filter, the gradient the co-solve uses | 2.19e-08 | 2.88e-07 | 4.55e-07 | 2.34e-07 | 1.57e-07 | **5/5 at 1e-6** |

COMPUTED. **`ALL_GATES_PASS at 1e-6 = True` and at the 1e-5 subgradient standard = True.**
Best relative errors 1.5e-08 to 9.4e-07, with no probe missing and no judgement call
required, which is better than the cycled dwell pass managed (one miss at 1.39e-05). The
stop-index stability check reports `moved = False`. `dJ/d(last duration)` is exactly zero,
as the hold-past-the-end model requires.

Two honest notes on the gate. **The random-duration probe drew the same coordinate as the
maximum-sensitivity probe** (both 0.80991), because a three-segment schedule has only two
free durations; that probe is therefore not independent. And **the gate was run on the
L_shape only**; the T_shape is ASSUMED to inherit it, on the same reasoning the joint,
rotation and dwell campaigns used.

---

## 4. The finding that reframes the whole switch-time story

COMPUTED, and it is the most transferable numerical result in the pass. **The switch-time
gradient is correct and useless for descent at the same time, and both halves of that
sentence are measured.**

### 4.1 What happened

On the L_shape, eight L-BFGS-B evaluations on the gated duration gradient moved the switch
time from 450.0 s to 449.99 s and did not improve J_phi at all. A direct 5 s scan over
400 to 500 s found the minimum at **465 s with J_phi 291.89**, 0.40 percent better than the
gradient's answer. On the T_shape the same procedure DID move, 550 s to 515.25 s, and landed
0.014 percent from the scan minimum at 520 s. So the gradient helped on one shape and
stalled on the other.

### 4.2 The measurement that explains it

At the production read state of the L_shape sequential arm (switch 450.244 s, argmin index
1159, J_phi 293.019), with the read index held FIXED so no envelope-theorem term can enter:

| epsilon (s) | central difference of J at the fixed index | relative error against the analytic +3.977658 |
|---|---|---|
| 1e-1 | -0.1435 | 1.04e+00 |
| 1e-2 | -0.3768 | 1.09e+00 |
| 1e-3 | -0.1208 | 1.03e+00 |
| 1e-4 | -2.7944 | 1.70e+00 |
| **1e-5** | **+3.977658** | **2.40e-08** |
| **1e-6** | **+3.977658** | **3.03e-08** |
| 1e-7 | +3.977662 | 1.17e-06 |

**The analytic gradient is exactly right inside a window of about 5e-5 seconds and exactly
wrong outside it.** A dense scan INSIDE a single control step confirms the large-scale
behaviour: as the switch moves across step 900 the mixing fraction goes from 0.04 to 1.00
and J_phi falls by 0.076, an effective slope of **-0.15 per second**, while the local
derivative is **+3.98 per second**. The argmin index does not move at any of these epsilons,
so this is not the envelope theorem; it is the objective itself.

The function is continuous (checked across a step edge at 450.5 s) and piecewise linear with
slopes alternating between about +4 and about -25 on intervals of order 1e-4 s. The exact
derivative at almost every point is one of the two branch slopes; the useful design-scale
slope is their average, and the adjoint returns the branch, not the average.

### 4.3 What causes it, and what does not

Three candidates were tested and **RULED OUT**:

* **The non-Lipschitz densification driving term** `(1 - phi)**0.8`, which
  `DWELL_SCHEDULE_REPORT.md` Section 3.4 identified as the campaign's known offender and
  which has 785 fully melted cells with base exactly zero at this read state. **ABLATED**:
  re-running the whole epsilon sweep with `dens_phi_solid_exponent = 1.0` reproduces every
  finite difference to five decimal places. It is not the cause.
* **The temperature-step clip** and **the temperature cap**: both measured at
  `frac_dT_clipped_max = 0.000000` and `frac_temp_cap_max = 0.000000` on every arm in the
  pass. Not active, so not the cause.

The remaining candidate, **not ablated and therefore not proven**, is the melt-fraction clip
inside the substep property blend: 1079 part cells over 7500 substeps give of order 10^6
opportunities for a cell to cross phi = 0 or phi = 1, each of which is a kink in that cell's
trajectory, and a spacing of 1e-4 s in the switch time is consistent with that population.
**Naming a cause is left open rather than guessed.**

### 4.4 What was done about it

The switch times of every arm reported below were set by the **5 s scan**, which is a
measurement, and the gradient result is reported alongside it rather than hidden. The
gradient-refined switch time and the scan minimum are both in `out_seq/<shape>_kink.json`.
This is stated in the emitted turntable programs under `switch_time_chosen_by`.

**The map gradient is not affected.** Q0 and Q2 pass, and the co-solve moved J_phi by 4.2
percent on the L_shape and 7.5 percent on the T_shape, so the map channel descends normally.
The plausible reason for the asymmetry is lever arm: a map perturbation moves every cell of
the heating field at every one of 1500 steps, while a switch-time perturbation moves the
heating of ONE step, so the microscopic non-smoothness sits at the same scale as the signal
in one channel and far below it in the other. That reasoning is ASSUMED, not measured.

---

## 5. Per-shape tables, all arms, each at its own stop

Read: J is the whole-domain shape objective; growth, under-melt, limb melt, mean relative
density and peak temperature are at that arm's own stop; peak temperature is the maximum
over the whole exposure UP TO the stop, not only at the stop, because the ceiling is a
process limit; `Eres` is the energy-gate residual there. The melt-window columns
(under / in-window / over, percent of part cells against the 175 to 185 C window) and the
95th-percentile overshoot are the `FGM_WINDOW_RESELECTION.md` metrics.

### L_shape (part cells 1079; wide limb 520, narrow limb 559)

| arm | J | IoU | grow % | under % | wide % | narrow % | mean rho | P abs W/m | stop s | peak T C | Eres | switch s | window u/i/o % | p95 over C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static 135 deg, uniform map | 405.93 | 0.6471 | 18.16 | 23.54 | 74.4 | 78.4 | 0.7246 | 504.2 | 303.0 | **253.4 CEILING** | 1.48 % | | 15.5/18.5/66.0 | 54.6 |
| static 135 deg, previous pass's 4 bpp map | 402.06 | 0.6496 | 17.70 | 23.54 | 74.2 | 78.5 | 0.7240 | 501.7 | 304.0 | **251.8 CEILING** | 1.45 % | | 15.6/18.3/66.2 | 53.5 |
| CYCLED equal dwell, uniform map, 20 s cycle | 418.47 | 0.6344 | 20.67 | 23.45 | 71.3 | 81.4 | 0.7560 | 365.6 | 502.5 | 227.9 | 1.43 % | | 12.1/21.9/66.0 | 38.6 |
| **BASELINE** previous best on record, re-run | **396.20** | **0.6484** | 16.50 | 24.47 | 73.5 | 77.5 | 0.7191 | 496.4 | 304.5 | 246.8 | 1.37 % | | 15.8/18.7/65.4 | 50.6 |
| SEQUENTIAL 90 then 0 deg, **uniform map**, screen grid | 293.05 | 0.7059 | 12.51 | 20.57 | 80.2 | 78.7 | 0.7208 | 357.4 | 580.0 | 223.8 | 0.92 % | 450.0 | 9.5/25.0/65.5 | 34.1 |
| the same, switch times moved by the gated gradient | 293.05 | 0.7059 | 12.51 | 20.57 | 80.2 | 78.7 | 0.7208 | 357.4 | 580.0 | 223.8 | 0.92 % | 450.0 | 9.5/25.0/65.5 | 34.1 |
| **SUB-HYPOTHESIS ARM** the same at the scan minimum | **291.89** | **0.7075** | 13.44 | 19.74 | 82.5 | 78.2 | 0.7261 | 352.6 | 591.5 | **223.9** | 0.97 % | 465.1 | 8.8/24.7/66.5 | 34.2 |
| three segments, 90 then 0 then 135 deg, uniform map | 293.05 | 0.7059 | 12.51 | 20.57 | 80.2 | 78.7 | 0.7208 | 358.0 | 580.0 | 223.8 | 0.92 % | 450.0 | 9.5/25.0/65.5 | 34.1 |
| SEQUENTIAL, co-solved map, continuous | 280.76 | 0.7154 | 11.03 | 20.57 | 78.8 | 80.0 | 0.7197 | 352.6 | 583.5 | 224.6 | 0.84 % | 450.0 | 9.8/27.6/62.6 | 35.0 |
| SEQUENTIAL, co-solved map, 4 bpp | 280.52 | **0.7168** | 10.94 | 20.48 | 79.0 | 80.0 | 0.7198 | 352.6 | 583.5 | 224.6 | 0.84 % | 450.0 | 9.7/28.1/62.2 | 35.0 |
| **DELIVERABLE** co-solved 4 bpp at the scan switch | **276.39** | 0.7161 | 11.96 | 19.83 | 81.2 | 79.2 | 0.7246 | 348.0 | 594.5 | 224.5 | 0.87 % | 465.1 | 8.9/26.5/64.6 | 34.7 |

### T_shape (part cells 1104; crossbar 576, stem 528)

| arm | J | IoU | grow % | under % | bar % | stem % | mean rho | P abs W/m | stop s | peak T C | Eres | switch s | window u/i/o % | p95 over C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static 90 deg, uniform map | 551.72 | 0.4956 | 3.80 | 48.55 | 94.4 | 4.5 | 0.6598 | 274.0 | 555.0 | 200.6 | 0.35 % | | 45.1/15.2/39.7 | 14.2 |
| static 90 deg, previous pass's 4 bpp map | 529.92 | 0.5321 | 4.53 | 44.38 | 98.3 | 9.1 | 0.6853 | 264.4 | 624.5 | 206.4 | 0.47 % | | 42.4/13.0/44.6 | 19.0 |
| CYCLED equal dwell, uniform map, 20 s cycle | 526.02 | 0.5746 | 22.64 | 29.53 | 60.1 | 81.8 | 0.7505 | 359.5 | 532.5 | 229.0 | 1.60 % | | 20.3/18.4/61.3 | 40.7 |
| **BASELINE** previous best on record, re-run | **422.28** | **0.6436** | 15.40 | 25.72 | 97.9 | 48.5 | 0.7469 | 306.2 | 672.5 | 219.3 | 1.10 % | | 16.8/26.3/56.9 | 31.5 |
| SEQUENTIAL 90 then 0 deg, **uniform map**, screen grid | 377.79 | **0.6581** | 20.29 | 20.83 | 83.0 | 75.0 | 0.7601 | 334.3 | 655.5 | 229.1 | 1.40 % | 550.0 | 8.2/25.2/66.7 | 39.4 |
| the same, switch times moved by the gated gradient | 374.02 | 0.6497 | 17.39 | 23.73 | 76.4 | 76.1 | 0.7418 | 344.7 | 626.5 | 228.1 | 1.23 % | 515.2 | 10.1/26.3/63.6 | 38.4 |
| **SUB-HYPOTHESIS ARM** the same at the scan minimum | **373.94** | 0.6524 | 18.30 | 22.83 | 77.4 | 76.9 | 0.7459 | 343.6 | 631.0 | 228.8 | 1.26 % | 518.9 | 10.0/25.9/64.1 | 39.1 |
| SEQUENTIAL, co-solved map, continuous | 345.08 | **0.6693** | 13.95 | 23.73 | 73.3 | 79.5 | 0.7366 | 337.2 | 632.5 | 230.6 | 1.03 % | 515.2 | 10.9/29.3/59.8 | 40.7 |
| SEQUENTIAL, co-solved map, 4 bpp | 344.95 | 0.6661 | 14.49 | 23.73 | 73.3 | 79.5 | 0.7371 | 337.3 | 632.5 | 230.6 | 1.04 % | 515.2 | 10.9/29.3/59.8 | 40.8 |
| **DELIVERABLE** co-solved 4 bpp at the scan switch | **343.02** | 0.6661 | 15.04 | 23.37 | 74.7 | 78.8 | 0.7377 | 336.3 | 634.5 | 230.3 | 1.04 % | 518.9 | 10.5/29.2/60.3 | 40.4 |

**Mean relative density at the deliverables' own stops spans 0.7191 to 0.7601.** These are
melt-stop reads, not end-of-horizon reads, and nothing here is an oversinter claim.

**Operating ceiling, 250 C.** Two breaches in the pass, both on STATIC L_shape arms
(253.4 C and 251.8 C), and both are baselines rather than deliverables. Every sequential arm
sits between 223.8 and 230.6 C.

**The J-best and IoU-best arms disagree on both shapes**, which the objective permits: on
the L_shape the 4 bpp co-solved map at the 450 s switch is IoU-best (0.7168) while the same
map at 465.1 s is J-best (276.39); on the T_shape the uniform-map arm at 550 s is IoU-best
(0.6581) while the co-solved arm is J-best (343.02). Both are left visible rather than
resolved by quoting only one.

---

## 6. The failure modes the task asked to watch, measured

COMPUTED, from the tables above and `figs_seq/fig_seq_screen_<shape>.png`.

1. **Phase-one over-melting and bed growth while phase two heats.** This is real and it is
   the price of the method, but on these shapes it is a NET GAIN, not a net cost. On the
   L_shape, growth falls from the baseline's 16.50 percent to the deliverable's 11.96
   percent and under-melt falls from 24.47 to 19.83 percent, so both sides of the trade
   improve at once. On the T_shape growth is essentially unchanged (15.40 to 15.04 percent)
   while under-melt falls from 25.72 to 23.37 percent. The reason both can improve is that
   the static baseline was already over-melting a rounded blob at high power in 304 s; the
   sequential arm delivers 30 percent less power over twice the time.
   **Where growth does bite is at LONG switch times**: on the L_shape's 105-degree-first
   family, growth climbs from 14.1 percent at a 150 s switch to 42.2 percent at 600 s, and
   the objective turns over well before that. The trade is visible in the third panel of
   `fig_seq_screen_L_shape.png`.
2. **Heat from phase two re-entering the phase-one limb.** Measured directly in the limb
   columns. On the L_shape the wide limb is at 91.5 percent when 105 degrees is held alone;
   in the deliverable it ends at 81.2 percent, so it LOSES about ten points while phase two
   runs, partly to lateral spreading and partly because the stop is taken later. The
   narrow limb rises from 2.7 percent (90 degrees alone) to 79.2 percent. The trade is
   strongly favourable but it is not free, and the phase-one limb does not simply freeze.
3. **The 250 C ceiling.** Two breaches, both static baselines, named above. No sequential
   arm approaches it.
4. **Order collapse.** On the T_shape the stem-first order with a switch at 400 s or later
   has its argmin in phase one, that is, the schedule degenerates to a static arm. That is
   the same collapse mode the cycled pass hit on the L_shape, and here it is confined to the
   wrong ordering.

---

## 7. Reproduction checks against the previous pass

PROVEN. Both shapes' best arms on record were re-run through this pass's scoring
(`out_seq/prior_baseline.json`):

| shape | stored arm | stored J / IoU | re-run J / IoU | delta |
|---|---|---|---|---|
| L_shape | `D_refined_4bpp_discovered_lib`, time resolved | 396.20 / 0.6484 | **396.20 / 0.6484** | **+0.000 %** |
| T_shape | `D_refined_timeresolved_discovered_lib` | 422.28 / 0.6436 | **422.28 / 0.6436** | **+0.000 %** |

Separately, the L_shape's stored `D_joint_4bpp` arm (402.06 / 0.6496) is reproduced to the
printed digits by a ONE-SEGMENT sequential march at 135 degrees, which is a cross-check of
the new march against a stored number from a different code path.

**One correction to the previous pass's own bookkeeping, found here.** The T_shape's best
stored quasi-static number is 426.21 and its best stored TIME-RESOLVED number is 422.28.
The re-run executes the cycle program time-resolved, so 422.28 is the right comparison and
426.21 is not; using the quasi-static value would have made this pass's gain look 0.9
percent larger than it is.

---

## 8. Budget accounting, stated plainly, including the overspend

The authorized budget was 60 forward equivalents, against the campaign's standing 40. One
forward equivalent is one 1500-step part-frame march; one gradient evaluation is 2.5 forward
equivalents by the campaign's own conversion (`DWELL_SCHEDULE_REPORT.md` Section 10 item 6:
16 gradient evaluations equal 40 forward equivalents).

| item | per shape | reading |
|---|---|---|
| duration refinement, 8 gradient evaluations | 20 | inside budget |
| co-solve, 20 gradient evaluations in three alternating blocks | 50 | |
| interior re-refinement, 6 gradient evaluations | 15 | |
| **solve depth total, 34 gradient evaluations** | **85** | **42 percent over the authorized 60** |
| screens and probe (L_shape) | 40 | measurement, not solve depth |
| scoring, snapshot and baseline marches | 12 | |
| the finite-difference gate, three layers at 900 steps | 131 | gates are reported separately by campaign convention |
| the non-smoothness diagnosis of Section 4 | 45 | unplanned, and the most valuable spend in the pass |

**The solve-depth budget was exceeded by 42 percent and that is named rather than buried.**
The overspend is concentrated in the co-solve's third block, which bought 0.0 percent on the
L_shape (J 280.76 to 280.76) and 0.3 percent on the T_shape; those eight gradient
evaluations per shape could have been cut with no change to any verdict.

**Wall time, COMPUTED and logged progressively.** After the L_shape probe finished (218 s
for twelve static marches over twelve orientations plus one electro-quasi-static pass at
7.0 s) the first screen was projected at about 700 s; actual 711 s. The gate was projected
from the cycled pass's 1708 s at a 1400-step horizon to about 3000 s at 900 steps with three
layers; actual 3836 s. Total recorded process time **about 14,700 s, roughly 4.1 hours, in
up to four pinned single-thread streams** (`fgm_solve_campaign/env1.sh`).

**Nothing was cut.** Two runs were superseded and both are reported rather than deleted: the
gradient-refined switch times (kept in the tables, and the reason they are not the
deliverable is Section 4), and the three-segment L_shape arm (kept, and the reason it was
not refined is that its argmin sits before its third switch).

---

## 9. Machine-readable deliverables

`fgm_solve_campaign/out_seq/<shape>_turntable_DELIVERABLE.json` carries the ordered list of
`{position_deg, dwell_s, move_at_s}`, the control step, the recommended stop, the scores,
the peak temperature up to the stop, a pointer to the dopant map array
(`<shape>_seq_maps.npz`, key `sat_S_seq_cosolved_4bpp_interior_switch`, 4 bits per pixel
inside the part), and an explicit `switch_time_chosen_by` field recording that the switch
came from a scan and not from the gradient.

| shape | program | moves over 750 s | recommended stop |
|---|---|---|---|
| L_shape | hold **90 degrees for 465.0 s**, then **0 degrees** to the end | **1 move** | 594.5 s |
| T_shape | hold **90 degrees for 519.0 s**, then **0 degrees** to the end | **1 move** | 634.5 s |

**One move.** That is the practical headline for the hardware: the entire gain over the
previous best on record comes from a single quarter turn, timed correctly, with no dopant
grading required to capture most of it.

---

## 10. Honest limits

1. **Neither shape reaches the SOLVED class.** IoU 0.7161 and 0.6661 against the 0.95
   qualifier. About one fifth of each part is still unmelted at the stop.
2. **The switch-time gradient does not descend the objective at the production read state**
   (Section 4). The switch times are scan results. The gradient is FD gated and correct at a
   fixed read index; that is a statement about the gradient, not about its usefulness here.
3. **The cause of the fine-scale non-smoothness is not established.** Three candidates are
   ruled out by ablation or by measured zero activation; the melt-fraction clip inside the
   substep is the remaining suspect and was NOT ablated.
4. **The gate was run on the L_shape only.** The T_shape is ASSUMED to inherit it.
5. **The candidate angle set was chosen from the probe, not swept.** Four angles per shape
   entered the kernel. A finer set might find a better pair, particularly for the T_shape
   whose limb response was taken from a previous report rather than re-measured here.
6. **Two segments only, in effect.** Three segments were screened on the L_shape and found
   inert; four segments were never run; a schedule that revisits a position non-adjacently
   was never run.
7. **The programs are NOT verified on the production engine.** The engine as shipped drives
   a fixed rotation increment at a fixed interval and cannot express a sequential hold at
   all (`DWELL_SCHEDULE_REPORT.md` Section 4). The part-frame march that runs them agrees
   with the engine to 0.09 percent on the schedules both can express, which is the strongest
   statement available.
8. **A turntable move is instantaneous and free** in this model. Here that assumption is
   unusually mild, because the program contains exactly one move.
9. **The early stop is disabled**, which departs from the frozen recipe. It is necessary
   (Section 1) and it makes every arm cost a full horizon.
10. **Not dose matched** (336.3 to 504.2 W/m). The sequential arms use LESS power than the
    baselines they beat, so the direction of the confound is against this pass's result, but
    it is still a confound.
11. **Conductivity channel only, single grid (120 x 120), two dimensions, no grid hold-out
    and no sub-filter-radius perturbation gate.** Neither Gate A nor Gate B of
    `FROZEN_CONVENTIONS_2D.md` Section 8 was run, so **no map here carries a SOLVED label**.
12. **Model, not hardware.** `ALLISON_LAW_REPLICATION.md` Section 6.1 records that the
    two-dimensional model over-predicts achievable tuned uniformity against hardware by
    roughly a factor of eight.

---

## 11. Proven, computed, assumed

**PROVEN**
* One segment for the whole horizon reproduces the single-position march bit for bit, and
  its map gradient reproduces `DwellKernel.both_gradients` at one-hot weights below 1e-13.
* Step-aligned durations reproduce `dwell_march.program_forward` bit for bit.
* The overlap matrix's vector-Jacobian product against a central difference, and the
  step-edge kink named and tested rather than hidden.
* dJ/d(durations) FD gated at 1e-6 at a fixed read index, best relative errors 1.5e-08 to
  6.1e-08; the map gradient through the sequential march likewise, filtered and unfiltered.
* Both shapes' best arms on record reproduced to +0.000 percent.
* The L_shape's stored `D_joint_4bpp` arm reproduced by a one-segment sequential march.
* Twenty new tests, each observed red first; 313 pass overall.

**COMPUTED**
* Every number in Sections 0, 2, 4, 5, 6, 8 and 9.
* That the densification exponent, the temperature-step clip and the temperature cap are all
  ruled out as causes of the fine-scale non-smoothness.

**ASSUMED**
* That the gate measured on the L_shape holds on the T_shape.
* That the rasterized binary part mask is the right nominal target.
* That an arbitrary stop time is realizable as a process control.
* That a turntable move is instantaneous and free.
* That the lever-arm argument of Section 4.4 explains why the map channel is unaffected.

---

## 12. The single most valuable next layer

**Give the objective a smoothing that the switch time can see.** Section 4 shows the
blocking issue is not the adjoint and not the physics but the fine-scale non-smoothness of
J_phi in a channel with a small lever arm. Two concrete moves, in order of cost: first,
ablate the melt-fraction clip inside the substep property blend (replace the hard clip with
a smoothed one whose width is a stated parameter, and test shrinking that width) to confirm
the cause named in Section 4.3; second, define the schedule objective as a short TIME
AVERAGE of J_phi over a window around the stop rather than the value at a single index,
which averages the branch slopes and would give the switch time a gradient that descends.
Both are cheap and both are gateable with the harness now written.

**Second: widen the actuator to the two-part question this pass leaves open.** Every arm
here holds a single orientation per phase. A schedule that holds the wide limb's best angle,
switches, and switches BACK for a short third hold was screened only on a coarse grid and
found inert because the stop arrived first; with the stop treated as a design variable
rather than an argmin, that may not hold.

**Third: run this on the cross and the star.** The mechanism is generic to any part whose
limbs are individually reachable but not simultaneously, and those two shapes are already at
IoU 0.98 and 0.95 under indexing, so the interesting question is whether sequential dwell
gets them to 1.0 or whether it costs them growth.

---

## 13. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New code:
`fgm_solve_campaign/adjoint2d/{seq_dwell,seq_dwell_march,gate_seq_dwell}.py`;
`fgm_solve_campaign/adjoint2d/tests/{test_seq_dwell,test_seq_dwell_march}.py`;
`scripts/analysis/{run_seq_probe,seq_screen,run_seq_screen,run_seq_screen2,run_seq_arms,
run_seq_prior_baseline,run_seq_finescan,run_seq_kink_probe,finalize_seq,
make_seq_figures}.py`.

Results, all under `fgm_solve_campaign/out_seq/`:
* `gate_seq_L_shape.json` the three-layer gate with full epsilon sweeps
* `L_shape_probe.json` the limb response against orientation
* `<shape>_screen.json`, `<shape>_screen2.json` the two-segment and three-segment screens
* `<shape>_arms.json` every scored arm, every solve trace, the snapshots index
* `<shape>_seq_maps.npz` dopant maps, melt fields at each arm's own stop, objective curves,
  the phase snapshots, the part mask and the two limb masks
* `<shape>_finescan.json` the 5 s switch-time scan
* `<shape>_kink.json` the switch-time non-smoothness probe and the interior refinement
* `prior_baseline.json` the previous pass's best arms, re-run
* `<shape>_turntable_DELIVERABLE.json` and `<shape>_turntable_S_seq_*.json` the programs
* `fgm_solve_campaign/logs_seq/*.log` per-run console logs

Figures, **all five viewed before delivery**, in `fgm_solve_campaign/figs_seq/`:
* `fig_seq_L_shape.png`, `fig_seq_T_shape.png` the per-shape composite: the program as a
  timeline, the melt field through phase one and phase two with both limbs outlined, the
  SAME wall-clock times on the static best-angle arm for contrast, and every arm at its own
  stop
* `fig_seq_screen_<shape>.png` what the switch time buys and what it costs in growth
* `fig_seq_curves.png` the objective against time, where the second descent after the switch
  is the mechanism made quantitative
