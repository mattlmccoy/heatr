# Asymmetric dwell scheduling: timing the rotations of the part

**Date:** 2026-08-01. **Scope:** overnight queue item 6b. The actuator is the TIMING OF THE
PART'S ROTATION. A turntable can be commanded to a fixed set of indexed orientations and
HELD at each one; the design variables are the dwell durations, solved jointly with the
dopant map, with the radio-frequency generator power held constant. Four shapes: cross and
square as regression controls, T_shape and L_shape as the targets. Grid 120 x 120
throughout. **Nothing was committed. No dissertation file was touched. `.claude/worktrees/`
was not read or written. No line of `rfam_eqs_coupled.py` was edited.**

**Acronyms, expanded on first use.** RFAM = radio-frequency additive manufacturing.
FGM = functionally graded material (a spatially varying dopant saturation map).
EQS = electro-quasi-static (the low-frequency Maxwell approximation the two-dimensional
solver uses). IoU = intersection over union. bpp = bits per pixel. L-BFGS-B = limited-memory
Broyden-Fletcher-Goldfarb-Shanno with box constraints. FD = finite difference. phi = melt
fraction. rho = relative density. W/m = watts per metre of depth. sigma = electrical
conductivity. eps_r = relative permittivity.

**Evidence tags.** PROVEN = unit tested, FD gated, or reproduced against a stored number.
COMPUTED = measured from a run in this pass. ASSUMED = a modelling choice or an inference
not measured here.

---

## 0. Conventions, stated once and carried on every number

**Objective.**

    J_phi(s, w, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

with `chi_part` the binary rasterized part mask in the PART frame, which is the frame the
design lives in and the frame the part never leaves in this formulation. Under-melting a
part cell and melting a bed cell cost the same.

**Stop.** `t_stop` = argmin of J_phi over that arm's OWN stored trajectory on a 1500-step
horizon (dt 0.5 s, 750 s), with early truncation 250 steps past the running minimum
(`library_solve.PATIENCE`). A minimum on the last stored step is flagged HORIZON and makes
that arm's J a BOUND, not a value. No arm quoted below is at the horizon. The melted region
for IoU, growth and under-melt is phi >= 0.5; J_phi itself uses no threshold.

**Grid qualifier, mandatory.** `SOLVE_ROBUSTNESS_VALIDATION.md` established that absolute
fidelity at grid 120 does not transfer to grid 160. Every IoU here is a property of the
method AT GRID 120 with an exactly reproduced dopant map. No grid-160 number appears
anywhere in this report and no grid hold-out was run.

**Design parameterization**, per `FROZEN_CONVENTIONS_2D.md`: normalized-convolution Gaussian
filter over the part only at a PHYSICAL radius of **1.0 mm = 1.983 cells at grid 120**, box
[0, 1] on the map, saturation held at 1.0 outside the part, conductivity channel only
(`eps_covary=False`), 4 bpp quantization inside the part on every deliverable. No
smoothed-Heaviside projection was used (beta = 0, the flag-off identity path).

**Dwell parameterization.** The dwell fractions are moved by the optimizer through a
**softmax**, which keeps them strictly positive and exactly normalized for any real logit
vector, so no constraint handling and no kink enters the FD gate. Its one cost is that an
exactly zero dwell is only reached in the limit; the emitted programs quantize to the 0.5 s
control step and a position below half a step drops out there. The **Euclidean simplex
projection** is implemented and unit tested alongside it (`dwell.project_to_simplex`, four
tests including a brute-force nearest-point check) and is used for snapping and reporting,
never inside the gated gradient chain, precisely because it is non-smooth on the faces.

**Energy gate.** Standing threshold 5 percent of integrated dose at the arm's own stop,
checked on every scored run. **Zero violations in this pass**; worst residual 1.55 percent.

**Absorbed power is NOT matched.** Deliverable arms span 313.1 to 496.4 W/m against the
500 W/m uniform calibration target. No comparison here is at equal delivered energy.

---

## 1. Verdict, one line per shape

| shape | role | dwell the solver chose (4 distinct positions) | equal-dwell control | dwell deliverable | verdict |
|---|---|---|---|---|---|
| **cross** | control | **0.500 / 0 / 0.500 / 0** at {0, 45, 90, 135} deg | J 117.72, IoU 0.8444 | **J 34.04, IoU 0.9829** | **CONTROL PASSES, and it wins.** Symmetry re-emerges EXACTLY (0.2501 / 0.2499 on the raw eight-vector) and the 45-degree family is rejected outright. J falls 71 percent and the cross enters the SOLVED class (IoU >= 0.95). |
| **square** | control | **0.250 / 0.250 / 0.250 / 0.250**, exactly equal | J 15.96, IoU 1.0000 | **J 15.96, IoU 1.0000** | **CONTROL PASSES.** Equal dwell re-emerges on all four distinct positions and the deliverable IS the control, to the last digit. Melt matches the nominal part exactly at grid 120. |
| **T_shape** | target | **0.210 / 0.028 / 0.735 / 0.028**, 2.94x equal at 90 deg | J 513.49, IoU 0.5710 | **J 422.11, IoU 0.6231** (quasi-static; best time-resolved arm 422.28 / 0.6436) | **PARTIAL RESCUE, honest.** Unequal dwell buys -17.8 percent J and +0.052 IoU over the equal-dwell control at the same budget, and beats the 0.572 best-on-record. It does NOT reach the SOLVED class: 27.5 percent of the part is still unmelted. |
| **L_shape** | target | **0 / 0 / 0 / 1.000**, all the exposure at 135 deg | J 407.81, IoU 0.6359 | **J 396.20, IoU 0.6484** | **NO RESCUE, and the schedule says so itself.** Given full freedom the optimizer collapses the schedule to a STATIC arm at 135 degrees, which is exactly the joint static campaign's winning angle for this shape. Gain over the control is -2.8 percent J and +0.013 IoU, inside the noise of this comparison. Its 246.8 C is the closest approach to the 250 C ceiling in the pass. |

COMPUTED, all four. Both columns are the TIME-RESOLVED execution of the actual cycle program
at the 20 s cycle, read at each arm's own stop, except the T_shape deliverable which is
quoted quasi-statically because that arm has no better time-resolved companion (its refined
time-resolved arm, 422.28 / 0.6436, is quoted alongside).

**The headline answer.** Asymmetric dwell scheduling is a REAL actuator and the regression
controls prove the machinery is sound: on the two shapes whose symmetry demands equal dwell,
equal dwell re-emerges to four decimal places from a free eight-parameter search, and on the
cross it simultaneously rejects the four 45-degree positions that were poisoning the
symmetric average. **But it does not rescue the stranded shapes.** On the T_shape it buys
a real, dose-honest +0.052 IoU by parking three quarters of the exposure at the one
orientation that heats the crossbar; on the L_shape it buys nothing, and the schedule
degenerates to "do not rotate". Both remain far below the 0.95 SOLVED class.

---

## 2. The structural finding that reframes every dwell number

COMPUTED, and it changes how the schedules must be read. **The part-frame heating at
orientation theta and at theta + 180 degrees is the SAME FIELD**, measured to a relative
maximum absolute difference of **3.3e-13 to 6.0e-13** on all four pairs of the candidate
set (L_shape, random map, `Q_k` arrays from `DwellKernel.averaged_Q`):

| pair | max abs difference | field scale | relative |
|---|---|---|---|
| 0 against 180 deg | 9.22e-06 | 2.28e+07 | **4.04e-13** |
| 45 against 225 deg | 3.19e-06 | 7.61e+06 | **4.20e-13** |
| 90 against 270 deg | 2.89e-06 | 8.88e+06 | **3.26e-13** |
| 135 against 315 deg | 7.39e-06 | 1.24e+07 | **5.97e-13** |

The reason is that the grounded parallel-plate drive is invariant under a half turn of the
whole system, so rotating the part by 180 degrees and rotating the resulting heating back
returns the original field. Three consequences, all of which are applied throughout this
report:

1. **The eight candidate positions carry only FOUR distinct heating patterns.** Every dwell
   vector is therefore determined only up to how it splits mass between theta and
   theta + 180. The raw eight-vector's "asymmetry" is partly an artefact of that gauge
   freedom, so **every dwell number here is reported on the four distinct positions**,
   `w4_k = w_k + w_{k+4}`.
2. **The emitted turntable program is issued in two forms**: the literal one and a REDUCED
   one on the distinct positions, which halves the number of moves and is physically
   identical. On the cross that is 150 moves against 75.
3. It is a free correctness check on the whole rotation stack, and it passes at 4e-13.

This also explains a result that would otherwise look like a bug: the L_shape's
time-resolved J is IDENTICAL to its quasi-static J to six figures at every cycle time from
10 s to 160 s, because its two kept positions are 135 and 315 degrees, one distinct pattern,
so the "schedule" is a static run.

---

## 3. What was built, and the finite-difference gate

### 3.1 New code, no existing solver internals modified

* `fgm_solve_campaign/adjoint2d/dwell.py` the pure logic: Euclidean simplex projection,
  softmax and its vector-Jacobian product, largest-remainder apportionment, the cycle
  scheduler that emits the machine-readable turntable program, and the asymmetry metrics.
* `fgm_solve_campaign/adjoint2d/dwell_kernel.py` the DWELL-WEIGHTED heating kernel,
  `Q_avg(s, w) = sum_k w_k R_(-theta_k)[Q_rf(R_(theta_k) s; theta_k)]`, and
  `both_gradients`, which returns dJ/ds and dJ/dw from ONE reverse march.
* `fgm_solve_campaign/adjoint2d/dwell_march.py` the TIME-RESOLVED execution of a program in
  the part frame, with no interpolation anywhere.
* `fgm_solve_campaign/adjoint2d/dwell_power.py` the secondary arm's power channel.
* `fgm_solve_campaign/adjoint2d/gate_dwell.py`, `gate_dwell_power.py` the FD gates.
* Drivers `scripts/analysis/run_dwell_solve.py`, `run_dwell_refine.py`,
  `run_dwell_engine_check.py`, `run_dwell_power_arm.py`, `finalize_dwell.py`,
  `make_dwell_figures.py`.

Read and imported, not modified: `adjoint2d/{forward,adjoint,eqs,gradops,shape_objective,
design_filter,printability,library_solve,energy_gate,pins,rot_frame,rot_kernel,schedule,
topopt}.py`, `scripts/analysis/turntable_glue.py`, `rfam_eqs_coupled.py`.

### 3.2 Red-first tests, 30 new, 281 passing overall

PROVEN. Each of the three new test files was observed failing with `ImportError` before its
module existed, then passing.

* `tests/test_dwell.py` (19). The simplex projection against a brute-force nearest-point
  search over 4000 random simplex points at 20 random inputs; softmax overflow safety at
  logits of 800; the softmax vector-Jacobian product against a central difference; the cycle
  scheduler's exposure conservation, monotone move times on the control-step grid,
  proportionality, zero-weight dropout, single-position merge, integer apportionment
  (14/13/13 by largest remainder), and rejection of a cycle too short to give each kept
  position a step. Then three physics tests: the **DEGENERACY GATE** (uniform weights
  reproduce `rot_kernel.AveragedKernel` with temperature, density and both averaged heating
  fields BIT IDENTICAL and the map gradient agreeing to below 1e-13 relative), the weight
  gradient against a central difference (below 1e-6), and the **quarter-turn equivariance
  identity** of the weighted kernel.
* `tests/test_dwell_march.py` (8). Step-schedule properties; a program that never moves
  reproducing the single-angle forward BIT FOR BIT; the one-step cycle agreeing with the
  quasi-static average to below 1e-3 in mean part temperature; and the ordering
  `gap(1 step) < gap(10) < gap(50)`, which is what makes the measured departure the
  quasi-static error and not something else.
* `tests/test_dwell_power.py` (3). The FLAG-OFF identity (a flat unit schedule reproduces
  the unscheduled kernel bit for bit and both gradients to below 1e-13 relative); the power
  gradient against a central difference; and the refusal to run when the heating cap binds.
* Whole suite: **281 tests pass** across `fgm_solve_campaign/adjoint2d/tests/` plus the four
  root-level test files, run as
  `PYTHONPATH=$PWD/fgm_solve_campaign:$PWD ./.venv312/bin/python -m pytest ...` (93.8 s).

### 3.3 The gate, four layers, on the cross and the T_shape

Central differences, epsilon swept over the campaign's eight values
(1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8), fixed read index at the base run's argmin.
Pass standard 1e-6 preferred, 1e-5 the campaign's documented subgradient standard. Raw
sweeps in `out_dwell/gate_dwell_{cross,T_shape}.json`.

Layers: **D0** the map gradient at UNIFORM weights, unfiltered (re-gates the inherited
gradient in the new code path); **D1** dJ/dw at NON-UNIFORM weights, probed along simplex
TANGENTS because a single-coordinate move off the simplex is not a legal design move;
**D2** the same through the softmax, which is the gradient L-BFGS-B receives; **D3** the map
gradient at non-uniform weights WITH the 1.0 mm filter.

| shape | layer | max-sensitivity | random | random direction | smooth direction | gradient direction | verdict |
|---|---|---|---|---|---|---|---|
| cross | D0 map, uniform w | 6.12e-07 | 9.37e-06 | 2.04e-07 | | 8.11e-07 | 3/4 at 1e-6, **4/4 at 1e-5** |
| cross | **D1 dwell w** | **1.34e-10** | **2.52e-09** | **5.18e-10** | | **9.02e-10** | **4/4 at 1e-6** |
| cross | **D2 softmax z** | **2.29e-09** | **2.19e-08** | **1.16e-09** | | **3.41e-09** | **4/4 at 1e-6** |
| cross | D3 map, filtered | 6.31e-08 | 9.04e-07 | 3.25e-07 | 8.78e-07 | 7.25e-07 | **5/5 at 1e-6** |
| T_shape | D0 map, uniform w | 5.64e-08 | 4.16e-08 | 3.07e-07 | | 4.20e-07 | **4/4 at 1e-6** |
| T_shape | **D1 dwell w** | **5.97e-10** | **1.47e-07** | **1.06e-08** | | **1.56e-10** | **4/4 at 1e-6** |
| T_shape | **D2 softmax z** | **1.04e-08** | **1.04e-08** | **5.53e-09** | | **1.27e-09** | **4/4 at 1e-6** |
| T_shape | D3 map, filtered | 2.90e-07 | **1.39e-05** | 3.03e-06 | 2.84e-06 | 6.80e-07 | 2/5 at 1e-6, **4/5 at 1e-5, ONE MISS** |

COMPUTED. **The two gradients this pass ADDED, D1 and D2, pass at 1e-6 on both shapes with
best relative errors between 1.6e-10 and 1.5e-07.** They are the cleanest gradients in the
campaign, and for the same reason the power-schedule gradient was: perturbing a dwell moves
every cell in the domain simultaneously, so the clipped melt fraction keeps a large smooth
interior population and the non-smooth boundary contribution is a vanishing share.

**The one miss, and its localization.** T_shape D3's random single-cell probe reads
1.39e-05 against the 1e-5 bar, 39 percent over. Three bisects, each one layer:

| bisect | result |
|---|---|
| same layer, same code, on the **cross** at non-uniform weights | **5/5 at 1e-6** (random cell 9.04e-07) |
| same layer, same shape, at **UNIFORM** weights (`gate_dwell_T_shape_D3b.json`) | **5/5 at 1e-6** (random cell 1.44e-07) |
| same layer, same weights, at a **different read index** (`_D3c.json`), 900 and 1300 | 4/5 and **5/5 at 1e-5**; WHICH probe misses changes, and at read 1300 the gradient-direction probe is 2.28e-08 |

The weighted-map code path is therefore verified independently on both axes, and the miss
MOVES with the read index. Its absolute errors are flat at 1.1e-06 to 1.6e-06 while the
analytic directional derivatives span 0.108 to 15.65, which is precisely the signature
`FROZEN_CONVENTIONS_2D.md` Section 5 item 7 tells us to read as a small-derivative probe
against a fixed evaluation bias, not as a chain-rule error. The decision-relevant
gradient-direction probe at that point is **6.80e-07, PASS at 1e-6**, and the
filter-smooth direction is 2.84e-06, PASS at 1e-5. **Proceeding was a judgement call and it
is named as one.**

### 3.4 A new numerical finding: the objective is not Lipschitz deep into full melt

COMPUTED, and it is the most transferable numerical result in the pass. The densification
solid-state driving term is `(1 - phi)**0.8` (`forward.py:258`,
`dens_phi_solid_exponent = 0.8`). An exponent below one is NOT Lipschitz where the base
reaches zero, and the base reaches exactly zero on every fully melted cell. At a read state
deep into full melt the central-difference error therefore **GROWS as epsilon shrinks** and
no epsilon sweep recovers the analytic derivative.

The first run of the power-schedule gate D4 landed on such a read state (the argmin of a
900-step full-power march sat on the horizon) and produced this, on a gradient that is
correct:

| probe | analytic | best relative error | sweep behaviour |
|---|---|---|---|
| max-sensitivity segment | -124.278 | 3.30e-10 | clean V |
| random segment | -60.092 | **1.99e-02** | fd swings -33.7, -36.0, -61.3, **+154.2** as epsilon falls |
| gradient direction | +224.65 | **1.27e-02** | fd swings 182, 195, 165, **-0.78** |

The decisive control: the **DWELL gradient, already gated at 1e-10 at its own production
read state, shows relative errors of 3.0e-02 to 3.0e+00 at that same 900-step horizon read
state** (fd jumping to 1748 against an analytic 442). The failure belongs to the read state,
not to any channel. Re-running D4 at the production horizon 1500 with the interior argmin
(read index 689 of 1500, `at_horizon = False`) gives:

| D4, production read state | max-sensitivity segment | random segment | random direction | gradient direction | verdict |
|---|---|---|---|---|---|
| power schedule with dwell in force | 3.53e-09 | 2.11e-08 | 1.02e-09 | 2.25e-08 | **4/4 at 1e-6, PASS** |

**Practical rule this establishes for the whole campaign: gate at the read state the
optimizer actually uses, and never at an over-melted horizon.** No optimization in this pass
was run on an ungated gradient; the secondary arm was held until D4 passed.

---

## 4. Real-data gate: the part-frame march against the production engine

The solve lives in the PART frame, where the part never moves and the heating pattern
switches. The production engine does the opposite, rotating the part in the LAB frame and
remapping the thermal fields at every event. The two are compared on the arms the engine can
execute WITHOUT interpolation error: equal dwell over the four multiples of 90 degrees at a
20 s cycle, which is exactly the engine's 90-degree indexing mode with an event every 5 s
(150 events over the horizon), on the two four-fold symmetric shapes. Engine runs go through
`turntable_glue.run_rotating` with the dopant map co-rotated; no engine file was edited.

| shape | map | part-frame J / IoU / stop | engine J / IoU / stop | J gap | IoU gap | engine energy residual |
|---|---|---|---|---|---|---|
| cross | uniform | 125.87 / 0.8803 / 317.0 s | 125.76 / 0.8803 / 316.5 s | **+0.09 %** | **+0.0000** | 0.85 % |
| cross | solved, 4 bpp | 123.81 / 0.8697 / 322.0 s | 123.78 / 0.8697 / 321.5 s | **+0.02 %** | **+0.0000** | 0.79 % |
| square | uniform | 78.56 / 0.9615 / 423.0 s | 78.57 / 0.9615 / 422.5 s | **-0.02 %** | **+0.0000** | 0.43 % |
| square | equal-dwell map, 4 bpp | 18.05 / 1.0000 / 507.0 s | 18.04 / 1.0000 / 506.5 s | **+0.07 %** | **+0.0000** | 0.19 % |
| square | joint map, 4 bpp | 18.05 / 1.0000 / 507.0 s | 18.04 / 1.0000 / 506.5 s | **+0.07 %** | **+0.0000** | 0.19 % |

PROVEN as a real-data check. **J agrees to within 0.09 percent on all five, IoU agrees
exactly, and the stop times agree to one outer step.** Two independent implementations of
the same physics, one of them the production engine.

**What this gate cannot say, stated plainly.** It cannot validate an ASYMMETRIC program,
because the production engine as shipped drives a FIXED rotation increment at a FIXED
interval (`rfam_eqs_coupled.py:2827-2836`) and has no way to express unequal dwells. Its
legacy `phases` branch does take arbitrary angles but spaces its events equally AND raises
`TypeError` on `len(_n_evts)` at `rfam_eqs_coupled.py:2819`, where an `int` is passed to
`len`, so it is unusable as shipped. That is reported rather than worked around, because
working around it would mean editing the engine. **The asymmetric arms are therefore
verified only against the part-frame time-resolved march, whose agreement with the engine is
established above on the schedules both can express.**

---

## 5. The quasi-static step, and the cycle time

The solve optimizes the dwell-WEIGHTED angle average, which is the limit of an infinitely
fast cycle. `dwell_march.program_forward` executes the actual program step by step in the
part frame with NO interpolation anywhere, so the only difference from the quasi-static
forward is the finite cycle time. COMPUTED, on the solve-stage maps
(`figs_dwell/fig_dwell_cycle_time.png`, `out_dwell/<shape>_dwell.json` key `cycle_sweep`):

| shape | cycle 10 s | 20 s | 40 s | 80 s | 160 s |
|---|---|---|---|---|---|
| cross | -2.04 % | -2.11 % | -2.20 % | -1.15 % | +2.88 % |
| square | -3.75 % | -3.83 % | -0.09 % | +15.19 % | **+51.25 %** |
| T_shape | -0.22 % | -0.82 % | -1.63 % | -3.31 % | -6.15 % |
| L_shape | +0.00 % | -0.00 % | +0.00 % | +0.00 % | -0.00 % |

**At the production cycle time of 20 s the quasi-static approximation is good to 4 percent
or better on every shape**, and every one of these arms PASSES the energy gate, which is a
direct improvement on the continuous-rotation pass: that pass measured 5 to 19 percent
energy loss at fast rotation and traced it to the engine's bilinear rotation-event remap.
Staying in the part frame removes that error class entirely. The L_shape's exact zeros are
the half-turn redundancy of Section 2, not a bug. The square degrades hardest at long cycle
times because its melt is nearly perfect and any azimuthal transient shows up immediately.

**Cycle time chosen: 20.0 s** = 40 control steps at dt 0.5 s, giving a dwell quantum of
2.5 percent of a cycle and 37.5 cycles over the 750 s horizon. Stated, not fitted.

---

## 6. Per-shape tables, all arms, at each arm's own stop

Read: J is the whole-domain shape objective; growth, under-melt, mean relative density and
maximum temperature are at that arm's own stop; `Eres` is the energy-gate residual there.
Dwell is on the FOUR DISTINCT positions {0, 45, 90, 135} degrees.

### cross (part cells 1036)

| arm | J | IoU | grow % | under % | mean rho | P abs W/m | stop s | max T C | Eres | dwell |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform map, equal dwell | 242.77 | 0.7760 | 18.92 | 7.72 | 0.8010 | 438.7 | 380.5 | 234.0 | 1.41 % | equal |
| stored zero-degree static map, equal dwell | 286.30 | 0.7274 | 17.57 | 14.48 | 0.8082 | 307.5 | 551.5 | 214.4 | 1.18 % | equal |
| **CONTROL** solved map, equal dwell, 4 bpp | 130.47 | 0.8384 | 8.69 | 8.88 | 0.7330 | 324.6 | 495.5 | 200.5 | 0.67 % | equal |
| **CONTROL** the same, time resolved | 117.72 | 0.8444 | 7.34 | 9.36 | 0.7236 | 324.6 | 487.5 | 199.3 | 0.48 % | equal |
| **DELIVERABLE** dwell schedule, 4 bpp | 34.20 | **0.9848** | 1.54 | **0.00** | 0.7255 | 364.9 | 470.0 | 195.7 | 0.37 % | 0.5/0/0.5/0 |
| **DELIVERABLE** the same, time resolved | **34.04** | **0.9829** | 1.74 | **0.00** | 0.7295 | 364.9 | 471.5 | 195.7 | 0.43 % | 0.5/0/0.5/0 |

### square (part cells 1600)

| arm | J | IoU | grow % | under % | mean rho | P abs W/m | stop s | max T C | Eres | dwell |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform map, equal dwell | 82.60 | 0.9579 | 4.00 | 0.38 | 0.7133 | 465.3 | 443.5 | 204.0 | 0.44 % | equal |
| stored zero-degree static map, equal dwell | 45.07 | 0.9667 | 1.50 | 1.88 | 0.7272 | 405.4 | 522.5 | 191.3 | 0.23 % | equal |
| **CONTROL** solved map, equal dwell, 4 bpp | 16.51 | **1.0000** | 0.00 | 0.00 | 0.7741 | 421.0 | 524.5 | 201.9 | 0.20 % | equal |
| **CONTROL / DELIVERABLE**, time resolved | **15.96** | **1.0000** | **0.00** | **0.00** | 0.7721 | 421.0 | 522.5 | 201.6 | 0.17 % | equal |

The square's deliverable and control are the SAME ARM: the free eight-parameter dwell search
returned exactly equal dwell, so the two collapse. That is the control passing, not a
missing run.

### T_shape (part cells 1104)

| arm | J | IoU | grow % | under % | mean rho | P abs W/m | stop s | max T C | Eres | dwell |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform map, equal dwell | 527.31 | 0.5721 | 21.92 | 30.25 | 0.7445 | 359.5 | 526.5 | 227.4 | 1.55 % | equal |
| stored zero-degree static map, equal dwell | 525.00 | 0.5721 | 21.92 | 30.25 | 0.7451 | 359.2 | 528.0 | 227.2 | 1.55 % | equal |
| **CONTROL** solved map, equal dwell, 4 bpp | 514.73 | 0.5747 | 20.11 | 30.98 | 0.7415 | 352.5 | 536.0 | 224.8 | 1.45 % | equal |
| **CONTROL** the same, time resolved | 513.49 | 0.5710 | 19.93 | 31.52 | 0.7391 | 352.5 | 532.5 | 224.8 | 1.42 % | equal |
| **DELIVERABLE** joint dwell + map, 4 bpp | **422.11** | 0.6231 | 16.30 | 27.54 | 0.7402 | 313.1 | 647.0 | 217.9 | 1.08 % | 0.21/0.03/0.74/0.03 |
| refined map at that dwell, 4 bpp | 426.21 | 0.6415 | 15.22 | 26.09 | 0.7450 | 306.2 | 671.5 | 218.7 | | 0.21/0.03/0.74/0.03 |
| refined, time resolved | 422.28 | **0.6436** | 15.40 | 25.72 | 0.7469 | 306.2 | 672.5 | 219.3 | | 0.21/0.03/0.74/0.03 |

The J-best and IoU-best T_shape arms disagree, which the objective permits and which is left
visible rather than resolved by quoting only one.

### L_shape (part cells 1079)

| arm | J | IoU | grow % | under % | mean rho | P abs W/m | stop s | max T C | Eres | dwell |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform map, equal dwell | 422.73 | 0.6319 | 19.83 | 24.28 | 0.7511 | 365.6 | 498.0 | 226.0 | 1.40 % | equal |
| stored zero-degree static map, equal dwell | 407.84 | 0.6344 | 17.61 | 25.39 | 0.7461 | 355.7 | 509.0 | 221.9 | 1.25 % | equal |
| **CONTROL** solved map, equal dwell, 4 bpp | 411.98 | 0.6327 | 18.35 | 25.12 | 0.7457 | 361.4 | 499.5 | 222.8 | 1.29 % | equal |
| **CONTROL** the same, time resolved | 407.81 | 0.6359 | 18.63 | 24.56 | 0.7491 | 361.4 | 502.5 | 224.2 | 1.31 % | equal |
| **DELIVERABLE** static at 135 deg, 4 bpp | **396.20** | **0.6484** | 16.50 | 24.47 | 0.7191 | 496.4 | 304.5 | **246.8** | 1.37 % | 0/0/0/1.0 |

The L_shape's deliverable runs at 496.4 W/m and reaches 246.8 C, the hottest arm in the
pass, 3.2 C under the 250 C operating ceiling. An earlier cold-start variant of the same arm
reached **251.7 C and is FLAGGED as over the ceiling**; it is not quoted as a deliverable.

**Operating ceiling, 250 C, across all scored arms.** One breach in the pass, named above.
Every arm quoted in Section 1 sits between 195.7 and 246.8 C.

**Mean relative density at the deliverables' own stops spans 0.7191 to 0.7741.** These are
melt-stop reads, not end-of-horizon reads, and nothing here is an oversinter claim.

---

## 7. The finding that decided the cross, and it is not about dwell

COMPUTED, and it is a methodological result the whole campaign should carry. The cross's map
solve is **START DOMINATED**, by a factor of 3.7 in J.

With the dwell FIXED at the schedule the optimizer discovered, and everything else identical
(16 gradient evaluations, 1.0 mm filter, box [0, 1]):

| start | first evaluation | best after 16 evaluations | deliverable IoU |
|---|---|---|---|
| cold, uniform saturation | 128.71 | **128.56**, a dead solve | 0.8768 |
| warm from the best stored ZERO-DEGREE library map | 187.39 | **34.39** | **0.9848** |

The cold arm moves 0.1 percent in sixteen evaluations. This exactly reproduces the
continuous-rotation pass's own trace for the matched four-angle kernel
(`out_rot/cross_rotavg_step90.json`: cold 128.71 to 126.80, warm 196.81 to 34.99), and this
pass's kernel reproduces that pass's stored map to **34.99 against 34.99**, so the two
implementations agree exactly and the gap is optimization, not physics.

Two controls rule out the obvious alternative explanation:

* **The filter radius is NOT the cause.** Re-running the cold arm at sigma = 1.5 cells
  (0.75 mm, the continuous-rotation pass's radius) instead of 1.983 cells (1.0 mm, the
  frozen convention) gives 126.80 against 128.56. A 1.4 percent effect, not a factor of 3.7.
* Every equal-dwell CONTROL in Section 6 was given the SAME library-warm start, so no
  comparison in this report is won by warm-starting one side only.

**Consequence for `FROZEN_CONVENTIONS_2D.md` Section 6**: the frozen recipe's "single cold
start" is the wrong default on the rotating and dwell kernels. The library-warm start is
worth more than the entire dwell actuator on the cross, and it costs the same 16 gradient
evaluations.

---

## 8. The one labelled secondary arm: does p(t) add anything on top of the dwell?

Power scheduling p(t) is DEPRIORITIZED. This is the single counterexample check, on the
cross, because that is the one shape whose optimized power schedule previously produced a
real 211 s generator OFF period whose benefit survived a dose-matched control
(`TEMPORAL_SCHEDULING_REPORT.md` Sections 5 and 8: +0.0413 IoU from temporal structure,
+0.0000 from dose). Run on the cross's dwell deliverable and its 4 bpp map, 12 segments over
750 s, box [0, 1.5], 12 gradient evaluations, gradient FD gated at 1e-6 (Section 3.4).

| arm | J | IoU | duty | switches | mean rho | stop s | max T C | structure |
|---|---|---|---|---|---|---|---|---|
| P0 dwell only, generator at nominal | 34.20 | 0.9848 | 1.000 | 0 | 0.7397 | 470.0 | 195.7 | none |
| P1 power schedule optimized, dwell fixed | **33.98** | **0.9885** | 0.983 | 2 | 0.7206 | 486.0 | 195.6 | FLAT |
| P2 power and dwell co-optimized | 33.98 | 0.9885 | 0.983 | 2 | 0.7206 | 486.0 | 195.6 | FLAT |
| P3 dose-matched control, constant p = 0.9828 | 34.69 | 0.9791 | 0.983 | 0 | 0.7287 | 482.0 | 195.6 | none |

COMPUTED. **Total gain +0.0038 IoU, decomposing as +0.0094 structure and -0.0057 dose.**
The optimized schedule sits at duty 0.983, that is essentially nominal power, makes only two
level changes, classifies FLAT, and contains **no generator OFF period at all**. The cross's
previous melt-erase-remelt cycle does not reappear.

**Verdict on the counterexample check: p(t) adds essentially nothing once the dwell schedule
is optimized.** The 0.6 percent J improvement is within the spread of a re-solve. This
supports the deprioritization rather than contradicting it. The honest caveat is that the
previous OFF-period result was found on a much weaker cross arm (IoU 0.75 at the 2500-step
horizon); at IoU 0.985 with zero under-melt there is very little for an eraser to erase.

---

## 9. Machine-readable deliverables, the slicer-to-machine path

Each shape emits two turntable programs as JSON, in
`fgm_solve_campaign/out_dwell/<shape>_turntable_{deliverable,equal_dwell_control}.json`.
Each carries the ordered list of `{position_deg, dwell_s, move_at_s}`, the cycle time, the
control step, the realized against requested dwell fractions, the recommended stop time, the
constant radio-frequency program with its calibrated drive voltage, and a pointer to the
dopant map array (`<shape>_dwell_maps.npz`, key `sat_<arm>`, 4 bits per pixel inside the
part). The REDUCED program on the distinct positions is emitted alongside the literal one.

| shape | reduced positions (deg) | dwell fraction | moves over 750 s | recommended stop |
|---|---|---|---|---|
| cross | 0, 90 | 0.500, 0.500 | 75 | 471.5 s |
| square | 0, 45, 90, 135 | 0.250 each | 150 | 522.5 s |
| T_shape | 0, 45, 90, 135 | 0.225, 0.025, 0.725, 0.025 | 151 | 647.0 s |
| L_shape | 135 | 1.000 | **1** | 304.5 s |

The L_shape's program is a single hold: the slicer's honest instruction for that geometry is
"orient at 135 degrees and do not rotate".

---

## 10. Honest limits

1. **One gate probe misses the 1e-5 subgradient standard** (T_shape D3, 1.39e-05). Localized
   by three bisects to the read state and not to the dwell channel, and the decision-relevant
   probe there is 6.80e-07. Proceeding was a judgement call.
2. **The gate was run on two shapes.** The square and the L_shape are ASSUMED to inherit it,
   on the same reasoning the joint and rotation campaigns used: the dwell changes the
   weights and the mask set, not the chain rule.
3. **The asymmetric programs are NOT verified on the production engine**, because the engine
   as shipped cannot execute an unequal dwell (Section 4). The part-frame march that runs
   them is verified against the engine on the schedules both can express, to 0.09 percent.
4. **The candidate set was 8 positions at 45 degrees, the same for every shape.** It was NOT
   swept. Section 2 shows it carries only four distinct patterns, so the real resolution is
   45 degrees over a half turn. A finer set might find a better park angle, particularly for
   the T_shape and the L_shape whose optima sit AT a candidate.
5. **The dwell durations are constant over the exposure.** The schedule repeats the same
   cycle from start to stop. A dwell pattern that CHANGES with time (park at 90 early, at 0
   late) is a strictly larger design space and was not searched. That is the most obvious
   thing the T_shape might want and it is not tested here.
6. **Budget accounting, stated rather than hidden.** The equal-budget joint arm is
   16 gradient evaluations, which is the campaign's 40 forward-equivalent budget. The
   DELIVERABLE arms quoted in Section 1 spent more: 16 for the control solve, 16 for the
   joint dwell discovery, and 16 more for the map refinement at the fixed dwell, from each of
   two starts. Every equal-dwell control was given the identical treatment, so the
   comparison is like for like, but no arm in Section 1 is a 40 forward-equivalent result.
7. **Not dose matched** (313.1 to 496.4 W/m on the deliverables).
8. **The cycle-time sweep was measured on the solve-stage maps**, before the refinement, so
   its percentages describe the approximation and not the final arms.
9. **Move duration and mechanical settling are not modelled.** A move is instantaneous and
   costs nothing. The T_shape program asks for 151 moves in 750 s.
10. **Conductivity channel only, single grid (120 x 120), two dimensions, no grid hold-out
    and no sub-filter-radius perturbation gate.** Neither Gate A nor Gate B of
    `FROZEN_CONVENTIONS_2D.md` Section 8 was run, so **no map here carries a SOLVED label**
    even though the cross and the square exceed the 0.95 threshold in grid.
11. **Model, not hardware.** `ALLISON_LAW_REPLICATION.md` Section 6.1 records that the
    two-dimensional model over-predicts achievable tuned uniformity against hardware by
    roughly a factor of eight.

---

## 11. Proven, computed, assumed

**PROVEN**
* The degeneracy of the weighted kernel at uniform weights against the already-gated
  `AveragedKernel`: temperature, density and both averaged heating fields bit identical,
  map gradient below 1e-13 relative.
* The flag-off identity of the power channel: bit identical forward, both gradients below
  1e-13 relative.
* A program that never moves reproducing the single-angle forward bit for bit.
* The dwell gradient dJ/dw and its softmax composition dJ/dz, FD gated at 1e-6 on two shapes,
  best relative errors 1.6e-10 to 1.5e-07, on simplex-tangent and unconstrained probes.
* The power-schedule gradient with a dwell in force, FD gated at 1e-6 at the production read
  state.
* The half-turn redundancy of the candidate set, to 4e-13 relative on all four pairs.
* The part-frame time-resolved march against the PRODUCTION ENGINE: J to 0.09 percent, IoU
  exact, stop times to one outer step, on five arm-and-map combinations.
* This pass's kernel reproducing the continuous-rotation pass's stored four-angle map score
  to 34.99 against 34.99.
* 281 tests, 30 of them new and each observed red first.

**COMPUTED**
* Every number in Sections 1, 2, 5, 6, 7, 8 and 9.

**ASSUMED**
* That the gate measured on the cross and the T_shape holds on the square and the L_shape.
* That the rasterized binary part mask is the right nominal target, carried over from the
  library, orientation, joint and rotation campaigns.
* That an arbitrary stop time is realizable as a process control.
* That the quasi-static weighted average is the right design model for a fast indexed
  turntable. MEASURED to 4 percent or better at the 20 s cycle in Section 5, and not
  measured outside the sampled cycle times.
* That a turntable move is instantaneous and free.

---

## 12. The single most valuable next layer

**Make the dwell schedule TIME VARYING.** Every limit in Section 10 that bites the stranded
shapes points the same way: the current design variable is one dwell vector held constant
for the whole exposure, and the T_shape's answer, 74 percent at 90 degrees, is a compromise
between two limbs that want different orientations at different times. The natural extension
is a dwell vector per time SEGMENT, `w(t)`, which is the outer product of the actuator this
pass proved and the segmentation `schedule.py` already provides. Its gradient is the same
angle-segment inner product restricted to each time segment, so it comes out of the SAME
backward sweep with no new physics and no new transpose, and the FD gate is the one already
written. This is the cheapest remaining move with a real chance on the T_shape.

**Second: change the campaign's default start.** Section 7 shows the library-warm start is
worth a factor of 3.7 in J on the cross at identical cost, and that the frozen "single cold
start" convention is what was hiding the cross's SOLVED-class result. Re-running the
continuous-rotation and joint-angle censuses with a two-start recipe is pure re-scoring.

**Third: fix `rfam_eqs_coupled.py:2819`** (`len` of an int) and give the turntable block a
per-event angle and interval list. It is a small change to a broken code path, it defaults
off, and it is the only thing standing between the asymmetric programs in Section 9 and a
production-engine verification.

---

## 13. Artifacts, absolute paths, and wall time

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New code:
`fgm_solve_campaign/adjoint2d/{dwell,dwell_kernel,dwell_march,dwell_power,gate_dwell,gate_dwell_power}.py`;
`fgm_solve_campaign/adjoint2d/tests/{test_dwell,test_dwell_march,test_dwell_power}.py`;
`scripts/analysis/{run_dwell_solve,run_dwell_refine,run_dwell_engine_check,run_dwell_power_arm,finalize_dwell,make_dwell_figures}.py`.

Results, all under `fgm_solve_campaign/out_dwell/`:
* `gate_dwell_cross.json`, `gate_dwell_T_shape.json`, `gate_dwell_T_shape_D3b.json`,
  `gate_dwell_T_shape_D3c.json`, `gate_dwell_power_cross.json` the gates with full epsilon
  sweeps
* `<shape>_dwell.json` every arm, every solve trace, the cycle sweep, the final selection
* `<shape>_dwell_maps.npz` dopant maps, melt fields at each arm's own stop, objective curves
* `<shape>_turntable_deliverable.json`, `<shape>_turntable_equal_dwell_control.json` the
  machine-readable turntable programs
* `<shape>_engine_check.json` the real-data gate against the production engine
* `cross_dwellpower.json`, `cross_dwellpower_fields.npz` the secondary power arm
* `fgm_solve_campaign/logs_dwell/*.log` per-run console logs

Figures, **all six viewed before delivery**, in `fgm_solve_campaign/figs_dwell/`:
* `fig_dwell_<shape>.png` (four) the per-shape composite: dwell schedule on the distinct
  positions, the turntable program as a timeline bar, the objective against time with each
  arm's own stop, the co-solved dopant map, and melt against nominal at the stop for the
  equal-dwell control and the dwell deliverable side by side
* `fig_dwell_census.png` all four shapes, where the turntable parks and what it buys
* `fig_dwell_cycle_time.png` the measured quasi-static approximation error

**Wall time, COMPUTED and logged progressively.** After the first shape finished (the cross
solve, 570 s for 48 gradient evaluations plus 12 scoring forwards and a 5-point cycle sweep)
the projection for the other three at 600 to 850 s each in parallel pinned single-thread
streams was 15 minutes; actual 694 to 824 s, inside that. The gates were projected from the
cross's 682 s at a 400-step horizon to 1700 s for the T_shape at 1400 steps; actual 1708 s.
**Recorded process time 7415 s across the JSON-stamped runs, plus roughly 3800 s of
refinement and bisect runs not individually stamped, about 3.1 hours of process time and
2.1 hours of wall clock in up to four pinned single-thread streams**
(`fgm_solve_campaign/env1.sh`, every numerical library pinned to one thread).

**Nothing was cut.** Two runs were discarded and re-run rather than reported: a first
multi-start refinement launch that died on a shell word-splitting error before doing any
numerical work, and a first secondary-power arm that ran against the pre-refinement cross
map. Both are named here and neither contributed a number to this report.
