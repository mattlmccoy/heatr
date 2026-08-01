# Continuous rotation and the ideal dopant map, two levels

**Date:** 2026-08-01. **Scope:** the direct test of whether a turntable, combined with a
dopant map solved for the rotating heating, can ELIMINATE rather than relocate the
perpendicular-limb penalty on the stranded shapes. Level 1 solves the map against a
rotationally averaged heating kernel in the part frame (the quasi-static limit). Level 2
verifies the winners on the production engine's own turntable machinery with the dopant
map co-rotated at every rotation event, and measures the quasi-static approximation error.
Five shapes: T_shape, L_shape, cross, star, and the square as the sanity control. Grid
120 x 120 throughout. **Nothing was committed. No dissertation file was touched.
`.claude/worktrees/` was not read or written. No line of `rfam_eqs_coupled.py` was
edited.**

**Acronyms, expanded on first use.** FGM = functionally graded material (a spatially
varying dopant saturation map). EQS = electro-quasi-static (the low-frequency Maxwell
approximation the two-dimensional solver uses). IoU = intersection over union. bpp = bits
per pixel. L-BFGS-B = limited-memory Broyden-Fletcher-Goldfarb-Shanno with box
constraints. FD = finite difference. phi = melt fraction. rho = relative density. W/m =
watts per metre of depth. sigma = electrical conductivity. eps_r = relative permittivity.

**Evidence tags.** PROVEN = unit tested, FD gated, or reproduced against a stored number.
COMPUTED = measured from a run in this pass. ASSUMED = a modelling choice or an inference
not measured here.

**Objective and stop convention, stated once and carried on every number.**

    J_phi(t) = sum over the WHOLE domain of (phi(x, t) - chi_part(x; theta(t)))^2

with `chi_part(.; theta(t))` the binary part mask the engine has rasterized at the
orientation in force at time t, so the nominal target CO-ROTATES with the part. Every
J_phi, IoU, growth, under-melt, mean relative density, maximum temperature and energy
residual below is read at that arm's OWN J-stop, `t_stop` = argmin of J_phi over that
arm's own stored trajectory on a 1500-step horizon (dt 0.5 s, 750 s). `HORIZON` is flagged
whenever the minimum sits on the last stored step, which makes that arm's J_phi a BOUND
and not a value. The melted region for IoU, growth and under-melt is phi >= 0.5; J_phi
itself uses no threshold. These are the exact conventions of `JOINT_ANGLE_MAP_REPORT.md`,
`SHAPE_LIBRARY_SOLVE_REPORT.md` and `MULTISTART_REPORT.md`.

**Grid qualifier, mandatory.** `SOLVE_ROBUSTNESS_VALIDATION.md` established that absolute
fidelity at grid 120 does not transfer to grid 160 and that the forward itself is not
converged in IoU between those grids. Every IoU here is a property of the method AT GRID
120 with an exactly reproduced dopant map, not a property of the method. No grid-160
number appears anywhere in this report.

**Energy gate, on every scored run.** Standing threshold 5 percent of integrated dose,
evaluated at that arm's own stop index. Gate failures are reported in the tables and no
verdict below rests on a failing arm.

---

## 1. Verdict per shape, first

The comparison the task asked for: the best STATIC arm on record (the joint campaign's
winning angle plus its winning map, re-measured here on the production engine) against the
best rotating arm that PASSES the energy gate. Both columns are engine numbers under
identical conventions.

| shape | best static arm | best gate-passing rotating arm | J change | IoU change | verdict |
|---|---|---|---|---|---|
| **cross** | 147.03, IoU 0.8514 | **34.82, IoU 0.9866** (90-degree indexing, solved 4-angle map) | **-76.3 %** | +0.135 | **rotation WINS decisively, and the cross crosses IoU 0.95 for the first time** |
| **star** | 106.28, IoU 0.7870 | **24.46, IoU 0.9527** (90-degree indexing, solved 4-angle map) | **-77.0 %** | +0.166 | **rotation WINS decisively, and the star crosses IoU 0.95** |
| **square** (control) | 25.56, IoU 0.9804 | **13.93, IoU 1.0000** (90-degree indexing, solved 4-angle map) | **-45.5 %** | +0.020 | **rotation WINS; melt region matches the nominal part exactly at grid 120** |
| **T_shape** | 522.28 (HORIZON, a bound), IoU 0.5356 | 467.33, IoU 0.5516 (continuous, period 720 s, solved 24-angle map) | -10.5 % | +0.016 | rotation wins slightly; the perpendicular-limb penalty is **NOT erased** |
| **L_shape** | 376.94, IoU 0.6693 | 352.97, IoU 0.6581 (continuous, period 360 s, solved 24-angle map) | -6.4 % | **-0.011** | **effectively a tie**: better J, worse IoU |

COMPUTED, all five. One-line reading per shape:

1. **cross: the biggest result in the pass.** J_phi falls from 147.03 to 34.82 and IoU
   rises from 0.8514 to 0.9866, which is the FIRST time anything in this campaign has put
   a stranded shape into the absolute SOLVED class (IoU >= 0.95) in the deployment-safe
   conductivity-only channel. Growth 1.16 percent, under-melt 0.19 percent, energy
   residual 0.37 percent. The 4 bpp printable version is 34.76 / 0.9847, so the win
   survives quantization essentially unchanged.
2. **star: the same story.** 24.46 / 0.9527 against the joint static winner's 106.28 /
   0.7870, energy residual 0.45 percent, 4 bpp 24.56 / 0.9527. On the star the DOPANT MAP
   is nearly inert under this actuator: the uniform arm at the same schedule gives 23.72 /
   0.9726, which is BETTER on J_phi. **The star's win belongs to the rotation, not to the
   grading.** That is stated plainly because it is the honest reading.
3. **square, the sanity control: it behaves exactly as the physics expectation predicted.**
   The averaged kernel is annular (Section 4 quantifies it), the solved map is a nearly
   uniform field with four corner corrections rather than the electrode-axis edge bands of
   the static map (see `fig_rot_maps_square.png`), and the melt region matches the nominal
   part with zero growth and zero under-melt.
4. **T_shape: the honest negative.** Rotation improves J_phi by 10.5 percent and IoU by
   0.016, but under-melt at the best rotating arm is still 42.91 percent of the part and
   IoU stays at 0.55. **Continuous rotation does NOT erase the perpendicular-limb penalty;
   it SPLITS it between the two limbs** (Section 3, with the limb-resolved numbers).
5. **L_shape: no result.** 6.4 percent better on J_phi, 0.011 worse on IoU, which is
   inside the noise of this comparison. The L_shape is where the joint static winner
   already balanced the two limbs, so there is nothing left for rotation to equalize.

**THE headline answer to the task's question.** Continuous rotation plus a solved map beats
the joint static winner on all four stranded shapes on J_phi, and by a factor of three or
more on two of them. But **the mechanism is not the one the question assumed.** Rotation
does not erase the perpendicular-limb penalty. It makes the heating kernel RADIAL, and a
radial kernel melts a rounded blob. That is a large gain on shapes that are close to
radially symmetric (star, square, cross) and almost nothing on shapes that are not
(T_shape, L_shape). Section 4 measures the residual azimuthal anisotropy of the averaged
kernel and it predicts the ranking exactly.

**At what rotation speed?** COMPUTED, and the answer is a genuine surprise:

* **Continuous 15-degree rotation is not the best actuator.** For every shape except the
  L_shape the best arm is **90-degree INDEXING**, a quarter turn every 2.0 s, not a fine
  continuous sweep. On the cross this is 34.82 against 97.27 for the best continuous arm,
  a factor of 2.8.
* Within continuous 15-degree rotation the optimum period is shape dependent and is set by
  two errors pulling in opposite directions (Section 5): star and square have a clear
  minimum near 24 to 60 s, the cross near 180 s, the T_shape at 720 s.
* **Every 15-degree arm faster than about 60 s FAILS the energy gate**, and Section 6
  proves that this is the engine's own rotation-event remap, not physics.

---

## 2. What was built, and the finite-difference gate

**New code (drivers, one new solver module, tests; no existing solver internals modified).**

* `fgm_solve_campaign/adjoint2d/rot_frame.py` the part-frame to lab-frame rotation as an
  ASSEMBLED sparse operator with an exact transpose, plus `averaging_angles`.
* `fgm_solve_campaign/adjoint2d/rot_kernel.py` the rotationally averaged kernel: the
  averaged forward, its reverse march, and the adjoint through both rotation transposes
  and the per-angle EQS adjoint.
* `fgm_solve_campaign/adjoint2d/gate_rot.py` the finite-difference gate.
* `scripts/analysis/turntable_glue.py` the run-script glue for the true rotating engine.
* Drivers: `run_rot_avg_solve.py`, `run_rot_verify.py`, `run_rot_speed_sweep.py`,
  `run_rot_headline.py`, `build_rot_tables.py`, `make_rot_figures.py`.

**Red-first tests, 40 passing.** PROVEN.

* `fgm_solve_campaign/adjoint2d/tests/test_rot_frame.py` (9). Observed failing with 9
  `ModuleNotFoundError` before `rot_frame.py` existed, then 9 passing. Covers identity at
  zero degrees, the EXACT pixel permutation at 90, 180 and 270 degrees, agreement with the
  already proven production-convention helper `rotate_sat_map` at a general angle to
  1e-9 in the interior, the dot-product transpose identity to 1e-10, the affine `outside`
  offset, the interior partition of unity to 1e-12, the round trip, and the angle-set
  rules.
  One real bug was caught by the red test and would have been invisible otherwise:
  `math.sin(math.pi)` is -1.2e-16, not zero, which pushed the source point of an edge cell
  a hair outside the array extent and silently zeroed an entire row, so the 180-degree case
  stopped being a pixel permutation. Fixed by exact trigonometry at multiples of 90 degrees.
* `fgm_solve_campaign/adjoint2d/tests/test_rot_kernel.py` (4). Red first. The strongest of
  these is a DEGENERACY gate: with a single averaging angle at zero degrees the averaged
  forward must reduce BIT FOR BIT to `adjoint2d.forward.forward` and its adjoint to
  `adjoint2d.adjoint.gradient`. Measured: temperature and density fields bit identical,
  gradient relative difference **below 1e-12**.
* `test_turntable_glue.py` (8). Red first, 7 `ModuleNotFoundError` before the module
  existed. Covers the metric arithmetic, the co-rotation at 90 degrees, the stop rule, the
  stop-time field snapshot, and two real-engine integration tests.
* `test_orientation_map_rotation.py` (5) and `test_joint_angle_lib.py` (14) still pass.
* **Whole-suite check: 213 tests pass** across `fgm_solve_campaign/adjoint2d/tests/` plus
  the four root-level test files, run as
  `PYTHONPATH=$PWD/fgm_solve_campaign:$PWD ./.venv312/bin/python -m pytest ...`. That
  PYTHONPATH is required by the PRE-EXISTING adjoint2d tests, which import `adjoint2d`
  directly and raise 15 collection errors without it; the two new test files set their own
  path and collect either way. Nothing in this pass changed that condition.

**The finite-difference gate on the averaged-kernel gradient.** Run on the cross, grid 120,
24 averaging angles, fixed read index, central differences, epsilon swept over
(1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8). Full sweeps in
`fgm_solve_campaign/out_rot/gate_rot_cross.json`.

| layer | probes at 1e-6 | probes at 1e-5 | gradient-direction probe | verdict |
|---|---|---|---|---|
| R0 single angle at zero degrees, unfiltered (the INHERITED path) | 3/4 | 3/4 | 8.46e-07 | see note |
| R1 full 24-angle set, unfiltered | **4/4** | **4/4** | **1.07e-07** | **PASS at 1e-6** |
| R2 full 24-angle set, WITH the design filter (the gradient the solve uses) | 4/5 | **5/5** | **1.68e-07** | **PASS at the 1e-5 subgradient standard** |

COMPUTED. Every probe shows the expected V shape, bottoming between 1e-6 and 1e-4 and
rising into the roundoff tail below.

Two things must be said plainly about this table. **The two ROTATION layers, R1 and R2, are
both cleaner than the inherited single-angle layer R0**, which is the right signature: the
angle average smooths the objective. R2's single miss at 1e-6 is the maximum-sensitivity
single-cell probe at **1.03e-06**, three percent over the bar, and its decision-relevant
gradient-direction probe is 1.68e-07. **R0's miss is at 1.13e-05 on the random single-cell
probe and it is NOT a rotation layer**: with one angle at zero degrees the kernel IS the
ordinary forward, so that number is a property of the inherited melt objective at this
design point, not of anything added here. Its analytic derivative is -0.569 against a
measured objective evaluation floor of 3.7e-11, which puts the arithmetic floor of that
probe at 3.3e-6, within a factor of three of what was measured. The campaign's documented
subgradient standard is 1e-5 (`SOLVE_ROBUSTNESS_VALIDATION.md` Section 7, 1.22e-05 on its
random-direction probe).

**Not gated:** the gate was run on the cross only, and the stop-index stability check
reports the base read index at the horizon of the 400-step gate run, so the gate is a
fixed-read-index gate (which is the correct layer, since the envelope theorem argument is
not needed at a fixed index). ASSUMED that the gate transfers to the other four shapes, on
the same reasoning the joint campaign used: the rotation changes the mask set and the
operator assembly per angle, not the chain rule.

---

## 3. Level 1: the rotationally averaged kernel, and the limb question

**Construction.** For each of 24 angles spaced 15 degrees over a FULL turn (the step of the
stored near-continuous turntable schedule, `configs/diamond_tt_15deg_48rot_nearcont.yaml`),
the part-frame design map is rotated into the lab frame, the part is rasterized at that
angle, the EQS problem is solved, and the resulting Q_rf is rotated BACK into the part
frame. The average is the kernel. It is an OPERATOR in the design variable, not a fixed
array: a fixed array evaluated once at a uniform map would make the design variable inert.
A full turn and not the shape's symmetry period, because the solved map is not constrained
to carry the shape's symmetry.

**Recipe.** Physical-length design filter at sigma = 1.5 cells (0.75 mm at this grid,
MANDATORY), box [0, 1], L-BFGS-B on the FD-gated filtered averaged-kernel gradient,
16 gradient evaluations per start (the campaign's 40 forward-equivalent depth budget), two
starts (cold uniform, warm from the best stored zero-degree map), better kept.
Conductivity channel only, which is the deployment-safe one given the open eps_r material
question in `EPS_CHANNEL_REPORT.md` Section 11.

| shape | uniform under the averaged kernel | best stored STATIC map under the averaged kernel | SOLVED averaged-kernel map | at 4 bpp |
|---|---|---|---|---|
| T_shape | 559.00, IoU 0.5543 | 557.82, 0.5528 | **540.20, 0.5577** | 540.18, 0.5577 |
| L_shape | 469.30, 0.6024 | 454.58, 0.6056 | **448.77, 0.6114** | 448.79, 0.6117 |
| cross | 258.29, 0.7660 | 284.86, 0.7361 | **192.02, 0.8085** | 192.08, 0.8085 |
| star | 27.10, 0.9468 | 81.91, 0.7973 | **24.63, 0.9534** | 24.67, 0.9500 |
| square | 29.36, 0.9877 | 43.26, 0.9537 | **16.04, 1.0000** | 16.07, 1.0000 |

COMPUTED. Energy gate passes on all 20 arms; maximum temperature clip fraction, temperature
clamp fraction and Q_rf cap fraction are all zero. Two readings worth naming.
**Re-solving matters:** on the star the stored static map is actively HARMFUL under
rotation, 81.91 against the uniform arm's 27.10, because it de-dopes the star's centre to
compensate a static field that no longer exists. And **the 4 bpp printable version costs
essentially nothing** anywhere in Level 1, between -0.2 and +0.2 percent.

### Does the T_shape crossbar melt? The limb-resolved answer

COMPUTED. The part is split by row width into the wide BAR (the T's crossbar, the L's foot;
576 and 520 cells) and the STEM (528 and 559 cells). Percentage of cells in each limb that
are melted (phi >= 0.5) at that arm's own J-stop:

| shape | arm | bar melted | stem melted |
|---|---|---|---|
| T_shape | static uniform at 0 degrees | **4.5 %** | 94.7 % |
| T_shape | joint static winner at 90 degrees, back-rotated | **98.3 %** | **8.5 %** |
| T_shape | rotationally averaged kernel, solved map | **56.9 %** | 78.8 % |
| L_shape | static uniform at 0 degrees | 13.7 % | 94.3 % |
| L_shape | joint static winner at 135 degrees, back-rotated | 74.0 % | 78.9 % |
| L_shape | rotationally averaged kernel, solved map | 64.8 % | 79.6 % |

This is the mechanism, in three rows. At zero degrees the T_shape's crossbar essentially
does not melt: 4.5 percent. **That is the perpendicular-limb penalty.** The joint static
winner does not remove it, it MOVES it: at 90 degrees the crossbar reaches 98.3 percent and
the stem collapses to 8.5 percent. The rotationally averaged kernel does something
genuinely different, it SPLITS the penalty, 56.9 percent and 78.8 percent. **But 43 percent
of the crossbar is still unmelted, so the answer to "does the crossbar melt" is: more than
before, not enough.** The L_shape is the counter-case where the joint static winner had
already balanced the limbs (74.0 against 78.9), and there rotation has nothing to add.

---

## 4. Why rotation helps some shapes and not others, measured

COMPUTED. The averaged kernel's residual AZIMUTHAL ANISOTROPY, defined as the mean over 23
rotations in 15-degree steps of the mean absolute difference between the kernel and its own
rotation, normalized by the kernel's mean, evaluated on the largest disc inscribed in the
part so the rotation stays inside the part. A perfectly annular kernel scores zero.

| shape | static 0-degree kernel | rotationally averaged kernel | reduction factor |
|---|---|---|---|
| star | 0.2058 | **0.0205** | 10.0 |
| square | 0.2079 | **0.0216** | 9.6 |
| cross | 0.2525 | **0.0389** | 6.5 |
| T_shape | 0.6590 | 0.2193 | 3.0 |
| L_shape | 0.4679 | **0.2840** | **1.6** |

**The ranking of this column is the ranking of the verdicts in Section 1.** Averaging over
a full turn drives the kernel to within 2 percent of annular on the star and the square,
4 percent on the cross, but leaves 22 and 28 percent residual anisotropy on the T_shape and
the L_shape. The reason is geometric and not numerical: the T and the L are far from
radially symmetric, so no amount of averaging in the part frame makes their kernel radial.
`fig_rot_kernels.png` shows this directly, including the radial profiles, and the difference
panels show exactly what the averaging moves: on the T_shape it takes energy OUT of the
stem (blue) and puts it into the crossbar (red); on the cross it takes energy out of the
limb tips and puts it into the centre; on the square it is nearly featureless.

The consequence is visible in every melt field in `fig_rot_headline.png`: **a radial kernel
melts a rounded blob**, and a rounded blob cannot reproduce a non-convex, high aspect-ratio
outline. That is the ceiling on what continuous rotation alone can do for the T_shape and
the L_shape, and it is a property of the actuator, not of the solve.

---

## 5. Level 2: the true rotating engine, and the rotation-speed question

Every Level-2 arm is a run of `rfam_eqs_coupled.run_sim` through the engine's own turntable
block (`rfam_eqs_coupled.py:2795-2837`), with the dopant map co-rotated at every rotation
event. Full tables, all arms, all periods, in `fgm_solve_campaign/out_rot/_tables.md`.

**Real-data driver gate, and it is a strong one.** PROVEN. With the turntable disabled the
production engine reproduces the adjoint2d prototype's stored numbers to the printed
digits on every shape:

| shape | engine S_uniform | stored prototype uniform | engine S_map0 | stored prototype map J |
|---|---|---|---|---|
| T_shape | 614.26 | 614.26 | 609.31 | 609.31 |
| L_shape | 538.42 | 538.42 | 521.16 | 521.16 |
| cross | 471.82 | 471.82 | 360.07 | 360.07 |
| star | 192.58 | 192.58 | 143.82 | 143.82 |
| square | 210.19 | (not stored) | 25.56 | 25.56 |

The joint static arms likewise reproduce (`S_joint`: T_shape 522.28 against 522.11,
L_shape 376.94 against 376.75, cross 147.03 against 146.80, star 106.28 against 106.27;
the small residual is that the stored numbers are the 4 bpp arms and these are the
continuous maps). **Two independent implementations of the same physics agree, so the
Level-2 numbers can be read against the Level-1 numbers without a translation step.**

**Rotation speed.** `fig_rot_speed.png`, left panel. Two errors pull in opposite directions
and the measured curves are their sum:

| shape | P = 12 s | 24 s | 60 s | 180 s | 360 s | 720 s |
|---|---|---|---|---|---|---|
| T_shape, solved map | 550.91 F | 548.55 | 532.70 | 504.05 | 512.15 | **467.33** |
| L_shape, solved map | 434.75 F | 419.36 | 397.64 | 378.68 | **352.97** | 402.27 |
| cross, solved map | 179.11 F | 151.69 F | 114.23 | **97.27** | 109.90 | 119.20 |
| star, solved map | 89.65 F | 51.59 F | 30.98 F | **85.64** | 180.76 | |
| square, solved map | 49.02 F | 25.15 F | **28.55** | 53.50 | | |

COMPUTED. `F` marks a failing energy gate; bold is the best GATE-PASSING period. The star
and the square have a clear interior minimum, the cross a shallower one near 180 s, and the
T_shape is still improving at 720 s, which is barely one turn over the exposure and is
therefore no longer meaningfully "continuous rotation".

**Quasi-static approximation error, for the 24-angle map.** COMPUTED, and it is LARGE and
non-monotone: the Level-1 forward predicts 24.63 for the star and the engine measures 30.98
at the star's best period (+25.8 percent) but 89.65 at P = 12 s (+263.9 percent); for the
cross the prediction is 192.02 and the engine measures between 97.27 and 179.11 depending
on period (-49.3 to -6.7 percent). The full table is in `_tables.md`. **On its own that
looks like a failure of the quasi-static assumption. Section 6 shows it is mostly not.**

---

## 6. Two engine findings, both isolated with controls

### 6.1 The rotation-event remap is not energy conserving, and it is the bilinear interpolation

COMPUTED, and this is the most consequential numerical finding in the pass. At a rotation
event the engine remaps temperature, relative density and melt fraction into the rotated
frame with `scipy.ndimage.map_coordinates` at order 1 (`rfam_eqs_coupled.py:2969-2985`) and
then resets the incremental energy baseline to the post-rotation state, so whatever energy
the interpolation destroys is never accounted. Over many events it accumulates.

The control that isolates it: **`C_null90`, a rotation schedule with 90-degree steps at the
SAME event rate.** At 90 degrees on this grid the remap coordinates land exactly on grid
points, so the interpolation is an exact pixel permutation and carries no error at all.

| shape | 15-degree steps, 1500 events | 90-degree steps, 1500 events (`C_null90`) |
|---|---|---|
| star | energy residual **15.59 %**, gate FAILS | **0.58 %**, gate PASSES |
| square | **10.30 %**, FAILS | **0.43 %**, PASSES |
| cross | **5.40 %**, FAILS | **0.81 %**, PASSES |
| T_shape | **5.55 %**, FAILS | **1.32 %**, PASSES |

Same event count, same physics, one difference. **The whole energy-gate violation is
bilinear interpolation error in the rotation remap**, roughly 0.007 to 0.010 percent of
dose per event on these shapes. It scales with the number of events, which is why fast
continuous rotation fails the gate and slow rotation does not.

**And it was hiding the quasi-static approximation, not revealing a failure of it.** The
decisive check is the square, whose part mask is invariant under 90 degrees, so the
90-degree control is EXACTLY the four-angle quasi-static average with the remap error
removed:

| square, uniform map | J_phi | IoU | growth % | under-melt % | stop s |
|---|---|---|---|---|---|
| four-angle averaged kernel, PREDICTED by Level 1 | 78.805 | 0.9615 | 4.00 | 0.00 | 424.0 |
| `C_null90` on the true engine, 1500 events, MEASURED | **78.54** | **0.9615** | **4.00** | **0.00** | **423.0** |

**J_phi agrees to 0.34 percent, IoU and both region metrics agree exactly, and the stop
times agree to one outer step.** PROVEN as a real-data check. The quasi-static
approximation is not the problem; the engine's 15-degree remap was.

### 6.2 The permittivity field never re-rasterizes at a rotation event

COMPUTED, and it is a SEPARATE gap this pass found. The relative-permittivity field is
built once at startup from the ORIGINAL fill fraction (`rfam_eqs_coupled.py:2532`) and is
never rebuilt at a rotation event; only conductivity is (`:2989-2997`). With eps_r 20
inside the part against 1 outside, that leaves a stationary dielectric ghost of the
un-rotated part in every post-event EQS solve. The glue can fix it in place
(`corotate_eps`, default OFF so the baseline arm is the engine as shipped). Measured on the
uniform arm at P = 12 s: square 39.36 to 72.49, cross 213.71 to 252.60, T_shape 580.00 to
541.01. **The fix changes J_phi by -7 to +84 percent, so this is not a small effect and it
is not a consistent-sign effect.** It is reported and left OFF everywhere else in this
report; every number outside this paragraph uses the engine as shipped. Naming which of
the two is physically right is a separate question and is not settled here.

### 6.3 Without the co-rotation fix a graded map is inert under rotation

COMPUTED. The control `C_avg_norotate` runs the solved map with the co-rotation switched
off, which is the engine as shipped. On the T_shape it gives J_phi 580.01 and IoU 0.5276,
against the co-rotated arm's 550.91 / 0.5352 and the UNIFORM arm's 580.00 / 0.5276.
**Identical to the uniform arm to five significant figures.** The lab-frame map washes out
completely as the part turns beneath it, so the dopant grading does nothing at all. On the
cross the effect is partial (187.20 against 179.11 co-rotated and 213.71 uniform) because
that map is close to four-fold symmetric. This is exactly the failure
`ORIENTATION_PIPELINE_ASSESSMENT.md` Section 3(b) predicted, now measured.

---

## 7. The result the campaign did not expect: symmetry-matched indexing beats continuous rotation

Section 6.1 showed that 90-degree stepping is numerically exact on this grid. That
suggested re-solving the map against the MATCHED four-angle averaged kernel
{0, 90, 180, 270} degrees and running it as 90-degree indexing on the true engine, one
quarter turn every 2.0 s (375 events over the horizon). COMPUTED:

| shape | Level-1 four-angle prediction | engine, uniform map | engine, SOLVED four-angle map | at 4 bpp | quasi-static error | energy residual |
|---|---|---|---|---|---|---|
| cross | 34.99, IoU 0.9829 | 125.93, 0.8697 | **34.82, 0.9866** | 34.76, 0.9847 | **-0.5 %** | 0.37 % |
| square | 13.92, 1.0000 | 78.43, 0.9615 | **13.93, 1.0000** | 13.88, 1.0000 | **+0.1 %** | 0.16 % |
| star | 33.68, 0.9319 | **23.72, 0.9726** | 24.46, 0.9527 | 24.56, 0.9527 | -27.4 % | 0.45 % |
| L_shape | 372.05, 0.6576 | 381.38, 0.6444 | **376.98, 0.6476** | 376.99, 0.6476 | +1.3 % | 0.95 % |
| T_shape | 450.03, 0.6131 | 516.40, 0.5661 | **500.14, 0.5720** | 500.14, 0.5752 | +11.1 % | 1.23 % |

Three things, all COMPUTED.

**One: this is the best arm on four of five shapes, and every one of them passes the energy
gate with a residual under 1.4 percent.** The cross and the square are better here than at
any continuous rotation period by factors of 2.8 and 2.1.

**Two: the quasi-static approximation is EXCELLENT when the actuator and the averaging set
match.** Predicted against measured: -0.5 percent on the cross, +0.1 percent on the square,
+1.3 percent on the L_shape. That is the honest verdict on Level 1 as a design model: it
is accurate to about one percent, and the large errors quoted in Section 5 came from
applying a 24-angle model to a schedule the engine could not execute cleanly. The two
remaining gaps are the T_shape (+11.1 percent) and the star (-27.4 percent, in the engine's
favour); the star's is confounded because a five-fold shape under four-fold indexing has a
part mask that genuinely changes at every event, so the target is not the same object the
part-frame average assumes.

**Three: more averaging is not better.** On the cross the 24-angle quasi-static solve
predicts 192.02 and the four-angle solve predicts 34.99 for the SAME shape and the same
budget. The four-angle set keeps the cross's limbs aligned with the electrode axis at every
sampled orientation, which is a good heating pattern for a cross; the 24-angle set includes
the 45-degree orientations, which are bad ones, and averages them in. **The right actuator
is an indexing schedule matched to the part's symmetry, not the finest rotation available.**

---

## 8. Health gates

COMPUTED over all 127 scored engine arm evaluations (the four headline arms per
shape were re-run to capture stop-time fields, so a few arms appear twice and reproduce
to the printed digits) and 20 Level-1 arms at the 24-angle kernel.

* **Energy-residual gate.** All Level-1 arms PASS (worst 0.9 percent). Among engine arms,
  every static arm passes (0.25 to 1.19 percent), every 90-degree indexing arm passes
  (0.16 to 1.36 percent), and **36 of the 15-degree rotating arm evaluations FAIL**, every one of them at
  300 events or more and none of them below, worst 18.96 percent (star, P = 12 s, stored zero-degree map). Section
  6.1 localizes the cause. **No verdict in Section 1 or Section 7 rests on a failing arm.**
* **Operating ceiling, 250 C.** One flag in the whole pass and it is not an optimum: the
  star's uniform arm at P = 360 s reaches 250.0 C, which is the maximum over all 127
  evaluations (the minimum is 186.5 C). Every arm named in Sections 1 and 7 sits between
  194.3 and 228.6 C.
* **Numerical clipping.** Zero on every Level-1 arm (temperature step clip, temperature
  clamp and Q_rf cap fractions all 0.0000). The engine's own dT-clip fraction is 0.0 on
  every engine arm at its own stop.
* **Horizon flags.** Six arm evaluations have their J_phi minimum on the last stored step and their
  J_phi is therefore a BOUND: T_shape `S_joint` (522.28), cross `R_joint_P12`, square
  `R_uniform_P12`, `R_avg_P12` and `C_avg_norotate`, plus the headline re-run of the T_shape static baseline. The one that matters is T_shape
  `S_joint`, and it is the comparison BASELINE, so treating it as a value makes the
  T_shape verdict harder to win, not easier.
* **Absorbed power is NOT matched.** Level-1 24-angle arms span 296.8 to 462.3 W/m against the
  500.0 W/m uniform calibration target. J_phi charges over-melt and under-melt
  symmetrically, which removes the crudest dose gaming, but **no comparison in this report
  is a comparison at equal delivered energy.** The engine arms do not record absorbed power
  in the per-step trace, so the dose spread on Level 2 is not quantified at all; that is a
  gap.
* **Mean relative density at the deliverables' own stops spans 0.6018 to 0.8894.** These
  are melt-stop reads, not end-of-horizon reads, and nothing here is an oversinter claim.
  The highest values (0.86 to 0.89) all occur on square arms whose stop is at or near the
  750 s horizon.

---

## 9. Honest limits

1. **The FD gate was run on one shape (the cross).** The other four are ASSUMED to inherit
   it. One gate run per shape is roughly 12 minutes each and was not done.
2. **The Level-1 maps used in the best CONTINUOUS arms are mismatched to their actuator.**
   They were solved against a 24-angle average, and the best continuous periods are 180 to
   720 s where the quasi-static assumption is poor. A map solved for the actual schedule
   would do better; how much better is not measured. The 90-degree indexing arms of Section
   7 do not have this problem, which is part of why they win.
3. **"90-degree indexing every 2.0 s" is not the same machine as a turntable.** It is 375
   quarter turns over 750 s. Whether the hardware can index that fast, and what the
   mechanical settling does to the powder bed, is outside this model entirely.
4. **The 90-degree exactness is a GRID property, not a physical one.** It holds because the
   rotation centre sits at index (n-1)/2 on an even grid, so the permutation is exact. At
   an odd grid, an off-centre part, or any other angle, the interpolation error returns.
   The engine's remap needs a conservative formulation before fast rotation of any kind is
   trustworthy.
5. **Not dose matched** (Section 8), and absorbed power is not even recorded on the engine
   arms.
6. **No rotation schedule was co-optimized.** The task explicitly excluded it and the
   speed sweep is a grid scan over a single constant period, four to six points per shape.
   The optimum could sit between sampled periods.
7. **Conductivity channel only.** The permittivity-co-varying channel is not touched here.
8. **Single grid (120 x 120), two dimensions.** `SOLVE_ROBUSTNESS_VALIDATION.md` shows
   grid-120 fidelity does not transfer to grid 160.
9. **Model, not hardware.** `ALLISON_LAW_REPLICATION.md` Section 6.1 records that the
   two-dimensional model over-predicts achievable tuned uniformity against hardware by
   roughly a factor of eight. Every IoU here is a statement about the model.
10. **One figure was CUT rather than shipped.** `fig_rot_summary.png` collapsed rotating
    arms measured at different rotation periods into a single bar per arm name, which is
    not a comparison; the headline figure replaces it with one explicit period per shape.
    No data was discarded, only that rendering.
11. **The star's Level-1 map is nearly inert under its winning actuator** (uniform 23.72
    against solved 24.46 at 90-degree indexing). The star's win is an ORIENTATION result,
    not a grading result, and it should be quoted that way.

---

## 10. Proven, computed, assumed

**PROVEN**
* The rotation operator's contract, red first: 9 tests including exact pixel permutations
  at 90, 180 and 270 degrees, the dot-product transpose identity to 1e-10, and agreement
  with the already proven production-convention helper to 1e-9.
* The degeneracy gate on the averaged kernel: one angle at zero degrees reproduces the
  ordinary forward bit for bit and its gradient to below 1e-12 relative.
* The glue, red first: 8 tests including two real-engine integration tests, one proving
  that melt fraction re-derived from temperature matches the engine's own per-step mean to
  1e-9 and one proving the co-rotation is the exact 90-degree permutation inside the part.
* The finite-difference gate on the averaged-kernel gradient: R1 4/4 at 1e-6, R2 5/5 at the
  1e-5 subgradient standard, gradient-direction probes 1.07e-07 and 1.68e-07.
* The real-data driver gate: the production engine reproduces the prototype's stored
  uniform and stored-map numbers to the printed digits on all five shapes.
* The square's four-angle quasi-static prediction against the exact-permutation engine run:
  J_phi 78.805 predicted against 78.54 measured, IoU and both region metrics identical.

**COMPUTED**
* Every number in Sections 1 and 3 through 8.

**ASSUMED**
* That the FD gate measured on the cross holds on the other four shapes.
* That the rasterized rotated binary part mask is the right nominal target at every
  instant, carried over from the library, orientation and joint campaigns.
* That an arbitrary stop time is realizable as a process control.
* That the quasi-static averaged kernel is the right design model for a fast turntable.
  Section 7 MEASURES this to about one percent on three shapes and does not measure it
  elsewhere.
* That the engine as shipped, with the permittivity field frozen at the original
  orientation, is the baseline worth reporting. Section 6.2 shows the alternative changes
  J_phi by up to 84 percent and does not settle which is right.

---

## 11. The single most valuable next layer

**Re-solve the map for the actual rotation schedule instead of for a static average.** The
averaged kernel is a model of the schedule; the schedule itself is available in the true
engine and in the prototype's own time loop. A time-resolved rotating forward inside
`adjoint2d`, with the part mask and the injected map rotating on the same event grid as the
engine, would remove the only remaining approximation between Level 1 and Level 2 and would
let the map be solved at the period that actually wins. Its adjoint is the existing reverse
march with a rotation transpose inserted at every event, and `rot_frame` already supplies
that transpose with an exact adjoint and a passing dot-product test.

**Second: fix the rotation-event remap to be conservative.** Section 6.1 quantifies a 5 to
19 percent energy loss at fast rotation and proves it is the bilinear interpolation. Until
that is fixed, no fast-rotation result at a non-multiple of 90 degrees can pass the
standing energy gate, and the whole fast-rotation regime is unmeasurable. This is engine
work and belongs to whoever owns `rfam_eqs_coupled.py`; it should be a flag defaulting to
the current behaviour, with the 90-degree control as its regression test.

**Third: settle the permittivity re-rasterization question** (Section 6.2). One of the two
behaviours is wrong and they differ by up to 84 percent of J_phi on a rotating run.

---

## 12. Artifacts, absolute paths, and wall time

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New code:
* `fgm_solve_campaign/adjoint2d/rot_frame.py`, `rot_kernel.py`, `gate_rot.py`
* `fgm_solve_campaign/adjoint2d/tests/test_rot_frame.py`, `tests/test_rot_kernel.py`
* `scripts/analysis/turntable_glue.py` and `test_turntable_glue.py`
* `scripts/analysis/run_rot_avg_solve.py`, `run_rot_verify.py`, `run_rot_speed_sweep.py`,
  `run_rot_headline.py`, `build_rot_tables.py`, `make_rot_figures.py`

Read and imported, not modified: `fgm_solve_campaign/adjoint2d/{forward,adjoint,eqs,
gradops,shape_objective,design_filter,printability,library_solve,energy_gate,pins}.py`,
`scripts/analysis/orientation_map_rotation.py`, `rfam_eqs_coupled.py`.

Results, all under `fgm_solve_campaign/out_rot/`:
* `gate_rot_cross.json` the finite-difference gate, full epsilon sweeps
* `<shape>_rotavg.json` and `<shape>_rotavg_maps.npz` Level 1, 24-angle kernel
* `<shape>_rotavg_step90.json` and `_maps.npz` Level 1, matched four-angle kernel
* `<shape>_verify.json`, `<shape>_speed.json`, `<shape>_headline.json`,
  `<shape>_index90.json` the engine arms
* `verify_fields/<shape>_{fields,headline,index90}.npz` melt fields; the headline and
  index90 files hold the field AT THE ARM'S OWN J-STOP
* `kernel_anisotropy.json`, `limb_analysis.json`, `quasistatic_4angle.json`
* `_tables.md` every table, every arm, every period
* `fgm_solve_campaign/logs_rot/*.log` per-run console logs

Figures, **all thirteen viewed before delivery**, in `fgm_solve_campaign/figs_rot/`:
* `fig_rot_headline.png` the verdict figure, five shapes by five arms, melt at each arm's
  own J-stop
* `fig_rot_kernels.png` the averaged kernel against the static kernel, with difference
  panels and radial profiles
* `fig_rot_speed.png` J_phi against rotation period, and the energy residual against event
  count with the 90-degree control marked
* `fig_rot_maps_<shape>.png` (five) the Level-1 solved map, its 4 bpp version and the melt
  fields
* `fig_rot_verify_<shape>.png` (five) every engine arm, end-of-horizon fields, captioned as
  such

**Wall time, COMPUTED and logged progressively.** After the first Level-1 solve finished
(star, 359 s for 32 gradient evaluations plus 4 scoring forwards) the projection for the
four larger shapes at 500 to 700 s each in five pinned single-thread streams was 12
minutes; actual 565 to 700 s, inside that. The first timing probe of the rotating engine
measured 23.6 s for 200 steps with 100 events against 6.0 s static, giving 0.176 s per
rotation event, from which the 1500-step arms were projected at 45 s plus 0.176 s per
event; actual 283 to 299 s at 1500 events, inside that. **Total 21910 seconds of process
time, about 3.5 hours of wall clock in up to six pinned single-thread streams**
(`fgm_solve_campaign/env1.sh`). Nothing was cut and no arm was skipped; the one figure that
was dropped is named in limit 10 with its reason.
