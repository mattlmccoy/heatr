# Joint per-angle dopant-map re-solve on the stranded shapes

**Date:** 2026-08-01. **Scope:** the direct test of the registered prediction that the best
orientation found with a FIXED dopant map is not the best orientation once the map is
RE-SOLVED at every angle. Four shapes (T_shape, L_shape, cross, star), the orientation
sweep's own symmetry-aware angle sets, a filtered box-constrained adjoint solve at every
angle from two starts, a depth check at a larger budget on the winning angles, and a
permittivity-co-varying arm on the cross only. Grid 120 x 120 throughout. **Nothing was
committed. No dissertation file was touched. `.claude/worktrees/` was not read or written.**

**Acronyms, expanded on first use.** FGM = functionally graded material (a spatially varying
dopant saturation map). EQS = electro-quasi-static (the low-frequency Maxwell approximation
the two-dimensional solver uses). IoU = intersection over union. bpp = bits per pixel.
L-BFGS-B = limited-memory Broyden-Fletcher-Goldfarb-Shanno with box constraints. FD = finite
difference. phi = melt fraction. rho = relative density. W/m = watts per metre of depth.
eps_r = relative permittivity. sigma = electrical conductivity.

**Evidence tags.** PROVEN = unit tested, FD gated, or reproduced against a stored number.
COMPUTED = measured from a run in this pass. ASSUMED = a modelling choice or an inference not
measured here.

**Objective and stop convention, stated once and carried on every number.**

    J_phi(s, theta, t_stop) = sum over the WHOLE domain of
                              (phi(x, t_stop) - chi_part(x; theta))^2

with `chi_part(.; theta)` the binary part mask rasterized by the production engine at
`geometry.part.rotation_deg = theta`, so the nominal target co-rotates with the part. Every
J_phi, IoU, growth, under-melt, mean relative density, absorbed power and maximum temperature
below is read at that arm's OWN J-stop, `t_stop` = argmin of J_phi over that arm's own stored
trajectory on a 1500-step horizon (dt 0.5 s, 750 s) with early truncation 250 steps past the
running minimum. `HORIZON` is flagged whenever the minimum sits on the last stored step, which
makes that arm's J_phi a BOUND and not a value. The melted region for IoU, growth and
under-melt is phi >= 0.5; J_phi itself uses no threshold. Absorbed power is the state-B value
in W/m. These are the exact conventions of `SHAPE_LIBRARY_SOLVE_REPORT.md`,
`MULTISTART_REPORT.md` and `ORIENTATION_OPTIMIZATION_REPORT.md`.

**Grid qualifier, mandatory.** `SOLVE_ROBUSTNESS_VALIDATION.md` established that absolute
fidelity at grid 120 does not transfer to grid 160 and that the forward itself is not
converged in IoU between those grids. Every IoU here is a property of the method AT GRID 120
with an exactly reproduced dopant map, not a property of the method. No grid-160 number
appears anywhere in this report.

---

## 1. Verdict per shape, first

| shape | actuator | prediction | fixed-map best | joint best, 15 fwd-equiv scan | joint best at depth | robust? |
|---|---|---|---|---|---|---|
| T_shape | conductivity | **REFUTED** | 90 deg | 90 deg (+0.0) | 90 deg | yes, but 90 and 135 are a near tie |
| L_shape | conductivity | **REFUTED** | 135 deg | 135 deg (+0.0) | 135 deg | yes |
| cross | conductivity | **CONFIRMED** | 30 deg | **45 deg (+15.0)** | **45 deg** | yes, and it strengthens with depth |
| star | conductivity | **CONFIRMED** | 0 deg | **18 deg (+18.0)** | **18 deg** | yes, and it strengthens with depth |
| cross | permittivity, MODEL ONLY | **REFUTED at depth** | 30 deg | 45 deg (+15.0) | **0 deg (+0.0 move)** | NO, the scan argmin reverses |

COMPUTED, all five. One-line reading per shape:

1. **T_shape: the angle does not move, and grading is worth 4.1 percent of J_phi at the best
   angle.** Joint 530.27 against the fixed-map arm's 552.91 at 90 degrees. At the depth budget
   90 degrees gives 522.11 (HORIZON, so a bound) against 135 degrees at 523.40, a gap of 0.25
   percent that sits INSIDE this campaign's measured 0.32 percent angle-to-angle reproducibility
   (Section 5). The honest T_shape statement is that 90 and 135 degrees are a statistical tie
   at depth and the fixed-map sweep's 90 is not overturned.
2. **L_shape: the angle does not move and re-solving buys almost nothing.** Joint 378.98
   against 380.42 at 135 degrees, 0.4 percent, rising to 0.96 percent at depth (376.75). The
   solved map at 135 degrees is saturation 1 almost everywhere except a small patch at the
   inner corner, visible in the figure: dopant grading is close to inert on this shape in the
   conductivity channel, which reproduces what the library and multi-start campaigns already
   measured at 0 degrees.
3. **cross, conductivity only: the angle MOVES from 30 to 45 degrees and the move survives
   depth.** At the scan budget 45 degrees gives 206.16 against 30 degrees at 222.12 (7.2
   percent). At the depth budget the gap widens to 146.80 (HORIZON, a bound) against 198.55, 26
   percent. The mechanism is visible in the figure: the FIXED-map curve turns UP between 30 and
   45 degrees while the re-solved curve keeps going DOWN, so the two curves cross and the argmin
   moves. **This is also the first time the cross crosses IoU 0.80 at grid 120 in the
   deployment-safe conductivity-only channel: 0.8048 at the scan budget and 0.8497 at depth.**
4. **star: the angle MOVES from 0 to 18 degrees, and this is the strongest case in the pass
   because the sign flips.** The fixed-map sweep concluded that "orientation buys exactly
   nothing" for the star and that 18 degrees is the WORST angle (uniform 245.74, rotated map
   220.55, both peaking at 18). The joint re-solve makes 18 degrees the BEST angle: 126.74 at
   the scan budget and 106.27 at depth, against 140.10 at 0 degrees. The fixed-map curve and the
   re-solved curve are inverted about the same abscissa. Joint at 18 degrees beats the sweep's
   best arm (157.39 at 0 degrees) by 19.5 percent at the scan budget and 32.5 percent at depth.
5. **cross, permittivity co-varying (MODEL ONLY): the apparent move is a budget artifact and
   REVERSES at depth.** The 15 forward-equivalent scan puts the optimum at 45 degrees (125.61
   against 226.30 at 0 degrees). The 40 forward-equivalent depth check puts it back at 0 degrees
   (100.22 against 109.09 at 45 degrees). The cause is localized and quantified in Section 7:
   the 0-degree solution in this channel is quantization fragile, costing +182.7 percent of
   J_phi when the continuous map is quantized to 4 bpp at the scan budget and +45.5 percent at
   depth, while every other point in the whole pass costs between -2.5 and +1.6 percent.

**Does any stranded shape now cross IoU 0.80 or 0.95 at grid 120?** COMPUTED. **0.80: yes,
the cross, in both channels.** Conductivity only, 0.8048 at the scan budget and 0.8497 at depth
(that one at the horizon). Permittivity co-varying, 0.8737 and 0.8931 (0.9280 on the
continuous, unprintable map at 0 degrees at depth). **0.95: no shape, in any arm.** The star
reaches 0.7852, the T_shape 0.5364 and the L_shape 0.6693. All four shapes stay outside the
absolute SOLVED class.

**How much does the joint optimum beat the fixed-map sweep's best, which was a lower bound?**

| shape | actuator | sweep best J_phi | joint best, scan | joint best, depth | improvement, scan | improvement, depth |
|---|---|---|---|---|---|---|
| T_shape | sigma | 552.91 | 530.27 | 522.11 (bound) | +4.1 % | +5.6 % |
| L_shape | sigma | 380.42 | 378.98 | 376.75 | +0.4 % | +1.0 % |
| cross | sigma | 257.77 | 206.16 | 146.80 (bound) | +20.0 % | +43.1 % |
| star | sigma | 157.39 | 126.74 | 106.27 | +19.5 % | +32.5 % |
| cross | eps | 257.77 | 125.61 | 100.22 | +51.3 % | +61.1 % |

COMPUTED. The two shapes whose angle moved are also the two shapes where re-solving buys the
most, which is the same statement twice: on T_shape and L_shape the dopant map is nearly inert,
so re-solving it cannot change where the orientation optimum sits.

---

## 2. Protocol

**Angles.** The orientation sweep's symmetry-aware sets, unchanged so the two curves are read
at the same abscissae: T_shape and L_shape [0, 180] in 22.5-degree steps (9 angles), cross
0/15/30/45, star 0/9/18/27/36. Angle differences are quoted modulo that shape's symmetry
period (T_shape and L_shape 180, cross 90, star 72), so that, for example, a cross result at 90
degrees would be a zero move.

**What is re-solved at each angle.** The design variable v on the part cells, box [0, 1], with
the MANDATORY physical-length design filter s = F(v) at sigma = 1.5 cells (0.75 mm at this
grid), optimized by L-BFGS-B on the filtered adjoint gradient dJ_phi/dv. This is exactly the
production recipe of `MULTISTART_REPORT.md`, whose FD gate on this composed gradient is the
gate this pass inherits (Section 6).

**Two starts per angle, better kept.**
* `cold`: uniform saturation 1.
* `warm`: the best known ZERO-DEGREE map for that shape in that actuator channel, ROTATED into
  the rotated part frame by the production convention. Conductivity-only warm sources, selected
  by their own stored J_phi: `out_lib/<shape>_maps.npz:A1_cont` for T_shape (609.31), L_shape
  (521.16) and cross (360.07), and `out_ms/star_maps.npz:MS_cont` for the star (143.82). The
  permittivity warm source is `out_eps/cross_maps.npz:EPS_best_cont` (76.25). Box [0, 1.5] arms
  were excluded because this campaign's box is [0, 1].

**The rotation, and the red-first test.** The solver injects saturation maps in the LAB frame
while `rotation_deg` rotates the part, so the warm map is rotated by
`scipy.ndimage.rotate(angle = -rotation_deg, reshape=False, order=1)`, clipped into the box
inside the rotated part and held at saturation 1 outside. That composition is a new helper,
`rotated_warm_start`, and it was written test first: `test_joint_angle_lib.py` was run before
`scripts/analysis/joint_angle_lib.py` existed and gave **14 failures, all
`ModuleNotFoundError: No module named 'scripts.analysis.joint_angle_lib'`**, then 14 passed.
The underlying rotation contract itself is already PROVEN in `test_orientation_map_rotation.py`
(5 tests, including the exact 90-degree pixel permutation and the zero-mismatch agreement with
the production part rasterization). **19 tests pass** across both files.

**Budget, stated exactly because the task named a number.** The scan gives **15 forward
equivalents PER START**, two starts per angle, so **28.1 to 31.6 forward equivalents per
angle** plus two scoring forwards (continuous map and 4 bpp map) and, at 0 degrees only, one
uniform gate forward. COMPUTED per shape: 6 gradient evaluations per start on T_shape, L_shape
and cross, 7 on the star. The conversion uses the per-shape adjoint-to-forward cost ratio
RECORDED BY THE LIBRARY CAMPAIGN (1.11 to 1.45), for the reason `MULTISTART_REPORT.md`
Section 3.3 gives: timing one forward and one adjoint on a loaded machine is unreliable and
would silently starve the solve. The **depth check** re-runs the top angles at **40 forward
equivalents per start** (16 to 18 gradient evaluations), which is the budget every previous
campaign used.

**Deliverable arm.** `JOINT_4bpp`, the winning start's filtered continuous map quantized to 4
bits per pixel inside the part through the production quantizer and re-run through the real
forward. That is the printable arm and it is the arm matched to the fixed-map sweep's quantized
graded arm. The continuous map `JOINT_cont` is reported alongside on every angle.

**Real-data driver gate, run at 0 degrees on every shape.** With s = 1 everywhere the uniform
arm must reproduce the stored library number and must be identical in both actuator channels.
COMPUTED: T_shape 614.26 against 614.26 (relative 7.02e-06), L_shape 538.42 against 538.42
(1.44e-06), cross 471.82 against 471.82 (2.55e-06, in BOTH channels), star 192.58 against
192.58 (2.15e-05).

**Determinism check, free.** The cross permittivity depth check was run twice, first on the top
two angles and then on all four. The 30-degree and 45-degree arms reproduce to the printed
digits across the two invocations (141.14 / 0.8638 and 109.09 / 0.8931 both times).

**Drive.** Per-shape calibrated voltage from the campaign configs, fixed across angles
(T_shape 1719.0 V, L_shape 1804.2 V, cross 2815.4 V, star 3223.5 V), grid 120, 27.12 MHz,
`enforce_generator_power: false`. Holding voltage fixed is the controlled comparison of the
actuator, but absorbed power then varies with orientation and with the map. Not dose matched;
Section 6 quotes the spread.

**Actuator choice.** The PRIMARY arm on all four shapes is conductivity only, because
`EPS_CHANNEL_REPORT.md` Section 11 leaves open whether the dopant moves relative permittivity
at all in the real binder, and until that is settled the conductivity-only census is the
deployable one. The permittivity-co-varying arm was run on the cross only, is labelled MODEL
ONLY everywhere it appears, and is not used for any deployment claim.

---

## 3. The angle tables

Full tables, all angles, both arms, with the fixed-map sweep columns alongside, are in
`fgm_solve_campaign/out_joint/_tables.md`. The deliverable rows at the scan budget:

| shape | actuator | angle | J joint 4 bpp | J fixed map | J uniform | IoU joint | grow % | under % | rho at stop | P_abs W/m | stop s | max T C | start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T_shape | sigma | **90** | **530.27** | 552.91 | 552.32 | 0.5339 | 4.35 | 44.29 | 0.6865 | 263.1 | 627.5 | 206.5 | warm |
| L_shape | sigma | **135** | **378.98** | 380.42 | 395.13 | 0.6669 | 14.21 | 23.83 | 0.7217 | 529.5 | 312.0 | 242.9 | warm |
| cross | sigma | **45** | **206.16** | 270.93 | 283.65 | 0.8048 | 11.13 | 10.57 | 0.7875 | 365.9 | 478.5 | 237.3 | warm |
| star | sigma | **18** | **126.74** | 220.55 | 245.74 | 0.7448 | 5.54 | 21.40 | 0.6391 | 378.9 | 222.5 | 199.9 | warm |
| cross | eps | **45** | **125.61** | 270.93 | 283.65 | 0.8737 | 7.55 | 6.04 | 0.7249 | 535.0 | 283.0 | 217.9 | cold |

The same rows at the depth budget:

| shape | actuator | angle | J joint 4 bpp | IoU | grow % | under % | rho at stop | P_abs W/m | stop s | max T C | start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T_shape | sigma | 90 | 522.11 **(HORIZON, a bound)** | 0.5364 | 3.17 | 44.66 | 0.6843 | 235.4 | 750.0 | 201.6 | cold |
| L_shape | sigma | 135 | 376.75 | 0.6693 | 13.94 | 23.74 | 0.7215 | 527.9 | 313.0 | 241.9 | warm |
| cross | sigma | 45 | 146.80 **(HORIZON, a bound)** | 0.8497 | 8.58 | 7.74 | 0.7943 | 263.5 | 750.0 | 203.7 | warm |
| star | sigma | 18 | 106.27 | 0.7852 | 4.80 | 17.71 | 0.6308 | 412.4 | 201.5 | 200.5 | cold |
| cross | eps | 0 | 100.22 | 0.8919 | 7.14 | 4.44 | 0.7206 | 360.4 | 480.0 | 212.0 | warm |

Two of the depth-budget optima sit at the horizon, which means their J_phi is an upper bound
and the true optimum for that arm is at least that good. Both bounds already beat the
comparison they are used for, so no verdict rests on the unmeasured part.

---

## 4. The depth check, and why it was added

A 15 forward-equivalent scan can move an argmin by itself, so the headline is only safe if the
ranking survives depth. The top angles of each scan, plus the fixed-map sweep's own best angle
in every case, were re-solved at 40 forward equivalents per start.

| shape | actuator | angles refined | scan best | refined best | argmin survives | J refined best | IoU refined best |
|---|---|---|---|---|---|---|---|
| T_shape | sigma | 90, 135 | 90 | 90 | **YES** | 522.11 (bound) | 0.5364 |
| L_shape | sigma | 135, 180 | 135 | 135 | **YES** | 376.75 | 0.6693 |
| cross | sigma | 30, 45 | 45 | 45 | **YES** | 146.80 (bound) | 0.8497 |
| star | sigma | 0, 18, 27 | 18 | 18 | **YES** | 106.27 | 0.7852 |
| cross | eps | 0, 15, 30, 45 | 45 | 0 | **NO** | 100.22 | 0.8919 |

COMPUTED. Four of five survive. The one that does not is the model-only permittivity arm, and
Section 7 localizes why. **The depth check was not in the original plan and it changed one of
the five verdicts, which is the argument for keeping it in any future orientation work.**

---

## 5. The campaign's own noise floor, measured rather than assumed

The forward physics is identical at angle pairs the shape symmetry makes equivalent, so any
disagreement between them is the SOLVE's angle-to-angle reproducibility, not a physical effect.
The uniform column is the fixed-map sweep's uniform arm at the same pair and shows how exactly
the equivalence holds in the forward itself.

| shape | pair, deg | J joint a | J joint b | relative gap % | uniform arm relative gap % |
|---|---|---|---|---|---|
| T_shape | 0 and 180 | 610.49 | 610.08 | 0.07 | 0.007 |
| T_shape | 22.5 and 157.5 | 669.21 | 669.23 | 0.00 | 0.004 |
| T_shape | 45 and 135 | 538.74 | 537.57 | 0.22 | 0.164 |
| T_shape | 67.5 and 112.5 | 658.86 | 656.75 | 0.32 | 0.722 |
| L_shape | 0 and 180 | 530.47 | 530.06 | 0.08 | 0.008 |
| star | 0 and 36 | 145.23 | 156.26 | **7.59** | 0.000 |
| star | 9 and 27 | 140.69 | 139.27 | 1.01 | 0.000 |

COMPUTED, and there are two readings.

**Reading one, the noise floor.** On T_shape and L_shape the re-solve reproduces itself across
symmetry-equivalent angles to **0.32 percent or better**, which is the resolution at which any
angle difference on those shapes can be believed. The T_shape depth-budget gap between 90 and
135 degrees is 0.25 percent, inside that floor, which is why Section 1 calls it a tie.

**Reading two, and it is a real limitation.** The star's 0-against-36 pair disagrees by 7.59
percent while its uniform arm agrees exactly. The cause is the warm start, not the physics: at
0 degrees the warm map is the stored map unrotated, while at 36 degrees it is a bilinearly
interpolated rotation of the same map, so the two solves start from genuinely different points
and land in different local minima. The pairs where BOTH members are interpolated (star 9 and
27, T_shape 22.5 and 157.5) agree to 1.01 percent and 0.00 percent. **The star's headline gap,
0 degrees against 18 degrees, is 12.7 percent at the scan budget and 24.0 percent at depth,
both larger than that 7.59 percent floor, which is why the star verdict stands.**

**One consequence that must be stated because it biases the pass in a known direction.** At 0
degrees the warm start is the actual stored 0-degree solution, which already absorbed 40 to 80
forward equivalents of earlier optimization at that exact angle; at every other angle the warm
start is a rotated approximation. **The whole design therefore FAVOURS 0 degrees.** Both
confirmed moves (cross 30 to 45, star 0 to 18) are moves AWAY from the favoured angle, so the
bias works against them. The one refutation that depends on 0 degrees winning (cross,
permittivity) is the one where this bias helps, and it should be read with that in mind.

---

## 6. Health gates

COMPUTED over all **88 scored arms** (continuous and 4 bpp, scan and depth, all five
shape-actuator pairs).

* **Energy-residual gate: zero violations.** Worst relative energy residual at any arm's own
  stop is **1.57 percent** (L_shape, 157.5 degrees) against the standing 5 percent threshold.
* **Numerical clipping: exactly zero.** Maximum temperature-step clip fraction, temperature
  clamp fraction and Q_rf cap fraction are **0.0000 on all 88 arms**.
* **Operating-ceiling flag (250 C):** **one location, and it is not an optimum.** The cross
  permittivity arm at 30 degrees reaches **252.1 C** (4 bpp) and 252.2 C (continuous) at its
  own J-stop, 2.1 C over. Its own optimum at 0 degrees reaches 212.0 C and the 45-degree arm
  220.6 C, both comfortably under. No conductivity-only arm anywhere in the pass exceeds the
  ceiling; the worst is the L_shape at 157.5 degrees, 249.8 C. For comparison, the fixed-map
  sweep's L_shape 135-degree uniform arm reached 254.5 C; the joint arm at that angle reaches
  242.9 C at the scan budget and 241.9 C at depth, so the re-solve REMOVED that exceedance.
* **Horizon flags:** T_shape 67.5 and 112.5 degrees, L_shape 45 and 67.5 degrees (scan), plus
  T_shape 90 degrees and cross conductivity 45 degrees at the depth budget. Those four J_phi
  values are bounds. Both depth-budget bounds already win their comparison.
* **Absorbed power is NOT matched**, and the spread across all 88 arms is **163.1 to 608.8
  W/m** against the 500.0 W/m uniform calibration target. J_phi charges over-melt and
  under-melt symmetrically, which removes the crudest dose gaming, but no J_phi or IoU
  comparison in this report is a comparison at equal delivered energy. Two specific cases worth
  naming: the star's move from 0 to 18 degrees raises absorbed power from 336.4 to 412.4 W/m at
  depth, so part of that win is dose; the cross conductivity move from 30 to 45 degrees LOWERS
  absorbed power from 370.1 to 263.5 W/m at depth while improving J_phi by 26 percent, so that
  win is not dose.
* **Mean relative density at the deliverables' own melt stops spans 0.6308 to 0.7943.** These
  are melt-stop reads, not end-of-horizon reads, and nothing here is an oversinter claim.
* **Gradient health.** No FD gate was re-run in this pass. The gradient is the filtered
  conductivity-channel gradient gated in `MULTISTART_REPORT.md` Section 2 (gradient-direction
  probe 1.84e-07 on the rectangle, 2.56e-06 on the square, filtered layer) and the filtered
  permittivity-channel gradient gated in `EPS_CHANNEL_REPORT.md` Section 3.2
  (gradient-direction probe 2.13e-07 on the square, 1.84e-07 on the cross). **What is NOT
  gated is whether those gates hold at a NONZERO rotation angle.** The rotation changes only
  the part mask and the rasterized geometry, not the operator assembly, the objective or the
  chain rule, so the gradient is expected to be equally correct; that expectation is ASSUMED,
  not measured, and it is the first thing to check if any of these numbers is ever questioned.

---

## 7. The quantization finding, which is what reversed the permittivity verdict

COMPUTED. The cost of quantizing the solved continuous map to 4 bits per pixel, measured on
every angle of every arm in the pass:

| where | quantization cost, percent of J_phi |
|---|---|
| all 27 conductivity-only scan angles | -0.8 to +1.6 |
| cross permittivity, 15, 30 and 45 degrees, both budgets | -2.5 to +1.2 |
| **cross permittivity, 0 degrees, 15 forward-equivalent scan** | **+182.7** (80.04 to 226.30) |
| **cross permittivity, 0 degrees, 40 forward-equivalent depth** | **+45.5** (68.89 to 100.22) |

One point in the whole pass is two orders of magnitude off the rest, and it is exactly the
point whose ranking reversed. The 0-degree permittivity map descends from the stored
`EPS_best_cont` solution, which itself carried a +39.7 percent quantization cost in
`EPS_CHANNEL_REPORT.md` (76.25 continuous to 106.50 at 4 bpp). Structure in that map is
load bearing below the printer's level resolution, and the design filter constrains the SPATIAL
length scale but says nothing about the AMPLITUDE resolution. At the shallow budget the solve
cannot move far enough from that fragile start to escape it, so the printable arm looks bad at
0 degrees and 45 degrees appears to win; at depth it escapes and 0 degrees wins again.

Two things follow, both COMPUTED rather than argued. **The 45-degree permittivity solution is
quantization robust (-0.5 percent) while the 0-degree one is not (+45.5 percent even at
depth), so on a printability criterion rather than a J_phi criterion the 45-degree orientation
is still the better engineering answer for the cross in that channel.** And the continuous
0-degree permittivity map at depth reaches **J_phi 68.89 and IoU 0.9280**, the closest anything
in this pass gets to the SOLVED bar, but it is not printable at 4 bits per pixel and the
printable version of it is 0.8919.

---

## 8. Honest limits

1. **The scan budget is small and it is known to matter.** 6 to 7 gradient evaluations per
   start on 542 to 1624 design variables. Every scan J_phi is an upper bound, the depth check
   improved every single angle it touched, and on one of five it changed the verdict. A
   full-depth scan of every angle was not run; it would cost roughly four times this pass.
2. **The design favours 0 degrees** (Section 5), because only at 0 degrees is the warm start
   the true stored optimum for that angle rather than a rotated approximation.
3. **The star's angle-to-angle reproducibility is 7.6 percent**, an order of magnitude worse
   than the T_shape's 0.32 percent, and it is caused by the interpolated warm start. The star
   verdict clears that floor by a factor of three at depth but the floor is not small.
4. **Not dose matched**, 163.1 to 608.8 W/m (Section 6).
5. **Four J_phi values are horizon bounds**, including two of the depth-budget optima.
6. **No FD gate at a nonzero rotation angle** (Section 6).
7. **The fixed-map comparison curve is not filtered.** The sweep's graded arm is the unfiltered
   `A1_4bpp` map rigidly rotated; the joint arm is filtered at sigma = 1.5 cells. They are
   therefore not the same feasible set, and the joint arm is WORSE than the fixed-map arm at 7
   of 31 angles (T_shape 0 and 180 by 0.1 to 0.2 percent, L_shape 0, 112.5 and 180 by 0.4 to
   1.8 percent, cross conductivity 0 and 15 by 8.8 and 4.9 percent). Every one of those is at
   an angle far from the optimum, and none of them is at any shape's best angle, but a
   re-solve losing to a rotated map is a filter cost and it is not hidden.
8. **The symmetry claim the angle sets rest on is only partly verified for the L_shape.** The
   orientation sweep's stated argument gives J(theta) = J(180 - theta), and its own uniform-arm
   data CONTRADICTS that for the L_shape (693.48 at 45 degrees against 395.13 at 135 degrees).
   What does hold for the L_shape, and is what the [0, 180] scan actually needs, is the y-flip
   equivalence J(0) = J(180), verified to 0.008 percent on the uniform arm and 0.08 percent
   here. The T_shape satisfies both relations. No angle outside [0, 180] was scanned on either
   shape and that gap is inherited from the sweep.
9. **The 22.5-degree, 15-degree and 9-degree angle steps are coarse.** Both confirmed moves
   land on the coarse grid's endpoint or near it (cross 45 is the end of its reduced range,
   star 18 is the midpoint of its). No refinement between grid points was run, so the true
   joint optimum could sit between sampled angles.
10. **Single grid (120 x 120), two dimensions, static angles only.** No turntable arms: combined
    turntable and graded dopant is physically wrong in the solver today because the saturation
    map does not co-rotate at rotation events.
11. **Model, not hardware.** `ALLISON_LAW_REPLICATION.md` Section 6.1 records that the
    two-dimensional model over-predicts achievable tuned uniformity against hardware by roughly
    a factor of eight. Every IoU here is a statement about the model.
12. **The permittivity arm is MODEL ONLY** and stays that way until the eps_r material question
    named in `EPS_CHANNEL_REPORT.md` Section 11 is settled.

---

## 9. Proven, computed, assumed

**PROVEN**
* The new helper logic, red first: `test_joint_angle_lib.py` was observed failing with 14
  `ModuleNotFoundError` failures before `joint_angle_lib.py` existed, then 14 passed. It covers
  the rotated warm start (identity at 0 degrees, exact pixel permutation at 90 degrees inside
  the part, saturation 1 outside, box respected), the argmin and tie rules, the symmetry-period
  angle wrap and its rejection of a non-positive period, the ceiling flag boundary, and
  agreement of the angle sets with the orientation sweep's.
* The underlying map rotation contract, carried over: 5 tests including the zero-mismatch match
  to the production part rasterization at 90 degrees.
* The uniform driver gate at 0 degrees on all four shapes, relative difference 1.44e-06 to
  2.15e-05 against the stored library values, in both actuator channels for the cross.
* Determinism: the cross permittivity depth arms at 30 and 45 degrees reproduce to the printed
  digits across two separate invocations.

**COMPUTED**
* Every number in Sections 1 and 3 through 8.

**ASSUMED**
* That the filtered gradient gates measured at 0 degrees hold at nonzero rotation angles.
* That the rasterized rotated binary part mask is the right nominal target at angle theta,
  carried over from the library and orientation reports.
* The y-flip angle equivalence theta ~ theta + 180 for the T_shape and L_shape, spot-checked
  here to 0.08 percent.
* That an arbitrary stop time is realizable as a process control.
* That the best stored zero-degree map is a fair warm start at other angles once rotated. It is
  not optimal there by construction; Section 5 measures the size of that effect.

---

## 10. The single most valuable next layer

**Refine the two confirmed moves between angle grid points, at the depth budget.** The cross
optimum sits at 45 degrees, the END of its symmetry-reduced range, so the true optimum is at 45
degrees exactly (a mirror point) or the range reduction is wrong; that is one check. The star
optimum sits at 18 degrees, the midpoint of its reduced range and also a mirror point, so
sampling 13.5 and 22.5 degrees would say whether the minimum is genuinely pinned to the mirror
or merely near it. Six solves at depth, roughly 30 minutes in parallel, and it converts "the
angle moved to a sampled point" into "the angle moved to this angle".

**Second: an FD gate at a nonzero rotation angle.** One gate run on the cross at 45 degrees
closes the only unverified link in the chain behind these numbers, and it is a re-run of
existing code (`gate_ms.py`) with one configuration line changed.

**Third: an amplitude-resolution term or a quantization-aware final pass.** Section 7 shows the
design filter controls the spatial length scale but not the printer's level resolution, and one
solution in this pass lost 45.5 percent of its quality at the quantizer. Solving with the
quantizer in the loop, or adding a penalty on level-crossing structure, would make the
continuous-to-printable gap small everywhere instead of small almost everywhere.

---

## 11. Artifacts, absolute paths, and wall time

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New code (drivers, helper and tests only; no solver internals were modified):
* `scripts/analysis/joint_angle_lib.py` the angle sets, symmetry periods, rotated warm start,
  argmin and tie rules, symmetry-period angle wrap, ceiling flag
* `test_joint_angle_lib.py` its 14 red-first tests
* `scripts/analysis/run_joint_angle_solve.py` the per-angle re-solve driver and the depth-check
  mode
* `scripts/analysis/build_joint_tables.py` the tables
* `scripts/analysis/make_joint_figures.py` the figures

Read and imported, not modified: `fgm_solve_campaign/adjoint2d/{forward,adjoint,gradops,
shape_objective,design_filter,printability,library_solve,control,energy_gate,pins}.py`,
`scripts/analysis/orientation_map_rotation.py`, `scripts/analysis/run_orientation_optimization.py`.

Results:
* `fgm_solve_campaign/out_joint/<shape>_<actuator>/results.json` per-angle metrics, both arms,
  both starts, every L-BFGS-B evaluation
* `fgm_solve_campaign/out_joint/<shape>_<actuator>/fields/ang*.npz` melt field at stop, part
  mask, solved continuous and 4 bpp maps, design variable, full J_phi curves
* `fgm_solve_campaign/out_joint/<shape>_<actuator>_refine/results_refine.json` the depth checks
* `fgm_solve_campaign/out_joint/cross_eps_refine/results_refine_top2.json.bak` the first
  two-angle permittivity depth check, kept because it is the determinism evidence quoted in
  Section 2
* `fgm_solve_campaign/out_joint/_tables.md` all tables including the full angle tables
* `fgm_solve_campaign/logs_joint/*.log` per-run console logs

Figures, **all six viewed before delivery**:
* `fgm_solve_campaign/figs_joint/fig_joint_<shape>_<actuator>.png` (five) the J_phi against
  angle curves with the fixed-map sweep overlaid and the depth-check points marked, plus the
  solved map and melt field at the joint optimum and the fixed-map melt at its own best angle
* `fgm_solve_campaign/figs_joint/fig_joint_summary.png` the five curves side by side

**Wall time, COMPUTED and logged progressively as required.** After the first three scans
finished (cross conductivity 581 s, cross permittivity 468 s, star 464 s at 4 angles, 4 angles
and 5 angles) the projection for the two nine-angle shapes at roughly 100 to 200 s per angle in
five parallel streams was 20 to 25 minutes; actual T_shape 1363 s and L_shape 1399 s, inside
that. Depth checks: T_shape 942 s, L_shape 542 s, cross conductivity 785 s, star 664 s, cross
permittivity 1052 s for all four angles. **Total 8262 s of process time, about 62 minutes of
wall clock in up to five pinned single-thread streams** (`fgm_solve_campaign/env1.sh`). Nothing
was cut and no angle was skipped.
