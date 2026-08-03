# Grid hold-out of the ROTATING, INDEXED and DWELL-SCHEDULED solved arms

**Date:** 2026-08-03. **Scope:** the four arms the dissertation bolds as SOLVED
whose actuator turns the part. Solved at grid 120, scored at grid 160.
**Forward runs only.** Nothing was re-solved: every arm takes a dopant map that
was already solved on the 120 grid, moves it to the hold-out grid in the
production convention, executes the arm's OWN stored turntable program there,
and re-optimizes only the stop.

`SOLVE_ROBUSTNESS_VALIDATION.md` established the static grid hold-out and its
honest framing. Its harness is static only, and it could not touch any arm whose
actuator rotates. This pass builds the missing half and runs it.

**Acronyms, expanded on first use.** FGM = functionally graded material (a
spatially varying dopant saturation map). EQS = electro-quasi-static (the
low-frequency Maxwell approximation the two-dimensional solver uses).
IoU = intersection over union. bpp = bits per pixel. phi = melt fraction.
J = the shape-fidelity objective. W/m = watts per metre of depth.

**Evidence tags.** PROVEN = unit-tested or reproduction-gated. COMPUTED =
measured from a real forward run in this pass. ASSUMED = a modelling choice or
an inference not measured here.

**Conventions, stated once and carried on every number.**

    J(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi(x))^2

with chi the sub-cell AREA FILL indicator (`adjoint2d/chi_area.py`), not the
binary raster. The binary raster's own area moves with the grid, which is
exactly the confound `SOLVE_ROBUSTNESS_VALIDATION.md` Section 3.2 named and
could not separate; the area fill does not (COMPUTED: the cross's chi area is
2.690941e-04 m2 at grid 120 and 2.690042e-04 m2 at grid 160, a difference of
0.033 percent, while the binary raster's area is 2.127 percent BELOW the area
fill at 120 and 1.637 percent ABOVE it at 160, a swing of 3.8 percentage
points). `J_raster_chi` against the binary raster and the IoU read at the
binary-raster stop are tabulated alongside every arm so the stored grid-120
numbers stay directly comparable.

`t_stop` = argmin of J over that arm's OWN trajectory, `HORIZON` flagged when
the minimum sits on the last stored step (index 1499, 750.0 s). The melted
region for IoU, growth and under-melt is phi >= 0.5. **J is a sum over cells and
is therefore NOT comparable between grids** (1036 part cells at 120 against 1920
at 160 on the cross); read rankings and margins. Absorbed power is the state-B
value in W/m. Conductivity channel only, 4 bpp inside the part, dopant map
outside the part held at the nominal 1.

---

## 1. The four verdicts, up front

| arm | grid-120 IoU | grid-160 IoU | SOLVED class (IoU >= 0.95) | rankings on J | verdict |
|---|---|---|---|---|---|
| **cross, 90-degree indexed, solved four-angle map** | **0.9700** | **0.8052** | **YES -> NO** | 3 of 3 preserved | **absolute fidelity does NOT survive; every ranking does** |
| **cross, co-solved asymmetric dwell** | **0.9811** | **0.8033** | **YES -> NO** | 3 of 3 preserved | **absolute fidelity does NOT survive; every ranking does** |
| **star, 90-degree indexed, solved four-angle map** | 0.9382 | 0.9170 | NO -> NO | 3 of 3 preserved | **transfers well in absolute terms and keeps every ranking, but it was never in the class under this execution model** |
| **keyhole, continuous-rotation co-solve** | **0.9684** | **0.9243** | **YES -> NO** | 3 of 3 preserved | **absolute fidelity degrades least of the four; every ranking survives** |

**The single headline. Not one of the three arms that was in the SOLVED class at
grid 120 is still in it at grid 160, and all twelve ranking tests survive.**
COMPUTED. This is the same verdict shape the static hold-out reached, reached
independently on a different actuator class: **the absolute IoU is a property of
the method AT GRID 120 with an exactly reproduced dopant map, and the durable
claim is the ranking.**

**Three rankings per arm, all preserved on all four arms, on J and on IoU
alike.** COMPUTED. The rotating solved arm beats, at BOTH grids: its own
rotating uniform-dopant comparator, the best static solved arm, and the static
uniform arm. Twelve of twelve on J and twelve of twelve on IoU.

**The margins shrink where the absolute fidelity falls, and they shrink a lot on
the cross.** COMPUTED. Margin of the rotating solved arm on J against its
rotating uniform comparator: cross indexed +73.7 percent at 120 to
+31.4 percent at 160; cross dwell +74.4 to +29.7 percent; keyhole +93.5 to
+80.9 percent; star +13.6 to +3.7 percent. **The star's win over rotating
uniform is 3.7 percent at grid 160 and is the one margin thin enough that a
modest change of convention could erase it.**

**Rotation itself is the large effect and the dopant map is the small one, on
three of the four arms.** COMPUTED at grid 160: the rotating UNIFORM arm alone
carries the cross from IoU 0.6152 (static uniform) to 0.7832; the solved map
then adds 0.0220. On the star, rotating uniform 0.9108, solved map adds 0.0062.
Only on the keyhole does the map do most of the work (rotating uniform 0.7648,
solved 0.9243, an addition of 0.1595). Naming this because the dissertation's
bolded numbers attach to the combination, and at the hold-out grid the actuator
and the map are not equal partners.

---

## 2. The harness reproduces every stored grid-120 number before it is trusted

PROVEN, in the sense of a reproduction gate against previously stored results.
This is the check that says the hold-out is measuring the grid and not measuring
a new code path. The grid-120 runs in this pass are NOT copied from the stored
reports; they are re-run through the new harness and compared.

| arm | stored grid-120 number | source | this harness at grid 120 | agreement |
|---|---|---|---|---|
| cross indexed, four-angle map, quasi-static | J 34.880, IoU 0.9829 | `out_rot/cross_rotavg_step90.json` `AVG_4bpp` | **J 34.88**, IoU 0.9719 | **J exact to 5 significant figures** |
| cross indexed, executed program | J 34.8177 / 34.7608 (4 bpp), IoU 0.9866 / 0.9847 | `CONTINUOUS_ROTATION_REPORT.md` Section 7, production engine | **J 34.7427, IoU 0.9847** | J to 0.05 percent, **IoU exact** |
| cross dwell, executed 150-move program | J 34.04, IoU 0.9829 | `DWELL_SCHEDULE_REPORT.md` Section 5 | **J 34.038, IoU 0.9829** | **both exact to the printed digits** |
| star indexed, four-angle map, quasi-static | J 33.654, IoU 0.9319 | `out_rot/star_rotavg_step90.json` `AVG_4bpp` | **J 33.65**, IoU 0.9348 | **J exact to 5 significant figures** |
| keyhole, twelve-angle quasi-static, area-fill chi | J 8.0759, IoU 0.9753 | `GEOMETRY_GENERALIZATION_REPORT.md` Section 6.1 | **J 8.08, IoU 0.9753** | **both exact to the printed digits** |

All J values in the reproduction table are against the BINARY RASTER chi except
the keyhole row, whose stored number was already against the area fill.

**One reproduction does NOT hold, and it is named rather than buried.** The
star's executed-program arm gives J 32.78 and IoU 0.9321 here, against the
production engine's 24.46 and 0.9527. This harness reproduces the star's
QUASI-STATIC four-angle kernel exactly (33.65 against the stored 33.654) and
does not reproduce the engine. That is the same -27.4 percent gap
`CONTINUOUS_ROTATION_REPORT.md` Section 7 already measured and already
explained: a five-fold shape under four-fold indexing has a part mask that
genuinely changes at every rotation event, so the lab-frame engine and any
part-frame formulation are not describing the same object.
**Consequence, stated plainly: the star row of this report is a hold-out of the
PART-FRAME MODEL of that arm, not of the engine arm the dissertation quotes at
0.9527.** The cross and the keyhole carry no such gap.

---

## 3. What was held out, and what was regenerated

The dopant map is the ONLY thing carried across the grid. Everything else is
regenerated at grid 160.

**3.1 The part mask and chi.** Rebuilt by the production domain builder at the
new grid. For the keyhole, which is an imported polygon and carries no stored
configuration, the whole intake is re-run at grid 160 from the same vertex list
the solve used (`scripts/analysis/novel_shapes.py`), so its mask and chi are
regenerated exactly as the library shapes' are.

**3.2 The map transfer, with the code cited.** `robust_rot.transfer_map`
reuses `robust.resample_map`, which reproduces the production map-injection call
(`rfam_eqs_coupled.py:374-380`): `scipy.ndimage.zoom` order 1, then clip, with
the production 1 percent dead band. Outside the part the map is held at the
nominal 1, which is the convention `rot_kernel.lab_map` assumes when it rotates
the map into the lab frame, so a transfer changes the dopant map and never the
sub-pixel geometry fill of a boundary cell. Inside the part the resampled map is
re-quantized at 4 bpp through the production quantizer
`printability.quantize_in_part`, because a bilinearly resampled 4 bpp map is no
longer on the printer's level grid.

**3.3 The co-rotation, and why it must be exact at the new grid.** The
part-frame kernel rotates the transferred map into the lab frame at every
sampled position, solves the EQS problem there against the part rasterized at
that position, and rotates the heating back. PROVEN at grid 160 by red-first
tests: the 90, 180 and 270-degree rotations are EXACT pixel permutations
(`np.array_equal` against `np.rot90`, zero cells differing, zero weight lost
outside the extent), the quarter-turn round trip is the exact identity, and
resampling 120 to 160 COMMUTES with co-rotation at those angles to 1e-12. That
matters because `CONTINUOUS_ROTATION_REPORT.md` Section 6.1 measured the
production engine's lab-frame bilinear remap at 0.007 to 0.010 percent of dose
per rotation event, enough to break the standing energy gate at 375 events. This
harness carries none of it: the part-frame march never interpolates a field.

**3.4 The drive, recalibrated at each grid.** The campaign convention is that
the UNIFORM arm absorbs 500.0 W/m in electrical state B
(`FROZEN_CONVENTIONS_2D` Section 3). That calibration is grid dependent. One EQS
solve, one exact quadratic rescale (`robust.recalibrated_voltage`), one
verification solve.

| shape | pinned drive | absorbed by the uniform static arm at 160, pinned drive | recalibrated drive at 160 | verified |
|---|---|---|---|---|
| cross | 2815.4 V | 456.3 W/m | 2947.0 V | 500.00 W/m |
| star | 3223.5 V | 491.6 W/m | 3251.0 V | 500.00 W/m |
| keyhole | 2428.2 V (template start) | 380.3 W/m | 2784.0 V | 500.00 W/m |

COMPUTED, and it is also a check on the stored configurations: re-running the
same calibration at grid 120 returns the pinned voltage to a relative 4e-14 on
the cross and the star, which confirms the stored configurations really are
calibrated at 120.

**3.5 The turntable program is carried across UNCHANGED.** A program is a
wall-clock object, a list of `{position_deg, dwell_s, move_at_s}`. The machine
would run the same program whatever the simulation grid, so the program belongs
to the arm and not to the discretization. PROVEN by unit test that the expansion
to per-outer-step positions is grid independent, and the harness REFUSES a
commanded position that is not in the candidate angle set rather than snapping
it silently, which is what `dwell_march.program_step_positions` does. COMPUTED:
the executed position histogram is identical at the two grids on all four arms.

---

## 4. Every arm, every grid

Seven arms scored per grid. `ROT` is the time-resolved execution of the stored
program in the part frame; `QS` is the quasi-static angle average, the infinitely
fast cycle; `equalW` uses equal angle weights, which is the averaged kernel every
one of these maps was actually SOLVED against; the plain `QS` uses the dwell
fractions the emitted program can actually realize on the control-step grid.
`Eres` is the standing energy-residual gate at that arm's own stop, threshold
5 percent.

### 4.1 cross, 90-degree indexing at 2.0 s, solved four-angle map

| grid | arm | J | J raster | IoU | IoU area | growth % | under % | stop s | P W/m | Eres % |
|---|---|---|---|---|---|---|---|---|---|---|
| 120 | ROT_solved | 25.89 | 34.74 | **0.9700** | 0.9121 | 3.09 | 0.00 | 464.5 | 369.4 | 0.47 |
| 120 | ROT_uniform | 98.46 | 125.95 | 0.8803 | 0.8578 | 9.65 | 3.47 | 317.0 | 500.0 | 0.82 |
| 120 | QS_solved_equalW | 26.17 | 34.88 | 0.9719 | 0.9118 | 2.90 | 0.00 | 465.5 | 369.4 | 0.45 |
| 120 | STATIC_solved | 316.15 | 360.18 | 0.6755 | 0.6906 | 16.60 | 21.24 | 750.0 HORIZON | 262.6 | 1.10 |
| 120 | STATIC_uniform | 463.25 | 471.82 | 0.5515 | 0.5450 | 5.02 | 42.08 | 226.5 | 500.0 | 0.74 |
| 160 | ROT_solved | 303.70 | 309.11 | **0.8052** | 0.7924 | 7.50 | 13.44 | 425.0 | 384.9 | 0.67 |
| 160 | ROT_uniform | 442.71 | 440.36 | 0.7832 | 0.7544 | 16.25 | 8.96 | 342.5 | 500.0 | 1.28 |
| 160 | QS_solved_equalW | 305.50 | 311.13 | 0.8037 | 0.7922 | 7.71 | 13.44 | 426.5 | 384.9 | 0.69 |
| 160 | STATIC_solved | 631.36 | 647.39 | 0.6667 | 0.6609 | 11.88 | 25.42 | 439.5 | 372.4 | 1.01 |
| 160 | STATIC_uniform | 724.72 | 746.22 | 0.6152 | 0.6034 | 6.67 | 34.38 | 258.5 | 500.0 | 0.70 |

### 4.2 cross, co-solved asymmetric dwell, stored 150-move 20 s-cycle program

| grid | arm | J | J raster | IoU | IoU area | growth % | under % | stop s | P W/m | Eres % |
|---|---|---|---|---|---|---|---|---|---|---|
| 120 | ROT_solved | 25.16 | 34.04 | **0.9811** | 0.9143 | 1.93 | 0.00 | 472.5 | 365.0 | 0.44 |
| 120 | ROT_uniform | 98.38 | 125.87 | 0.8803 | 0.8573 | 9.65 | 3.47 | 317.0 | 500.0 | 0.85 |
| 120 | QS_solved_equalW | 25.49 | 34.20 | 0.9737 | 0.9148 | 2.70 | 0.00 | 475.5 | 364.9 | 0.43 |
| 120 | STATIC_solved | 316.15 | 360.18 | 0.6755 | 0.6906 | 16.60 | 21.24 | 750.0 HORIZON | 262.6 | 1.10 |
| 120 | STATIC_uniform | 463.25 | 471.82 | 0.5515 | 0.5450 | 5.02 | 42.08 | 226.5 | 500.0 | 0.74 |
| 160 | ROT_solved | 311.15 | 317.04 | **0.8033** | 0.7882 | 7.50 | 13.65 | 429.0 | 382.9 | 0.67 |
| 160 | ROT_uniform | 442.85 | 440.10 | 0.7821 | 0.7551 | 16.67 | 8.75 | 344.0 | 500.0 | 1.30 |
| 160 | QS_solved_equalW | 313.31 | 319.30 | 0.8000 | 0.7879 | 7.81 | 13.75 | 430.5 | 382.8 | 0.68 |
| 160 | STATIC_solved | 631.36 | 647.39 | 0.6667 | 0.6609 | 11.88 | 25.42 | 439.5 | 372.4 | 1.01 |
| 160 | STATIC_uniform | 724.72 | 746.22 | 0.6152 | 0.6034 | 6.67 | 34.38 | 258.5 | 500.0 | 0.70 |

The cross's dwell deliverable realizes equal dwell on four positions
{0, 90, 180, 270} degrees, so its quasi-static realized-weight and equal-weight
arms are bit identical (J 25.49 both, at both grids). It differs from the
indexed arm in the CYCLE TIME (20 s against 8 s) and in the MAP, not in the
weights.

### 4.3 star, 90-degree indexing at 2.0 s, solved four-angle map

| grid | arm | J | J raster | IoU | IoU area | growth % | under % | stop s | P W/m | Eres % |
|---|---|---|---|---|---|---|---|---|---|---|
| 120 | ROT_solved | 22.24 | 32.78 | 0.9382 | 0.8967 | 1.48 | 4.80 | 174.5 | 483.3 | 0.32 |
| 120 | ROT_uniform | 25.73 | 35.36 | 0.9170 | 0.8835 | 2.21 | 6.27 | 158.0 | 508.4 | 0.29 |
| 120 | QS_solved_equalW | 23.31 | 33.65 | 0.9348 | 0.8948 | 1.85 | 4.80 | 175.0 | 483.4 | 0.38 |
| 120 | STATIC_solved | 132.56 | 157.39 | 0.7032 | 0.6857 | 4.43 | 26.57 | 164.0 | 448.5 | 0.49 |
| 120 | STATIC_uniform | 165.42 | 192.58 | 0.6483 | 0.6237 | 7.01 | 30.63 | 142.0 | 500.0 | 0.77 |
| 160 | ROT_solved | 49.03 | 69.42 | 0.9170 | 0.8817 | 3.78 | 4.83 | 179.0 | 479.4 | 0.50 |
| 160 | ROT_uniform | 50.90 | 70.76 | 0.9108 | 0.8768 | 3.57 | 5.67 | 166.0 | 500.0 | 0.49 |
| 160 | QS_solved_equalW | 51.33 | 72.10 | 0.9172 | 0.8801 | 3.99 | 4.62 | 179.5 | 479.4 | 0.49 |
| 160 | STATIC_solved | 286.08 | 307.82 | 0.6798 | 0.6528 | 6.93 | 27.31 | 155.0 | 474.6 | 0.77 |
| 160 | STATIC_uniform | 323.85 | 347.33 | 0.6429 | 0.6201 | 8.82 | 30.04 | 144.0 | 500.0 | 0.93 |

**The star is the best absolute transfer of the four and the weakest result.**
IoU falls only 0.0212 across the grid change, which is a third of the keyhole's
fall and an eighth of the cross's. But its margin over its own rotating uniform
comparator is +13.6 percent of J at 120 and **+3.7 percent at 160**, and on IoU
+0.0212 and +0.0062. On this shape the actuator is doing nearly all the work and
the solved map is a rounding correction, which is what
`CONTINUOUS_ROTATION_REPORT.md` Section 7 already saw when the star's uniform
engine arm (0.9726) beat its solved engine arm (0.9527).

### 4.4 keyhole, continuous-rotation co-solve, stored 448-move twelve-position program

| grid | arm | J | J raster | IoU | IoU area | growth % | under % | stop s | P W/m | Eres % |
|---|---|---|---|---|---|---|---|---|---|---|
| 120 | ROT_solved | 12.49 | 24.79 | **0.9684** | 0.9558 | 1.46 | 1.75 | 718.5 | 251.7 | 0.14 |
| 120 | ROT_uniform | 191.50 | 211.07 | 0.7876 | 0.7838 | 14.27 | 10.00 | 560.5 | 310.5 | 0.98 |
| 120 | QS_solved (realized dwell) | 13.99 | 27.82 | 0.9620 | 0.9533 | 2.14 | 1.75 | 723.5 | 251.7 | 0.21 |
| 120 | QS_solved_equalW (design model) | **8.08** | 23.32 | **0.9753** | 0.9656 | 2.14 | 0.39 | 747.5 | 245.7 | 0.20 |
| 120 | STATIC_solved | 168.35 | 191.73 | 0.8163 | 0.8162 | 14.17 | 6.80 | 501.5 | 367.2 | 0.98 |
| 120 | STATIC_uniform | 509.96 | 524.99 | 0.5750 | 0.5659 | 30.68 | 24.85 | 336.0 | 500.0 | 2.27 |
| 160 | ROT_solved | 78.54 | 97.85 | **0.9243** | 0.9202 | 4.56 | 3.36 | 701.0 | 264.8 | 0.34 |
| 160 | ROT_uniform | 412.06 | 436.76 | 0.7648 | 0.7638 | 16.43 | 10.95 | 554.0 | 322.2 | 1.13 |
| 160 | QS_solved (realized dwell) | 81.39 | 104.23 | 0.9209 | 0.9168 | 4.23 | 4.01 | 694.5 | 264.8 | 0.31 |
| 160 | QS_solved_equalW (design model) | 63.58 | 87.36 | 0.9331 | 0.9277 | 3.80 | 3.15 | 717.0 | 258.5 | 0.28 |
| 160 | STATIC_solved | 351.91 | 376.13 | 0.7991 | 0.7945 | 14.43 | 8.57 | 440.0 | 404.0 | 1.05 |
| 160 | STATIC_uniform | 759.35 | 781.97 | 0.6328 | 0.6293 | 26.14 | 20.17 | 339.0 | 500.0 | 1.94 |

---

## 5. A finding this pass did not go looking for: the keyhole's emitted machine program cannot realize its own design

COMPUTED, and it is independent of the grid. The keyhole's SOLVED-class number
0.9753 is the twelve-angle average with EQUAL weights, which is the kernel the
map was solved against. The turntable program the pipeline emitted for it
allocates 40 control steps per 20 s cycle across 12 positions, which does not
divide, so it realizes dwell fractions
**0.100 / 0.100 / 0.100 / 0.100 / 0.075 x 8** instead of 1/12 each. At grid 120,
in the quasi-static limit and with everything else held fixed, that costs

    J 8.08 -> 13.99   (+73.2 percent),   IoU 0.9753 -> 0.9620

**and 0.9620 is still inside the SOLVED class, so the arm survives its own
program at grid 120.** At grid 160 the same gap is +28.0 percent on J. The
finite cycle time costs a further +12.0 percent of J at 120 and +3.6 percent at
160 (the program march against its own quasi-static limit). None of this is a
grid effect; it is a program-emission effect that was invisible until the
program was actually executed. **ASSUMED and worth one cheap experiment: a cycle
length that is an integer multiple of 12 control steps would remove the gap
entirely.**

The other three arms have essentially no realization gap: 0.3 percent or less on
J on the cross indexed arm, exactly zero on the cross dwell arm, -0.3 percent on
the star. Their programs divide evenly across four positions.

The finite cycle time is small everywhere: the program march is worse than its
own quasi-static limit by +1.4 percent (cross indexed), +1.3 percent (cross
dwell), +4.5 percent (star) and +12.0 percent (keyhole) of J at grid 120.
**The quasi-static design model is therefore a good model of the executed
program on all four arms**, which is a positive result for the solve method and
is measured here at two grids rather than assumed.

---

## 6. Mechanism: what actually fails at grid 160

COMPUTED, and visible in the figure. On both cross arms the failure is
**limb starvation**. Under-melt of the part goes from 0.00 percent at grid 120 to
13.4 percent at grid 160 while growth into the bed FALLS from 9.65 to 8.96
percent on the uniform arm and rises only modestly on the solved arm. At grid
120 the solved cross melts its limbs completely and spills slightly past the
tips; at grid 160 the limb tips do not melt at all. The keyhole shows the same
sign much more weakly (under-melt 1.75 to 3.36 percent). This is the same
starvation signature the static hold-out reported for the square.

**But the forward model is not grid converged in this metric, and that limits
what any grid hold-out can prove.** COMPUTED, and it is the same caveat as the
static pass, re-measured on this actuator class. Between the grids, at fixed
uniform dopant and with the drive recalibrated at each grid:

| arm, uniform dopant | IoU at 120 | IoU at 160 | change |
|---|---|---|---|
| cross, rotating | 0.8803 | 0.7832 | **-0.0971** |
| cross, static | 0.5515 | 0.6152 | **+0.0637** |
| star, rotating | 0.9170 | 0.9108 | -0.0062 |
| star, static | 0.6483 | 0.6429 | -0.0054 |
| keyhole, rotating | 0.7876 | 0.7648 | -0.0228 |
| keyhole, static | 0.5750 | 0.6328 | **+0.0578** |

The uniform arm, which contains no solved map at all, moves by up to 0.097 IoU
points and moves in BOTH DIRECTIONS on the same shape. **A grid hold-out is
therefore a joint test of map transfer AND discretization convergence, and this
pass cannot separate the two.** Saying so is more useful than a verdict that
pretends it can. Note also that the cross's static uniform arm gets BETTER at
160 while its rotating uniform arm gets worse, so the convergence error is not
even of one sign within a shape.

---

## 7. Gates

**Energy gate: clean on all 56 forward runs.** COMPUTED. The maximum relative
energy residual at any arm's own stop is **2.27 percent** (keyhole static
uniform at grid 120) against the 5 percent threshold, and
`energy_gate_violations` is EMPTY in all four result files. The rotating arms
are the cleanest in the set, 0.14 to 1.30 percent, which is the expected
consequence of a part-frame march that interpolates no field.

**Temperature ceiling.** Two arms exceed the 250 C ceiling flag, both static
keyhole comparators (static uniform 290.5 C at grid 120 and 268.7 C at 160,
static solved 253.1 C at 120 and 259.6 C at 160). No rotating arm on any shape
at either grid exceeds 217.8 C. This is recorded because the static comparators
are references, not deliverables.

**Unit tests: 14 new, written red first.** `adjoint2d/tests/test_robust_rot.py`.
The `ImportError` for the absent `robust_rot` module was observed before the
module existed. The tests cover: the 90, 180 and 270-degree rotations as exact
pixel permutations at grid 160 against `np.rot90` with zero cells differing; the
quarter-turn round trip as the exact identity at grid 160; zero rotation weight
lost outside the extent at both grids; resample and co-rotation commuting at
1e-12 at the three exact angles; the transferred map holding the nominal 1
outside the part and lying on the 4 bpp level grid inside it; the same-grid
transfer being a quantization only, which is the production 1 percent dead band;
the indexing program expanding to four outer steps per position; the stored-move
expansion being grid independent; the refusal of a commanded position absent
from the candidate angle set; and the drive recalibration at grid 160 landing on
500.0 W/m to 1e-6 by a verification EQS solve. **90 tests pass** across
`test_robust_rot`, `test_robust`, `test_rot_frame`, `test_rot_kernel`,
`test_dwell_march`, `test_dwell`, `test_chi_area`, `test_printability` and
`test_geometry_calibrate`.

**No gradient was computed and therefore none was re-gated.** This pass is
forward scoring only. `forward.py`, `adjoint.py`, `rot_kernel.py`,
`dwell_kernel.py` and `dwell_march.py` were READ and not modified. The standing
finite-difference gate of the previous reports applies unchanged, and it remains
a SUBGRADIENT gate whose random-direction probe bottoms at 1.22e-05 because the
pinned population is the cold powder bed. Every solved map scored here inherits
that caveat.

---

## 8. Cost and wall time

COMPUTED, and logged from the first arm onwards as required. The first arm
(cross indexed, eight arms across two grids in its first form) cost **198.8 s**
of wall clock. That projected to about **13 minutes** for four arms, far under
the ~6 hour line at which the star was to be dropped, so **nothing was cut and
the star arm was run.** The arm set was then extended twice (adding the
quasi-static realized-weight and equal-weight comparators, then the stored melt
fields for the figure) and the whole set re-run each time; only the final
re-run's numbers appear above.

| arm | forward runs | wall, final pass |
|---|---|---|
| cross_index90 | 14 | 161.7 s |
| cross_dwell | 14 | 163.3 s |
| star_index90 | 14 | 97.7 s |
| keyhole_cont | 14 | 224.3 s |
| **total, final pass** | **56** | **647.1 s** |

Four streams ran concurrently under `fgm_solve_campaign/env1.sh` single-thread
pinning, so the wall clock of the final pass was about 4 minutes. Including the
two superseded passes and the probe runs, total real compute for this report is
under 25 minutes.

---

## 9. Proven, computed, assumed

**PROVEN**
* The 90, 180 and 270-degree rotations are exact pixel permutations at grid 160,
  with zero cells differing from `np.rot90` and zero weight lost outside the
  array extent, so the indexing schedules carry no remap error at the hold-out
  grid.
* Resampling 120 to 160 and co-rotating commute to 1e-12 at those angles.
* The transferred map holds the nominal value outside the part and lies exactly
  on the 4 bpp printer level grid inside it; a same-grid transfer is a
  quantization only.
* A stored turntable program expands to the same per-outer-step positions
  whatever the grid, and a commanded position outside the candidate angle set is
  refused rather than snapped.
* The drive recalibration is the exact square-root law and the recalibrated
  uniform arm absorbs 500.00 W/m at grid 160 on all three shapes, verified by a
  second EQS solve.
* 14 new tests, red first; 90 tests pass across the nine relevant modules.
* Reproduction gate: the harness reproduces the stored grid-120 numbers of all
  four arms, exactly on the cross dwell program (34.038 against 34.04), the
  keyhole design model (8.08 against 8.0759), the cross four-angle kernel
  (34.88 against 34.880) and the star four-angle kernel (33.65 against 33.654).

**COMPUTED**
* Every number in Sections 1, 3 through 8.
* The reproduction FAILURE on the star's engine arm, and its size.
* The keyhole program-realization gap, +73.2 percent of J at grid 120.

**ASSUMED**
* That bilinear resampling with a clip is the right way to move a solved map to
  a new grid. It is the production convention and it is what the engine would
  do, but an area-conserving or level-set transfer might transfer better. Not
  tested, and carried over unchanged from the static hold-out.
* That the part-frame march is the right execution model. It is exact in its
  energy bookkeeping and free of remap error, and it reproduces the production
  engine on the cross to 0.05 percent, but Section 2 shows it is NOT the same
  object as the engine for a part whose mask changes at a rotation event.
* That the sub-cell area fill is the right nominal target. It is grid
  independent to 0.033 percent here, which the binary raster is not, but no
  bench measurement says a melt boundary should be scored against a cell-average
  indicator.
* That an arbitrary stop time is realizable as a process control.
* That the move between indexed positions is instantaneous. Unchanged from the
  dwell and rotation reports.

---

## 10. Honest limits

1. **A grid hold-out at 120 against 160 is not a clean test of map transfer,
   because the forward model is not grid converged in this metric.** Section 6:
   the rotating uniform cross arm alone moves 0.097 IoU points, and the static
   uniform cross arm moves the OTHER WAY by 0.064. The right version of this
   test needs a grid-convergence study of the forward first, which this pass did
   not run. This is the same limit the static hold-out named and it is not
   narrowed here.
2. **Four arms on three shapes.** No claim is made about any other rotating arm
   in the campaign.
3. **The star arm is a hold-out of the part-frame model, not of the engine
   arm.** Section 2. Its grid-120 fidelity here is 0.9382, not the 0.9527 the
   dissertation quotes, and the two numbers come from different execution
   models.
4. **Not dose matched.** The drive is calibrated so the uniform STATIC arm
   absorbs 500.0 W/m at each grid. The rotating arms absorb 251.7 to 508.4 W/m
   across this pass and are NOT equalized against each other. The objective
   penalizes over-melting as well as under-melting, which removes the crudest
   dose gaming, but no comparison here is at equal delivered energy.
5. **One horizon flag**, on the cross's static solved comparator at grid 120
   (750.0 s), so that reference's J is an upper bound and the margin against it
   at grid 120 can only be better than quoted.
6. **The adjoint arms are conductivity only; the historical stored masks
   co-vary permittivity.** Unchanged, still the largest actuator gap, and no
   permittivity-co-varying comparator appears in this pass at all.
7. **Nothing was re-solved at grid 160**, so this pass cannot say whether a map
   solved AT 160 would recover the SOLVED class. That is the decisive experiment
   and it is named in Section 11.
8. **The model over-predicts achievable tuned uniformity by roughly a factor of
   eight against hardware** (`ALLISON_LAW_REPLICATION.md` Section 6.1). Every
   IoU here is a statement about a two-dimensional model, not about a printed
   part.

---

## 11. The single most valuable next layer

**Re-solve the cross's four-angle map AT GRID 160 and compare the two solved
maps directly.** This pass shows the transferred map loses 0.165 IoU points and
cannot say whether that is transfer or convergence. If the 160-solved map
recovers IoU >= 0.95 on the cross, the failure is transfer and the fix is a
better transfer convention. If it does not, the failure is convergence and no
absolute IoU at any grid should be quoted without a convergence study behind it.
That is one rotating co-solve, about 400 to 900 s at grid 160 on the evidence of
`out_rot/cross_rotavg_step90.json` (402.6 s at grid 120), and it is decisive.

**Second, and nearly free: re-emit the keyhole's turntable program with a cycle
length divisible by its position count.** Section 5 measured a +73.2 percent J
penalty at grid 120 that is pure program-emission arithmetic and has nothing to
do with the physics. One forward run confirms or refutes it.

**Third: a forward grid-convergence study on the ROTATING uniform arm**, at 96,
120, 160 and 200 with the drive recalibrated at each grid. Section 6 shows the
rotating uniform cross arm moves 0.097 IoU points between two grids and the
static one moves the other way, so the convergence behaviour of the rotating
forward is not the same as the static one and has never been measured.

---

## 12. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New code:
* `fgm_solve_campaign/adjoint2d/robust_rot.py` the hold-out harness for rotating,
  indexed and dwell arms: map transfer with co-rotation, program expansion,
  per-grid drive recalibration, and the three scorers (executed program,
  quasi-static average, static)
* `fgm_solve_campaign/adjoint2d/tests/test_robust_rot.py` 14 tests, red first
* `scripts/analysis/run_rot_holdout.py` the driver and the arm registry
* `scripts/analysis/make_rot_holdout_figure.py` the figure

New results, one JSON and one npz per arm, with full configuration pins
(grid, part-cell count, drive voltage pinned and recalibrated, verified absorbed
power, chi provenance and area, raster-against-area delta, the executed program
and its position histogram, and the source npz and key of every map):
* `fgm_solve_campaign/out_rot_holdout/cross_index90.json` and `_maps.npz`
* `fgm_solve_campaign/out_rot_holdout/cross_dwell.json` and `_maps.npz`
* `fgm_solve_campaign/out_rot_holdout/star_index90.json` and `_maps.npz`
* `fgm_solve_campaign/out_rot_holdout/keyhole_cont.json` and `_maps.npz`
* `fgm_solve_campaign/logs_rot_holdout/*.log` per-run console logs with wall times

Figure, viewed before delivery:
* `fgm_solve_campaign/figs_rot_holdout/fig_rot_holdout_class_change.png`
  the three arms whose class changes. Top row: the melted region at grid 120 and
  at grid 160 on the same axes against the nominal outline, which shows the
  cross losing its limb tips. Bottom row: the intersection-over-union ladder at
  the two grids with the 0.95 SOLVED line drawn. The star has no panel because
  its class does not change.

Read, not modified:
* `SOLVE_ROBUSTNESS_VALIDATION.md`, `CONTINUOUS_ROTATION_REPORT.md`,
  `DWELL_SCHEDULE_REPORT.md`, `GEOMETRY_GENERALIZATION_REPORT.md`
* `fgm_solve_campaign/out_rot/*.json`, `*_maps.npz`
* `fgm_solve_campaign/out_dwell/cross_turntable_deliverable.json`,
  `cross_dwell_maps.npz`
* `fgm_solve_campaign/out_lib/{cross,star}_maps.npz`
* `fgm_solve_campaign/out_intake/keyhole_novel.json`, `keyhole_maps.npz`
* `fgm_solve_campaign/adjoint2d/{forward,adjoint,rot_kernel,dwell_kernel,dwell_march,rot_frame,chi_area,robust,printability,geometry_calibrate,topopt_objective}.py`
* `outputs_eqs/fgm_calibrated_control/configs/{cross,star}_m0p0500.yaml`
* `scripts/analysis/novel_shapes.py`

A partial static cross-at-160 run from an earlier session
(`fgm_solve_campaign/out_robust_holdout2/cross.log`, four arms, pinned drive,
binary raster chi, no JSON emitted) was found and read. It is NOT reused in any
table here because its conventions differ from this pass on three axes at once
(pinned rather than recalibrated drive, binary raster rather than area-fill chi,
and a different static solved map arm), and the static comparators were re-run
under this pass's conventions instead. Its numbers are consistent in sign with
the static rows of Section 4.1.
