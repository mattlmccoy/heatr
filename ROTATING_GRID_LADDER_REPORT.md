# The rotating forward grid ladder: is the two-dimensional forward model grid converged on the rotating actuator?

**Date:** 2026-08-03. **Scope:** the measurement that
`ROTATING_HOLDOUT_REPORT.md` Section 11 named third and
`HOLDOUT_FOLLOWUP_REPORT.md` Section 6 promoted to the blocking experiment.
**Forward runs only.** Nothing was solved in this pass, no gradient was
computed, and therefore no finite-difference gate was re-run. Nothing in either
prior report is overwritten; every number here is in a new artifact.

**Acronyms, expanded on first use.** FGM = functionally graded material (a
spatially varying dopant saturation map). EQS = electro-quasi-static (the
low-frequency Maxwell approximation the two-dimensional solver uses).
IoU = intersection over union. bpp = bits per pixel. phi = melt fraction.
J = the shape-fidelity objective. W/m = watts per metre of depth.

**Evidence tags.** PROVEN = unit-tested, red first, or reproduction-gated.
COMPUTED = measured from a real forward run in this pass. ASSUMED = a modelling
choice or an inference not measured here.

**Conventions, carried on every number and unchanged from the two prior
reports.**

    J(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi(x))^2

with chi the sub-cell AREA FILL indicator (`adjoint2d/chi_area.py`). **J is a
sum over cells and is NOT comparable between grids**; a `J_per_part_cell`
normalization is emitted and used only as a sanity reading, never as a
convergence proof. `t_stop` = argmin of J over the arm's own trajectory,
HORIZON flagged when the minimum sits on the last stored step (there are NO
horizon flags anywhere in this pass). The melted region for IoU, growth and
under-melt is phi >= 0.5. IoU is against the BINARY part mask, which is the
reading the SOLVED class threshold of 0.95 has always been quoted on; the
grid-independent area-fill IoU is tabulated alongside it and used as a control.
Absorbed power is the state-B value in W/m. Conductivity channel only; the
transferred map is 4 bpp inside the part with the dopant held at the nominal 1
outside. **The drive is recalibrated at EVERY grid** by one EQS solve and one
exact quadratic rescale, verified by a second solve, so the uniform STATIC arm
absorbs 500.00 W/m in electrical state B. Rotating arms are NOT dose matched
against each other.

**Every IoU in this report carries its grid qualifier.** That is the point of
the document.

---

## 1. The verdict, up front

**The rotating forward model is grid converged on the keyhole and is NOT grid
converged on the cross, at any grid reachable in this study.** COMPUTED, on
seven grids per shape (n = 96, 120, 160, 180, 200, 240, 360), with a uniform
dopant map so that no solved map and no map transfer is present in the primary
arm at all.

| shape, rotating arm, UNIFORM dopant | IoU spread over n >= 160 | IoU spread over n >= 200 | last ladder step | **verdict** |
|---|---|---|---|---|
| **keyhole**, continuous rotation | **0.0079** | **0.0016** | +0.0012 | **CONVERGED by grid 160** |
| **cross**, 90-degree indexing | **0.1228** | **0.0644** | -0.0237 | **NOT CONVERGED at grid 360** |

The two sequences, read straight off the ladder (COMPUTED):

    cross,   rotating uniform IoU:  0.7539  0.8803  0.7832  0.8566  0.7338  0.7982  0.7745
    keyhole, rotating uniform IoU:  0.8452  0.8026  0.7749  0.7669  0.7720  0.7724  0.7736
    grid n:                             96     120     160     180     200     240     360

**The keyhole's answer, stated so it can be quoted.** Its rotating uniform arm
settles on **IoU 0.772 +/- 0.002 for n >= 200** (0.7720 at 200, 0.7724 at 240,
0.7736 at 360). The three finest grids agree to 0.0016, which is about 15 part
cells out of 9368 at grid 360. The ladder step falls from 0.0426 (96 to 120) to
0.0012 (240 to 360), a factor of 35. **On this shape the ladder LIFTS the
ceiling: an absolute rotating IoU is now quotable, with the grid named.**

**The cross's answer, stated just as plainly.** Its rotating uniform arm changes
sign at every single one of the six ladder steps (+0.1264, -0.0971, +0.0735,
-0.1228, +0.0644, -0.0237) and still moves 0.0237 between grid 240 and grid 360,
where the part is already resolved by 9680 cells, more than nine times the
grid-120 count. There is no asymptotic range. **On this shape the ladder
CONFIRMS the ceiling: no absolute rotating IoU on the cross is quotable at any
grid, and the durable claim remains the ranking.**

**And the finding that matters most for the dissertation.** COMPUTED. **Grid
120, the grid every dopant map in this campaign was solved on, is the single
most favourable grid in the entire cross ladder for the rotating arm and the
single least favourable for the static arm.** Its rotating uniform IoU 0.8803 is
the ladder MAXIMUM and its static uniform IoU 0.5515 is the ladder MINIMUM, so
the measured benefit of rotation at grid 120 is 0.3288 IoU points against a mean
of **0.1886** over the finer grids, a factor of **1.74**. The same effect is
present on the keyhole and is milder: 0.2275 at grid 120 against a mean of
0.1481 over n >= 160, a factor of 1.54.

**Consequence for the cross's SOLVED-class number, stated without softening.**
The cross's transferred four-angle solved map clears the 0.95 class line at
**exactly one grid out of the seven** (0.9700 at grid 120). At the next-best
grid it reads 0.9446 (grid 180) and elsewhere 0.7750 to 0.8484. That grid-120
number is not wrong, but it sits at the top of a scatter of 0.195 IoU points, and
this report is the evidence that it must never be quoted without its grid.

**All 56 ranking tests survive the whole ladder.** COMPUTED, 28 per shape:
at every one of the seven grids, on J and on IoU alike, the rotating arm beats
its static comparator and the transferred solved map beats its own rotating
uniform comparator. The prior reports established this at two grids. It now
holds at seven, up to a grid nine times finer in cell count. **This is the
claim the dissertation already makes, and the ladder strengthens rather than
threatens it.**

---

## 2. What was run, and the gate it passed first

### 2.1 The arms

`scripts/analysis/run_rot_grid_ladder.py`. Per shape, per grid, four arms:

* `ROT_uniform`, the rotating actuator at saturation 1 everywhere. **The primary
  arm.** It contains no solved map and no transfer, so anything it does across
  the ladder is discretization and nothing else.
* `STATIC_uniform`, the same uniform map with the part held still. The contrast
  arm the task asked for; its 120 to 160 movement was already known at +0.0637
  IoU on the cross.
* `QS_uniform`, the quasi-static angle average of the rotating uniform arm, the
  infinitely-fast-cycle limit. It separates any grid movement from the finite
  cycle time.
* `ROT_transfer`, the four-angle map solved at grid 120, moved to this grid by
  the production resample and co-rotated. **This arm carries a solved map and is
  therefore NOT evidence about the forward on its own**; it is here so the map's
  transfer curve rides the same ladder.

The actuator is the arm's own stored turntable program, carried across every
grid unchanged, because a program is a wall-clock object the machine would run
whatever the simulation grid: the cross's 90-degree indexing at 2.0 s on four
candidate angles, and the keyhole's divisor-aware re-emitted twelve-position
continuous-rotation program from `HOLDOUT_FOLLOWUP_REPORT.md` Section 2.4.

### 2.2 The reproduction gate, run before any new grid was trusted

PROVEN, in the sense of a reproduction gate against previously stored results,
and read from the stored JSON at run time rather than transcribed, so a
transcription error cannot pass it.

| shape | grid | arm | stored J | this ladder J | stored IoU | this ladder IoU |
|---|---|---|---|---|---|---|
| cross | 120 | ROT_uniform | 98.46 | **98.46** | 0.8803 | **0.8803** |
| cross | 120 | STATIC_uniform | 463.25 | **463.25** | 0.5515 | **0.5515** |
| cross | 160 | ROT_uniform | 442.71 | **442.71** | 0.7832 | **0.7832** |
| cross | 160 | STATIC_uniform | 724.72 | **724.72** | 0.6152 | **0.6152** |
| keyhole | 120 | ROT_uniform | 173.10 | **173.10** | 0.8026 | **0.8026** |
| keyhole | 120 | STATIC_uniform | 509.96 | **509.96** | 0.5750 | **0.5750** |
| keyhole | 160 | ROT_uniform | 384.03 | **384.03** | 0.7749 | **0.7749** |
| keyhole | 160 | STATIC_uniform | 759.35 | **759.35** | 0.6328 | **0.6328** |

Eight comparisons, all PASS, exact to the printed digits, against tolerances of
5e-3 relative on J and 5e-4 absolute on IoU. Sources:
`out_rot_holdout/cross_index90.json` and `out_rot_holdout/keyhole_cont_fixedprog.json`.
The transferred-map arms reproduce too and are not part of the formal gate:
cross J 25.89 and IoU 0.9700 at grid 120 and J 303.70 and IoU 0.8052 at grid 160;
keyhole J 7.84 and IoU 0.9828 at grid 120 and J 62.39 and IoU 0.9447 at grid 160.

**One gate defect was found and fixed before any result was read.** The first
version of the reproduction gate returned `ALL_PASS: true` on a ladder that
overlapped the stored grids in ZERO places, because an empty conjunction is
true. That is a false green of exactly the kind that makes a diagnostic worse
than no diagnostic. The gate now counts its comparisons and reports
`NOT CHECKED` when it made none, distinctly from `PASS`. The alignment-probe
files correctly read `NOT CHECKED, no overlap between this ladder and the stored
grids (0 comparisons)`.

### 2.3 The drive, recalibrated at every grid

COMPUTED. The pinned drive absorbs very different power at different grids, so
this is not optional.

| grid n | cross: pinned 2815.4 V absorbs | cross: recalibrated drive | keyhole: pinned 2428.2 V absorbs | keyhole: recalibrated drive |
|---|---|---|---|---|
| 96 | 462.0 W/m | 2928.9 V | 439.6 W/m | 2589.6 V |
| 120 | 500.0 W/m | 2815.4 V | 426.1 W/m | 2630.3 V |
| 160 | 456.3 W/m | 2947.0 V | 380.3 W/m | 2784.0 V |
| 180 | 520.3 W/m | 2759.8 V | 366.0 W/m | 2838.2 V |
| 200 | 370.0 W/m | 3272.7 V | 359.2 W/m | 2865.0 V |
| 240 | 483.4 W/m | 2863.3 V | 349.1 W/m | 2906.0 V |
| 360 | 435.8 W/m | 3015.6 V | 333.3 W/m | 2974.2 V |

Every recalibrated drive is verified at **500.00 W/m** by a second EQS solve.
The grid-120 cross row returning the pinned voltage unchanged is a check that
the stored configuration really is calibrated at grid 120, and the grid-160 rows
reproduce `ROTATING_HOLDOUT_REPORT.md` Section 3.4 to the printed digit on both
shapes. Note that the pinned drive absorbs 520.3 W/m at grid 180 and 370.0 W/m
at grid 200: the absorbed power at fixed voltage is itself not monotone in the
grid, which is an independent sighting of the same rasterization sensitivity
Section 4 discusses.

---

## 3. The full ladder

### 3.1 cross, 90-degree indexing at 2.0 s

| grid | arm | J | IoU | IoU area | growth % | under % | stop s | P W/m | Eres % |
|---|---|---|---|---|---|---|---|---|---|
| 96 | ROT_uniform | 160.44 | 0.7539 | 0.7326 | 13.69 | 14.29 | 319.0 | 500.0 | 1.04 |
| 96 | STATIC_uniform | 250.01 | 0.6383 | 0.6211 | 11.90 | 28.57 | 287.5 | 500.0 | 1.06 |
| 96 | ROT_transfer | 109.47 | 0.8090 | 0.7760 | 5.95 | 14.29 | 404.5 | 382.3 | 0.47 |
| 120 | ROT_uniform | 98.46 | **0.8803** | 0.8578 | 9.65 | 3.47 | 317.0 | 500.0 | 0.82 |
| 120 | STATIC_uniform | 463.25 | **0.5515** | 0.5450 | 5.02 | 42.08 | 226.5 | 500.0 | 0.74 |
| 120 | ROT_transfer | 25.89 | **0.9700** | 0.9121 | 3.09 | 0.00 | 464.5 | 369.4 | 0.47 |
| 160 | ROT_uniform | 442.71 | 0.7832 | 0.7544 | 16.25 | 8.96 | 342.5 | 500.0 | 1.28 |
| 160 | STATIC_uniform | 724.72 | 0.6152 | 0.6034 | 6.67 | 34.38 | 258.5 | 500.0 | 0.70 |
| 160 | ROT_transfer | 303.70 | 0.8052 | 0.7924 | 7.50 | 13.44 | 425.0 | 384.9 | 0.67 |
| 180 | ROT_uniform | 298.94 | 0.8566 | 0.8232 | 10.08 | 5.70 | 321.0 | 500.0 | 0.96 |
| 180 | STATIC_uniform | 1038.99 | 0.5575 | 0.5476 | 3.47 | 42.31 | 229.0 | 500.0 | 0.50 |
| 180 | ROT_transfer | 106.66 | 0.9446 | 0.8853 | 4.46 | 1.32 | 432.5 | 386.7 | 0.57 |
| 200 | ROT_uniform | 824.97 | **0.7338** | 0.7306 | 20.00 | 11.94 | 347.0 | 500.0 | 1.51 |
| 200 | STATIC_uniform | 1108.80 | **0.6464** | 0.6403 | 14.31 | 26.11 | 297.5 | 500.0 | 1.29 |
| 200 | ROT_transfer | 633.25 | 0.7750 | 0.7622 | 11.11 | 13.89 | 468.0 | 354.9 | 0.90 |
| 240 | ROT_uniform | 741.59 | 0.7982 | 0.7935 | 13.15 | 9.68 | 338.5 | 500.0 | 1.14 |
| 240 | STATIC_uniform | 1703.50 | 0.5774 | 0.5737 | 2.01 | 41.10 | 228.0 | 500.0 | 0.26 |
| 240 | ROT_transfer | 409.92 | 0.8484 | 0.8422 | 6.30 | 9.82 | 427.0 | 390.6 | 0.62 |
| 360 | ROT_uniform | 2188.23 | 0.7745 | 0.7741 | 19.38 | 7.54 | 358.0 | 500.0 | 1.47 |
| 360 | STATIC_uniform | 3790.84 | 0.6067 | 0.5999 | 4.34 | 36.69 | 242.0 | 500.0 | 0.57 |
| 360 | ROT_transfer | 1384.23 | 0.8213 | 0.8185 | 9.71 | 9.90 | 458.0 | 375.3 | 0.80 |

Part cells: 672, 1036, 1920, 2420, 2880, 4380, 9680.

### 3.2 keyhole, continuous rotation, divisor-aware twelve-position program

| grid | arm | J | IoU | IoU area | growth % | under % | stop s | P W/m | Eres % |
|---|---|---|---|---|---|---|---|---|---|
| 96 | ROT_uniform | 75.89 | 0.8452 | 0.8366 | 10.61 | 6.52 | 542.0 | 310.5 | 0.58 |
| 96 | STATIC_uniform | 238.88 | 0.6368 | 0.6319 | 18.48 | 24.55 | 302.0 | 500.0 | 1.55 |
| 96 | ROT_transfer | 9.53 | 0.9702 | 0.9515 | 1.82 | 1.21 | 692.5 | 258.0 | 0.06 |
| 120 | ROT_uniform | 173.10 | 0.8026 | 0.7997 | 13.59 | 8.83 | 580.0 | 302.0 | 0.91 |
| 120 | STATIC_uniform | 509.96 | 0.5750 | 0.5659 | 30.68 | 24.85 | 336.0 | 500.0 | 2.27 |
| 120 | ROT_transfer | 7.84 | **0.9828** | 0.9663 | 1.36 | 0.39 | 745.0 | 245.9 | 0.13 |
| 160 | ROT_uniform | 384.03 | **0.7749** | 0.7709 | 13.45 | 12.09 | 554.5 | 314.0 | 0.95 |
| 160 | STATIC_uniform | 759.35 | 0.6328 | 0.6293 | 26.14 | 20.17 | 339.0 | 500.0 | 1.94 |
| 160 | ROT_transfer | 62.39 | 0.9447 | 0.9318 | 4.01 | 1.74 | 722.0 | 258.7 | 0.31 |
| 180 | ROT_uniform | 492.44 | **0.7669** | 0.7661 | 12.32 | 13.86 | 517.0 | 322.6 | 0.91 |
| 180 | STATIC_uniform | 950.95 | 0.6389 | 0.6330 | 25.49 | 19.83 | 338.5 | 500.0 | 1.91 |
| 180 | ROT_transfer | 78.79 | 0.9351 | 0.9302 | 3.78 | 2.96 | 681.5 | 266.2 | 0.31 |
| 200 | ROT_uniform | 631.44 | **0.7720** | 0.7668 | 14.12 | 11.90 | 538.0 | 322.9 | 1.01 |
| 200 | STATIC_uniform | 1243.58 | 0.6244 | 0.6223 | 27.48 | 20.40 | 346.0 | 500.0 | 2.01 |
| 200 | ROT_transfer | 116.95 | 0.9304 | 0.9216 | 3.68 | 3.54 | 685.0 | 265.6 | 0.30 |
| 240 | ROT_uniform | 900.15 | **0.7724** | 0.7693 | 12.50 | 13.10 | 539.0 | 321.5 | 0.92 |
| 240 | STATIC_uniform | 1834.51 | 0.6139 | 0.6119 | 25.39 | 23.03 | 335.5 | 500.0 | 1.93 |
| 240 | ROT_transfer | 199.64 | 0.9212 | 0.9151 | 4.21 | 4.00 | 704.5 | 264.2 | 0.33 |
| 360 | ROT_uniform | 2091.36 | **0.7736** | 0.7716 | 13.95 | 11.85 | 560.5 | 321.7 | 0.99 |
| 360 | STATIC_uniform | 4153.89 | 0.6091 | 0.6096 | 23.59 | 24.72 | 326.5 | 500.0 | 1.84 |
| 360 | ROT_transfer | 586.78 | 0.9074 | 0.9034 | 4.92 | 4.79 | 724.0 | 263.7 | 0.37 |

Part cells: 660, 1030, 1844, 2330, 2882, 4152, 9368.

`QS_uniform` is in the artifacts for every row and is omitted from these tables
for width; Section 5 reports what it says.

---

## 4. Convergence commentary, including where a Richardson reading is not available

### 4.1 The keyhole is in a usable asymptotic range and the cross is not

COMPUTED. Successive ladder steps in IoU on the rotating uniform arm:

| step | cross | keyhole |
|---|---|---|
| 96 to 120 | **+0.1264** | -0.0426 |
| 120 to 160 | **-0.0971** | -0.0277 |
| 160 to 180 | **+0.0735** | -0.0079 |
| 180 to 200 | **-0.1228** | +0.0051 |
| 200 to 240 | **+0.0644** | +0.0005 |
| 240 to 360 | **-0.0237** | **+0.0012** |

The keyhole's steps shrink by a factor of 35 across the ladder and the last
three grids lie inside 0.0016 of one another. The cross's steps alternate in
sign at every step and the largest of them (0.1228, from grid 180 to grid 200)
occurs in the SECOND HALF of the ladder, which is the opposite of convergent
behaviour.

### 4.2 Why no Richardson order is quoted as physics

**A fitted order is available and it is not credible, and saying so is the
honest reading.** The ladder 96, 120, 160, 180, 200, 240, 360 has no constant
refinement ratio, so the textbook three-grid Richardson formula does not apply.
The driver instead fits an observed order on the error against the finest grid
and reports it only when the successive differences are monotone and of one
sign. On the keyhole's monotone sub-sequence 96, 120, 160, 180 that fit returns
**p = 4.5**, and the driver's own per-file fits return 6.4 and 6.1 on the two
keyhole segments. **An order of 4 to 6 is not physically plausible** for a
scheme whose part boundary is a staircase rasterization on a Cartesian grid,
which is first order in the cell size at best. The fit is dominated by the
choice of reference grid on a short sequence, not by an asymptotic error law.

**What is defensible instead is the plain empirical statement**, which is what
Section 1 quotes: the keyhole's rotating uniform IoU is 0.772 +/- 0.002 for
n >= 200, and the cross's has no limit within the study. No extrapolated limit
is claimed for either shape.

### 4.3 The scatter is in the melt field, not in the binary target

COMPUTED, and this is the control that matters, because the obvious objection to
the cross result is that the binary part mask the IoU is read against changes
with the grid. It does, and it is not the explanation. The grid-independent
area-fill IoU scatters just as badly:

| shape, rotating uniform | binary IoU spread, n >= 160 | area-fill IoU spread, n >= 160 |
|---|---|---|
| cross | 0.1228 | **0.0926** |
| keyhole | 0.0079 | **0.0055** |

The area-fill target's own area is grid independent to 0.033 percent
(`ROTATING_HOLDOUT_REPORT.md` Section 2). **The cross's scatter therefore lives
in the computed melt field and not in the metric.**

### 4.4 The finite cycle time is not the cause

COMPUTED. `QS_uniform`, the infinitely-fast-cycle limit of the same arm, tracks
the executed program at every grid on both shapes: the cross's quasi-static IoU
sequence is 0.7539, 0.8737, 0.7796, 0.8498, 0.7338, 0.7960, 0.7760 against the
executed 0.7539, 0.8803, 0.7832, 0.8566, 0.7338, 0.7982, 0.7745, with a spread
over n >= 160 of 0.1161 against the executed 0.1228. **The scatter survives
removing the cycle time entirely**, so it is not a program-execution effect and
it is not a residue of the program-emission bug that
`HOLDOUT_FOLLOWUP_REPORT.md` fixed.

### 4.5 A candidate mechanism, offered as ASSUMED and not proven

ASSUMED. The cross is a RECTILINEAR polygon: the production shape builder puts
its unique coordinates at +/- 11.000 mm (the limb half-length) and
+/- 3.667 mm (the arm half-width). Measured by
`scripts/analysis/rot_ladder_raster_geometry.py` against the SAME production
rasterizer the forward uses, neither boundary lands on a cell edge at any grid
in the ladder, and the whole-cell rounding of both is at the 0.3 to 3.7 percent
level and is NOT monotone in the grid:

| grid n | cell size mm | limb half, exact to raster cells | limb error % | arm half, exact to raster cells | arm error % |
|---|---|---|---|---|---|
| 96 | 0.6316 | 17.417 to 17.0 | **-2.39** | 5.806 to 6.0 | **+3.35** |
| 120 | 0.5042 | 21.817 to 22.0 | **+0.84** | 7.272 to 7.0 | **-3.74** |
| 160 | 0.3774 | 29.150 to 29.0 | -0.51 | 9.717 to 10.0 | +2.92 |
| 180 | 0.3352 | 32.817 to 33.0 | +0.56 | 10.939 to 11.0 | +0.56 |
| 200 | 0.3015 | 36.483 to 36.0 | -1.32 | 12.161 to 12.0 | -1.32 |
| 240 | 0.2510 | 43.817 to 44.0 | +0.42 | 14.606 to 15.0 | +2.70 |
| 360 | 0.1671 | 65.817 to 66.0 | +0.28 | 21.939 to 22.0 | +0.28 |

Across the seven grids the cross's rotating uniform IoU correlates with the
LIMB rounding error at Pearson r = **+0.773**, with the arm rounding error at
r = -0.465, and with the raster-against-area area error at r = +0.056.
**Seven points and one plausible predictor is suggestive and is not a
mechanism.** The correlation is reported because it is the only geometric
quantity in hand that tracks the scatter, and it is labelled ASSUMED because a
correlation on seven points with a competing monotone confound (the grid number
itself) cannot establish causation. What IS established is that the geometric
representation error does not shrink monotonically along this ladder, so a
non-monotone IoU sequence is not evidence of a broken solver.

The keyhole, by contrast, is an imported polygon with curved boundary segments
whose raster-against-area error is between +0.116 and +0.720 percent at every
grid in its ladder, a factor of three tighter than the cross's, and it converges.

---

## 5. What this changes for the two prior reports

### 5.1 The cross verdict of `HOLDOUT_FOLLOWUP_REPORT.md` Section 3.2 is UPHELD and sharpened

That report named the cross's grid-120 to grid-160 class loss FORWARD
NON-CONVERGENCE, with an explicit at-this-budget caveat, on the evidence of a
native re-solve at grid 160 that recovered only 42.4 percent of the lost IoU.
**This ladder confirms the diagnosis independently and removes the budget
caveat from the forward half of it**, because the primary arm here contains no
solve at all: the rotating uniform cross arm, with no dopant map and nothing to
optimize, moves 0.1228 IoU points over n >= 160 and 0.0644 over n >= 200. The
question "is the cross's SOLVED-class loss at grid 160 a modelling artifact that
finer grids resolve, or a persistent model property?" has an answer:
**it is a persistent model property, and finer grids do not resolve it.** Grid
360 is 3.0 times finer in cell size than grid 120 and 9.3 times larger in part
cell count, and the cross's rotating uniform arm is still moving 0.0237 per
step there.

### 5.2 The keyhole verdict of `HOLDOUT_FOLLOWUP_REPORT.md` Section 3.3 is UPGRADED from inference to measurement

That report called the keyhole's class loss MAP TRANSFER, on the evidence that a
natively solved grid-160 map returned the arm to the class. It could not rule out
that some of the loss was the forward. **This ladder rules it out.** With the
keyhole's rotating forward converged to 0.0079 of IoU over n >= 160, the
transferred grid-120 map nonetheless keeps falling monotonically over exactly
that range: **0.9447 (160), 0.9351 (180), 0.9304 (200), 0.9212 (240), 0.9074
(360)**, a fall of 0.0373 IoU points across grids where the forward itself moves
0.0079. The transfer penalty is therefore real, it is the dominant term on this
shape, and it GROWS with the refinement ratio rather than saturating. COMPUTED.

**A second-order consequence, named because it is not comfortable.** The
keyhole's transferred map is inside the SOLVED class at grid 96 (0.9702) and
grid 120 (0.9828) and outside it at all five finer grids. Its class membership
is therefore also a grid-120 statement, even though its forward converges.

### 5.3 The ranking claim is strengthened

COMPUTED. `ROTATING_HOLDOUT_REPORT.md` Section 1's headline was that absolute
IoU does not survive a grid change while every ranking does, measured at two
grids. **All 28 ranking tests per shape survive all seven grids**, on J and on
IoU, up to grid 360. The dissertation's position, that rankings are the durable
claim, is the correct one and is now supported by a seven-point ladder rather
than a two-point hold-out.

---

## 6. Gates

**Energy gate: clean on all 56 forward runs.** COMPUTED. `energy_gate_violations`
is EMPTY in all four result files. The maximum relative energy residual at any
arm's own stop is **2.27 percent** (keyhole static uniform at grid 120) against
the 5 percent threshold. The cross's maximum is 1.51 percent, at grid 200. The
residual does NOT grow with refinement on either shape, which is the expected
behaviour of a part-frame march that interpolates no field.

**Stop times: no horizon flags anywhere.** COMPUTED. Every one of the 56 runs
found its J minimum strictly inside the 750.0 s horizon, so no J in this report
is an upper bound. This is a cleaner position than either prior report, both of
which carried horizon flags.

**Temperature ceiling, including one flag that does not fit the prior reports'
pattern.** COMPUTED. Ten of the 56 runs exceed the 250 C ceiling flag. Nine are
STATIC comparators, which is the expected pattern and is why they are references
rather than deliverables: every keyhole static uniform arm exceeds it, at 260.2,
290.5, 268.7, 270.2, 275.2, 273.3 and 273.2 C for grids 96 through 360, and the
cross static uniform arm exceeds it at grid 200 only (252.2 C).

**The tenth is a ROTATING arm and it is named rather than buried:** the cross's
rotating uniform arm reaches **259.5 C at grid 200**, and its quasi-static twin
reaches the same. The prior reports both stated that no rotating arm exceeds the
ceiling, and that statement was true of the grids they ran; it is not true at
grid 200. Grid 200 is also the grid at which the cross's rotating uniform IoU is
lowest in the whole ladder (0.7338) and its growth into the bed is highest
(20.00 percent), so the ceiling flag and the fidelity minimum coincide there. No
rotating arm on either shape exceeds the ceiling at any other grid.

**No gradient was computed and therefore none was re-gated.** This pass is
forward scoring only. `forward.py`, `adjoint.py`, `rot_kernel.py`,
`dwell_kernel.py`, `dwell_march.py` and `robust_rot.py` were READ and NOT
modified, as were `scripts/solve_fgm.py` and everything under `webui/`. The
standing finite-difference gate of the prior reports applies unchanged to the
one arm that carries a solved map, and it remains a SUBGRADIENT gate whose
random-direction probe bottoms at 1.22e-05 because the pinned population is the
cold powder bed.

---

## 7. Cost and wall time, and the cuts that were not needed

COMPUTED and logged from the first grid onwards, as the run rules require.

The first grid finished in **9.1 s** (cross at n = 96, four arms). The driver's
own projection at that moment, on the fourth-power cost model the CFL-limited
substep count implies, was **4.4 minutes** for the remaining three grids of the
main ladder; the actual was 3.0 minutes. That projection was three orders of
magnitude under the 8 hour line at which the task asked for cuts, so **nothing
was dropped: the keyhole was run, grid 200 was run, and the transferred-map arm
was run on both shapes.**

Because the measured cost was so far under budget, **the ladder was EXTENDED
rather than cut**, from the four requested grids to seven, adding n = 180, 240
and 360. That extension is what turned the keyhole result from "the 160 to 200
step happens to be small" into a three-grid agreement, and it is what showed the
cross still moving at grid 360.

| stream | grids | forward runs | wall |
|---|---|---|---|
| cross main ladder | 96, 120, 160, 200 | 16 | 188.8 s |
| keyhole main ladder | 96, 120, 160, 200 | 16 | 281.2 s |
| cross extension | 180, 240, 360 | 12 | 2092.4 s |
| keyhole extension | 180, 240, 360 | 12 | 3025.8 s |
| **total** | **7 grids, 2 shapes** | **56** | **5588 s** |

Two streams at a time under `fgm_solve_campaign/env1.sh` single-thread pinning,
so about 55 minutes of wall clock for 93 minutes of compute. The measured cost
growth matches the fourth-power model closely: the cross's per-grid wall is
9.1, 19.3, 54.9, 105.6, and 296.9 s for the rotating uniform arm alone at grid
360, against the outer step and horizon being pinned and grid independent.

---

## 8. Proven, computed, assumed

**PROVEN**
* The reproduction gate: eight comparisons against the stored grid-120 and
  grid-160 numbers of `out_rot_holdout/cross_index90.json` and
  `out_rot_holdout/keyhole_cont_fixedprog.json`, read from the stored JSON and
  not transcribed, all agreeing to the printed digits.
* The drive recalibration at every grid is the exact square-root law and the
  recalibrated uniform static arm absorbs 500.00 W/m at all seven grids on both
  shapes, each verified by a second EQS solve.
* The stored turntable program expands to the same wall-clock schedule at every
  grid, and a commanded position outside the candidate angle set is refused
  rather than snapped. Inherited from `adjoint2d/tests/test_robust_rot.py`,
  90 tests, unchanged by this pass.

**COMPUTED**
* Every number in Sections 1 through 7.
* The two convergence verdicts and the spreads they are read from.
* That grid 120 is the ladder maximum for the cross's rotating uniform arm and
  the ladder minimum for its static uniform arm, and the 1.74 factor on the
  rotation margin that follows.
* That the scatter survives the grid-independent area-fill target (Section 4.3)
  and survives removing the finite cycle time (Section 4.4), so it is neither a
  metric artifact nor a program-execution artifact.
* That the keyhole's transferred map falls 0.0373 IoU points over a range where
  its own forward moves 0.0079.
* All 56 ranking tests preserved across seven grids.
* The cross's rasterization rounding table and the r = +0.773 correlation.

**ASSUMED**
* That the cross's scatter is caused by the whole-cell rounding of its
  rectilinear boundaries. Suggestive on seven points with one competing
  confound, and explicitly NOT established. The decisive version of that
  experiment is named in Section 10.
* That grid 360 is fine enough to speak for "reachable grids". It is 3.0 times
  finer in cell size than the solve grid and costs about 300 s per rotating
  forward run under single-thread pinning; a two-dimensional study can go
  further, but not while remaining a routine part of a campaign loop.
* That bilinear resampling with a clip is the right transfer convention, that
  the sub-cell area fill is the right nominal target, that an arbitrary stop
  time is realizable as a process control, and that the move between indexed
  positions is instantaneous. All unchanged from
  `ROTATING_HOLDOUT_REPORT.md` Section 9.
* That the fitted orders of 4 to 6 in Section 4.2 are fitting artifacts rather
  than a real convergence order. The alternative, that the scheme really is
  fourth order at a staircase boundary, is not credible but was not
  independently disproved here.

---

## 9. Honest limits

1. **Two shapes.** The convergence verdict SPLITS between them, which is the
   strongest possible reason not to generalize it to the star, the square, the
   T or the L. In particular, nothing here says the star's rotating arm
   converges, and the star is the shape whose margin over its own rotating
   uniform comparator was already the thinnest.
2. **The cross verdict is a statement about reachable grids, not a proof of
   non-convergence.** "Still moving 0.0237 per step at grid 360" is what was
   measured. "Never converges" is not, and a grid far finer than 360 was not
   run.
3. **The transferred-map arm is not evidence about the forward.** It carries a
   grid-120 solved map and is included only so its transfer curve rides the same
   ladder. No statement in Section 1 rests on it.
4. **Nothing was re-solved at any grid in this pass**, so this report says
   nothing about whether a deeper solve at grid 360 would do better. The
   budget caveat on `HOLDOUT_FOLLOWUP_REPORT.md` Section 3.2's SOLVE half
   stands untouched.
5. **The mechanism for the cross's scatter is not established.** Section 4.5.
6. **The adjoint arms are conductivity only; the historical stored masks
   co-vary permittivity.** Unchanged, still the largest actuator gap.
7. **The model over-predicts achievable tuned uniformity by roughly a factor of
   eight against hardware** (`ALLISON_LAW_REPLICATION.md` Section 6.1). Every
   IoU here is a statement about a two-dimensional model, not a printed part.

---

## 10. The single most valuable next layer

**Re-run the cross ladder on a cross whose boundaries are exactly representable
at every grid, and on the same cross with a smoothed melt indicator.** Section
4.5 offers one candidate mechanism and cannot test it, because no grid in the
present ladder puts either cross boundary on a cell edge. Two cheap, decisive
variants:

1. **Snap the geometry to the grid.** Build the cross at each grid from a
   width chosen so that both the limb half-length and the arm half-width are an
   exact whole number of cells. If the scatter collapses, the cross's
   non-convergence is boundary rasterization and the fix is a geometry
   convention, not a solver change. If it does not, the cause is in the melt
   physics and the search moves there. Forward runs only; on the evidence of
   this pass, under twenty minutes for a seven-grid ladder.
2. **Score against a smoothed melt indicator.** The phi >= 0.5 threshold is a
   Heaviside on a field whose boundary layer is a few cells wide, so the melted
   set can gain or lose a whole ring of cells for a small change of field. A
   regularized indicator whose width is then shrunk is the standard test for
   whether a non-smooth threshold is manufacturing the scatter.

Second, and independent: **run the star and the square through the same
ladder.** The verdict split between two shapes and the campaign quotes absolute
rotating IoU on more than two.

---

## 11. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

**New code, all under `scripts/analysis/`. No solver module was modified.**
* `scripts/analysis/run_rot_grid_ladder.py` the ladder driver, its arm registry,
  its reproduction gate and its convergence reader
* `scripts/analysis/rot_ladder_raster_geometry.py` the rasterization measurement
  of Section 4.5
* `scripts/analysis/make_rot_ladder_figure.py` the figure and the merged table

**New results, with full configuration pins per grid** (grid number, part cell
count, cell size, outer step, thermal substep count, pinned and recalibrated
drive voltage, verified absorbed power, chi provenance and area,
raster-against-area delta, the executed program and its position histogram, and
the source npz and key of every map):
* `fgm_solve_campaign/out_rot_ladder/cross_ladder.json` and `_maps.npz`
* `fgm_solve_campaign/out_rot_ladder/keyhole_ladder.json` and `_maps.npz`
* `fgm_solve_campaign/out_rot_ladder/cross_align_probe_ladder.json` and `_maps.npz`
* `fgm_solve_campaign/out_rot_ladder/keyhole_align_probe_ladder.json` and `_maps.npz`
* `fgm_solve_campaign/out_rot_ladder/ladder_merged_table.json` the seven-grid
  merged table both shapes are read from in Sections 3 and 4
* `fgm_solve_campaign/out_rot_ladder/cross_raster_geometry.json` Section 4.5
* `fgm_solve_campaign/logs_rot_ladder/*.log` per-run console logs with the
  per-arm wall times and the running cost projection

**Figure, viewed before delivery:**
* `fgm_solve_campaign/figs_rot_ladder/fig_rot_grid_ladder.png`
  Panel A, the cross: intersection over union against grid number for the
  rotating uniform, static uniform and transferred-map arms, with the 0.95
  SOLVED line, the grid-120 solve grid marked, and the rotating arm's spread
  over n >= 120 shaded and labelled at 0.146. Panel B, the same axes for the
  keyhole, spread 0.036. Panel C, the size of each successive ladder step on a
  logarithmic axis for the two rotating uniform arms: the keyhole walks down by
  a factor of 35, the cross does not walk down.

**Read, not modified:**
* `ROTATING_HOLDOUT_REPORT.md`, `HOLDOUT_FOLLOWUP_REPORT.md`
* `fgm_solve_campaign/adjoint2d/{robust_rot,forward,adjoint,rot_kernel,dwell_kernel,dwell_march,dwell,chi_area,robust,printability,geometry_calibrate,topopt_objective,pins,library_solve}.py`
* `fgm_solve_campaign/out_rot_holdout/{cross_index90,keyhole_cont_fixedprog,keyhole_program_fixed}.json`
* `fgm_solve_campaign/out_rot/cross_rotavg_step90_maps.npz`,
  `fgm_solve_campaign/out_intake/keyhole_maps.npz`
* `outputs_eqs/fgm_calibrated_control/configs/cross_m0p0500.yaml`
* `scripts/analysis/novel_shapes.py`, `scripts/analysis/run_rot_holdout.py`
* `rfam_eqs_coupled.py` (the production shape builder, for the cross polygon
  coordinates quoted in Section 4.5)

`scripts/solve_fgm.py` and everything under `webui/` were NOT touched, as two
other sessions own them.
