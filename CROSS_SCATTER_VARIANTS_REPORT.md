# Is the cross's rotating grid scatter rasterization, the metric, or the model?

**Date:** 2026-08-03. **Scope:** the two variants
`ROTATING_GRID_LADDER_REPORT.md` Section 10 named as the decisive test of the
ASSUMED mechanism its Section 4.5 offers for the cross's non-converging
rotating intersection-over-union sequence. **Forward runs only.** Nothing was
solved in this pass, no gradient was computed, and therefore no
finite-difference gate was re-run. Nothing in the prior report is overwritten;
every number here is in a new artifact with a variant suffix.

**Acronyms, expanded on first use.** FGM = functionally graded material (a
spatially varying dopant saturation map). EQS = electro-quasi-static (the
low-frequency Maxwell approximation the two-dimensional solver uses).
IoU = intersection over union. bpp = bits per pixel. phi = melt fraction.
J = the shape-fidelity objective. W/m = watts per metre of depth.

**Evidence tags.** PROVEN = unit-tested, red first, or reproduction-gated.
COMPUTED = measured from a real forward run or a real re-scoring in this pass.
ASSUMED = a modelling choice or an inference not measured here.

**Conventions, carried unchanged from the three prior reports.**

    J(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi(x))^2

with chi the sub-cell AREA FILL indicator (`adjoint2d/chi_area.py`). J is a sum
over cells and is NOT comparable between grids. `t_stop` = argmin of J over the
arm's own trajectory, horizon flagged (there are NO horizon flags anywhere in
this pass). The melted region for the published IoU is phi >= 0.5 against the
BINARY part mask. The drive is recalibrated at EVERY grid by one EQS solve and
one exact quadratic rescale, verified by a second solve, so the uniform STATIC
arm absorbs 500.00 W/m in electrical state B. Rotating arms are NOT dose
matched against each other. **Every IoU below carries its grid number.**

---

## 1. The verdict, one word: NEITHER

**The cross's rotating grid movement is not a rasterization artifact and not a
metric artifact. It survives both variants and it stands as a property of the
two-dimensional model at the grids reachable in this study.** COMPUTED, on
seven grids (n = 96, 120, 160, 180, 200, 240, 360), uniform dopant, rotating
uniform arm.

| cross, rotating uniform arm | IoU spread, n >= 160 | IoU spread, n >= 200 | last ladder step | sign changes in the six steps |
|---|---|---|---|---|
| **ORIGINAL** (published ladder) | **0.1228** | 0.0644 | 0.0237 | **5 of 5** |
| **VARIANT A**, snapped geometry | **0.1270** | 0.0664 | 0.0416 | **0 of 5** |
| **VARIANT B**, sub-cell melt metric | **0.1227** | 0.0598 | 0.0164 | **5 of 5** |
| **VARIANT A and B together** | **0.1180** | 0.0581 | 0.0342 | **0 of 5** |
| keyhole, original, for reference | 0.0079 | 0.0016 | 0.0012 | 1 of 5 |

The three cross sequences, read straight off the artifacts (COMPUTED):

    grid n:               96      120     160     180     200     240     360
    ORIGINAL:           0.7539  0.8803  0.7832  0.8566  0.7338  0.7982  0.7745
    VARIANT A:          0.9711  0.9384  0.9023  0.8574  0.8416  0.8168  0.7753
    VARIANT B:          0.7288  0.8664  0.7523  0.8536  0.7308  0.7906  0.7742

**What each variant did and did not remove, stated separately, because the two
halves of the answer are different.**

* **Variant A removes the ALTERNATION and not the MOVEMENT.** With both cross
  boundaries snapped to whole cell multiples, so the rasterization rounding the
  correlation was computed against is identically zero at every grid, the
  sequence becomes strictly monotone: six steps, all negative, zero sign
  changes, against five sign changes in five opportunities in the original.
  **But the spread over n >= 160 is 0.1270, which is LARGER than the original's
  0.1228, and the last ladder step is 0.0416 against the original's 0.0237.**
  The scatter did not collapse. It straightened.
* **Variant B removes nothing at all.** Scoring the melt front by the same
  sub-cell area-fill convention the target chi already uses moves the spread
  over n >= 160 from 0.1228 to **0.1227**, a change of 0.0001, and leaves all
  five sign changes in place. The correlation with the limb rounding gets
  STRONGER under the smoothed metric, not weaker: Pearson r = **+0.835**
  against the published **+0.773**.
* **Both together leave 0.1180**, which is 96 percent of the original spread.

**The correlation, re-computed where the task asked for it.** Neither variant
collapsed, so the re-computation is reported on both. Under variant B the
Pearson correlation of the rotating uniform IoU against the limb-boundary
rounding error over the seven grids is **+0.835** (published: +0.773, which
this pass reproduces to three decimals as an independent check of the analysis
path). Under variant A the predictor is identically zero at all seven grids, so
the correlation is **not defined** and is written as null in the artifact
rather than as a number. The predictor that replaces it under variant A, the
snapped part's own area, explains little: r = **-0.315** of the IoU against the
part area, and the area sequence is non-monotone while the IoU sequence is
monotone.

**The one-line reading.** The rasterization rounding was tracking the SIGN of
the cross's grid-to-grid movement, not its SIZE. Remove it and the same
0.13 IoU points of movement is still there, now pointing one way.

---

## 2. What variant A actually changes, measured rather than assumed

COMPUTED, and this is the finding that makes the variant interpretable.

**The part MASK is bit-identical between the original and the snapped cross at
every one of the seven grids.** Snapping moves each boundary to where the
rasterizer had already put it, so the set of cells the domain builder calls
part does not move at all:

| grid n | part cells, original | part cells, snapped | mask identical | fractional-fill cells, original | fractional-fill cells, snapped | max change in a cell's fill fraction |
|---|---|---|---|---|---|---|
| 96 | 672 | 672 | yes | 140 | 0 | 0.375 |
| 120 | 1036 | 1036 | yes | 172 | 0 | 0.4375 |
| 160 | 1920 | 1920 | yes | 236 | 0 | 0.25 |
| 180 | 2420 | 2420 | yes | 88 | 0 | 0.125 |
| 200 | 2880 | 2880 | yes | 292 | 0 | 0.5 |
| 240 | 4380 | 4380 | yes | 348 | 0 | 0.453125 |
| 360 | 9680 | 9680 | yes | 176 | 0 | 0.125 |

(`fgm_solve_campaign/out_rot_ladder/cross_snapA_mask_identity.json`.)

So variant A is precisely this experiment: **remove the sub-cell boundary
information from BOTH the material model and the target, and see whether the
movement goes away.** After the snap the material fill fraction is binary
everywhere, and the sub-cell area-fill target chi collapses onto the binary
raster, measured at **+0.0000 percent** raster-minus-area at every one of the
seven grids (`cross_raster_geometry_snapA.json`; the original ladder's
corresponding numbers run from -0.334 to +0.322 percent).

**The price of the variant, quoted as the task required.** The snap holds the
raster exact and lets the PART move by up to half a cell. Per grid:

| grid n | cell size mm | limb half: cells, shift in cells, shift in mm, percent of the dimension | arm half: cells, shift in cells, shift in mm, percent of the dimension |
|---|---|---|---|
| 96 | 0.6316 | 17, -0.417, -0.2632, **-2.39 %** | 6, +0.194, +0.1228, **+3.35 %** |
| 120 | 0.5042 | 22, +0.183, +0.0924, +0.84 % | 7, -0.272, -0.1373, **-3.74 %** |
| 160 | 0.3774 | 29, -0.150, -0.0566, -0.51 % | 10, +0.283, +0.1069, +2.92 % |
| 180 | 0.3352 | 33, +0.183, +0.0615, +0.56 % | 11, +0.061, +0.0205, +0.56 % |
| 200 | 0.3015 | 36, -0.483, -0.1457, -1.32 % | 12, -0.161, -0.0486, -1.32 % |
| 240 | 0.2510 | 44, +0.183, +0.0460, +0.42 % | 15, +0.394, +0.0990, +2.70 % |
| 360 | 0.1671 | 66, +0.183, +0.0306, +0.28 % | 22, +0.061, +0.0102, +0.28 % |

Every shift is at most half a cell by construction (PROVEN,
`test_snap_moves_each_boundary_by_at_most_half_a_cell`), which is 0.263 mm at
the coarsest grid and 0.031 mm at the finest. In percent of the dimension the
worst case is the arm half-width at grid 120, -3.74 percent. The part area
moves with it, between -2.62 and +2.67 percent of the nominal 268.89 mm2.

**The internal consistency check that says the snap is doing what it claims.**
The perturbation must vanish as the cell shrinks, and the two ladders must
approach each other. They do, at the finest grid: the snapped and original
rotating uniform arms differ by **0.2172 IoU points at grid 96 and by 0.0008 at
grid 360**. The approach is not monotone (the difference is 0.0008 at grid 180
and 0.1078 at grid 200), which is itself another sighting of the same
non-monotone behaviour this report is about.

---

## 3. What variant B actually changes, and the width-shrink diagnostic

**Variant B re-runs no physics.** COMPUTED. Every melt field it scores was
already stored by the published ladder at each arm's own optimal stop, so the
trajectory, the stop time, the drive and the target are bit-identical to the
published run by construction. The recomputed binary reading agrees with the
stored one at **all 28 arm-and-grid comparisons with an absolute difference of
exactly 0.00e+00**, despite the melt fields being archived as 32-bit floats.
That is the re-scoring gate and it is PASS.

**Which smoothed indicator, and why that one.** The campaign's target chi is
the sub-cell area fraction of each cell that lies inside the part boundary. The
defensible melt analogue under that convention is the sub-cell area fraction of
each cell that lies inside the MELT FRONT, evaluated by bilinear reconstruction
of phi on the same offset pattern `chi_area.area_fill_poly` uses. Both sides of
the overlap are then the same kind of object, and the indicator inherits chi's
own degeneracy: a front already sitting on cell edges gives back the binary set
exactly (PROVEN,
`test_melt_area_fill_reproduces_a_straight_binary_front_exactly`). The measured
limit of that reconstruction is at a right-angle corner, where the corner cell
of a convex corner reads 0.906 rather than 1.0 (PROVEN and pinned,
`test_melt_area_fill_rounds_a_convex_corner_by_a_measured_amount`); on the
cross that is eight convex and four concave corners, and the errors are of
opposite sign.

**The regularizer width was shrunk, and it behaves exactly as a regularizer
should.** The Gaussian-smoothed alternative is not defensible against chi (it
has a width with no counterpart in the target), so it is reported only as the
width-shrink diagnostic the discipline asks for. Spread of the cross's rotating
uniform arm over n >= 160, at four widths:

| melt indicator | spread over n >= 160 |
|---|---|
| Gaussian, sigma = 1.0 cell | 0.1025 |
| Gaussian, sigma = 0.5 cell | 0.1146 |
| Gaussian, sigma = 0.25 cell | 0.1221 |
| Gaussian, sigma = 0 (the original threshold) | **0.1228** |
| sub-cell melt area fill against chi | **0.1227** |

The width-zero limit returns the published metric bit for bit (PROVEN,
`test_gaussian_melt_indicator_at_zero_width_is_the_binary_threshold`), and the
sequence walks back to it monotonically as the width shrinks. A full cell of
Gaussian smoothing buys 0.0203 of the 0.1228, which is 17 percent, at the cost
of a melt front deliberately blurred over a scale the model does not have.
**There is no width at which the scatter goes away and the metric still means
anything.**

**The keyhole control, run because a metric change must not break the shape
that already converges.** COMPUTED. The keyhole's rotating uniform arm reads
spread 0.0079 over n >= 160 under the published binary metric, **0.0068** under
the sub-cell melt area fill and 0.0085 under one cell of Gaussian smoothing. It
stays converged under all three. The metric is not what separates the two
shapes.

---

## 4. Gates

**Reproduction gate.** The published cross ladder's own gate is unchanged and
still reads PASS on 4 comparisons (grids 120 and 160, rotating uniform and
static uniform, read from `out_rot_holdout/cross_index90.json` at run time).
**Variant A cannot use it and says so**: its status is
`NOT APPLICABLE, the geometry was deliberately modified by the snap, so the
stored numbers describe a different part`, with 0 comparisons made, which is
reported distinctly from PASS and from NOT CHECKED. This is the same
false-green discipline the prior pass installed. The substitute control is
PROVEN rather than run at cost: at grid 181 the snap is arithmetically a no-op
(the cell size is exactly 1/3 mm, so 11.000 mm is 33 cells and 11/3 mm is 11),
and a snapped configuration and an unsnapped one there produce an identical
part mask, an identical material fill fraction and an identical chi
(`test_snapped_and_unsnapped_cases_are_identical_where_the_snap_is_a_no_op`).
Grid 181 is deliberately NOT in the ladder: with an odd grid number the domain
builder puts a grid point at the origin, so a boundary at a whole multiple of
the cell size lands ON grid points, which is the ambiguous case and not the
exact one.

**Variant B and the combined variant carry their own re-scoring gate**, PASS at
28 comparisons each on the cross, 28 on the keyhole, tolerance 5e-4 absolute in
IoU, actual maximum deviation 0.00e+00.

**Energy gate: clean on all 28 new forward runs.** COMPUTED.
`energy_gate_violations` is EMPTY in the variant A result file. Its maximum
relative energy residual at any arm's own stop is **1.50 percent** against the
5 percent threshold, against 1.51 percent in the original cross ladder.

**Stop times: no horizon flags anywhere**, in any variant, so no J here is an
upper bound.

**Temperature ceiling, and one thing that got better.** COMPUTED. The original
cross ladder flags three arms over 250 C, all three at grid 200 (rotating
uniform 259.5 C, static uniform 252.2 C, quasi-static 259.5 C). **The snapped
cross ladder flags none at any grid.** That is a real difference and it is
named rather than buried: the grid-200 ceiling excursion the prior report
called out was tied to the boundary representation at that grid, not to the
part.

**Rankings.** COMPUTED. The original cross ladder passes all 28 ranking tests
(rotation beats static, and the transferred map beats rotating uniform, on J
and on IoU, at all seven grids), and so does the keyhole. **The snapped ladder
passes 27 of 28.** The single failure is at grid 96: the grid-120-solved
transferred map has J = 53.89 against the snapped uniform arm's J = 23.04, so
the map does not improve J there, while it still wins on IoU (0.9912 against
0.9711). The cause is visible in the number: with the boundary snapped, the
uniform rotating arm at grid 96 is already at IoU 0.9711, inside the SOLVED
class line of 0.95, so there is very little for a map solved on a different
geometry to add. **This is one ranking test in a corner case and it is reported
because the campaign's durable claim is the ranking; it does not touch the
original geometry's 28 of 28.**

---

## 5. What this changes for the prior reports

### 5.1 `ROTATING_GRID_LADDER_REPORT.md` Section 4.5 is DEMOTED

That section offers the whole-cell rounding of the cross's rectilinear
boundaries as an ASSUMED mechanism, on a Pearson correlation of +0.773 over
seven points, and states in its own words that it is suggestive and not a
mechanism. **It is now measured and it is not the cause of the movement.**
Removing the rounding entirely leaves 0.1270 of spread over n >= 160 where
there was 0.1228. What the correlation was picking up is the SIGN pattern: with
the rounding removed the sequence stops alternating and becomes monotone, zero
sign changes against five. That is a real and reportable effect of the
rasterization, and it is a different claim from the one the correlation was
being read as.

### 5.2 `ROTATING_GRID_LADDER_REPORT.md` Section 1's cross verdict is UPHELD and hardened

The verdict was that the cross's rotating forward is not grid converged at any
grid reachable in the study, and that no absolute rotating IoU on the cross is
quotable at any grid. **Both variants leave that standing.** The snapped ladder
is still moving 0.0416 IoU points between grid 240 and grid 360, which is
larger than the original ladder's 0.0237 at the same step, on a part resolved
by 9680 cells. The smoothed metric is still moving 0.0164 there.

### 5.3 The keyhole's convergence is metric independent

COMPUTED and new: the keyhole's rotating uniform arm converges under the binary
threshold, under the sub-cell melt area fill and under a cell of Gaussian
smoothing alike (spreads 0.0079, 0.0068, 0.0085 over n >= 160). The split
between the two shapes is not an artifact of how the melted set is counted.

---

## 6. The consequence for the dissertation, and exactly which sentences change

**The cross's non-convergence SURVIVES as a model property.** It does not
resolve into a rasterization artifact. The Chapter 5 wording the writer is
about to patch should therefore change in exactly one sentence class, and
should NOT change in three others.

**CHANGES: any sentence that attributes the cross's grid scatter to
rasterization rounding, or that hedges the non-convergence as probably a
boundary-representation artifact.** This includes any sentence built on the
r = +0.773 correlation, and any phrasing of the form "the cross's
non-monotonicity is likely an artifact of the staircase boundary". Those must
either be deleted or narrowed to the claim that survives: *the rasterization
rounding controls the SIGN of the grid-to-grid movement, not its size; with the
boundaries snapped to the grid the sequence becomes monotone and the movement
is unchanged.*

**DOES NOT CHANGE, and is now better supported:**

1. Every sentence saying an absolute rotating IoU on the cross must carry its
   grid number, and that the grid-120 solved-class number 0.9700 is a
   grid-120 statement. Two independent attempts to make the cross behave
   failed.
2. Every sentence resting on RANKINGS rather than absolute values. The original
   geometry's 28 of 28 ranking tests are untouched by this pass.
3. Every sentence about the keyhole converging and its transferred map decaying
   with refinement. The keyhole is unmoved under the new metric.

**A sentence class that can now be written that could not be before:** the
scatter is not an artifact of the non-smooth melt threshold either. That was an
open objection to the whole rotating hold-out reading and it is now closed by
measurement, at 0.0001 of IoU spread, on melt fields that re-ran no physics.

---

## 7. Cost and wall time

COMPUTED and logged from the first grid onwards, as the run rules require. No
cuts were needed and none were made silently.

| stream | grids | forward runs | wall |
|---|---|---|---|
| variant A, snapped cross ladder, four arms per grid | 96, 120, 160, 180, 200, 240, 360 | 28 | **2547.0 s** |
| variant B, cross re-scoring | the same seven, from stored fields | 0 | 6.7 s |
| variant B, keyhole re-scoring | seven | 0 | 6.2 s |
| variant A and B combined re-scoring | seven | 0 | 2.3 s |
| snapped rasterization measurement | eight, including the 181 control | 0 | 12 s |
| unit tests, 23 of them | n/a | 0 | 5.3 s |

Variant A per grid: 21.4, 39.5, 134.0, 161.0, 227.2, 479.1, 1484.8 s, single
thread pinned under `fgm_solve_campaign/env1.sh`. The task's estimate was about
twenty minutes per variant; variant A took 42.5 minutes and variant B took
7 seconds, and the total including the controls was under 45 minutes of
compute.

---

## 8. Proven, computed, assumed

**PROVEN** (23 tests, red first, `scripts/analysis/test_rot_ladder_variants.py`)
* The snap puts both cross boundaries on whole cell multiples at every ladder
  grid, and moves each by at most half a cell.
* The snap is arithmetically a no-op at grid 181 and produces an identical
  case there: identical part mask, identical material fill fraction, identical
  target chi. This is the substitute for the reproduction gate that variant A
  cannot use.
* The sub-cell melt area fill returns a straight binary front exactly, and its
  only departure is at corners, where a convex corner cell reads 0.906.
* The Gaussian melt indicator at zero width is the published binary threshold,
  bit for bit.
* The overlap function is the campaign's own `topopt_objective.area_iou` and
  not a second copy of it.

**COMPUTED**
* Every number in Sections 1 through 7.
* Both variant verdicts and the spreads they are read from.
* That the part mask is bit-identical between the original and the snapped
  cross at all seven grids, and that the snapped raster-minus-area delta is
  +0.0000 percent at all seven.
* That the published +0.773 and -0.465 correlations reproduce exactly through
  this pass's own analysis path.
* That the snapped ladder has zero temperature-ceiling flags where the original
  has three, all at grid 200.
* The one ranking test that fails under the snapped geometry, at grid 96, on J
  and not on IoU.

**ASSUMED**
* That a monotone sequence with steps that do not shrink is better described as
  an unconverged discretization trend than as a physical limit. No extrapolated
  limit is claimed for the snapped ladder either, and its steps
  (-0.0327, -0.0361, -0.0449, -0.0157, -0.0248, -0.0416) are not monotone in
  magnitude, so no observed order is quoted.
* That grid 360 speaks for reachable grids. Unchanged from the prior report and
  still a statement about reachable grids, not a proof of non-convergence.
* That the transferred-map arm is not evidence about the forward. It carries a
  map solved at grid 120 on the ORIGINAL geometry, and under variant A it is
  additionally being transferred onto a slightly different part, which is named
  and is why its ranking failure at grid 96 is reported but not weighted.
* Everything the prior reports assume about bilinear transfer, the sub-cell
  area-fill target, an arbitrary realizable stop time and instantaneous moves
  between indexed positions.

---

## 9. Honest limits

1. **One shape.** This pass tests the cross, with the keyhole carried only as a
   metric control. It says nothing about the star, the square, the T or the L.
2. **Variant A does not hold the part fixed.** It holds the raster exact and
   moves the part by up to half a cell, up to 3.74 percent of the arm
   half-width at grid 120. The measured correlation of the result against that
   perturbation is weak (r = -0.315) and the perturbation vanishes at the fine
   end, where the two ladders agree to 0.0008, but a half-cell geometry change
   is a real change and is not zero.
3. **Variant B changes the metric only, not the stop rule.** The stop time is
   still the argmin of the original J, which is scored against the sub-cell
   area-fill target. A variant that also re-optimized the stop against the
   smoothed metric was not run.
4. **The mechanism for the cross's movement is still not established.** This
   pass rules two candidates out. It does not name the third. The next
   candidates in line are the melt front's interaction with the arm width in
   cells (the arm half-width spans 6 to 22 cells across this ladder) and the
   bed-growth term, which runs from 2.98 percent at grid 96 to 19.79 percent at
   grid 360 in the snapped ladder and is the largest single change along it.
5. **Nothing was re-solved.** No statement here bears on whether a deeper solve
   at a finer grid would do better.
6. **The adjoint arms are conductivity only.** Unchanged, still the largest
   actuator gap.
7. **The model over-predicts achievable tuned uniformity by roughly a factor of
   eight against hardware** (`ALLISON_LAW_REPLICATION.md` Section 6.1). Every
   IoU here is a statement about a two-dimensional model, not a printed part.

---

## 10. The single most valuable next layer

**Measure where in the melt field the movement lives.** Both cheap explanations
are now dead, so the next step is not another variant of the metric or the
geometry but a spatial decomposition of the difference between two adjacent
grids: which cells of the melted set change between grid 200 and grid 240, and
whether they sit at the arm tips, at the concave corners, or in the bed growth
outside the part. The bed-growth number moving from 2.98 percent to 19.79
percent along the snapped ladder while the part under-melt stays between 0.00
and 7.13 percent says the movement is mostly OUTSIDE the part, which would make
this a question about the powder bed's melt front and not about the part's
boundary at all. That is a re-scoring of fields already stored, so it costs
seconds, and it is the first thing that would change the search.

Second, and independent: **run the star and the square through the same
ladder.** The verdict still splits between two shapes and the campaign quotes
absolute rotating IoU on more than two.

---

## 11. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

**New code, all under `scripts/analysis/`. No solver module was modified.**
* `scripts/analysis/rot_ladder_variants.py` the snap, the sub-cell melt area
  fill, the Gaussian indicator, and the correlation and spread helpers
* `scripts/analysis/test_rot_ladder_variants.py` 23 tests, red first
* `scripts/analysis/run_ladder_smooth_iou.py` the variant B re-scoring driver
* `scripts/analysis/make_cross_variants_figure.py` the figure, the merged
  table, the ranking check and the gate reader

**Extended, not forked:**
* `scripts/analysis/run_rot_grid_ladder.py` gains `--snap-geometry` and a
  reproduction gate that reports NOT APPLICABLE when the geometry was
  deliberately changed
* `scripts/analysis/rot_ladder_raster_geometry.py` gains `--snap` and an
  explicit grid list

**New results:**
* `fgm_solve_campaign/out_rot_ladder/cross_snapA_ladder.json` and `_maps.npz`
* `fgm_solve_campaign/out_rot_ladder/cross_smoothB.json`
* `fgm_solve_campaign/out_rot_ladder/keyhole_smoothB.json`
* `fgm_solve_campaign/out_rot_ladder/cross_snapA_smoothB.json`
* `fgm_solve_campaign/out_rot_ladder/cross_snapA_mask_identity.json`
* `fgm_solve_campaign/out_rot_ladder/cross_raster_geometry_snapA.json`
* `fgm_solve_campaign/out_rot_ladder/cross_variants_table.json` the merged
  table every number in Sections 1 through 4 is read from
* `fgm_solve_campaign/logs_rot_ladder/cross_snapA.log` the per-arm wall times
  and the running cost projection

**Figure, viewed before delivery:**
* `fgm_solve_campaign/figs_rot_ladder/fig_cross_scatter_variants.png`
  Panel A, the three cross ladders plus the keyhole reference, with the spread
  over n >= 160 and the sign-change count for each. Panel B, the size of each
  successive ladder step on a logarithmic axis, where only the keyhole walks
  down. Panel C, the rotating IoU against the limb rounding error, with the
  original and variant B scattered against the predictor and variant A pinned
  at zero rounding error, still spanning 0.1958 over the whole ladder and
  0.1270 over n >= 160.

**Read, not modified:**
* `ROTATING_GRID_LADDER_REPORT.md`, `ROTATING_HOLDOUT_REPORT.md`,
  `HOLDOUT_FOLLOWUP_REPORT.md`
* `fgm_solve_campaign/adjoint2d/{robust_rot,forward,adjoint,rot_kernel,dwell_kernel,chi_area,topopt_objective,pins,library_solve}.py`
* `shapes.py`, `rfam_eqs_coupled.py` for the cross polygon and the domain grid
* `outputs_eqs/fgm_calibrated_control/configs/cross_m0p0500.yaml`

`scripts/solve_fgm.py`, `rfam_gui_server.py`, everything under `webui/` and
everything under `.claude/worktrees/` were NOT touched, as other sessions own
them.
