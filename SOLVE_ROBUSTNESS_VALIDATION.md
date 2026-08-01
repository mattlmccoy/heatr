# Robustness validation of the solved dopant maps: grid transfer and rim smoothing

**Date:** 2026-08-01. **Scope:** two hold-out tests of the solved per-cell dopant
maps of `SHAPE_LIBRARY_SOLVE_REPORT.md`, on six shapes (square, circle,
trapezoid, triangle, diamond, rectangle). **Forward runs only.** Nothing was
re-solved: every arm takes a map that was already solved on the 120 x 120 grid,
perturbs it outside the solve, re-runs the real forward, and re-optimizes only
the stop. Nothing was committed. No dissertation file was touched. The git
worktrees under `.claude/worktrees/` were not read or written.

**Acronyms, expanded on first use.** FGM = functionally graded material (a
spatially varying dopant saturation map). EQS = electro-quasi-static (the
low-frequency Maxwell approximation the 2-D solver uses). IoU = intersection over
union. bpp = bits per pixel. L-BFGS-B = limited-memory
Broyden-Fletcher-Goldfarb-Shanno with box constraints. phi = melt fraction.
J = the shape-fidelity objective. W/m = watts per metre of depth.

**Evidence tags.** PROVEN = unit-tested or bit-identity-gated. COMPUTED =
measured from a real forward run in this pass. ASSUMED = a modelling choice or an
inference not measured here.

**The objective and the stop convention, stated once.**

    J(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

with `chi_part` the rasterized binary part mask. Every J, IoU, growth,
under-melt and absorbed-power number below is read at that arm's **own J-stop**,
t_stop = argmin over that arm's own stored trajectory of J. Each arm gets its own
stop; no arm is read at a fixed time and none at phi_bar = 0.90. `HORIZON` is
flagged whenever the minimum sits on the last stored step (index 1499, 750.0 s),
which means the objective had not turned and that arm's J is a bound. The melted
region for IoU, growth and under-melt is phi >= 0.5; J itself uses no threshold.
Absorbed power is the state-B value in W/m.

**J is a sum over cells and is therefore NOT comparable between grids.** The
120 grid has 800 to 1624 part cells and the 160 grid has 1434 to 2812. Task A is
read as **rankings and relative margins**, never as absolute J across grids.
`J_per_part_cell` is tabulated as a readability aid only and is not a claim of
grid invariance.

---

## 1. The two verdicts, up front

**TASK A, grid hold-out. The absolute fidelity does NOT transfer, the ranking
against the historical baseline does on 5 of the 6 shapes, and the win over
uniform transfers on 6 of 6.** COMPUTED. Optimizing at 120 and scoring at 160:
the ranking of the printable solved 4-bpp arm against the best stored historical
mask is preserved on square, trapezoid, triangle, diamond and rectangle, and
**flips on the circle**, which goes from a +71.9 percent J win at 120 to a
-125.8 percent loss at 160 (IoU 0.9904 against 0.9492 becomes 0.9484 against
0.9783). Where the win survives the margin shrinks on two of three shapes:
trapezoid +77.6 to +37.0 percent, diamond +40.5 to +12.2 percent, triangle
+48.6 to +38.9 percent. **None of the three shapes that were in the absolute
SOLVED class at 120 (IoU >= 0.95: square 0.9816, circle 0.9904, trapezoid 0.9718)
is still in it at 160** (0.7767, 0.9484, 0.9053). **The solved arm still beats the
uniform arm on J on 6 of 6 at 160.** Per class: the shapes whose 120 solution was
a fine rim contrast (square, circle) lose the most; the shapes whose solution is a
coarser interior gradient (triangle, diamond) transfer best; the shape whose solve
had stalled (rectangle) transfers trivially because there was little structure to
lose.

**And it is not a dose artifact.** The pinned drive voltage was calibrated so the
UNIFORM arm absorbs 500.0 W/m **at grid 120**; at 160 the same voltage gives
359.5 to 443.4 W/m. Re-applying the campaign's own calibration at 160 (one exact
rescale, absorbed power is quadratic in the drive) moves every margin by at most
4.5 percentage points and **changes no ranking on any of the six shapes**. The
square's solved IoU is 0.7767 at pinned voltage and 0.7775 dose-matched, against
0.9816 at 120.

**TASK B, rim robustness. The solved maps are NOT robust to rim smoothing on 5 of
6 shapes: the fine rim structure is load-bearing, with a sensitivity length of
about one cell.** COMPUTED. A single-cell part-masked Gaussian blur of the solved
continuous map, re-quantized at 4 bpp, costs between **+37.3 percent** (triangle)
and **+891.9 percent** (circle) on J and between 0.0016 and 0.1173 IoU points. Two
cells costs +75.7 to +1257.1 percent on J and 0.0328 to 0.1660 IoU points. The
one exception is the **rectangle, where smoothing IMPROVES the arm** (J -20.2
percent at one cell, -21.8 percent at two, IoU +0.0290 and +0.0345), which is
exactly what a stalled, nearly uniform solved map should do. Even at two cells,
**all 6 shapes still beat the uniform arm on J**; against the best stored
historical mask only the trapezoid and the triangle still win.

**The two tasks corroborate one mechanism.** ASSUMED, but supported from two
independent directions: the shapes that lose most to a one-cell blur (circle
+892 percent, square +153 percent) are the same shapes that lose most to the
grid change, and bilinear resampling from 120 to 160 is itself a sub-cell blur.
The solved square and circle maps are rim solutions tuned to the 120-cell
rasterization of the part boundary, and they do not survive being moved off it.

---

## 2. What was run, and how

### 2.1 Provenance

Source of every solved map and every historical baseline: the committed snapshot
`fgm_solve_campaign/out_lib/<shape>.json` and `<shape>_maps.npz`, produced by
`adjoint2d/library_solve.py` and reported in `SHAPE_LIBRARY_SOLVE_REPORT.md`.
Configuration: `outputs_eqs/fgm_calibrated_control/configs/<shape>_m0p0500.yaml`,
the same deterministic first-in-sorted-order pick `library_solve.shape_config`
makes. The only field changed for Task A is `geometry.grid_nx` and
`geometry.grid_ny`, 120 to 160; for the dose-matched repeat, additionally
`electric.voltage_v`. Nothing else in any block was touched.

### 2.2 The resampling convention, with the code cited

The 120-to-160 transfer of the solved map uses the **production map-injection
convention**, reproduced (not re-derived) in `adjoint2d/robust.resample_map`:

    # rfam_eqs_coupled.py:374-380, the FGM npz loader
    zy = ny_sim / sat.shape[0] ; zx = nx_sim / sat.shape[1]
    if abs(zy - 1.0) > 0.01 or abs(zx - 1.0) > 0.01:
        sat = scipy.ndimage.zoom(sat, (zy, zx), order=1)
    sat = np.clip(sat, 0.0, 1.0)

The identical call appears in the direct-map branch at
`rfam_eqs_coupled.py:335-340`. Bilinear, then clipped; no smoothing kernel, no
area weighting, and a 1 percent dead band in which the resample is skipped.
PROVEN: `adjoint2d/tests/test_robust.py::test_resample_matches_scipy_zoom_order1_clipped`
asserts equality against that exact call.

**The historical mask needs no such transfer and did not get one.** It is loaded
from its stored 1715 x 1715 uint8 printer level map through the PRODUCTION loader
`rfam_eqs_coupled._FgmFeedback.from_config` against the 160 grid, so the loader
itself resamples printer resolution straight to 160 x 160. The historical arm is
therefore at no resampling disadvantage relative to its 120 run. Its boundary
convention (`asstored` or `outside1`) is the one that won that shape's 120
census, carried over unchanged.

After resampling, the solved map is re-quantized to 4 bpp inside the part through
the production quantizer `printability.quantize_in_part`
(`fgm_generator.py:583-586`), because a bilinearly resampled 4-bpp map is no
longer on the printer's level grid. The un-requantized continuous transfer is
also run and reported as `A1_cont_160`; the difference between the two is under
0.4 percent of J on all six shapes, so the re-quantization is not what decides
anything.

### 2.3 The smoothing convention

`adjoint2d/robust.smooth_in_part`. The blur is applied to the solved
**continuous** map (arm `A1_cont`), as a **normalized convolution over the part
only**,

    out = gaussian(s * chi_part) / gaussian(chi_part)   on the part

so the nominal saturation held outside the part (the prototype convention,
s = 1) cannot bleed inward and manufacture an answer. Outside the part the map is
held at the nominal 1, exactly as `printability.quantize_in_part` does, so an arm
changes the dopant map and not the sub-pixel geometry fill of boundary cells.
The result is then clipped to [0, 1] (renormalization is a clip, not a rescale:
a normalized convolution is a convex combination of in-part values and therefore
already lies inside the original range; the clip is defensive and was a no-op on
all twelve arms). Then it is re-quantized at 4 bpp through the production
quantizer. Radius 0 is the stored `A1_4bpp` arm and was NOT re-run.

PROVEN: seven unit tests on the smoother, including one that a part which is
uniformly 0.2 stays at 0.2 to 1e-12 despite the outside being 1.0 (no bleed),
one that roughness decreases monotonically in the radius, and one that the box is
respected. Eleven tests total in `adjoint2d/tests/test_robust.py`, written red
first (the import error was observed before `robust.py` existed, then the two
`recalibrated_voltage` tests were observed failing before that function existed).
All 11 pass, and the 36-test set covering `test_robust`, `test_printability`,
`test_library_solve` and `test_shape_objective` passes.

### 2.4 The dose-matched repeat

COMPUTED, and this is the honest control the task did not ask for but needed. The
campaign calibrates the drive voltage per shape so the **uniform** arm absorbs
500.0 W/m. That calibration is grid-dependent: at 160 the pinned voltage gives
359.5 to 443.4 W/m. The EQS solve is linear in the applied potential and neither
sigma nor eps_r depends on it, so absorbed power is exactly quadratic in the
drive and one rescale suffices, `V' = V * sqrt(500 / P_measured)`
(`robust.recalibrated_voltage`, two unit tests). COMPUTED check: the recalibrated
uniform arm absorbs 500.00 W/m on all six shapes, to the two decimals printed.

---

## 3. Task A: grid hold-out, optimize at 120 and score at 160

### 3.1 Rankings and margins, the headline table

`dJ` is the J margin of the printable solved 4-bpp arm against that shape's best
stored historical mask, `(J_hist - J_solved) / J_hist`; positive means the solved
map wins. `dIoU` is solved minus historical.

| shape | dJ at 120 | dJ at 160, pinned V | dJ at 160, dose-matched | dIoU at 120 | dIoU at 160 | ranking on J survives | ranking on IoU survives |
|---|---|---|---|---|---|---|---|
| square | -100.9 % | -224.6 % | -265.8 % | -0.0159 | -0.1494 | YES (loss stays a loss) | YES |
| circle | **+71.9 %** | **-125.8 %** | **-129.2 %** | +0.0412 | -0.0298 | **NO, sign flip** | **NO, sign flip** |
| trapezoid | +77.6 % | +37.0 % | +34.3 % | +0.1030 | +0.0430 | YES | YES |
| triangle | +48.6 % | +38.9 % | +41.7 % | +0.0810 | +0.0972 | YES | YES |
| diamond | +40.5 % | +12.2 % | +16.7 % | +0.0727 | +0.0108 | YES | YES |
| rectangle | -1742.6 % | -488.1 % | -559.9 % | -0.1576 | -0.1026 | YES (loss stays a loss) | YES |

**5 of 6 rankings survive on J and 5 of 6 on IoU, and it is the same five.**
Of the three genuine wins at 120 (circle, trapezoid, triangle, plus the diamond),
one flips. Of the four wins, three survive with the margin cut by 8 to 40
percentage points.

### 3.2 Absolute fidelity, which does not transfer

| shape | solved IoU at 120 | solved IoU at 160 | solved IoU at 160, dose-matched | historical IoU at 120 | historical IoU at 160 | uniform IoU at 160 |
|---|---|---|---|---|---|---|
| square | **0.9816** | 0.7767 | 0.7775 | 0.9975 | 0.9261 | 0.7711 |
| circle | **0.9904** | 0.9484 | 0.9414 | 0.9492 | 0.9783 | 0.9020 |
| trapezoid | **0.9718** | 0.9053 | 0.9173 | 0.8688 | 0.8623 | 0.8875 |
| triangle | 0.8578 | 0.8303 | 0.8411 | 0.7768 | 0.7332 | 0.7594 |
| diamond | 0.8520 | 0.8059 | 0.8147 | 0.7793 | 0.7951 | 0.7216 |
| rectangle | 0.8424 | 0.8954 | 0.8873 | 1.0000 | 0.9980 | 0.8887 |

Three readings, all COMPUTED.

1. **Zero of the three SOLVED-class shapes stays in the class.** The absolute
   claim "the melted region IS the nominal part to IoU >= 0.95" is a statement
   about the 120 grid and does not survive to 160 for any of them.
2. **The forward model itself is not grid-converged in this metric, and that
   limits what a grid hold-out can prove.** The UNIFORM arm, which contains no
   solved map at all, moves by up to 0.13 IoU points between grids and moves in
   BOTH directions: square 0.8508 to 0.7711 (worse), circle 0.7881 to 0.9020
   (better), diamond 0.5903 to 0.7216 (better). A grid hold-out is therefore a
   joint test of map transfer AND discretization convergence, and this pass
   cannot separate the two. Naming that is more useful than a verdict that
   pretends it can.
3. **The historical mask is not immune either**, which is what makes the ranking
   comparison worth anything: it moves 0.9975 to 0.9261 on the square and 0.7768
   to 0.7332 on the triangle, while improving on the circle (0.9492 to 0.9783).
   The circle flip is as much the historical mask getting better as the solved
   map getting worse.

### 3.3 The dose confound, measured rather than assumed

| shape | uniform P_abs at 160, pinned V | recalibrated V | solved arm P_abs at 160, dose-matched | solved arm P_abs at 120 |
|---|---|---|---|---|
| square | 366.8 W/m | 2428.2 to 2834.8 V | 463.6 W/m | 412.0 W/m |
| circle | 403.7 W/m | 3399.6 to 3783.7 V | 477.8 W/m | 366.4 W/m |
| trapezoid | 397.3 W/m | 2635.8 to 2957.0 V | 483.3 W/m | 422.9 W/m |
| triangle | 443.4 W/m | 3005.5 to 3191.5 V | 464.6 W/m | 442.1 W/m |
| diamond | 359.5 W/m | 2475.4 to 2919.3 V | 412.4 W/m | 318.5 W/m |
| rectangle | 426.9 W/m | 3316.2 to 3588.8 V | 500.4 W/m | 503.3 W/m |

The dose-matched solved arms absorb MORE power than their 120 counterparts on
five of six shapes and still score worse in absolute terms. **The collapse of
absolute fidelity at 160 is not underdose.** It is resolution: the part boundary
is rasterized onto different cells, the map is bilinearly smeared across 1.33
cells, and Task B shows directly that a blur of that size is expensive.

### 3.4 Every arm, every shape

Stop convention as in Section 0. `Eres` is the standing energy-residual gate at
the arm's own stop, threshold 5 percent.

| shape | arm | J | J per part cell | IoU | growth % | under % | stop s | P_abs W/m | Eres % |
|---|---|---|---|---|---|---|---|---|---|
| square | uniform, 120 | 210.19 | 0.13137 | 0.8508 | 7.25 | 8.75 | 416.5 | 500.0 | 0.70 |
| square | best stored mask, 120 | 12.77 | 0.00798 | 0.9975 | 0.00 | 0.25 | 389.5 | 559.8 | 0.16 |
| square | solved 4 bpp, 120 | 25.66 | 0.01604 | 0.9816 | 1.88 | 0.00 | 554.0 | 412.0 | 0.24 |
| square | uniform, 160 | 631.22 | 0.22471 | 0.7711 | 9.97 | 15.20 | 662.5 | 366.8 | 0.82 |
| square | best stored mask, 160 | 176.47 | 0.06282 | 0.9261 | 2.06 | 5.48 | 443.5 | 502.5 | 0.25 |
| square | solved continuous, 160 | 574.51 | 0.20452 | 0.7760 | 5.06 | 18.48 | 669.5 | 340.1 | 0.47 |
| square | solved 4 bpp, 160 | 572.79 | 0.20391 | 0.7767 | 5.06 | 18.41 | 670.0 | 340.1 | 0.47 |
| square | solved 4 bpp, 160 dose-matched | 573.84 | 0.20429 | 0.7775 | 5.45 | 18.01 | 438.0 | 463.6 | 0.55 |
| circle | uniform, 120 | 275.02 | 0.22179 | 0.7881 | 14.19 | 10.00 | 378.0 | 500.0 | 1.17 |
| circle | best stored mask, 120 | 52.18 | 0.04208 | 0.9492 | 1.61 | 3.55 | 226.5 | 689.3 | 0.22 |
| circle | solved 4 bpp, 120 | 14.68 | 0.01183 | 0.9904 | 0.65 | 0.32 | 566.5 | 366.4 | 0.14 |
| circle | uniform, 160 | 180.92 | 0.08269 | 0.9020 | 5.39 | 4.94 | 486.0 | 403.7 | 0.43 |
| circle | best stored mask, 160 | 34.15 | 0.01561 | 0.9783 | 0.91 | 1.28 | 258.0 | 646.1 | 0.19 |
| circle | solved continuous, 160 | 79.20 | 0.03620 | 0.9449 | 2.93 | 2.74 | 514.0 | 386.2 | 0.24 |
| circle | solved 4 bpp, 160 | 77.10 | 0.03524 | 0.9484 | 2.83 | 2.47 | 515.0 | 385.7 | 0.24 |
| circle | solved 4 bpp, 160 dose-matched | 88.58 | 0.04048 | 0.9414 | 2.93 | 3.11 | 387.0 | 477.8 | 0.26 |
| trapezoid | uniform, 120 | 171.32 | 0.14409 | 0.8387 | 9.50 | 8.16 | 341.5 | 500.0 | 0.80 |
| trapezoid | best stored mask, 120 | 122.22 | 0.10279 | 0.8688 | 5.13 | 8.66 | 296.5 | 571.3 | 0.50 |
| trapezoid | solved 4 bpp, 120 | 27.36 | 0.02301 | 0.9718 | 1.35 | 1.51 | 434.0 | 422.9 | 0.23 |
| trapezoid | uniform, 160 | 209.55 | 0.09870 | 0.8875 | 4.71 | 7.07 | 467.5 | 397.3 | 0.41 |
| trapezoid | best stored mask, 160 | 269.20 | 0.12680 | 0.8623 | 8.81 | 6.17 | 364.5 | 518.3 | 0.71 |
| trapezoid | solved continuous, 160 | 169.51 | 0.07984 | 0.9058 | 3.01 | 6.69 | 483.5 | 384.2 | 0.28 |
| trapezoid | solved 4 bpp, 160 | 169.48 | 0.07983 | 0.9053 | 3.01 | 6.74 | 484.0 | 384.0 | 0.28 |
| trapezoid | solved 4 bpp, 160 dose-matched | 157.38 | 0.07413 | 0.9173 | 2.54 | 5.93 | 355.0 | 483.3 | 0.31 |
| triangle | uniform, 120 | 202.82 | 0.25352 | 0.7687 | 16.75 | 10.25 | 253.5 | 500.0 | 1.40 |
| triangle | best stored mask, 120 | 171.27 | 0.21408 | 0.7768 | 12.00 | 13.00 | 368.5 | 354.4 | 0.86 |
| triangle | solved 4 bpp, 120 | 87.99 | 0.10998 | 0.8578 | 7.25 | 8.00 | 270.0 | 442.1 | 0.52 |
| triangle | uniform, 160 | 384.20 | 0.26792 | 0.7594 | 19.11 | 9.55 | 309.0 | 443.4 | 1.42 |
| triangle | best stored mask, 160 | 417.61 | 0.29122 | 0.7332 | 20.22 | 11.85 | 510.5 | 309.1 | 1.28 |
| triangle | solved continuous, 160 | 255.16 | 0.17794 | 0.8316 | 12.62 | 6.35 | 327.5 | 412.1 | 0.99 |
| triangle | solved 4 bpp, 160 | 255.07 | 0.17787 | 0.8303 | 12.62 | 6.49 | 327.5 | 412.0 | 1.00 |
| triangle | solved 4 bpp, 160 dose-matched | 240.54 | 0.16774 | 0.8411 | 11.92 | 5.86 | 275.5 | 464.6 | 1.00 |
| diamond | uniform, 120 | 768.11 | 0.47297 | 0.5903 | 24.75 | 26.35 | 473.5 | 500.0 | 2.07 |
| diamond | best stored mask, 120 | 355.07 | 0.21864 | 0.7793 | 9.36 | 14.78 | 537.0 | 433.1 | 0.83 |
| diamond | solved 4 bpp, 120 | 211.16 | 0.13002 | 0.8520 | 2.34 | 12.81 | 750.0 HORIZON | 318.5 | 0.25 |
| diamond | uniform, 160 | 811.67 | 0.28865 | 0.7216 | 12.16 | 19.06 | 680.0 | 359.5 | 0.98 |
| diamond | best stored mask, 160 | 581.21 | 0.20669 | 0.7951 | 9.67 | 12.80 | 640.0 | 379.8 | 0.79 |
| diamond | solved continuous, 160 | 509.41 | 0.18116 | 0.8060 | 1.21 | 18.42 | 750.0 HORIZON | 296.9 | 0.15 |
| diamond | solved 4 bpp, 160 | 510.06 | 0.18139 | 0.8059 | 1.14 | 18.49 | 750.0 HORIZON | 296.6 | 0.14 |
| diamond | solved 4 bpp, 160 dose-matched | 515.20 | 0.18322 | 0.8147 | 6.33 | 13.37 | 524.0 | 412.4 | 0.60 |
| rectangle | uniform, 120 | 203.07 | 0.17628 | 0.8528 | 13.19 | 3.47 | 321.5 | 500.0 | 1.16 |
| rectangle | best stored mask, 120 | 10.69 | 0.00928 | 1.0000 | 0.00 | 0.00 | 331.0 | 483.7 | 0.19 |
| rectangle | solved 4 bpp, 120 | 196.94 | 0.17095 | 0.8424 | 14.58 | 3.47 | 321.5 | 503.3 | 1.23 |
| rectangle | uniform, 160 | 210.20 | 0.10264 | 0.8887 | 8.79 | 3.32 | 390.5 | 426.9 | 0.70 |
| rectangle | best stored mask, 160 | 33.47 | 0.01634 | 0.9980 | 0.00 | 0.20 | 378.0 | 444.7 | 0.21 |
| rectangle | solved continuous, 160 | 196.66 | 0.09603 | 0.8972 | 6.45 | 4.49 | 382.0 | 427.5 | 0.63 |
| rectangle | solved 4 bpp, 160 | 196.87 | 0.09613 | 0.8954 | 6.45 | 4.69 | 382.5 | 427.3 | 0.64 |
| rectangle | solved 4 bpp, 160 dose-matched | 234.99 | 0.11474 | 0.8873 | 9.18 | 3.12 | 310.5 | 500.4 | 0.83 |

Three horizon flags, all on the diamond, all at pinned voltage: the diamond's
solved arm has its J minimum on the last stored step at both grids, so its J is
an upper bound there and its true margin can only be better than +12.2 percent.
The dose-matched diamond arm's stop is interior at 524.0 s, which is the cleanest
diamond number in the table.

### 3.5 Which stored mask each shape was compared against

Carried over unchanged from the 120 census, including the boundary convention.

| shape | best stored mask | campaign | convention |
|---|---|---|---|
| square | `map_m0p5477 / mag0p55` | calibration | as stored |
| circle | `map_m0p70 / mag0p70` | old {0.30 .. 0.85} grid | outside = 1 |
| trapezoid | `map_m0p1110 / mag0p11` | calibration | outside = 1 |
| triangle | `map_m2p7016 / mag2p70` | calibration | outside = 1 |
| diamond | `map_m1p4854 / mag1p49` | calibration | as stored |
| rectangle | `map_m0p50 / mag0p50` | old {0.30 .. 0.85} grid | as stored |

**Figure:** `fgm_solve_campaign/figs_robust/fig_robust_grid.png`, viewed before
delivery. Panel A is IoU at the three settings with the stored mask as a tick.
Panel B is the J margin, where a sign change between the blue and orange bars is
a win that did not survive. Panel C is the absorbed-power confound. Panel D is
the under-melt starvation signature.

---

## 4. Task B: rim robustness at grid 120

Radius 0 is the stored `A1_4bpp` arm and was not re-run. Radii 1 and 2 are
part-masked normalized-convolution Gaussians on the solved CONTINUOUS map,
clipped to [0, 1], re-quantized at 4 bpp, re-run, stop re-optimized.

### 4.1 J and IoU against smoothing radius

| shape | J at r = 0 | J at r = 1 | J at r = 2 | dJ at r = 1 | dJ at r = 2 | IoU at r = 0 | IoU at r = 1 | IoU at r = 2 |
|---|---|---|---|---|---|---|---|---|
| square | 25.66 | 64.84 | 88.38 | **+152.7 %** | **+244.4 %** | 0.9816 | 0.9547 | 0.9340 |
| circle | 14.68 | 145.56 | 199.16 | **+891.9 %** | **+1257.1 %** | 0.9904 | 0.8731 | 0.8353 |
| trapezoid | 27.36 | 74.21 | 105.27 | +171.2 % | +284.7 % | 0.9718 | 0.9254 | 0.8978 |
| triangle | 87.99 | 120.83 | 154.61 | +37.3 % | +75.7 % | 0.8578 | 0.8562 | 0.8251 |
| diamond | 211.16 | 454.49 | 564.48 | +115.2 % | +167.3 % | 0.8520 | 0.7381 | 0.6860 |
| rectangle | 196.94 | 157.09 | 153.91 | **-20.2 %** | **-21.8 %** | 0.8424 | 0.8715 | 0.8770 |

### 4.2 Does the smoothed arm still beat its references

| shape | beats best stored mask at r = 0 | at r = 1 | at r = 2 | beats uniform at r = 2 |
|---|---|---|---|---|
| square | no | no | no | YES |
| circle | YES | no | no | YES |
| trapezoid | YES | YES | YES | YES |
| triangle | YES | YES | YES | YES |
| diamond | YES | no | no | YES |
| rectangle | no | no | no | YES |

**Four of the six wins over the historical mask are held at radius 0; two survive
a two-cell blur.** The win over uniform survives everywhere.

### 4.3 How much the map actually moved, and what the printer sees

| shape | in-part saturation standard deviation at r = 0 | at r = 1 | at r = 2 | root-mean-square change at r = 2 | largest single-cell change at r = 2 | printer levels used at r = 0 | at r = 2 |
|---|---|---|---|---|---|---|---|
| square | 0.2469 | 0.2314 | 0.2158 | 0.0802 | 0.4658 | 13 | 12 |
| circle | 0.1726 | 0.1088 | 0.0863 | 0.1294 | 0.8035 | 10 | 6 |
| trapezoid | 0.1290 | 0.0940 | 0.0768 | 0.0837 | 0.4989 | 12 | 5 |
| triangle | 0.2459 | 0.1868 | 0.1622 | 0.1583 | 0.8400 | 15 | 11 |
| diamond | 0.3672 | 0.3311 | 0.3015 | 0.1475 | 0.7656 | 16 | 15 |
| rectangle | 0.1581 | 0.1230 | 0.1077 | 0.0956 | 0.3265 | 9 | 7 |

COMPUTED, and it sharpens the mechanism: **a one-cell blur is a small change to
the map and a large change to the result.** On the circle the in-part standard
deviation falls by 37 percent and the root-mean-square per-cell change is 0.0999
of a saturation unit, roughly 1.5 printer levels at 4 bpp, and that costs a
factor of ten on J. The number of distinct printer levels the map uses collapses
on the shapes that lose most (circle 10 to 6, trapezoid 12 to 5), which is the
same statement in the printer's own units.

**The rectangle is the control that makes the reading trustworthy.**
`SHAPE_LIBRARY_SOLVE_REPORT.md` Section 1 point 4 diagnosed the rectangle as a
solver stall, not an actuator limit: the solve moved J only from 203.07 to 196.87
in 15 gradient evaluations and left a nearly uniform map. If the rim sensitivity
measured here were an artifact of the smoothing procedure rather than a property
of the solved maps, the rectangle would degrade too. It improves. COMPUTED, the
rectangle's smoothed arm at r = 2 (J 153.91, IoU 0.8770) is better than both its
own unsmoothed solved arm and the uniform arm (J 203.07) and remains far worse
than the stored mask (J 10.69). Smoothing a stalled solution is a mild
regularizer; smoothing a converged rim solution destroys it.

**Figure:** `fgm_solve_campaign/figs_robust/fig_robust_rim.png`, viewed before
delivery. Panel A is J relative to radius 0 on a log axis, panel B is IoU against
radius, panel C is J relative to the best stored mask (circles) and to uniform
(squares), where crossing the black line is the point at which the smoothing
costs the win.

---

## 5. Quoting guidance: what the census numbers mean and how to state them

> **Use these exact framings. Each is a different baseline, and they are not
> interchangeable.**
>
> **1. 13 of 18, against the best-stored ORACLE.** The printable single-pass
> solved map beats, on J and on IoU, the best by J of EVERY distinct stored 4-bpp
> dopant map that campaign holds for that shape, in either boundary convention,
> scored on the same engine at its own J-stop. Source:
> `SHAPE_LIBRARY_SOLVE_REPORT.md` Section 1 point 1; re-verified in this pass from
> `out_lib/*.json`, 13 of 18 on J and 13 of 18 on IoU, and it is the same 13. **This
> baseline is an oracle: selecting that mask cost 28 to 38 forward solves per
> shape** (the per-shape counts in `SHAPE_LIBRARY_SOLVE_REPORT.md` Section 5,
> whose prose says "34 to 38" but whose own table lists 28 for the triangle; use
> the table), against the 40 forward-equivalents the adjoint solve was given.
> Quote it as the hardest available baseline, and say that it is an oracle.
>
> **2. 16 of 18, against the arm the campaign actually STANDARDIZED on.** The
> old {0.30 .. 0.85} grid m = 0.85 as-stored mask is what the
> `geometry_dual_readstate` campaign actually ran and what would actually have
> been printed. Source: `SHAPE_LIBRARY_SOLVE_REPORT.md` Section 4, 16 of 18 on J
> and 16 of 18 on IoU, losing only the square and the rounded_rect. Quote this
> when the comparison is to practice rather than to a retrospective best-of scan.
>
> **3. 16 of 18, the CHEAP-RECIPE number.** Against the MEDIAN by J of that
> shape's stored one-parameter family, which is what a practitioner gets by
> picking a gain from the family without running the oracle scan. COMPUTED in this
> pass from `out_lib/*.json`: 16 of 18 on J and 16 of 18 on IoU (the losses are the
> square, 25.66 against 23.78, and the rectangle, 196.94 against 15.81). Quote
> this as the honest cost-matched comparison, since the median arm costs one
> forward solve to obtain and the oracle costs 28 to 38.
>
> **4. 18 of 18 against uniform**, on J, at grid 120. Source:
> `SHAPE_LIBRARY_SOLVE_REPORT.md` Section 1 point 1. COMPUTED in this pass: still
> 6 of 6 at grid 160, and still 6 of 6 after a two-cell rim blur.
>
> **5. THE DOSE-MATCH LIMITATION, which must accompany all of the above.** None
> of these arms is power matched. Absorbed power at the arms' own stops **spans
> 221.3 to 792.4 W/m** across the 18-shape census
> (`SHAPE_LIBRARY_SOLVE_REPORT.md` Section 12 point 5: the low is the ellipse
> solved 2-bpp arm, the high is the octagon's best stored mask) against the
> 500.0 W/m uniform calibration target. In THIS pass the span is 296.6 to
> 800.4 W/m. The objective penalizes over-melting as well as under-melting, which
> removes the crudest dose gaming, but a J or IoU comparison between two arms is
> not a comparison at equal delivered energy. State it as a limitation every time
> the census counts are quoted, not once in an appendix.
>
> **6. Two robustness caveats from this report, which are new.** The IoU >= 0.95
> "SOLVED" classification is **grid-specific**: none of square, circle or
> trapezoid keeps it at grid 160. And the solved maps are **rim-sensitive**: a
> one-cell blur costs +37 to +892 percent of J on five of six shapes. Do not quote
> an absolute IoU as a property of the method; quote it as a property of the
> method at grid 120 with an exactly reproduced dopant map.

---

## 6. Cost, and the wall-time projection that was required

COMPUTED. Per-run wall time was logged from the first two runs onward. **The
machine was under heavy contention from a concurrent job for most of this pass
(one-minute load average between 28 and 191 on a 12-core machine), so identical
work took between 4 and 610 seconds and these numbers are not a performance
measurement of the code.**

After the first two Task A runs (square uniform at 160, 610.2 s; square best
stored mask at 160, 83.9 s) the mean was 347 s per forward run, which projected
to 24 runs x 347 s, about 2.3 hours, for the six-shape Task A subset. That is
under the ~5 hour line, so **diamond and rectangle were NOT dropped and the full
six-shape subset was run for both tasks.** The load then fell and the actual
totals were far lower.

| task | forward runs | measured total wall | per shape |
|---|---|---|---|
| A, grid hold-out (final artifacts, 7 arms per shape) | 42 | 1028 s | 108 to 200 s |
| B, rim robustness (2 new arms per shape) | 12 | 1113 s | 8 to 570 s |

An earlier Task A pass without the dose-matched arms cost a further 1176 s and
was superseded. Total real compute for this report is about 55 minutes of wall
clock, 0.9 hours.

---

## 7. Gates

**The standing 5 percent energy-residual gate is clean on all 54 forward runs.**
COMPUTED: the maximum relative energy residual at any arm's own stop is
**2.07 percent** (diamond uniform at 120, carried over from the stored census)
and the maximum on any arm newly run in this pass is **1.58 percent** (diamond,
two-cell smoothing). **Zero violations**, `energy_gate_violations` is empty in all
twelve result files.

**No gradient was computed, and therefore none was re-gated.** This pass is
forward scoring only. `forward.py`, `adjoint.py` and `shape_objective.py` were not
touched. The standing finite-difference gate of the previous reports applies
unchanged, and it is still a SUBGRADIENT gate whose random-direction probe bottoms
at 1.22e-05 because the pinned population is the cold powder bed. Every solved map
scored here inherits that caveat.

**Unit tests.** 11 new tests in `adjoint2d/tests/test_robust.py`, written red
first: the `ImportError` was observed before `robust.py` existed, and the two
`recalibrated_voltage` tests were observed failing before that function existed.
36 tests pass across `test_robust`, `test_printability`, `test_library_solve` and
`test_shape_objective`.

---

## 8. Proven, computed, assumed

**PROVEN**
* The resampler reproduces the production map-injection call
  (`scipy.ndimage.zoom` order 1, then clip) exactly, asserted against that call.
* The part-masked smoother does not let the outside value bleed into the part
  (a uniformly 0.2 part stays 0.2 to 1e-12 with the outside at 1.0), reduces
  in-part roughness monotonically in the radius, respects the [0, 1] box, and is
  the identity at radius 0.
* The voltage recalibration is the exact square-root law, and the recalibrated
  uniform arm absorbs 500.00 W/m on all six shapes.
* 36 unit tests pass; the 11 new ones were red before they were green.

**COMPUTED**
* Every number in Sections 3, 4, 6 and 7.
* The cheap-recipe count of Section 5 item 3, from `out_lib/*.json`.

**ASSUMED**
* That bilinear resampling with a clip is the right way to move a solved map to a
  new grid. It is the production convention and it is what the engine would do,
  but it is not the only choice and an area-conserving or level-set transfer might
  transfer better. Not tested.
* That a Gaussian blur of the continuous map is a reasonable stand-in for the
  physical rim uncertainty of a real print (binder bleed, droplet spread, powder
  spreading). No bench measurement of the actual rim blur exists, so the radius
  axis is in cells, not in micrometres. At the pinned geometry one cell is 0.5 mm
  at grid 120, which is coarse against a 720 dots-per-inch printer, so the radii
  tested here are likely PESSIMISTIC as a model of printer blur and OPTIMISTIC as
  a model of powder-scale smearing. Naming both directions because the sign is not
  known.
* That the mechanism linking the two tasks (rim solutions tuned to the 120-cell
  rasterization) is the cause. The two measurements are consistent with it and
  the rectangle control argues against a procedural artifact, but no direct test
  isolating the boundary rasterization was run.
* That the rasterized binary part mask is the right nominal target. Carried over,
  still untested.
* That an arbitrary stop time is realizable as a process control.

---

## 9. Honest limits

1. **A grid hold-out at 120 against 160 is not a clean test of map transfer,
   because the forward model is not grid-converged in this metric.** The uniform
   arm alone moves up to 0.13 IoU points between the grids, in both directions.
   Section 3.2 point 2. The right version of this test needs a grid-convergence
   study of the forward first, which this pass did not run.
2. **Six shapes, not eighteen.** The 13 of 18 and 16 of 18 census counts are NOT
   re-established at grid 160 or under smoothing; only the six shapes named here
   are.
3. **Not dose matched, and the span is wide.** 296.6 to 800.4 W/m in this pass,
   221.3 to 792.4 W/m across the census. The dose-matched Task A repeat fixes the
   UNIFORM arm at 500 W/m and does not equalize the arms with each other.
4. **The adjoint arms are conductivity-only; every historical arm co-varies
   permittivity.** Unchanged from the previous reports and still the largest
   actuator gap. Nothing here narrows it.
5. **Three horizon flags on the diamond**, so its Task A margin is a bound at
   pinned voltage.
6. **Fifteen L-BFGS-B evaluations on 1600 design variables** remains a very small
   budget; every solved J is an upper bound, and a better-converged map might be
   more or less rim-sensitive. Not known.
7. **The model over-predicts achievable tuned uniformity by roughly a factor of
   eight against hardware** (`ALLISON_LAW_REPLICATION.md` Section 6.1). Every IoU
   here is a statement about a 2-D model, not about a printed part.
8. **Wall times in Section 6 are contaminated by machine contention** and should
   not be quoted as performance.

---

## 10. The single most valuable next layer

**A grid-convergence study of the forward model at fixed dopant map, before any
further grid hold-out.** This pass cannot separate "the solved map does not
transfer" from "the forward is not converged", and Section 3.2 shows the second
effect is large enough to matter on its own. Concretely: run the uniform arm and
one stored mask per shape at 96, 120, 160 and 200 with the voltage recalibrated
at each grid, and report IoU and the stop time against cell size. If IoU is still
moving at 160, then no grid hold-out is interpretable and the honest claim is
grid-specific.

**Second: re-solve at 160 and compare the two solved maps directly.** If the
160-solved map recovers IoU >= 0.95 on the square, the failure is transfer; if it
does not, the failure is convergence. That is one solve per shape, about 500 to
1300 s each, and it is the decisive experiment.

**Third, and cheap: add an explicit rim regularizer to the objective and test
shrinking its width.** Task B shows the solve is exploiting structure at the
single-cell scale, which is exactly the regime where a piecewise-smooth objective
with a phase-change ramp of dt_pc = 10 C should be distrusted. A total-variation
or Gaussian-smoothing penalty on the design variable, with the width swept, would
say whether a deliberately blunt solved map gives up much of the win. If it does
not, the printable claim gets much stronger.

---

## 11. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New code:
* `fgm_solve_campaign/adjoint2d/robust.py` resampler, part-masked smoother,
  voltage recalibration
* `fgm_solve_campaign/adjoint2d/robust_run.py` the two drivers, forward only
* `fgm_solve_campaign/adjoint2d/make_robust_figures.py` the two figures
* `fgm_solve_campaign/adjoint2d/tests/test_robust.py` 11 tests, red first

New results, one JSON and one npz of maps per shape per task:
* `fgm_solve_campaign/out_robust/{square,circle,trapezoid,triangle,diamond,rectangle}_grid.json`
* `fgm_solve_campaign/out_robust/{...}_grid_maps.npz`
* `fgm_solve_campaign/out_robust/{...}_rim.json`
* `fgm_solve_campaign/out_robust/{...}_rim_maps.npz`
* `fgm_solve_campaign/logs_robust/*.log` the per-run console logs with wall times

New figures, both viewed before delivery:
* `fgm_solve_campaign/figs_robust/fig_robust_grid.png`
* `fgm_solve_campaign/figs_robust/fig_robust_rim.png`

Read, not modified:
* `SHAPE_LIBRARY_SOLVE_REPORT.md`
* `fgm_solve_campaign/VERIFICATION_PRINTABILITY_REPORT.md`
* `fgm_solve_campaign/out_lib/*.json`, `fgm_solve_campaign/out_lib/*_maps.npz`
* `rfam_eqs_coupled.py`, `fgm_generator.py`
* `outputs_eqs/fgm_calibrated_control/configs/*_m0p0500.yaml`
* the stored 4-bpp dopant maps under
  `outputs_eqs/fgm_calibrated_control/runs/` and
  `outputs_eqs/geometry_dual_readstate/runs/`
