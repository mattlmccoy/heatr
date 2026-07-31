# Gain-Calibrated Heuristic Control: P0 result on the trusted 2-D engine

**Date:** 2026-07-31. **Scope:** P0 of the solve workstream, answering Q1 of
`FGM_INVERSE_DESIGN_ASSESSMENT.md` Section 6 ("the missing control"). Nothing was committed.
No dissertation file was touched.

**Acronyms, expanded on first use.** FGM = functionally graded material (a spatially varying
dopant saturation map). EQS = electro-quasi-static (the low-frequency Maxwell approximation the
2-D solver uses). CAD = computer-aided design. sigma_T = the 2-D uniformity metric
`ui_rms_part(t) * (T_bar_part(t) - 23 C)`, in deg C, always quoted with its read state.
phi_bar = mean part melt fraction.

**Evidence tags.** PROVEN = unit-tested or reproduced to machine precision. COMPUTED = measured
from a real forward solve in this campaign or read from a stored artifact. ASSUMED = a modelling
choice or an inference not measured here.

---

> **UPDATE 2026-07-31, read Section 7 before quoting Section 1.** An unseeded re-run (v2) shows
> the v1 triangle result was partly a warm-start artifact: the fit metric is bimodal and the v1
> warm start sat on the wrong side of a local maximum. Corrected there: triangle is
> **-7.1 % better** than uniform, not +79.8 % worse, and L_shape does have feasible gains.
> Confirmed there: H_shape, trapezoid and T_shape remain harmful under a global search. The
> 12-of-18 hold-out win rate is unchanged. Section 7 has the full v2 tables.

## 1. Verdict

**1. Calibrating the gain is worth doing, but it is not the fix the assessment predicted.**
COMPUTED over 18 shapes: the calibrated-gain arm beats the existing best-of-four in-sample arm
on the hold-out read state in **12 of 18** shapes. It leaves the harmful class almost untouched:
**5 shapes** were harmful versus uniform under best-of-four, **4 shapes** are still harmful under
calibration. Exactly **one** shape was rescued (equilateral_triangle, +5.9 % worse than uniform
becomes -43.8 % better). The assessment's expectation, "this alone should remove the harmful
cases, which are essentially all of the current negative results", is **not supported**.

**2. Where calibration pays, it pays because the stored 4-point grid was truncated, not because
the gain was mis-scaled.** The three large wins are all shapes whose fit metric was still falling
at m = 0.85, the top of the stored grid: cross (best-of-four 31.40 C, calibrated **13.86 C**,
-55.9 %), equilateral_triangle (43.93 -> **23.31 C**, -46.9 %), diamond (26.64 -> **15.95 C**,
-40.1 %). The optimal gains there are m = 1.09, 1.49 and 2.12, all outside the grid
`{0.3, 0.5, 0.7, 0.85}` that every prior 2-D FGM number in this project was selected from.
**This is the single most actionable finding: the existing magnitude sweep does not span the
optimum for high-perimeter and sharp-cornered shapes.**

**3. For the shapes where the correction direction is wrong, the line search walks the gain to
the domain floor and the shape stays harmful.** triangle, H_shape and trapezoid all selected
m ~ 0.11, the smallest gain the search would take, and all three remain worse than uniform at
melt-onset (+79.8 %, +67.6 %, +48.7 %). A one-dimensional line search on a scalar gain cannot
repair a search direction; it can only shrink the step. This confirms the assessment's Section 9
two-zone measurement (the heuristic direction is only 63 % aligned with steepest descent) as the
operative mechanism, and it identifies the remaining harmful class as the domain where a solve,
not a better gain, is required.

**4. The hold-out is doing real work and it sometimes costs you.** On rectangle the gain chosen
at the heating peak (m = 0.30) is worse at melt-onset than the in-sample pick (3.89 C against
2.36 C, +65.0 %). Three more shapes lose slightly (pentagon +1.8 %, rounded_rect +2.9 %). This is
the honest price of not selecting on the reported metric, and it bounds how much of the published
best-of-four benefit is in-sample optimism: for 14 of 17 melting shapes the two selections agree
closely, so the published percentages are **not** dominated by selection bias, but rectangle shows
the bias is not zero either.

**5. The square gap to the per-node iterated optimizer barely moves.** COMPUTED: uniform 4.931 C,
best-of-four 3.768 C, calibrated **3.669 C**, per-node iterated best 2.49 C
(`PERNODE_RESULTS.md`). Gap closure to the per-node number goes from 47.6 % to **51.7 %**.
Calibrating the one-shot gain buys 4 percentage points of that gap. The remaining 48 % is
structure the scalar gain cannot express.

**6. L_shape is a clean demonstration of why the feasibility rule was pre-registered.** COMPUTED:
all 8 gains fail to reach phi_bar = 0.90 within the 750 s horizon, so the reported result is
**NOT_REACHED**, not a final-step fallback. Worse, the *unconstrained* fit-metric winner is
m = 1.7632 with heating-peak sigma_T 10.215 C against a 58.707 C uniform baseline, which reads as
an 82.6 % improvement. That part reaches **final phi_bar = 0.0**: it never melts at all. Selecting
on a uniformity metric without a feasibility constraint would have reported a spectacular fake
result here.

---

## 2. What was run, and the configuration pins

**Engine.** `rfam_eqs_coupled.py`, the trusted 2-D EQS plus thermal engine.

**Configuration**, identical to `outputs_eqs/geometry_dual_readstate/run_sweep.py` so every number
is directly comparable to that campaign:

| pin | value |
|---|---|
| grid | 120 x 120 |
| size normalization | as-sized (no equal-area scaling) |
| drive mode | voltage drive, `enforce_generator_power: false`, per-shape `v_cal` calibrated so the UNIFORM baseline absorbs 500.0 W/m |
| horizon | 1500 steps (750 s) |
| map | one-shot PROPORTIONAL inverse (`use_delta_correction` off), proxy `T_phi90`, `invert=True`, `bpp=4`, `baseline_saturation=0.5`, `dead_band=0.05` |
| design variable | the scalar `magnitude` m, domain [0.05, 2.50] |
| fit metric | heating-peak sigma_T (max over steps with phi_bar < 0.90) |
| hold-out metric | melt-onset sigma_T at the first phi_bar >= 0.90 crossing |

The protocol, the two metrics, the feasibility rule and the verdict criteria were frozen in
`outputs_eqs/fgm_calibrated_control/PREREGISTRATION.md` before the first new solve.

**Cache validity, PROVEN.** The 4 stored arms per shape (m = 0.30, 0.50, 0.70, 0.85) are reused as
warm-start evaluations. To show the reuse is exact rather than assumed, the square m = 0.85 point
was re-solved from scratch and compared to the stored artifact
(`outputs_eqs/fgm_calibrated_control/cache_verification.json`):

```
heating-peak: stored 6.992556339431026  fresh 6.992556339431086  |diff| = 6.04e-14
melt-onset:   stored 3.768226446207588  fresh 3.768226446207653  |diff| = 6.48e-14
```

That is bit-identical to floating-point roundoff, so cached and new evaluations sit on the same
curve.

**Budget spent, COMPUTED.** 4 warm-start evaluations plus at most 4 new solves per shape, so 8
evaluations per shape, within the 5-to-8 pre-registered budget. **72 new solves in total**
(4 per shape on all 18), **6 830 s of new-solve wall time** plus the verification solve. Per-solve
cost was 60-100 s on an idle machine; three shapes ran 500-700 s per solve while an unrelated
`pytest test_heatr3d_s1.py` job (not started by this campaign, 439 % CPU) shared the machine.

**Coverage, stated explicitly.** 18 of the 19 standardized shapes were run: square, circle,
hexagon, triangle, L_shape (the five required) plus rectangle, rounded_rect, octagon, pentagon,
diamond, ellipse, star, star6, cross, trapezoid, equilateral_triangle, H_shape, T_shape.
**gt_logo was NOT run.** Its geometry is built from a scalable vector graphics file and
`shapes.make_shape_from_svg` imports `cv2`, which is not installed in `.venv312`
(`ModuleNotFoundError: No module named 'cv2'`). The environment was left unmodified rather than
mutated mid-campaign. Nothing was silently truncated.

---

## 3. Results

### Table 1. Melt-onset sigma_T (the HOLD-OUT read state), deg C

Arm (a) uniform and arm (b) best-of-four are read from stored `geometry_dual_readstate`
artifacts and were not re-run. Arm (c) is this campaign.

| shape | (a) uniform | (b) best-of-four in-sample | (c) calibrated gain | c vs a | c vs b | b gain m | c gain m | new solves |
|---|---|---|---|---|---|---|---|---|
| square | 4.931 | 3.768 | 3.669 | -25.6 % | -2.6 % | 0.85 | 0.8775 | 4 |
| circle | 17.428 | 7.134 | 7.117 | -59.2 % | -0.2 % | 0.70 | 0.6644 | 4 |
| hexagon | 18.083 | 6.788 | 6.726 | -62.8 % | -0.9 % | 0.85 | 0.8327 | 4 |
| triangle | 23.043 | 45.334 | 41.423 | +79.8 % | -8.6 % | 0.30 | 0.1090 | 4 |
| L_shape | 58.729 | NOT_REACHED | NOT_REACHED | n/a | n/a | none | NOT_REACHED | 4 |
| H_shape | 6.562 | 11.077 | 10.996 | +67.6 % | -0.7 % | 0.30 | 0.1090 | 4 |
| T_shape | 60.149 | 71.039 | 71.039 | +18.1 % | +0.0 % | 0.30 | 0.3000 | 4 |
| cross | 43.771 | 31.401 | **13.858** | **-68.3 %** | **-55.9 %** | 0.85 | 1.0927 | 4 |
| diamond | 32.234 | 26.637 | **15.948** | **-50.5 %** | **-40.1 %** | 0.85 | 2.1208 | 4 |
| ellipse | 22.046 | 8.684 | 8.603 | -61.0 % | -0.9 % | 0.70 | 0.7429 | 4 |
| equilateral_triangle | 41.480 | 43.933 | **23.310** | **-43.8 %** | **-46.9 %** | 0.85 | 1.4854 | 4 |
| octagon | 18.248 | 7.045 | 6.959 | -61.9 % | -1.2 % | 0.50 | 0.4513 | 4 |
| pentagon | 24.036 | 14.617 | 14.880 | -38.1 % | +1.8 % | 0.50 | 0.3000 | 4 |
| rectangle | 6.971 | 2.356 | 3.888 | -44.2 % | **+65.0 %** | 0.85 | 0.3000 | 4 |
| rounded_rect | 7.589 | 4.612 | 4.748 | -37.4 % | +2.9 % | 0.70 | 0.5877 | 4 |
| star | 24.858 | 24.168 | 24.168 | -2.8 % | +0.0 % | 0.85 | 0.8500 | 4 |
| star6 | 15.927 | 8.272 | 6.945 | -56.4 % | -16.0 % | 0.85 | 0.9375 | 4 |
| trapezoid | 10.842 | 16.735 | 16.124 | +48.7 % | -3.7 % | 0.30 | 0.1090 | 4 |

### Table 2. Heating-peak sigma_T (the FIT read state), deg C

| shape | (a) uniform | (b) best-of-four in-sample | (c) calibrated gain | c vs a | c vs b |
|---|---|---|---|---|---|
| square | 11.049 | 6.993 | 6.923 | -37.3 % | -1.0 % |
| circle | 19.961 | 12.136 | 11.991 | -39.9 % | -1.2 % |
| hexagon | 19.934 | 12.083 | 11.853 | -40.5 % | -1.9 % |
| triangle | 24.831 | 45.269 | 41.360 | +66.6 % | -8.6 % |
| L_shape | 58.707 | NOT_REACHED | NOT_REACHED | n/a | n/a |
| H_shape | 14.869 | 16.595 | 16.064 | +8.0 % | -3.2 % |
| T_shape | 60.127 | 71.020 | 71.020 | +18.1 % | +0.0 % |
| cross | 43.757 | 31.679 | 15.562 | -64.4 % | -50.9 % |
| diamond | 32.226 | 29.299 | 19.996 | -38.0 % | -31.8 % |
| ellipse | 22.018 | 12.982 | 12.891 | -41.5 % | -0.7 % |
| equilateral_triangle | 41.431 | 43.808 | 24.704 | -40.4 % | -43.6 % |
| octagon | 20.787 | 11.693 | 11.370 | -45.3 % | -2.8 % |
| pentagon | 24.098 | 19.435 | 19.264 | -20.1 % | -0.9 % |
| rectangle | 14.943 | 9.689 | 8.881 | -40.6 % | -8.3 % |
| rounded_rect | 10.817 | 8.524 | 7.867 | -27.3 % | -7.7 % |
| star | 26.967 | 25.156 | 25.156 | -6.7 % | +0.0 % |
| star6 | 18.346 | 11.315 | 9.378 | -48.9 % | -17.1 % |
| trapezoid | 14.867 | 20.463 | 19.521 | +31.3 % | -4.6 % |

Note the calibrated arm wins the FIT metric on every shape by construction (a line search cannot
select a step that makes its own objective worse). Table 1 is the number that carries information;
Table 2 is the objective the gain was chosen on.

### Table 3. Search diagnostics and standing gates

`P_abs` is absorbed power at the melt-onset read state; the voltage-drive protocol leaves it free,
so it must be reported per arm. `residual frac` is |energy residual| / integrated dose at that
read state, the standing gate from `HEATR_STANDARD_PARAMETERS.md` Section 4.

| shape | new gains evaluated | stop reason | status | infeasible gains | P_abs at melt, W/m (a / c) | residual frac (c) | dT clip frac (c) |
|---|---|---|---|---|---|---|---|
| square | 1.0927, 0.8775, 0.8970, 0.8807 | budget exhausted | OK | none | 500.0 / 564.8 | 0.000149 | 0.0000 |
| circle | 0.6235, 0.6644, 0.6629, 0.6801 | budget exhausted | OK | none | 500.0 / 764.7 | 0.002936 | 0.0000 |
| hexagon | 1.0927, 0.8327, 0.7934, 0.8084 | budget exhausted | OK | none | 500.0 / 746.7 | 0.004835 | 0.0000 |
| triangle | 0.0500, 0.2045, 0.1455, 0.1090 | budget exhausted | OK | none | 500.0 / 604.0 | 0.025510 | 0.0000 |
| L_shape | 1.0927, 1.4854, 2.1208, 1.7632 | budget exhausted | **NOT_REACHED** | all 8 | 500.0 / NOT_REACHED | n/a | n/a |
| H_shape | 0.0500, 0.2045, 0.1455, 0.1090 | budget exhausted | OK | none | 500.0 / 570.7 | 0.014133 | 0.0000 |
| T_shape | 1.0927, 1.4854, 2.1208, 2.5000 | budget exhausted | OK | 7 of 8 | 500.0 / 474.4 | **0.051990** | 0.0000 |
| cross | 1.0927, 1.4854, 1.1855, 1.1184 | budget exhausted | OK | 1.4854 | 500.0 / 397.9 | 0.018349 | 0.0000 |
| diamond | 1.0927, 1.4854, 2.1208, 2.5000 | budget exhausted | OK | 2.5000 | 500.0 / 376.6 | 0.011773 | 0.0000 |
| ellipse | 0.7429, 0.7624, 0.7164, 0.7472 | budget exhausted | OK | none | 500.0 / 717.8 | 0.007200 | 0.0000 |
| equilateral_triangle | 1.0927, 1.4854, 2.1208, 1.6243 | budget exhausted | OK | none | 500.0 / 518.1 | 0.013633 | 0.0000 |
| octagon | 0.4591, 0.4407, 0.4513, 0.4502 | budget exhausted | OK | none | 500.0 / **1053.9** | 0.005479 | 0.0000 |
| pentagon | 0.0500, 0.3277, 0.2478, 0.2935 | budget exhausted | OK | none | 500.0 / 669.8 | 0.009199 | 0.0000 |
| rectangle | 0.0500, 0.2703, 0.3630, 0.3131 | budget exhausted | OK | none | 500.0 / 491.2 | 0.000366 | 0.0000 |
| rounded_rect | 0.6028, 0.6047, 0.5877, 0.5569 | budget exhausted | OK | none | 500.0 / 643.4 | 0.000262 | 0.0000 |
| star | 1.0927, 0.8558, 0.8370, 0.8419 | budget exhausted | OK | none | 500.0 / 535.3 | 0.029054 | 0.0000 |
| star6 | 1.0927, 0.9089, 0.9375, 0.9644 | budget exhausted | OK | none | 500.0 / 594.3 | 0.007907 | 0.0000 |
| trapezoid | 0.0500, 0.2045, 0.1455, 0.1090 | budget exhausted | OK | none | 500.0 / 593.6 | 0.005518 | 0.0000 |

**Gate readings.** The temperature-step clip fraction is 0.0000 on every selected arm, so the
non-smooth clip term is dormant throughout. The energy residual gate (< 5 % of integrated dose) is
met on 16 of 17 melting shapes. **T_shape fails it at 5.20 %** and its number should carry that
qualifier. star (2.91 %) and triangle (2.55 %) are the next highest.

### Table 4. Pre-registered verdict per shape

| shape | (b) harmful vs uniform? | (c) harmful vs uniform? | (c) beats (b) on hold-out? | rescued? |
|---|---|---|---|---|
| square | no | no | YES | no |
| circle | no | no | YES | no |
| hexagon | no | no | YES | no |
| triangle | YES | YES | YES | no |
| L_shape | n/a (no arm melts) | NOT_REACHED | n/a | n/a |
| H_shape | YES | YES | YES | no |
| T_shape | YES | YES | no | no |
| cross | no | no | YES | no |
| diamond | no | no | YES | no |
| ellipse | no | no | YES | no |
| equilateral_triangle | YES | no | YES | **YES** |
| octagon | no | no | YES | no |
| pentagon | no | no | no | no |
| rectangle | no | no | no | no |
| rounded_rect | no | no | no | no |
| star | no | no | no | no |
| star6 | no | no | YES | no |
| trapezoid | YES | YES | YES | no |

Totals over 18 shapes: calibration beats best-of-four on the hold-out in **12**; harmful versus
uniform at melt-onset, **5** for (b) and **4** for (c); **1** rescued.

---

## 4. What calibration fixed, and what it did not

### Fixed (COMPUTED)

- **The truncated search domain.** Five shapes have their optimum above the stored grid's top point
  m = 0.85: diamond (2.12), equilateral_triangle (1.49), cross (1.09), star6 (0.94), square (0.88).
  The first three gain 40 to 56 % on the hold-out metric against the best-of-four arm. Every published 2-D FGM percentage for these shapes
  understates the achievable benefit of the same map shape.
- **The one genuine rescue.** equilateral_triangle goes from +5.9 % worse than uniform to
  -43.8 % better. It was in the harmful class only because the grid stopped short.
- **Small, consistent gains where the heuristic already worked.** square, circle, hexagon, ellipse,
  octagon all improve by 0.2 to 2.6 % on the hold-out. Real but not decision-changing.
- **A defensible selection rule.** Nothing in this campaign is selected on the metric it is
  reported on, so the reported percentages are free of in-sample optimism.

### Not fixed (COMPUTED)

- **The harmful class survives.** triangle +79.8 %, H_shape +67.6 %, trapezoid +48.7 %,
  T_shape +18.1 %, all still worse than doing nothing at melt-onset. In all four the search walked
  the gain down to or near the floor and the fit metric was still flat or rising, which is the
  signature of a wrong direction rather than a wrong step length.
- **The gain cannot reach the uniform arm.** Because `sat_scaled = 0.5 + m * (sat_raw - 0.5)`, the
  m -> 0 limit is a part uniformly doped at sat = 0.5, not the uniform baseline at sat = 1.0. So
  the line search has no "do nothing" option: the smallest step it can take still halves the
  dopant. ASSUMED consequence, not separately measured: some of the residual harm on the harmful
  class is this dose offset rather than the map shape. Testing that needs a two-parameter search
  over (magnitude, baseline_saturation), which was out of the pre-registered budget.
- **The gap to the per-node iterated optimizer.** Square gap closure 47.6 % -> 51.7 %.
- **L_shape.** No gain melts. Higher gain makes it strictly worse: final phi_bar falls
  monotonically 0.8955 (m = 0.30) -> 0.8089 (0.85) -> 0.5192 (1.09) -> 0.0000 (m >= 1.4854).

### Newly measured numerical findings

1. **The fit metric is genuinely non-monotone in the gain, with a sharp interior optimum for some
   shapes.** cross heating-peak sigma_T: 43.99 (m = 0.30), 42.15, 38.07, 31.68 (0.85),
   **15.56 (1.09)**, 16.16 (1.12), 16.43 (1.19), 28.18 (1.49, and no melt). A 28 % change in gain
   either side of the optimum costs a factor of about 1.8. Any campaign that fixes the magnitude by
   hand is sampling a sharply varying function.
2. **Absorbed power varies by a factor of 2.8 across calibrated arms at fixed drive voltage**:
   octagon 1 053.9 W/m against diamond 376.6 W/m, both from a 500.0 W/m uniform baseline. The
   voltage-drive protocol is the pre-registered choice (matching `geometry_dual_readstate`), but
   these arms are NOT dose matched and must not be read as pure redistribution results. The
   power-matched re-analysis that `POWER_MATCHED.md` performs for the stored grid has not been done
   for the calibrated arms.
3. **A uniformity metric without a feasibility constraint is exploitable.** L_shape's
   unconstrained fit winner (m = 1.7632) reports heating-peak 10.215 C against a 58.707 C baseline
   while reaching final phi_bar = 0.0.
4. **Selection-rule bias is real but small.** Over the 17 melting shapes, switching from in-sample
   melt-onset selection to hold-out heating-peak selection changes the reported melt-onset number
   by more than 5 % on only 4 shapes, and only one of those (rectangle, +65.0 %) is a loss.

---

## 5. Honest limitations

- **PROVEN** only: the pure line-search and hold-out-selection logic (16 unit tests) and the
  bit-identical cache reproduction (6.0e-14). No gradient exists in this workstream, so there is no
  finite-difference gate to report; the forward solves are the verification gate for the numbers.
- **Single grid.** Everything here is 120 x 120. The 2-D mesh-convergence risk named in
  `FGM_INVERSE_DESIGN_ASSESSMENT.md` Section 3.2 risk 1 is untouched by this campaign.
- **Single metric definition.** sigma_T here is the 2-D `ui_rms * (T_bar - 23)` convention. It must
  not be compared numerically with heatr3d `std(T)` values.
- **Not dose matched.** See finding 2 above.
- **T_shape violates the 5 % energy-residual gate** (5.20 %).
- **gt_logo not run** (missing `cv2`).
- **The 2.49 C per-node reference** is from `PERNODE_RESULTS.md` at Smax = 0.06 with a two-sided
  actuator, which is a different actuation range from this campaign's [0, 1] clipped map. The gap
  closure figure is therefore indicative, not a like-for-like budget comparison.
- **ASSUMED:** that `magnitude` is the right single scalar to calibrate. The alternative
  one-parameter family, `baseline_saturation` at fixed magnitude, was not tested.

---

## 6. The single most valuable next layer

**A two-parameter calibration over (magnitude, baseline_saturation) on the four surviving harmful
shapes** (triangle, H_shape, trapezoid, T_shape) plus L_shape. It costs about 12 to 16 solves per
shape, roughly 90 minutes total at the measured 60-100 s per solve, and it separates the two
mechanisms this campaign could not: whether the residual harm is the dose offset baked into the
m -> 0 limit, or the map direction itself. If dose is the cause, the harmful class is still a
calibration problem. If it is not, the harmful class is proven to need a real solve, and that is
the evidence that justifies the 2-D adjoint prototype of assessment Section 7.

---

## 7. Unseeded re-run (v2), 2026-07-31

**This section supersedes the harmful-class conclusion of Section 1 item 3 for one shape and
confirms it for three. Sections 1 to 6 are left exactly as originally written; nothing above has
been rewritten.**

### 7.1 Why v2 exists

The step-3 adjoint prototype (`ADJOINT_PROTOTYPE_REPORT.md` Section 7.1) measured a dense 26-point
gain scan on the triangle and found the fit metric is **bimodal**: a local maximum of 48.82 C at
m = 0.638 and a global minimum of 23.01 C at m = 2.402. The v1 search above was warm-started from
the stored grid `{0.30, 0.50, 0.70, 0.85}`, which lies **entirely on the left flank of that local
maximum**. A bracket search seeded there correctly sees the objective rising to its right and
falling to its left, and walks to the domain floor. The v1 triangle result (+79.8 % worse than
uniform) is therefore partly a **warm-start artifact, not a property of the map family.**

That failure mode is now covered by a unit test. `test_seeded_search_is_trapped_by_the_triangle_local_maximum`
drives the v1 routine with a fixture interpolated from the captured 26-point scan and asserts it
walks below m = 0.35; it passed on first run, reproducing the artifact in the pure logic before any
solve was spent.

### 7.2 What changed, and what did not

**Changed (one thing only).** The search is now UNSEEDED: a 7-point log-spaced coarse scan across
the full domain, then up to 3 refinement probes around the best bracket
(`gain_calibration.calibrate_gain_unseeded`, `coarse_scan_gains`). The domain was extended from
[0.05, 2.50] to **[0.05, 6.00]**, because the measured triangle global minimum at m = 2.402 sat
right at the old cap. Log spacing was chosen because useful gains span more than two decades.

**Unchanged.** Every configuration pin of Section 2; the fit metric (heating-peak sigma_T); the
hold-out metric (melt-onset sigma_T at phi_bar = 0.90); the feasibility rule (reject any gain that
does not cross phi_bar = 0.90 in the horizon, and never substitute a final-step fallback); the
energy-residual gate; absorbed power reported per arm.

**Cost, COMPUTED.** **157 new solves** plus **20 cache hits** across 18 shapes, 10 637 s
(2.95 h) of new-solve wall time. Per shape: 10 evaluations, of which 2 to 10 were new.
gt_logo remains unrun (`cv2` missing).

**TDD.** 6 new unit tests, written red first (5 failed on the skeleton for the intended reason,
the 6th passed immediately and is the artifact reproduction described above). **22 tests total,
all passing.**

### 7.3 Verdict of the unseeded re-run

**1. Exactly ONE of the five previously-harmful shapes is rescued: the triangle.** Melt-onset
sigma_T 41.423 C (v1) becomes **21.401 C** (v2), which is **-7.1 % versus the 23.043 C uniform
baseline**, where v1 reported +79.8 % worse. The direction of that correction is confirmed
independently by the step-3 scan (-5.1 %). **H_shape, trapezoid and T_shape survive an unseeded
global search over the whole domain and remain harmful** (+67.6 %, +48.7 %, +14.9 %), so for those
three the Section 1 item 3 conclusion stands and is now stronger: it is not a warm-start artifact.

**2. L_shape's feasibility case is resolved, and the answer is bad.** v2 found two feasible gains
(m = 0.0500 and 0.1110) that DO reach phi_bar = 0.90, so v1's NOT_REACHED was itself a
search-coverage artifact. But the best feasible arm reads **74.108 C against a 58.729 C uniform
baseline, +26.2 % worse**. The other 8 of 10 gains remain infeasible, and the failure is
monotone in the gain: final phi_bar goes 0.9055 (m = 0.111), 0.8990 (0.247), 0.8749 (0.548),
0.0688 (1.216), then **0.0000 for every m >= 2.31**. L_shape is not rescued; it is now
measurably harmful rather than unmeasurable. **Caveat: the L_shape v2 arm fails the 5 %
energy-residual gate at 6.12 %**, so that number carries a health warning.

**3. The 12-of-18 hold-out win rate against best-of-four does NOT change: it is 12 of 18 again.**
The composition changes, and not only favourably. Unseeded search buys global coverage and pays
for it in local precision at a fixed budget: **star regresses from -2.8 % to +9.4 % versus
uniform** (its 0.85 basin was missed by the coarse scan), and diamond loses 19.2 % relative to v1.
So the count of shapes harmful versus uniform goes **4 (v1) to 5 (v2)**. Trading one rescue for one
regression is not progress on its own.

**4. The right procedure is neither v1 nor v2 alone, and it costs nothing extra.** Applying the
same hold-out rule to the UNION of every gain already solved (Table 5b, zero additional solves)
gives **4 shapes harmful and 13 of 18 beating best-of-four**, which dominates both arms. A global
coarse scan is needed to find the basin; local refinement is needed to sit in it. Any future
campaign should do both and select once over the union.

### Table 5. Seeded (v1) versus unseeded (v2), melt-onset sigma_T (HOLD-OUT read state), deg C

| shape | (a) uniform | (b) best-of-four | (c1) v1 seeded | (c2) v2 unseeded | c2 vs a | c2 vs c1 | m (v1) | m (v2) | new solves (v2) |
|---|---|---|---|---|---|---|---|---|---|
| triangle | 23.043 | 45.334 | 41.423 | **21.401** | **-7.1 %** | -48.3 % | 0.1090 | 6.0000 | 6 |
| H_shape | 6.562 | 11.077 | 10.996 | 10.996 | +67.6 % | +0.0 % | 0.1090 | 0.1095 | 9 |
| trapezoid | 10.842 | 16.735 | 16.124 | 16.124 | +48.7 % | +0.0 % | 0.1090 | 0.1093 | 9 |
| T_shape | 60.149 | 71.039 | 71.039 | 69.106 | +14.9 % | -2.7 % | 0.3000 | 0.1110 | 10 |
| L_shape | 58.729 | NOT_REACHED | NOT_REACHED | 74.108 | +26.2 % | n/a | NOT_REACHED | 0.1110 | 10 |
| square | 4.931 | 3.768 | 3.669 | 3.647 | -26.0 % | -0.6 % | 0.8775 | 0.9033 | 2 |
| circle | 17.428 | 7.134 | 7.117 | 7.159 | -58.9 % | +0.6 % | 0.6644 | 0.6402 | 10 |
| hexagon | 18.083 | 6.788 | 6.726 | 6.773 | -62.5 % | +0.7 % | 0.8327 | 0.8217 | 10 |
| cross | 43.771 | 31.401 | 13.858 | 14.519 | -66.8 % | +4.8 % | 1.0927 | 1.2164 | 10 |
| diamond | 32.234 | 26.637 | 15.948 | 19.005 | -41.0 % | +19.2 % | 2.1208 | 1.2164 | 10 |
| ellipse | 22.046 | 8.684 | 8.603 | 8.485 | -61.5 % | -1.4 % | 0.7429 | 0.6713 | 10 |
| equilateral_triangle | 41.480 | 43.933 | 23.310 | 23.346 | -43.7 % | +0.2 % | 1.4854 | 1.6219 | 5 |
| octagon | 18.248 | 7.045 | 6.959 | 7.002 | -61.6 % | +0.6 % | 0.4513 | 0.5052 | 8 |
| pentagon | 24.036 | 14.617 | 14.880 | 14.787 | -38.5 % | -0.6 % | 0.3000 | 0.3424 | 9 |
| rectangle | 6.971 | 2.356 | 3.888 | 3.075 | -55.9 % | -20.9 % | 0.3000 | 1.2164 | 9 |
| rounded_rect | 7.589 | 4.612 | 4.748 | 4.771 | -37.1 % | +0.5 % | 0.5877 | 0.5756 | 10 |
| star | 24.858 | 24.168 | 24.168 | **27.204** | **+9.4 %** | +12.6 % | 0.8500 | 0.5477 | 10 |
| star6 | 15.927 | 8.272 | 6.945 | 7.233 | -54.6 % | +4.1 % | 0.9375 | 0.9028 | 10 |

Over 18 shapes: v2 beats best-of-four on the hold-out in **12** (same count as v1); harmful versus
uniform at melt-onset, **4** for v1 and **5** for v2; **1** shape rescued by going unseeded.

### Table 5b. Union arm (same hold-out rule, all gains already solved, ZERO extra solves)

| shape | (a) uniform | (c1) v1 seeded | (c2) v2 unseeded | (c3) union | c3 vs a | m (union) | evaluations |
|---|---|---|---|---|---|---|---|
| triangle | 23.043 | 41.423 | 21.401 | 21.401 | -7.1 % | 6.0000 | 14 |
| H_shape | 6.562 | 10.996 | 10.996 | 10.996 | +67.6 % | 0.1090 | 17 |
| trapezoid | 10.842 | 16.124 | 16.124 | 16.124 | +48.7 % | 0.1090 | 17 |
| T_shape | 60.149 | 71.039 | 69.106 | 69.106 | +14.9 % | 0.1110 | 18 |
| L_shape | 58.729 | NOT_REACHED | 74.108 | 74.108 | +26.2 % | 0.1110 | 18 |
| square | 4.931 | 3.669 | 3.647 | 3.647 | -26.0 % | 0.9033 | 18 |
| circle | 17.428 | 7.117 | 7.159 | 7.117 | -59.2 % | 0.6644 | 18 |
| hexagon | 18.083 | 6.726 | 6.773 | 6.773 | -62.5 % | 0.8217 | 18 |
| cross | 43.771 | 13.858 | 14.519 | 13.858 | -68.3 % | 1.0927 | 18 |
| diamond | 32.234 | 15.948 | 19.005 | 15.948 | -50.5 % | 2.1208 | 18 |
| ellipse | 22.046 | 8.603 | 8.485 | 8.485 | -61.5 % | 0.6713 | 18 |
| equilateral_triangle | 41.480 | 23.310 | 23.346 | 23.310 | -43.8 % | 1.4854 | 18 |
| octagon | 18.248 | 6.959 | 7.002 | 6.959 | -61.9 % | 0.4513 | 18 |
| pentagon | 24.036 | 14.880 | 14.787 | 14.880 | -38.1 % | 0.3000 | 17 |
| rectangle | 6.971 | 3.888 | 3.075 | 3.075 | -55.9 % | 1.2164 | 17 |
| rounded_rect | 7.589 | 4.748 | 4.771 | 4.771 | -37.1 % | 0.5756 | 18 |
| star | 24.858 | 24.168 | 27.204 | 24.168 | -2.8 % | 0.8500 | 18 |
| star6 | 15.927 | 6.945 | 7.233 | 6.945 | -56.4 % | 0.9375 | 18 |

Union arm over 18 shapes: **4** harmful versus uniform at melt-onset; **13 of 18** beat
best-of-four on the hold-out.

### Table 6. v2 search diagnostics and gates

| shape | selected m | evaluations (new / cached) | stop reason | status | infeasible | P_abs at melt, W/m (uniform / v2) | residual frac (v2) | dT clip frac (v2) |
|---|---|---|---|---|---|---|---|---|
| triangle | 6.0000 | 6 / 1 | domain edge reached | OK | none | 500.0 / 353.7 | 0.013704 | 0.0000 |
| H_shape | 0.1095 | 9 / 1 | budget exhausted | OK | none | 500.0 / 570.7 | 0.014133 | 0.0000 |
| trapezoid | 0.1093 | 9 / 1 | budget exhausted | OK | none | 500.0 / 593.6 | 0.005518 | 0.0000 |
| T_shape | 0.1110 | 10 / 0 | budget exhausted | OK | 7 of 10 | 500.0 / 484.8 | **0.051698** | 0.0000 |
| L_shape | 0.1110 | 10 / 0 | budget exhausted | OK | 8 of 10 | 500.0 / 518.4 | **0.061154** | 0.0000 |
| square | 0.9033 | 2 / 8 | budget exhausted | OK | none | 500.0 / 563.9 | 0.000154 | 0.0000 |
| circle | 0.6402 | 10 / 0 | budget exhausted | OK | none | 500.0 / 763.5 | 0.002961 | 0.0000 |
| hexagon | 0.8217 | 10 / 0 | budget exhausted | OK | none | 500.0 / 745.4 | 0.004823 | 0.0000 |
| cross | 1.2164 | 10 / 0 | budget exhausted | OK | 4 of 10 | 500.0 / 325.3 | 0.021542 | 0.0000 |
| diamond | 1.2164 | 10 / 0 | budget exhausted | OK | 5 of 10 | 500.0 / 466.6 | 0.012300 | 0.0000 |
| ellipse | 0.6713 | 10 / 0 | budget exhausted | OK | none | 500.0 / 708.6 | 0.006986 | 0.0000 |
| equilateral_triangle | 1.6219 | 5 / 5 | budget exhausted | OK | none | 500.0 / 504.9 | 0.014182 | 0.0000 |
| octagon | 0.5052 | 8 / 2 | budget exhausted | OK | none | 500.0 / 1067.9 | 0.005766 | 0.0000 |
| pentagon | 0.3424 | 9 / 1 | budget exhausted | OK | none | 500.0 / 672.7 | 0.009214 | 0.0000 |
| rectangle | 1.2164 | 9 / 1 | budget exhausted | OK | none | 500.0 / 458.8 | 0.003129 | 0.0000 |
| rounded_rect | 0.5756 | 10 / 0 | budget exhausted | OK | none | 500.0 / 641.8 | 0.000258 | 0.0000 |
| star | 0.5477 | 10 / 0 | budget exhausted | OK | 5 of 10 | 500.0 / 590.7 | **0.035783** | 0.0000 |
| star6 | 0.9028 | 10 / 0 | budget exhausted | OK | none | 500.0 / 608.3 | 0.009056 | 0.0000 |

**Gate readings for v2.** The temperature-step clip fraction is 0.0000 on every selected arm. The
5 % energy-residual gate is met on 16 of 18 shapes; **L_shape fails at 6.12 % and T_shape fails at
5.17 %**, and both numbers must be quoted with that qualifier. star is the next highest at 3.58 %.

### 7.4 The v2 triangle curve, measured under this campaign's own pins

The bimodality reproduces here, not only in the step-3 conventions. Heating-peak sigma_T against
melt-onset sigma_T, deg C, uniform baseline 24.831 / 23.043:

| m | 0.0500 | 0.1110 | 0.2466 | 0.5477 | 1.2164 | 2.7016 | 6.0000 |
|---|---|---|---|---|---|---|---|
| heating-peak | 41.41 | 41.36 | 44.24 | **48.68 (local max)** | 27.77 | 22.69 | **22.44** |
| melt-onset | 41.47 | 41.42 | 44.30 | 48.75 | 27.63 | 21.61 | **21.40** |

Two caveats on the triangle rescue. First, the search stopped at the **domain edge** m = 6.00, so
the true optimum may lie beyond it; the curve is nearly flat there (22.69 at m = 2.70 against 22.44
at m = 6.00), so the remaining headroom is small but the number is a boundary value, not an
interior minimum. Second, that arm absorbs **353.7 W/m against the 500.0 W/m uniform baseline**,
a 29 % dose reduction at fixed drive voltage. The triangle rescue is therefore **not dose matched**
and part of it may be a dose effect rather than redistribution.

### 7.5 What this changes in the Section 1 verdict

- Item 3 ("the harmful class survives") is **corrected for the triangle** and **confirmed for
  H_shape, trapezoid and T_shape** against a global search over [0.05, 6.00].
- Item 6 (L_shape NOT_REACHED) is **corrected**: feasible gains exist at m <= 0.111; they are
  harmful, and the arm fails the energy-residual gate.
- Item 1's headline count is **unchanged at 12 of 18** for the unseeded arm, and improves to
  **13 of 18** for the union arm.
- The rescued count over the whole campaign rises from 1 (equilateral_triangle, v1) to **2**
  (equilateral_triangle and triangle) once both searches are pooled.
- **New methodological finding:** a warm start from a hand-chosen grid can invert the sign of a
  reported compensation result. The v1 triangle number was +79.8 % worse than uniform; the same map
  family with the same budget, searched globally, is 7.1 % better. Any future gain sweep in this
  project should begin with a coarse scan spanning the full feasible domain.

---

## 8. Artifacts (absolute paths)

- Pre-registration: `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/outputs_eqs/fgm_calibrated_control/PREREGISTRATION.md`
- Calibration module: `.../outputs_eqs/fgm_calibrated_control/gain_calibration.py`
- Unit tests (22, all passing): `.../outputs_eqs/fgm_calibrated_control/test_gain_calibration.py`
- Runner (`run` = v1 seeded, `run2` = v2 unseeded, `verify` = cache check): `.../outputs_eqs/fgm_calibrated_control/run_calibrated.py`
- Table builders: `.../outputs_eqs/fgm_calibrated_control/{build_tables.py, build_tables_v2.py}`
- Cache verification: `.../outputs_eqs/fgm_calibrated_control/cache_verification.json`
- Per-shape results: `.../outputs_eqs/fgm_calibrated_control/<shape>.json` (v1, 18 files) and `<shape>.v2.json` (v2, 18 files)
- Generated tables: `.../outputs_eqs/fgm_calibrated_control/{tables.md, tables_v2.md}`
- Solver run directories and maps: `.../outputs_eqs/fgm_calibrated_control/runs/<shape>/`
- Solver configurations actually executed: `.../outputs_eqs/fgm_calibrated_control/configs/`
- Logs: `.../outputs_eqs/fgm_calibrated_control/{groupA.log, groupB.log, v2_A.log ... v2_F.log}`
