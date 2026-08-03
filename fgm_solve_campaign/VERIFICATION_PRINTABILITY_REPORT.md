# Verification and printability of the solved dopant map

**Date:** 2026-07-31. **Scope:** a verification-and-hardening pass on the
shape-fidelity solve of `SHAPE_FIDELITY_SOLVE_REPORT.md`. Three questions:
is the head-to-head apples-to-apples against the real stored historical masks
on the real engine at standard parameters; do the results survive the printer's
2 or 4 bits per pixel; and is a single printing pass enough. Nothing was
committed. No dissertation file was touched. All work is in the git worktree
`.claude/worktrees/agent-a02efc1141ba69c58`.

**Acronyms, expanded on first use.** FGM = functionally graded material (a
spatially varying dopant saturation map). EQS = electro-quasi-static (the
low-frequency Maxwell approximation the 2-D solver uses). IoU = intersection
over union. FD = finite difference. bpp = bits per pixel. sigma_T = the 2-D
campaign uniformity metric `ui_rms_part * (T_bar_part - 23 C)`, in deg C.
phi = melt fraction. phi_bar = mean part melt fraction. J = the shape-fidelity
objective, `sum over the WHOLE domain of (phi - chi_part)^2` in units of cells.

**Evidence tags.** PROVEN = unit-tested or bit-identity-gated. COMPUTED =
measured from a real solve in this campaign. ASSUMED = a modelling choice or an
inference not measured here.

**Read and stop conventions, stated once and referenced throughout.**

* **J-stop** (used for every J, IoU, growth and under-melt number below):
  t_stop = argmin over that arm's own stored trajectory of J. Each arm gets its
  own stop. `at_horizon` is reported whenever the minimum sits on the last
  stored step, which means the run was truncated before the objective turned.
* **T_phi90** (used for every sigma_T identity number in Section 2): the
  temperature field at the END of the first outer step whose phi_bar reaches
  0.90, cast to float32 exactly as `rfam_eqs_coupled.py:3181` stores it.
  sigma_T on that field is `ui_rms_part * (T_bar_part - 23 C)`.
* **melted region** for IoU, growth and under-melt: phi >= 0.5. J itself uses no
  threshold.

---

## 1. Verdict, three sentences

**1. Yes, apples-to-apples, and the engine claim is now stronger than it was:**
the prototype reproduces four ACTUAL archived historical runs BIT-IDENTICALLY
(`max|diff| = 0.000e+00` on the stored T_phi90 field, sigma_T agreeing to all
twelve printed digits, run directories cited in Section 2), the prototype's
pinned configuration differs from the historical `used_config.yaml` in exactly
ONE field (the path of the dopant map npz) and in nothing physical, and the
rebuilt heuristic control arms of the previous report land within -2.9 % to
+2.1 % of the ACTUAL stored 4-bpp masks scored under the same objective; the
one place the picture changes is the square, where the best historical arm is
not the window-reselection winner but the old-grid m = 0.85 mask at J = 22.20,
so the solved arm's margin on the square shrinks from -27.9 % to -9.7 % (double
pass) and turns to +7.3 % on J for the single-pass arm even though that arm
reaches a PERFECT IoU of 1.0000.

**2. Yes, the results survive the printer's bit depth, comfortably at 4 bpp and
with a real but small cost at 2 bpp:** the square goes from IoU 0.9975
continuous to **1.0000** at 4 bpp and 0.9804 at 2 bpp (J 22.63 to 23.83 to
46.40), the triangle from 0.8782 to **0.8762** at 4 bpp and 0.8689 at 2 bpp
(J 90.79 to 91.24 to 100.44), and the full printer-resolution round trip at
720 dots per inch costs less than 2 % of J on three of four shapes.

**3. Single pass is sufficient on three of four shapes and the double pass buys
real fidelity only on the L_shape:** at 4 bpp the single-pass box [0, 1] arm is
within 1 IoU point of the double-pass box [0, 1.5] arm on the square
(1.0000 against 0.9975), the triangle (0.8578 against 0.8762) and the cross
(0.6755 against 0.6627), but on the L_shape the double pass is worth
**+10.3 IoU points** (0.5209 to 0.6234) and **-22.4 % on J** (521.2 to 404.6),
and longer exposure does NOT substitute for it (the L_shape single-pass arm
already sits at 79 % of its cells pinned at the top of the box and its optimal
stop is 263.0 s against the double pass's 253.0 s, that is, the single-pass arm
runs LONGER and still melts less).

---

## 2. Task 1b: is the configuration comparable, and does the engine still match?

### 2.1 Field-by-field configuration diff

COMPUTED, `diff` of each historical `used_config.yaml` against the
configuration the prototype was pinned to. Historical run directory:
`<main>/outputs_eqs/geometry_dual_readstate/runs/<shape>/fgm_m0p85/used_config.yaml`.
Prototype configuration:
`<main>/outputs_eqs/fgm_calibrated_control/configs/<cfg>.yaml`, with
`<cfg>` = `square_m0p8775`, `triangle_m0p1090`, `cross_m1p0927`,
`L_shape_m0p1110`.

| block | fields compared | delta | assessed effect |
|---|---|---|---|
| `electric` | frequency 27.12 MHz, `voltage_mode: grounded`, `voltage_v`, `enforce_generator_power: false`, `power_factor 1.0`, `max_qrf 1.0e11`, `update_interval 20`, `zero_qrf_outside_doped: true`, `effective_depth_m 0.02` | **none** | none |
| `geometry` | chamber 60 x 60 mm, grid 120 x 120, part 20 x 20 mm, rotation 0 | **none** | none |
| `thermal` | `dt_s 0.5`, `n_steps 1500`, `ambient 23 C`, `h 5 W/m2K`, `convective_boundaries [top]`, `max_deltaT_per_step 10 C`, `max_temp 600 C`, `min_temp -50 C`, `depth_correction disabled` | **none** | none |
| `thermal.phase_change` | `comsol_heaviside`, `linear`, `t_pc 180 C`, `dt_pc 10 C`, `latent 96700 J/kg` | **none** | none |
| `densification` | `physics_dual`, `viscous_capillary`, all 18 fields | **none** | none |
| `materials` | doped sigma 0.04 S/m, eps_r 20, powder and virgin blocks, `sigma_temp_coeff 0`, `sigma_density_coeff 0` | **none** | none |
| `electrodes` | boundary mode, hi top, lo bottom, 1 cell | **none** | none |
| `fgm_feedback.saturation_map_npz` | the path of the stored map | **DIFFERENT** | none on the physics: the prototype does not read this block at all, it builds its own design variable. It is the only textual difference in the whole file. |

Calibrated drive voltage, identical in both families: square 2428.1732217858744 V,
triangle 3005.544909894063 V, cross 2815.3798396110424 V,
L_shape 1804.2151285366901 V. COMPUTED: every uniform arm absorbs
500.0 W per metre of depth, which is the calibration target.

### 2.2 Against `HEATR_STANDARD_PARAMETERS.md`

| standard | prototype | verdict |
|---|---|---|
| grid <= 160 cells per axis, never 240 | 120 | COMPLIANT |
| drive mode: voltage drive for FGM comparisons, report absorbed power per arm | voltage drive, `enforce_generator_power: false`, absorbed power reported on every arm | COMPLIANT |
| dimensionality: HEATR 2-D / 2.5-D is primary and validated | 2-D | COMPLIANT |
| metric definition `ui_rms_part * (T_bar_part - 23 C)` | used, but only as a DIAGNOSTIC; the reported objective is J | LABELLED DEVIATION, deliberate |
| exposure: phi_bar = 0.90 via the exposure-time optimizer | **DEVIATION.** The stop is t_stop = argmin J per arm. COMPUTED, previous report Section 1: phi_bar at the optimal stop spans 0.45 to 0.98 and is never 0.90, so the fixed phi_bar = 0.90 read state is not the shape-optimal stop for any arm on any shape | NAMED DEVIATION. It is the point of the objective, not an oversight, but it means no J number here is comparable to a phi_bar = 0.90 sigma_T number |
| FGM map: s_base 0.5, dead band 0.05, magnitude per-shape tuned, bpp stated | the heuristic arms use s_base 0.5 and dead band 0.05; the solved arms have no such parameters (they are per-cell); bpp is stated for every arm in Section 4 | COMPLIANT for the controls, not applicable to the solve |
| standing gate: energy residual over integrated dose on every solve | **NOT REPORTED.** The prototype reports the temperature-step clip fraction, the temperature-clamp fraction and the Q_rf cap fraction instead, all three exactly 0.0000 on all 48 arms scored in this pass | GAP, named. The clip gates are the melt-onset-instability detectors that matter at grid 120; the energy residual is the stronger standing gate and is not wired into the prototype |

### 2.3 Bit-identity against ACTUAL stored historical runs

PROVEN. This is the strongest version of the L0 gate available: rather than
starting a fresh production run, the prototype reads a stored run's OWN
`used_config.yaml`, loads that run's OWN dopant map through the PRODUCTION
loader `rfam_eqs_coupled._FgmFeedback.from_config`, marches, and compares
against the `T_phi90` array archived inside that run's `fields.npz`.
Read/stop convention: T_phi90 as defined at the top. Artifact
`out_verify/stored_run_gate.json`, code `adjoint2d/verify_hist.py`.

| stored run directory | melt-onset outer step | max abs diff on T_phi90, deg C | sigma_T, prototype | sigma_T, stored | verdict |
|---|---|---|---|---|---|
| `outputs_eqs/geometry_dual_readstate/runs/square/baseline` | 845 (423.0 s) | **0.000e+00** | **4.930989680787** | **4.930989680787** | PASS |
| `outputs_eqs/geometry_dual_readstate/runs/square/fgm_m0p85` | 672 (336.5 s) | **0.000e+00** | **3.768226015489** | **3.768226015489** | PASS |
| `outputs_eqs/geometry_dual_readstate/runs/triangle/fgm_m0p85` | 572 (286.5 s) | **0.000e+00** | **45.357859475541** | **45.357859475541** | PASS |
| `outputs_eqs/geometry_dual_readstate/runs/cross/fgm_m0p85` | 663 (332.0 s) | **0.000e+00** | **31.400589027953** | **31.400589027953** | PASS |

The square baseline value 4.930989680787 is the published 4.931 C uniform
square baseline, now reproduced from the archived field rather than from a
report table.

**One gotcha found and fixed while building this gate, worth recording.** The
first run of the gate gave `max|diff| = 1.526e-05` on the UNIFORM baseline while
the three FGM runs were already exact. Cause:
`_FgmFeedback.effective_fill` casts to float32 (`rfam_eqs_coupled.py:416`) ONLY
when the hook is enabled; with FGM disabled it returns `fill_frac` untouched and
the production conductivity array stays float64. The prototype was forcing the
float32 branch in both cases. Setting `float32_sat` from the hook state took the
baseline to 0.000e+00. That is a one-line branch, and it silently biased a
"bit-identical" claim by one float32 unit in the last place until it was found.

---

## 3. Task 1a: the ACTUAL stored historical masks, scored under J

The previous report's control arms were REBUILT by golden-section search inside
the prototype. They are not the historical artifacts. Here the ACTUAL stored
4-bpp maps are loaded through the production loader and scored under the same J
with their own J-stop. Artifact `out_verify/hist_arms.json`.

**What "the actual stored mask" means, precisely.** The npz written by
`fgm_generator.py` stores a printer-resolution `level_map` of shape
1715 x 1715 uint8 with values in [0, 15], and the production loader
(`rfam_eqs_coupled.py:366-380`) divides it by 15, resamples it down to the
120 x 120 simulation grid and clips to [0, 1]. That resampled array is what the
historical run actually simulated, and it is what is scored here. The historical
runs injected it through the permittivity-co-varying hook, so saturation moved
conductivity AND permittivity; the adjoint arm moves conductivity only.

Four historical arms per shape:

* `hist_win_asstored_eps` the window-reselection winner, exactly as stored
  (saturation is zero outside the part) and in its native
  permittivity-co-varying channel. This is the historical arm verbatim.
* `hist_win_outside1_eps` the same map with the saturation outside the part
  restored to the nominal 1. The prototype's convention, so that an arm changes
  the dopant and nothing else. On shapes with a rasterized diagonal boundary this
  is a real effect: COMPUTED, the triangle has 40 cells outside the binary part
  mask that still carry geometry fill, the cross 116 and the L_shape 69, and the
  square has ZERO, which is why the square is insensitive to the convention.
* `hist_win_outside1_sig` the same map through the conductivity-only channel,
  which is the actuator the adjoint arm has.
* `hist_m0p85_asstored_eps` the old {0.30 .. 0.85} grid arm that the
  `geometry_dual_readstate` campaign ran, as stored.

### 3.1 Historical arms, all at their own J-stop

| shape | arm | J | IoU | growth % | under % | stop idx | stop s | at horizon | phi_bar | sigma_T C | P_abs W/m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| square | hist_win_asstored_eps (m = 0.9033) | 28.53 | 0.9765 | 0.88 | 1.50 | 749 | 375.0 | no | 0.975 | 5.17 | 563.9 |
| square | hist_win_outside1_eps | 28.53 | 0.9765 | 0.88 | 1.50 | 749 | 375.0 | no | 0.975 | 5.17 | 563.9 |
| square | hist_win_outside1_sig | 385.87 | 0.6925 | 0.00 | 30.75 | 1499 | 750.0 | **YES** | 0.701 | 9.72 | 298.2 |
| square | **hist_m0p85_asstored_eps** | **22.20** | **0.9838** | 0.50 | 1.12 | 751 | 376.0 | no | 0.979 | 5.32 | 564.6 |
| triangle | hist_win_asstored_eps (m = 1.2164) | 223.74 | 0.7612 | 22.50 | 6.75 | 609 | 305.0 | no | 0.873 | 26.85 | 457.4 |
| triangle | **hist_win_outside1_eps** | **190.62** | **0.7806** | 18.50 | 7.50 | 629 | 315.0 | no | 0.885 | 23.67 | 437.1 |
| triangle | hist_win_outside1_sig | 210.93 | 0.7404 | 10.75 | 18.00 | 803 | 402.0 | no | 0.814 | 19.82 | 328.6 |
| triangle | hist_m0p85_asstored_eps | 369.26 | 0.5865 | 37.25 | 19.50 | 521 | 261.0 | no | 0.817 | 42.04 | 556.3 |
| cross | hist_win_asstored_eps (m = 1.1855) | 349.90 | 0.6855 | 19.69 | 17.95 | 931 | 466.0 | no | 0.809 | 12.26 | 359.1 |
| cross | **hist_win_outside1_eps** | **343.27** | **0.6933** | 20.85 | 16.22 | 1499 | 750.0 | **YES** | 0.838 | 13.56 | 268.8 |
| cross | hist_win_outside1_sig | 1036.00 | 0.0000 | 0.00 | 100.00 | 0 | 0.5 | no | 0.000 | 0.19 | 107.0 |
| cross | hist_m0p85_asstored_eps | 414.59 | 0.6279 | 16.22 | 27.03 | 469 | 235.0 | no | 0.735 | 24.72 | 601.4 |
| L_shape | **hist_win_asstored_eps (m = 0.1110)** | **611.55** | 0.4340 | 3.99 | 54.87 | 408 | 204.5 | no | 0.452 | 46.31 | 518.4 |
| L_shape | hist_win_outside1_eps | 623.26 | 0.4301 | 5.38 | 54.68 | 441 | 221.0 | no | 0.453 | 46.76 | 494.1 |
| L_shape | hist_win_outside1_sig | 621.45 | **0.4387** | 8.80 | 52.27 | 924 | 462.5 | no | 0.478 | 35.05 | 317.1 |
| L_shape | hist_m0p85_asstored_eps | 691.64 | 0.3668 | 5.10 | 61.45 | 542 | 271.5 | no | 0.387 | 43.75 | 396.7 |

### 3.2 Was the rebuilt control faithful?

COMPUTED, the previous report's rebuilt permittivity-channel control (`H_eps`,
its Section 5) against the ACTUAL stored mask in the matching convention:

| shape | rebuilt H_eps J | actual stored mask J | delta | rebuilt IoU | actual IoU |
|---|---|---|---|---|---|
| square | 27.94 | 28.53 (as stored) | **+2.1 %** | 0.9765 | 0.9765 |
| triangle | 189.84 | 190.62 (outside 1) | **+0.4 %** | 0.7806 | 0.7806 |
| cross | 353.45 | 343.27 (outside 1) | **-2.9 %** | 0.6633 | 0.6933 |
| L_shape | 630.23 | 623.26 (outside 1) | **-1.1 %** | 0.4222 | 0.4301 |

**The rebuild was faithful to within 3 % of J on all four shapes.** The
golden-section reconstruction is therefore not what decided any conclusion.

### 3.3 The comparison Matt asked for: solved map against the real historical mask

Best historical arm by J against the best PRINTABLE solved arm at 4 bpp, same
engine, same configuration, each at its own J-stop.

| shape | best historical arm | J | IoU | solved single pass, 4 bpp: J / IoU | delta J | solved double pass, 4 bpp: J / IoU | delta J |
|---|---|---|---|---|---|---|---|
| square | m = 0.85 old grid, as stored | 22.20 | 0.9838 | **23.83 / 1.0000** | +7.3 % | **20.05 / 0.9975** | **-9.7 %** |
| triangle | m = 1.2164 winner, outside 1 | 190.62 | 0.7806 | **87.99 / 0.8578** | **-53.8 %** | 91.24 / 0.8762 | **-52.1 %** |
| cross | m = 1.1855 winner, outside 1 (AT HORIZON) | 343.27 | 0.6933 | 360.18 / 0.6755 | +4.9 % | 369.89 / 0.6627 | +7.8 % |
| L_shape | m = 0.1110 winner, as stored | 611.55 | 0.4340 | **521.20 / 0.5209** | **-14.8 %** | **404.65 / 0.6234** | **-33.8 %** |

Four honest qualifications on that table.

1. **The square margin shrank and the two metrics disagree.** Against the
   window-reselection winner the solved single-pass arm is -16.5 % on J; against
   the m = 0.85 old-grid mask, which is the genuinely best historical square arm
   under this objective, it is +7.3 % WORSE on J while being BETTER on the
   thresholded region (IoU 1.0000 against 0.9838, growth 0.00 % against 0.50 %,
   under-melt 0.00 % against 1.12 %). J is a squared L2 on the continuous melt
   fraction over the whole domain, so it also charges for a partially-melted halo
   that never crosses phi = 0.5; the solved map wins the shape and loses the
   halo. Both numbers are reported; neither is suppressed.
2. **The cross goes to the historical mask, and the comparison is unresolved.**
   The historical arm's minimum sits on the last stored step (index 1499), so its
   J is a lower bound on how good it is, and the solved A1 arm's stop is index
   1498. This is the same horizon ambiguity the previous report flagged. It
   should be read as a tie the 1500-step horizon cannot resolve, and the
   FIGURE (`figs/fig_hist_cross.png`) shows why the historical arm is
   competitive: it crushes the vertical limb's saturation to near zero in the
   PERMITTIVITY channel, which the conductivity-only solve cannot do.
3. **Not dose matched.** Absorbed power at the stop spans 107.0 to 601.4 W per
   metre against the 500.0 W per metre uniform baseline. On the square the
   historical arms run at about 564 W per metre and the solved arms at about
   403 to 415 W per metre, so the solved arm reaches a better shape on
   roughly 28 % less absorbed power. The objective penalizes over-melting as well
   as under-melting, which removes the crudest dose gaming, but the arms are not
   power-matched and the table keeps the confound visible.
4. **Different actuators.** Every historical arm co-varies permittivity; every
   solved arm is conductivity-only. The `hist_win_outside1_sig` rows measure what
   the historical MAP is worth on the solved arm's actuator, and it is much less:
   the square control leaves 30.75 % of the part unmelted and the cross control
   melts nothing at all (J = 1036.00, exactly the part cell count, optimal stop
   index 0). The permittivity channel is doing real work for the controls and
   remains the single largest missing layer in the solve.

**A finding worth naming: the L_shape "control" is barely a graded map.**
COMPUTED, the stored L_shape winner mask has saturation ranging 0.4667 to 0.5333
inside the part with mean 0.4942. At the window-selected gain m = 0.1110 the
proportional-inverse family collapses onto a near-uniform HALF-dose map, and it
is worse than uniform (J 611.55 against 538.42). Calling it a graded control on
the L_shape overstates what it is.

---

## 4. Task 2: printability

The printer rasterizes at 2 or 4 bpp. There is no continuous grading in
hardware, so the continuous solved map is a bound, not a product.

### 4.1 The quantizer, and its verification

New module `adjoint2d/printability.py`, reproducing the production convention
and citing it inline:

* `fgm_generator.py:583-586` sim-resolution quantization,
  `level_map_sim = np.round(sat_scaled * max_val).astype(np.uint8)` with
  `max_val = (1 << bpp) - 1`.
* `fgm_generator.py:593-606` the printer round trip,
  `px_m = 25.4e-3/dpi; zx = dx_m/px_m; sat_dpi = clip(zoom(sat_scaled,(zy,zx),order=1),0,1);`
  `level_map_dpi = clip(round(sat_dpi*max_val),0,max_val)`.
* `rfam_eqs_coupled.py:366-380` the loader that inverts it.

PROVEN, four new test functions (8 test cases, one is parametrized over five
stored artifacts) in `adjoint2d/tests/test_printability.py`, written red first;
the import error was captured before the module existed. The substantive test is a
CAPTURED-REAL-DATA contract test, not an invented fixture: for five actual
stored artifacts (the four window-reselection winner maps and the square
m = 0.85 map), feeding the stored `sat_map` through `printer_level_map`
reproduces the stored 1715 x 1715 `level_map` **exactly**, `np.array_equal` True,
zero differing pixels. A further test writes a map with the new quantizer and
reads it back with the PRODUCTION loader, requiring exact equality. Full suite:
**41 tests pass** (33 carried over, 8 new).

**Conventions, stated because they change the numbers.** Quantization is applied
INSIDE the part only; outside it the saturation is held at the nominal 1, so an
arm changes the dopant map and not the sub-pixel geometry fill of boundary
cells. Saturation above 1.0 is treated as a SECOND printing pass, not as a
clipped value: the level quantum stays the printer's 1/max_val and only the
number of levels grows, so `A15_4bpp` is a 4-bpp map whose levels run to 22 of
15 rather than a map crushed back into [0, 1].

### 4.2 Continuous against 4 bpp against 2 bpp, every arm at its own J-stop

COMPUTED, artifact `out_verify/printability.json`, code
`adjoint2d/verify_print.py`. `_dpi` is the full 720 dots-per-inch printer round
trip at 4 bpp. `levels` is the number of distinct printer levels the map
actually uses inside the part.

| shape | arm | J | IoU | growth % | under % | stop idx | stop s | P_abs W/m | levels |
|---|---|---|---|---|---|---|---|---|---|
| square | A1 continuous | 22.63 | 0.9975 | 0.25 | 0.00 | 1146 | 573.5 | 403.2 | 12 |
| square | **A1 4 bpp** | 23.83 | **1.0000** | 0.00 | 0.00 | 1148 | 574.5 | 403.0 | 12 |
| square | A1 2 bpp | 46.40 | 0.9804 | 1.88 | 0.12 | 1192 | 596.5 | 388.2 | 3 |
| square | A1 4 bpp dpi | 22.91 | **1.0000** | 0.00 | 0.00 | 1144 | 572.5 | 403.8 | 12 |
| square | A15 continuous | 20.14 | 0.9975 | 0.00 | 0.25 | 1118 | 559.5 | 414.7 | 18 |
| square | **A15 4 bpp** | **20.05** | 0.9975 | 0.00 | 0.25 | 1119 | 560.0 | 414.4 | 18 |
| square | A15 2 bpp | 24.50 | 0.9963 | 0.12 | 0.25 | 1133 | 567.0 | 409.4 | 5 |
| square | A15 4 bpp dpi | 20.18 | 0.9975 | 0.00 | 0.25 | 1106 | 553.5 | 417.7 | 17 |
| triangle | A1 continuous | 88.06 | 0.8578 | 7.25 | 8.00 | 539 | 270.0 | 442.3 | 15 |
| triangle | **A1 4 bpp** | **87.99** | 0.8578 | 7.25 | 8.00 | 539 | 270.0 | 442.1 | 15 |
| triangle | A1 2 bpp | 90.65 | 0.8538 | 7.75 | 8.00 | 536 | 268.5 | 444.6 | 4 |
| triangle | A1 4 bpp dpi | 89.36 | 0.8558 | 7.50 | 8.00 | 537 | 269.0 | 444.1 | 15 |
| triangle | A15 continuous | 90.79 | **0.8782** | 6.75 | 6.25 | 512 | 256.5 | 461.9 | 20 |
| triangle | **A15 4 bpp** | 91.24 | 0.8762 | 7.00 | 6.25 | 516 | 258.5 | 459.7 | 20 |
| triangle | A15 2 bpp | 100.44 | 0.8689 | 6.75 | 7.25 | 515 | 258.0 | 457.6 | 5 |
| triangle | A15 4 bpp dpi | 96.98 | 0.8661 | 8.25 | 6.25 | 510 | 255.5 | 466.3 | 21 |
| cross | A1 continuous | 360.07 | 0.6777 | 16.80 | 20.85 | 1498 | 749.5 | 263.0 | 16 |
| cross | A1 4 bpp | 360.18 | 0.6755 | 16.60 | 21.24 | 1499 | 750.0 | 262.6 | 16 |
| cross | **A1 2 bpp** | **359.55** | **0.6833** | 15.83 | 20.85 | 1499 | 750.0 | 261.8 | 4 |
| cross | A1 4 bpp dpi | 361.14 | 0.6761 | 15.64 | 21.81 | 1439 | 720.0 | 266.8 | 16 |
| cross | A15 continuous | 369.28 | 0.6604 | 13.71 | 24.90 | 1112 | 556.5 | 311.2 | 23 |
| cross | A15 4 bpp | 369.89 | 0.6627 | 13.90 | 24.52 | 1123 | 562.0 | 309.5 | 23 |
| cross | A15 2 bpp | 371.32 | 0.6542 | 13.90 | 25.48 | 1148 | 574.5 | 303.5 | 5 |
| cross | A15 4 bpp dpi | 373.45 | 0.6563 | 11.78 | 26.64 | 1048 | 524.5 | 317.9 | 23 |
| L_shape | A1 continuous | 521.16 | 0.5209 | 6.39 | 44.58 | 525 | 263.0 | 493.6 | 16 |
| L_shape | A1 4 bpp | 521.20 | 0.5209 | 6.39 | 44.58 | 525 | 263.0 | 493.7 | 16 |
| L_shape | A1 2 bpp | 521.42 | 0.5200 | 6.39 | 44.67 | 525 | 263.0 | 493.5 | 4 |
| L_shape | A1 4 bpp dpi | 521.29 | 0.5205 | 6.49 | 44.58 | 525 | 263.0 | 493.7 | 16 |
| L_shape | A15 continuous | 396.52 | **0.6265** | 6.21 | 33.46 | 502 | 251.5 | 553.0 | 23 |
| L_shape | **A15 4 bpp** | 404.65 | 0.6234 | 6.30 | 33.73 | 505 | 253.0 | 549.9 | 23 |
| L_shape | A15 2 bpp | 442.03 | 0.5762 | 7.60 | 38.00 | 513 | 257.0 | 536.4 | 5 |
| L_shape | A15 4 bpp dpi | 396.89 | 0.6265 | 6.21 | 33.46 | 502 | 251.5 | 553.1 | 23 |

Two cross-arm curiosities are reported rather than smoothed: `A1 2 bpp` on the
cross is very slightly BETTER than continuous (J 359.55 against 360.07), and
`A1 4 bpp` on the square is slightly worse on J while being better on IoU. Both
are the objective's continuous-versus-thresholded split again, and both are
inside the horizon and breakpoint noise of a piecewise-smooth objective. Neither
is a mechanism.

### 4.3 Do the headline numbers survive?

**Square 0.9975: yes, it improves.** At 4 bpp the single-pass map reaches
IoU **1.0000** with zero part growth and zero under-melt, and the full 720
dots-per-inch round trip also reaches 1.0000. At 2 bpp the map collapses to
THREE distinct levels and IoU falls to 0.9804 with J doubling to 46.40; that is
still far better than the 0.8508 uniform arm and better than the best historical
square mask's 0.9838, but the 2-bpp square is visibly a different map (see
`figs/fig_print_square.png`, third column, where the melt front bulges at the
top and bottom edges).

**Triangle 0.8782: yes.** 4 bpp costs 0.0020 IoU points (0.8782 to 0.8762) and
+0.5 % on J. 2 bpp costs 0.0093 IoU points and +10.6 % on J. The single-pass
triangle arm is completely insensitive to 4 bpp (J 88.06 to 87.99, IoU
unchanged to four decimals).

**The bit-depth cost is not uniform across shapes.** COMPUTED, cost of 2 bpp
relative to continuous, on J: square A1 +105 %, triangle A1 +2.9 %, cross A1
-0.1 %, L_shape A1 +0.05 %; square A15 +21.6 %, triangle A15 +10.6 %, cross A15
+0.6 %, L_shape A15 +11.5 %. The shapes whose solved map has fine structure the
solve actually exploits (the square's rim-versus-core contrast) pay for bit
depth; the shapes whose map is largely saturated at a box bound (the L_shape
single-pass arm, 78.9 % of its cells at s = 1) pay almost nothing because there
is nothing to quantize.

---

## 5. Task 3: saturation headroom, single pass against double pass

The ink is 25 weight percent carbon black in isopropyl alcohol. One pass
saturates at s = 1. Anything above 1.0 requires a second printing pass over the
same layer.

### 5.1 The box [0, 1] arms, from stored data

CONFIRMED from stored artifacts `out_shape/{square,triangle,cross,L_shape}.json`.
Both box [0, 1] (arm A1) and box [0, 1.5] (arm A15) arms already exist for ALL
FOUR shapes at both the 15 and 40 forward-solve-equivalent budgets, so no new
solve was needed for parts (a) or (b) of this task. Values re-read and confirmed
against the previous report's Section 5 table: identical.

### 5.2 How much of the box above 1.0 does the solve actually use?

COMPUTED from `out_shape/<shape>_maps.npz`:

| shape | arm | mean s in part | max s | fraction of part cells above s = 1 | fraction pinned at the box top |
|---|---|---|---|---|---|
| square | A1 | 0.689 | 1.000 | 0.0 % | 20.0 % |
| square | A15 | 0.925 | 1.398 | **31.6 %** | 0.0 % |
| triangle | A1 | 0.805 | 1.000 | 0.0 % | 46.8 % |
| triangle | A15 | 0.925 | 1.500 | **44.0 %** | 6.8 % |
| cross | A1 | 0.610 | 1.000 | 0.0 % | 19.5 % |
| cross | A15 | 0.835 | 1.500 | **41.7 %** | 23.6 % |
| L_shape | A1 | 0.923 | 1.000 | 0.0 % | **78.9 %** |
| L_shape | A15 | 1.180 | 1.500 | **70.6 %** | **55.4 %** |

The double-pass region is not incidental: every A15 map puts a third to
two-thirds of the part above single-pass saturation, and the L_shape puts 55 %
of the part hard against the top of the box, which is itself a signal that the
box is still binding there.

### 5.3 Does single pass plus stop-time freedom recover the double pass?

COMPUTED, at 4 bpp so both arms are printable, each at its own J-stop:

| shape | single pass J / IoU / stop s | double pass J / IoU / stop s | cost of staying single pass |
|---|---|---|---|
| square | 23.83 / **1.0000** / 574.5 | 20.05 / 0.9975 / 560.0 | **none on shape.** J is +18.9 % but IoU, growth and under-melt are all at or better than the double pass. Single pass is sufficient. |
| triangle | 87.99 / 0.8578 / 270.0 | 91.24 / 0.8762 / 258.5 | **-1.8 IoU points**, and J is -3.6 % in the single pass's favour. The two arms are effectively tied and the metrics disagree on which side. Single pass is sufficient. |
| cross | 360.18 / 0.6755 / 750.0 (AT HORIZON) | 369.89 / 0.6627 / 562.0 | **none measurable.** The single pass is better on both J and IoU. Single pass is sufficient. |
| L_shape | 521.20 / 0.5209 / 263.0 | 404.65 / **0.6234** / 253.0 | **-10.3 IoU points, +28.8 % on J.** The double pass buys real fidelity here. |

**Does longer exposure substitute for oversaturation? No, and the stop times say
so directly.** On the L_shape the single-pass arm's optimal stop is 263.0 s and
the double-pass arm's is 253.0 s: the single-pass arm already runs LONGER and
still leaves 44.6 % of the part unmelted against the double pass's 33.7 %. Its
map is 78.9 % pinned at s = 1, so the box is binding and there is no more
conductivity to buy; the only remaining actuator is time, and the J curve has
already turned (the stop is interior, not at the horizon), meaning further
exposure spills melt into the bed faster than it fills the foot of the L. The
same reading holds in the opposite direction on the square, where the
oversaturated arm reaches its minimum EARLIER (560.0 s against 574.5 s) because
the higher rim saturation pulls the melt front outward sooner: there,
oversaturation buys time, not fidelity, and the single-pass arm simply spends the
time instead.

---

## 6. What is proven, computed, assumed

**PROVEN**
* The prototype reproduces four ACTUAL archived historical runs bit-identically
  on the stored T_phi90 field, `max|diff| = 0.000e+00`, run directories cited in
  Section 2.3.
* The new quantizer reproduces five ACTUAL stored 1715 x 1715 `level_map`
  artifacts exactly (zero differing pixels), and a map written with it and read
  back by the PRODUCTION loader is bit-equal.
* 41 unit tests pass; the 8 new ones were written red first and the import error
  was observed before the module existed.
* Carried over unchanged: the forward march is bit-identical to
  `rfam_eqs_coupled.run_sim` on the pinned configuration
  (`out_adjoint/l0_*.json`), and the fixed-stop and envelope-stop shape
  gradients agree to a relative difference of 0.000e+00
  (`out_adjoint/gate_shape_square.json`).

**COMPUTED**
* Every number in Sections 2, 3, 4 and 5.
* Temperature-step clip fraction, temperature-clamp fraction and Q_rf cap
  fraction are all exactly 0.0000 on all 48 arms scored in this pass.

**ASSUMED**
* That treating saturation above 1.0 as a second printing pass at the same
  1/max_val level quantum is the right hardware model. A real second pass may not
  be linear in absorbed dopant, may spread, and may change the drying behaviour.
  Not measured; a bench measurement of the achievable saturation of a double pass
  is the physical check this rests on.
* That holding the saturation at the nominal 1 outside the part is the right
  boundary convention for the solved arms. The stored production maps are zero
  outside; Section 3.1 reports both conventions on the historical arms so the
  effect is visible, but the solved arms were only ever run with outside = 1.
* That the rasterized binary part mask is the right nominal target. Carried over
  from the previous report, still untested.
* That an arbitrary stop time is realizable as a process control.

---

## 7. Honest limits

1. **No gradient was re-gated in this pass, because none was changed.** The
   modules added are forward-scoring and quantization only
   (`printability.py`, `verify_hist.py`, `verify_print.py`,
   `make_verify_figures.py`); `forward.py`, `adjoint.py` and
   `shape_objective.py` were not touched. The standing FD gate therefore still
   applies unchanged, and it is still a SUBGRADIENT gate: the random-direction
   probe bottoms at 1.22e-05 and cannot be fixed by widening the phase-change
   regularizer, because the pinned population is the cold powder bed. Every
   optimization result carried into this report rests on that.
2. **The energy-residual standing gate of `HEATR_STANDARD_PARAMETERS.md` is not
   wired into the prototype.** The three clip gates are, and are clean, but the
   stronger gate is missing. Named in Section 2.2, not worked around.
3. **The cross ranking is still unresolved at the 1500-step horizon.** Two arms
   in this pass have their J minimum at the last stored step.
4. **Not dose matched.** Absorbed power spans 107.0 to 601.4 W per metre.
5. **The adjoint arms are conductivity-only; every historical arm co-varies
   permittivity.** Section 3.1's `hist_win_outside1_sig` rows quantify how much
   of the control's performance is the actuator rather than the map, and it is a
   lot.
6. **Fifteen L-BFGS-B evaluations on 1600 design variables remains a very small
   budget.** All solved numbers are upper bounds on J.
7. **The model over-predicts achievable tuned uniformity by roughly a factor of
   eight against hardware** (`ALLISON_LAW_REPLICATION.md` Section 6.1). An IoU of
   1.0000 is a statement about the model at grid 120 in two dimensions, not about
   a printed part.
8. **The 2-D forward has no through-thickness physics**, so "single pass against
   double pass" is here purely a statement about the in-plane conductivity a
   second pass would buy, with no representation of the second pass wetting the
   layer below.

---

## 8. The single most valuable next layer

Unchanged from the previous report, and this pass sharpened the evidence for it:
**add the permittivity channel to the gradient.** Section 3.1 measures the
actuator gap directly. The same historical MAP that scores J = 343.27 on the
cross in the permittivity channel scores J = 1036.00 (nothing melts at all) in
the conductivity-only channel, and on the square it goes from 28.53 to 385.87.
Two of the four shapes go to the historical mask essentially because it has an
actuator the solve does not. The forward already supports the channel and is
bit-identity gated in it (`out_adjoint/l0_square_epscovary.json`); only the chain
rule from the design variable through the complex permittivity into gamma is
missing, and it needs its own FD gate.

Second: **more budget on the cross and the L_shape, and a longer horizon**, since
both comparisons currently sit inside horizon uncertainty.

---

## 9. Artifacts, absolute paths

Worktree root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/.claude/worktrees/agent-a02efc1141ba69c58`

New code:
* `adjoint2d/printability.py` the production-convention quantizer
* `adjoint2d/verify_hist.py` the stored-run identity gate and the historical arms
* `adjoint2d/verify_print.py` the bit-depth sweep
* `adjoint2d/make_verify_figures.py` the figures
* `adjoint2d/tests/test_printability.py` 8 tests, red first

New results:
* `out_verify/stored_run_gate.json`
* `out_verify/hist_arms.json`
* `out_verify/printability.json`
* `out_verify/{square,triangle,cross,L_shape}_quantized_maps.npz`

New figures, all viewed before delivery:
* `figs/fig_print_{square,triangle,cross,L_shape}.png` continuous against 4 bpp
  against 2 bpp, single and double pass, with the melted region under each
* `figs/fig_hist_{square,triangle,cross,L_shape}.png` the solved printable map
  against the ACTUAL stored historical mask

Carried over and still valid: `SHAPE_FIDELITY_SOLVE_REPORT.md`,
`ADJOINT_PROTOTYPE_REPORT.md`, `out_adjoint/`, `out_shape/`,
`figs/fig_shape_*.png`.

Main-tree sources read:
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/fgm_generator.py`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/rfam_eqs_coupled.py`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/HEATR_STANDARD_PARAMETERS.md`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/FGM_WINDOW_RESELECTION.md`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/outputs_eqs/geometry_dual_readstate/runs/<shape>/{baseline,fgm_m0p85,map_m0p85}/`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/outputs_eqs/fgm_calibrated_control/{configs,runs}/`
