# Geometry generalization: the solve stack on an arbitrary imported outline

**Date:** 2026-08-02. **Scope:** the layer that removes "the eighteen-shape library" from every
result the campaign has produced. An imported outline (a binary mask array, a polygon vertex
list, or a portable network graphics mask file) now becomes a fully solvable HEATR 2-D case:
grid-independent target indicator, production part mask, automatically calibrated drive,
symmetry analysis, heating-kernel anisotropy spectrum, actuator recommendation, filtered
gradient solve, mode co-solve, 4 bits per pixel deliverable, and a turntable program. Grid
120 x 120 throughout. **Nothing was committed. No dissertation file was touched.
`.claude/worktrees/`, `deck_gifs/`, `rfam_gui_server.py` and `webui/` were not read or
written. `scripts/solve_fgm.py` was READ ONLY and not modified. No line of
`rfam_eqs_coupled.py` was edited.**

**Acronyms, expanded on first use.** RFAM radio-frequency additive manufacturing. FGM
functionally graded material (a spatially varying dopant saturation map). EQS
electro-quasi-static (the low-frequency Maxwell approximation the two-dimensional solver
uses). IoU intersection over union. bpp bits per pixel. L-BFGS-B limited-memory
Broyden-Fletcher-Goldfarb-Shanno with box constraints. FD finite difference. PNG portable
network graphics. STL stereolithography. phi melt fraction. W/m watts per metre of depth.

**Evidence tags.** PROVEN = unit tested red first, FD gated, or reproduced against a stored
number. COMPUTED = measured from a run in this pass. ASSUMED = a modelling choice or an
inference not measured here.

**Conventions, stated once and carried on every number.**

    J(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi(x))^2

with `chi` the grid-independent sub-cell AREA FILL indicator (`adjoint2d.chi_area`), so **J is
NOT numerically comparable to any J quoted against the binary raster** in `out_lib`,
`out_rot`, `out_dwell` or `out_seq`; `J_raster_chi` is recorded next to every arm for that
comparison. `t_stop` = argmin of J over that arm's OWN stored trajectory on a 1500-step
horizon (dt 0.5 s, 750 s), with `HORIZON` flagged whenever the minimum sits on the last
stored step, which makes that arm's J a bound. The melted region for IoU, growth and
under-melt is phi >= 0.5. **Grid qualifier, mandatory:** `SOLVE_ROBUSTNESS_VALIDATION.md`
established that grid-120 fidelity does not transfer to grid 160, so every fidelity number
here is a property of the method AT GRID 120. Conductivity channel only (the deployable one).
**Absorbed power is calibrated to 500 W/m on the UNIFORM arm only; solved arms are NOT dose
matched** and the measured spread on the deliverable arms of this pass is 245.7 to 500.0 W/m.
Standing energy-residual gate 5 percent of integrated dose at each arm's own stop: **zero
violations in this pass, worst residual 2.27 percent.**

---

## 1. Verdict, first

| question | verdict |
|---|---|
| **Does the pipeline handle an arbitrary imported geometry end to end?** | **YES, PROVEN on two geometries that are not in the library, with no hand-coded knowledge of either shape anywhere in the code path.** The keyhole ran intake, calibration, symmetry, anisotropy spectrum, recommendation, static solve, rotation co-solve, 4 bpp deliverable and a 448-move turntable program in 891 s and reached **J 8.08, IoU 0.9753 at grid 120**, which is inside the campaign's SOLVED band (IoU >= 0.95). The gear ran the same path in 612 s. |
| **Is the intake faithful?** | **PROVEN, 18 of 18.** Every library shape pushed back through the intake as a bare vertex list reproduces its stored configuration's part mask with **0 cells differing** and its stored calibrated drive voltage to **0.000000 percent**. |
| **Do the three intake routes agree?** | **COMPUTED.** The PNG route reproduces the mask route **exactly** (maximum cell difference 0.0). A 400 x 400 mask import reproduces the polygon route's target area to **-0.094 percent**, with a maximum single-cell difference of 0.120 that sits entirely on the boundary staircase. |
| **Does the symmetry detector recover the known orders?** | **COMPUTED, 15 of 16** shapes whose order the library defines. Cross 4, star 5, star6 6, hexagon 6, equilateral triangle 3, L none and no mirror, T none but exactly one mirror, square 4 with 4 mirrors. **One miss: the octagon is reported as 10-fold** (it is nearly circular and its 36-degree mismatch falls under the measured threshold). |
| **Does the actuator classifier reproduce the campaign's outcomes?** | **5 of 5 on the five shapes the rotation campaign measured, but those five are the points the bands were CALIBRATED on, so this is threshold reproduction and not out-of-sample validation.** The out-of-sample evidence is a Spearman rank correlation of **-0.79 (p = 9.1e-5, n = 18)** between the best mode's residual anisotropy and the library's own static-solve 4 bpp IoU, and **1 of 2** on the novel end-to-end tests. |
| **Was the gradient re-gated on the new geometry?** | **PROVEN.** The FD gate was re-run on the novel gear: the filtered layer the co-solve actually descends passes **5 of 5 probes at 1e-6**; the two unfiltered layers pass 2 of 4 at 1e-6 and **4 of 4 at the campaign's 1e-5 subgradient standard**. The read index does not move under the probes. |

**The one honest negative, and it is the most useful result in the pass.** On the gear the
classifier said MODE_SUFFICES via continuous rotation and **the static solved map won anyway**
(J 132.36 against 138.59, IoU 0.8458 against 0.8273 at 4 bpp). Rotation did exactly what the
anisotropy predicted, it cut the UNIFORM arm's J from 207.89 to 138.59, a 33 percent
improvement at 8 percent less absorbed power. What it did not do is beat a dopant map solved
against the static field. **The classifier answers "does this actuator improve the heating",
not "does this actuator beat a solved map", and those are different questions.** Section 6
names the fix.

---

## 2. What was built

All new code. No existing solver internals were modified.

**Modules,** `fgm_solve_campaign/adjoint2d/`:

* `geometry_intake.py` the three intake routes, the area-fill delegation, the emitted
  production configuration, the self-intersection refusal, and the layer-wise AGGREGATION
  CONTRACT in the module docstring.
* `geometry_contour.py` exact mask-to-polygon tracing (the staircase boundary of the pixel
  region, built from unshared pixel edges and stitched into loops) plus a dependency-light
  even-odd inside test for lanes without matplotlib.
* `geometry_symmetry.py` rotational order, continuity flag and mirror axes by angular
  autocorrelation of chi about its own area centroid; the universal half-turn GAUGE rule; the
  symmetry-reduced candidate angle set; the divisor-compatible indexing orders.
* `geometry_actuator.py` the part-frame heating-kernel anisotropy spectrum per actuator mode,
  the classifier bands with their calibration table, and the recommendation.
* `geometry_calibrate.py` the automatic drive calibration to 500 W/m.

**Tests, red first, 70 new, whole adjoint2d suite 383 passing** (107 s,
`PYTHONPATH=$PWD/fgm_solve_campaign:$PWD ./.venv312/bin/python -m pytest
fgm_solve_campaign/adjoint2d/tests/`). Each new test file was observed failing with
`ImportError: cannot import name ...` before its module existed, then passing.
`tests/test_geometry_contour.py` (7), `test_geometry_intake.py` (21),
`test_geometry_symmetry.py` (20), `test_geometry_actuator.py` (14),
`test_geometry_calibrate.py` (8).

**`tests/fill_contract.py`, the shared fill contract, written as importable FUNCTIONS and not
as a test class**, at the three-dimensional lane's request. It states the convention (cells
centred on grid points, even-odd inside test so vertex winding is irrelevant, the production
sub-cell offsets at `CHI_N_SUB = 32`, boundary samples decided by the inside test and not
clipped analytically) and provides `assert_circle_area`, `assert_rotated_rect_area`,
`assert_winding_invariance`, `assert_grid_independence` and
**`assert_extrusion_slice_reduction`**, which is the statement a three-dimensional VOLUME
fill must satisfy on an interior extrusion slice: cell for cell, it must equal the
two-dimensional area fill. Every one takes the fill implementation as an argument, so the
three-dimensional lane runs the identical assertions against its own evaluator.

**Drivers,** `scripts/analysis/`: `run_intake_library.py`, `run_intake_novel.py`,
`run_intake_gate.py`, `novel_shapes.py`, `build_intake_tables.py`, `make_intake_figures.py`.

**Read and imported, not modified:** `adjoint2d/{chi_area, forward, adjoint, eqs, gradops,
design_filter, topopt, topopt_objective, printability, library_solve, energy_gate, pins,
robust, rot_frame, rot_kernel, gate_rot, gate_rho, gate_ms, dwell, control}.py`,
`rfam_eqs_coupled.py`, `shapes.py`, `scripts/solve_fgm.py`.

**How this reaches `scripts/solve_fgm.py` without touching it.** The intake emits a normal
production configuration whose part is `shape: polygon` with `polygon_points`, a branch
`shapes.make_shape` has supported all along (shapes.py:855-866) and which
`rfam_eqs_coupled._single_part_mask_and_fill` rotates and offsets like any other shape. So
`solve_fgm.run_solve(config_path, ...)` consumes an imported geometry today, unchanged: it
calls `chi_area.chi_from_cfg` on that configuration, which this pass verified returns the
intake's own chi to 1e-15 (`test_chi_from_the_emitted_config_equals_the_intake_chi`). The
warm-start `auto` branch finds no `out_lib` entry for an imported name and starts cold, which
is the correct behaviour and is logged loudly by that script.

---

## 3. Intake fidelity, the real-data gate

COMPUTED, full table in `fgm_solve_campaign/out_intake/_tables.md` Table A. For each of the
eighteen library shapes the polygon is taken from the production domain builder, pushed
through `geometry_intake.from_polygon` as if it were an imported outline, and the result is
compared against that shape's own stored calibrated configuration.

* **Part mask: 18 of 18 match, 0 cells differing.** The intake's configuration builds the
  same `Case` the library builds.
* **Drive voltage: 18 of 18 match to 0.000000 percent.** The automatic calibration
  rediscovers every stored per-shape voltage, from 1718.98 V (T_shape) to 4900.01 V
  (ellipse).
* The binary raster over-states or under-states the area-fill target by **-3.26 percent
  (octagon) to +3.23 percent (diamond)**, which reproduces the spread
  `FROZEN_CONVENTIONS_2D.md` Section 2 recorded and is the reason the target indicator is the
  area fill and not the raster.

**The three routes,** `fig_intake_routes.png` and `out_intake/route_equivalence.json`.
COMPUTED on the gear: the PNG route is bit-equal to the mask route (maximum cell difference
**0.0**, which it should be, since the PNG carries the same pixels); the 400 x 400 mask import
differs from the polygon route by **-0.094 percent of area** with a maximum single-cell
difference of **0.120**, concentrated on the boundary, which is the staircase of the import
resolution and not an error in the fill.

**The analytic anchors,** PROVEN through `fill_contract.py`: the circle's area fill matches
the closed-form area of the inscribed 720-gon the sampler actually sees to better than 0.1
percent; the rotated rectangle at 31 degrees matches w times h; reversing the vertex ring
changes not one cell; and the filled area moves less than 0.2 percent between grid 120 and
grid 160.

---

## 4. Symmetry, the gauge, and the candidate set

COMPUTED, Table B. `rotation_mismatch(chi, alpha)` is the normalized angular autocorrelation
defect about the area centroid. The acceptance threshold was MEASURED rather than assumed:
across the library, exact symmetries at non-multiples of 90 degrees score 0.0082 (circle) to
0.0241 (five-pointed star) because of the bilinear resampling floor, and the smallest
non-symmetry scores 0.0351 (octagon at 120 degrees). `SYMMETRY_TOL = 0.030` sits in that gap.

| target from the task | detected |
|---|---|
| cross 4 | **4** (C4v, 4 mirrors) |
| star 5 | **5** (C5v, 5 mirrors) |
| hexagon 6 | **6** (C6v, 6 mirrors) |
| L none | **1, C1, zero mirrors** |
| T mirror only | **1, C1v, exactly one mirror** |

Also recovered: square 4, diamond 4, star6 6, pentagon 5, equilateral triangle 3, H_shape 2,
rectangle 2, ellipse 2, rounded rectangle 2 (its configuration is 22 mm by 18 mm, so 2 is
correct), triangle 1, trapezoid 1, circle flagged continuously symmetric. **The one failure is
the octagon, reported as 10-fold against its defined 8.** It is a near-circular shape and its
36-degree mismatch falls below the measured threshold; the consequence for the recommendation
is nil (continuous is chosen either way) but the detector over-reports on near-circular parts
and that is a named limit.

**The gauge rule is applied universally.** `gauge_reduce` folds every orientation modulo 180
degrees, because `DWELL_SCHEDULE_REPORT.md` Section 2 measured that the part-frame heating at
theta and theta + 180 is the same field to 4e-13 relative. Its effect is visible in the emitted
programs: the keyhole's 24-angle continuous schedule becomes **12 distinct positions**, and
the gear's 8-fold indexing becomes **4 distinct positions** at 0, 45, 90 and 135 degrees. The
candidate span is 360 / n for even n and 180 / n for odd n, because an odd rotation group does
not contain the half turn.

---

## 5. The anisotropy spectrum and the actuator classifier

### 5.1 The metric, and why it is a re-derivation

**This must be said before any number.** `CONTINUOUS_ROTATION_REPORT.md` Section 4 quotes five
anisotropy values (star 0.0205, square 0.0216, cross 0.0389, T_shape 0.2193, L_shape 0.2840)
and `out_rot/kernel_anisotropy.json` still holds them, but **the script that wrote that file
does not exist anywhere in the repository** (a grep over the whole tree finds no code that
writes it). Four reconstructions of the definition its prose gives were tried in this pass and
none reproduced the stored values; the same definition also returns an UNDEFINED value on the
L_shape, whose part does not cover the rotation centre, so it cannot be the definition that
produced the stored L_shape number. COMPUTED.

The metric here is therefore explicit and new. With `R_k` the part-frame rotations of the
24-angle reference group, part mask `P` and part-frame kernel `K`:

    w    = mean_k R_k P            the annular OCCUPANCY of the part
    Kbar = mean_k R_k K            the annular part of the kernel
    A(K) = sum(w |K - Kbar|) / sum(w Kbar)

It is zero exactly when the kernel is invariant under the sampled rotation group, it is
invariant to the drive scale, and the occupancy weight is what makes it defined for every
geometry instead of requiring an inscribed disc. **The bands are calibrated against the
campaign's OUTCOMES, which are ground truth, not against its numbers, which cannot be
re-derived.** Both are reported side by side.

### 5.2 The bands

COMPUTED on the five calibration shapes, continuous mode, uniform dopant map, electrical
state B, grid 120:

| shape | A (this metric) | A (stored, not reproducible) | campaign outcome |
|---|---|---|---|
| square | 0.277 | 0.0216 | rotation wins (J 25.56 to 13.93, IoU 0.9804 to 1.0000) |
| cross | 0.419 | 0.0389 | rotation wins (J 147.03 to 34.82, IoU 0.8514 to 0.9866) |
| star | 0.462 | 0.0205 | rotation wins (J 106.28 to 24.46, IoU 0.7870 to 0.9527) |
| T_shape | 0.837 | 0.2193 | rotation fails (-10.5 percent J, 42.9 percent still unmelted) |
| L_shape | 1.098 | 0.2840 | rotation fails (effectively a tie) |

`A_MODE_SUFFICES = 0.50`, `A_PHYSICAL_LIMIT = 0.80`, and a mode must reduce the anisotropy by
at least `MIN_REDUCTION = 1.15` before rotation is recommended at all. The gap between the
worst winner (star, 0.462) and the best failure (T_shape, 0.837) is a factor of 1.81 and both
thresholds sit inside it. **The star's margin to the first threshold is 8 percent. That
thinness is the honest limit of this classifier.**

### 5.3 The eighteen-shape confusion, reported honestly

`fig_intake_classifier.png`, Tables C and D.

**Against the five MEASURED rotation outcomes: 5 of 5.** Square, cross and star are called
MODE_SUFFICES and rotation won on all three; T_shape and L_shape are called PHYSICAL_LIMIT and
rotation failed on both. **This is threshold reproduction, not validation:** those five points
are exactly the points the bands were fitted to. A confusion table on the calibration set can
only be diagonal or broken, and it is diagonal.

**The out-of-sample evidence, which is the number worth quoting.** COMPUTED across all
eighteen shapes: the Spearman rank correlation between the best mode's residual anisotropy and
the library campaign's own static-solve 4 bpp IoU is **-0.79 (p = 9.1e-5)**. The two shapes the
classifier calls PHYSICAL_LIMIT (T_shape, L_shape) are the two lowest-fidelity shapes in the
whole library (IoU 0.4444 and 0.5209). Two shapes with poor static fidelity are called
MODE_SUFFICES (cross 0.6755, star 0.7032) and **those are precisely the two the rotation
campaign rescued to 0.9866 and 0.9527**, so the classifier separates "hard statically but
fixable by an actuator" from "hard, full stop", which is the distinction the Studio
suggest-feature needs to make.

Thirteen shapes carry no measured rotation outcome at all, so **thirteen of the eighteen rows
in Table C are PREDICTIONS with no ground truth behind them.** Naming that is more useful than
a confusion matrix built by substituting the static-solve class for a rotation outcome.

---

## 6. The novel geometries, end to end

Two outlines that are not in the library, generated parametrically in
`scripts/analysis/novel_shapes.py` and handed to the pipeline as bare vertex lists. Full arm
tables in `out_intake/_tables.md` Table F; figures `fig_intake_novel_gear8.png` and
`fig_intake_novel_keyhole.png`, both viewed before delivery.

### 6.1 The keyhole: the classifier was right, and the arm reaches the SOLVED band

COMPUTED. Intake 98 vertices, 1030 part cells, 261.4 mm^2, symmetry **C1v** (no rotational
symmetry, one mirror), calibrated drive **2630.33 V** for 500.0000 W/m in two EQS solves.
Spectrum: static 0.6110, continuous 0.5156. Recommendation **MAP_PLUS_MODE via continuous,
reduction 1.18, rotation recommended**.

| arm | J | IoU | growth pct | under-melt pct | stop s | P_abs W/m | max T C | energy residual |
|---|---|---|---|---|---|---|---|---|
| uniform, static | 509.96 | 0.5750 | 30.68 | 24.85 | 336.0 | 500.0 | **290.5** | 2.27 pct |
| solved static, 4 bpp | 168.35 | 0.8163 | 14.17 | 6.80 | 501.5 | 367.2 | 253.1 | 0.98 pct |
| uniform, continuous | 176.35 | 0.7973 | 13.01 | 9.90 | 573.0 | 301.3 | 223.9 | 0.89 pct |
| **solved continuous, 4 bpp** | **8.08** | **0.9753** | **2.14** | **0.39** | 747.5 | 245.7 | 212.4 | 0.20 pct |

**Prediction HELD.** Against the solved static arm the recommended mode buys **-95.2 percent
of J and +0.159 IoU**, and the melted region is the nominal part to within 2.1 percent growth
and 0.4 percent under-melt at grid 120. Quantization to 4 bpp costs nothing (8.08 against the
continuous map's 8.15). Three cautions, all COMPUTED: the winning arm stops at 747.5 s of a
750 s horizon, so it is 2.5 s from being a HORIZON bound; it absorbs 245.7 W/m against the
uniform arm's calibrated 500, so **the comparison is not at equal delivered energy and the
rotating arm wins with roughly half the dose spread over twice the time**; and the uniform
static arm reaches **290.5 C, above the 250 C operating ceiling**, while every solved arm sits
at 212 to 253 C.

**This arm is NOT labelled SOLVED.** The campaign's SOLVED label requires Gate A (the grid-160
hold-out) and Gate B (the sub-filter-radius blur), and neither was run here. IoU 0.9753 is a
grid-120 in-grid number.

### 6.2 The gear: the classifier was wrong, and the failure is informative

COMPUTED. Intake 72 vertices, 1144 part cells, 295.9 mm^2, symmetry **C8v**, calibrated drive
**3098.69 V** for 500.0000 W/m. Spectrum: static 0.4440, continuous 0.2280, index2 0.4440,
index4 0.2888, index8 0.2411. Recommendation **MODE_SUFFICES via continuous, reduction 1.95,
rotation recommended**; index8 is within 6 percent of continuous, so it was co-solved as a
near tie.

| arm | J | IoU | growth pct | under-melt pct | stop s | P_abs W/m | max T C | energy residual |
|---|---|---|---|---|---|---|---|---|
| uniform, static | 207.89 | 0.8042 | 8.92 | 12.41 | 320.0 | 500.0 | 214.0 | 0.81 pct |
| **solved static, 4 bpp** | **132.36** | **0.8458** | 8.30 | 8.39 | 412.5 | 409.0 | 208.4 | 0.69 pct |
| solved continuous, 4 bpp | 138.59 | 0.8273 | 12.85 | 6.64 | 332.5 | 460.6 | 218.6 | 1.00 pct |
| solved index8, 4 bpp | 148.63 | 0.8215 | 13.64 | 6.64 | 353.0 | 465.2 | 223.5 | 1.02 pct |

**Prediction DID NOT HOLD.** The static solved map wins on both J and IoU, by 4.5 percent and
0.019. Two mechanisms, both COMPUTED and both worth carrying forward.

1. **Rotation did what the anisotropy said it would.** It cut the uniform arm's J from 207.89
   to 138.59, a 33 percent improvement, at 8 percent less absorbed power. What the classifier
   compares is actuator against no actuator; the pipeline's real competitor is a SOLVED map,
   and on a nearly radially symmetric part a solved static map is already very good.
2. **The dopant map is INERT under rotation on this part.** The co-solved continuous arm
   returns J 138.59, identical to the rotating UNIFORM arm to six figures. This is the same
   phenomenon `CONTINUOUS_ROTATION_REPORT.md` Section 1 reported on the star ("the star's win
   belongs to the rotation, not to the grading") and the mechanism was localized here: at the
   uniform design v = 1 the entire filtered gradient under the rotating kernel is NEGATIVE
   (measured range -0.645 to -0.0087), meaning more dopant everywhere would help, while every
   component is already pinned at the upper rail of the box. The cold start is therefore a
   CONSTRAINED stationary point and L-BFGS-B correctly stops after one evaluation. The
   pipeline now runs a second start warm from the static solved design, which is what
   `run_rot_avg_solve.py` did with a stored map; on the gear that start converges back to the
   uniform arm, and on the keyhole it drives J from 271.26 to 8.15.

**The named fix.** The classifier should predict against the SOLVED static arm rather than
against the uniform arm. The measurable version is one extra number per mode, the residual
anisotropy of the kernel **after the static map is applied**, which is one forward per mode
and requires the static solve to have run first. That is the single most valuable next layer.

### 6.3 Arrow, spectrum only

COMPUTED, no solve run, and labelled as such: 7 vertices, 890 part cells, C1v, calibrated
3830.08 V, static 1.1266, continuous 0.5312, recommendation MAP_PLUS_MODE via continuous with
a reduction of 2.12.

---

## 7. The finite-difference gate on the novel geometry

PROVEN. `out_intake/gate_gear8.json`, driver `scripts/analysis/run_intake_gate.py`, which
reuses `gate_rot.gate_layer` unchanged. **The generalization layer introduces no new
gradient**, so this is not a gate on new mathematics; it is the existing gradients re-run on
geometry they have never seen, which is exactly the assumption
`CONTINUOUS_ROTATION_REPORT.md` limit 1 flagged as unverified. Central differences, epsilon
swept over the campaign's eight values, four to five probes per layer, 400-step gate runs,
fixed read index.

| layer | max-sensitivity cell | random cell | random direction | smooth direction | gradient direction | verdict |
|---|---|---|---|---|---|---|
| R0 single angle at zero degrees, unfiltered (this IS the static solve's gradient, proven bit-identical to `forward.forward` by `test_rot_kernel.py`) | 4.38e-07 | 1.03e-06 | 1.63e-06 | n/a | **1.37e-07** | 2/4 at 1e-6, **4/4 at 1e-5** |
| R1 full 24-angle set, unfiltered | 1.61e-08 | 2.06e-06 | 5.80e-07 | n/a | 1.36e-06 | 2/4 at 1e-6, **4/4 at 1e-5** |
| **R2 full 24-angle set WITH the 1.0 mm filter, the gradient the co-solve descends** | 4.62e-09 | 6.43e-08 | 8.50e-08 | 1.91e-07 | **9.42e-08** | **5/5 at 1e-6** |

COMPUTED. Every probe shows the expected V shape. **The filtered layer is the cleanest by one
to two orders of magnitude**, which is the same signature the rotation campaign's own gate
showed (its R2 was cleaner than its inherited R0) and is what the filter is for: it removes
the cell-scale content that the clipped melt fraction's kinks corrupt. The two unfiltered
layers miss the 1e-6 bar on the random-cell and random-direction probes and clear the
campaign's documented 1e-5 subgradient standard, which is the standard that applies because
the objective reads a CLIPPED melt fraction. **The read index does not move under any probe**
(`stop_index_stability.moved = false`), so the fixed-index gate is the correct layer and no
envelope-theorem argument is being leaned on. Wall time 695 s.

---

## 8. Two robustness findings from using the tool

1. **A self-intersecting outline was silently reinterpreted, and is now refused.** COMPUTED.
   The first keyhole generator walked the full head circle and then appended the stem corners;
   the two crossed at the join, and the even-odd fill rule turned the overlap into a
   V-shaped notch cut out of the head. The intake accepted it, calibrated it, solved it and
   produced a perfectly plausible figure of the wrong part. The even-odd rule is TOTAL: it
   never errors, it just answers a different question. `geometry_intake.find_self_intersection`
   now refuses a crossed outline with the offending edge pair in the message, consecutive
   duplicate vertices are dropped first (a repeated point is a zero-length edge and made the
   crossing test report a false positive), and both behaviours are covered red first. **The
   run on the crossed outline was discarded and is not quoted anywhere in this report.**
2. **Holes are refused, loudly, rather than filled.** `chi_area.area_fill_union` combines parts
   with `np.maximum`, reproducing `rfam_eqs_coupled.py:1595`, and a maximum cannot express a
   hole: an annulus would come back a filled disc. A clockwise interior loop from a mask import
   therefore raises `UnsupportedGeometry` with that reasoning in the message.

---

## 9. The layer-wise aggregation contract, documented not implemented

`geometry_intake.LAYER_AGGREGATION_CONTRACT`, quoted in the module docstring so a consumer
cannot miss it. In one line: a build is one exposure on one turntable, so the deliverable is a
**PER-BUILD** turntable program and a per-build actuator recommendation; per-layer results are
combined **DOSE-WEIGHTED by default** (layer part area times exposure time); the **WORST layer
is always computed and carried as a disagreement flag**, and a worst layer whose recommended
actuator class differs from the dose-weighted one is a STRONG DISAGREEMENT that must be
surfaced to the user rather than averaged away; and the whole per-layer aggregation is a
**HEURISTIC pending a full three-dimensional verification run**, because z-coupling is real
(the `stl_compensation_tool` analysis measured a sub-linear z-gain, roughly g to the power
-0.7, which is direct evidence that layers are not independent). Stored FGM maps are
printer-resolution rasters and must be resampled by the production convention before reuse.
**ASSUMED and named: none of this aggregation is implemented or measured in this pass.** It is
an API-shape and documentation commitment only.

---

## 10. Honest limits

1. **The anisotropy metric is a re-derivation.** The campaign's stored numbers cannot be
   reproduced because the producing script is gone, so the thresholds are calibrated on five
   OUTCOMES and the classifier's 5 of 5 on those five is in-sample.
2. **Two novel end-to-end tests, and one of them missed.** One hit (keyhole), one miss (gear).
   That is the out-of-sample accuracy of the classifier as an end-to-end predictor: **1 of 2**.
3. **The classifier answers the wrong comparison** (actuator against uniform, not actuator
   against a solved map). Section 6.2 measures the consequence and names the fix.
4. **Grid 120 only.** No Gate A grid hold-out, no Gate B sub-filter blur, so the keyhole's
   0.9753 is an in-grid number and carries no SOLVED label.
5. **Not dose matched.** Deliverable arms span 245.7 to 500.0 W/m. The keyhole's winning arm
   uses about half the calibrated dose and twice the time.
6. **The rotation arms are QUASI-STATIC.** They use `rot_kernel.AveragedKernel`, not the
   production engine's turntable block. `CONTINUOUS_ROTATION_REPORT.md` Section 7 measured
   that approximation at about 1 percent when the actuator and the averaging set match and up
   to 27 percent when they do not, and the emitted programs here were NOT executed on the real
   engine.
7. **The emitted turntable programs are untested as programs.** They are produced by
   `dwell.cycle_program` on equal dwell over the gauge-distinct positions and are not
   optimized; no dwell schedule was solved for either novel shape.
8. **The symmetry detector over-reports on near-circular parts** (octagon 10 against 8) and
   cannot see a symmetry broken by less than roughly 3 percent of the part area.
9. **Single part only, no holes.** Multi-part imports are refused deliberately, because every
   rotation and dwell convention in the campaign was established on a single part.
10. **The self-intersection check is skipped above 3000 vertices** for cost, and that is
    recorded in the provenance rather than assumed away. Mask-derived staircases cannot
    self-intersect by construction.
11. **Model, not hardware.** `ALLISON_LAW_REPLICATION.md` Section 6.1 records that the
    two-dimensional model over-predicts achievable tuned uniformity against hardware by
    roughly a factor of eight.

---

## 11. Proven, computed, assumed

**PROVEN**
* The intake round trip on all eighteen library shapes: part mask 0 cells differing, drive
  voltage 0.000000 percent apart, 18 of 18.
* The fill contract: circle area against the closed form, rotated-rectangle area, winding
  invariance to the last cell, grid-120 to grid-160 area stability, and bit-identity of the
  intake's `area_fill` with `chi_area.area_fill_poly`.
* The dependency-light even-odd inside test against `matplotlib.path.Path`, 4000 random points
  on a concave star, zero disagreements.
* The mask tracer's exactness: the staircase polygon's shoelace area equals the pixel count
  times the cell area to 1e-12, corners preserved, holes refused.
* The FD gate on the novel gear: R2 5 of 5 at 1e-6, R0 and R1 4 of 4 at the 1e-5 subgradient
  standard, read index stable.
* 383 tests passing across `adjoint2d/tests/`, 70 of them new and each observed red first.

**COMPUTED**
* Every number in Sections 3 through 8.

**ASSUMED**
* That the campaign's five measured rotation outcomes are the right ground truth for
  calibrating the actuator bands.
* That the quasi-static averaged kernel is the right design model for the rotating arms, at
  the accuracy the rotation campaign measured.
* That the area-fill target indicator is the right target for an imported geometry, carried
  over from `FROZEN_CONVENTIONS_2D.md` Section 2.
* That an arbitrary stop time is realizable as a process control.
* The entire layer-wise aggregation contract of Section 9.

---

## 12. The single most valuable next layer

**Make the classifier predict against the solved static arm, not against the uniform arm.**
Section 6.2 measured the gap: on the gear, rotation improves the heating exactly as the
anisotropy says and still loses to a solved static map, because the classifier has no model of
what a map can buy. The concrete change is one extra number per mode, the residual anisotropy
of that mode's kernel evaluated with the STATIC SOLVED map injected rather than a uniform map,
which costs one forward per mode and is available as soon as the static solve has run. The
recommendation then becomes "spend on the actuator" only when the actuator still helps after
the map has done its work, which is the decision a user actually faces.

Second: **run Gate A and Gate B on the keyhole's winning arm.** It is at IoU 0.9753 in grid,
the first arm this pipeline has produced inside the SOLVED band on a geometry the campaign has
never seen, and it cannot be called SOLVED until it survives the grid-160 hold-out and the
sub-filter-radius blur.

Third: **execute one emitted turntable program on the production engine.** The programs are
machine readable and gauge reduced but nothing in this pass ran one, so the quasi-static
numbers have no Level-2 companion on a novel geometry.

---

## 13. Artifacts, absolute paths, and wall time

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New modules:
* `fgm_solve_campaign/adjoint2d/geometry_intake.py`, `geometry_contour.py`,
  `geometry_symmetry.py`, `geometry_actuator.py`, `geometry_calibrate.py`
* `fgm_solve_campaign/adjoint2d/tests/fill_contract.py` (importable contract, not a test class)
* `fgm_solve_campaign/adjoint2d/tests/test_geometry_{contour,intake,symmetry,actuator,calibrate}.py`

New drivers:
* `scripts/analysis/run_intake_library.py`, `run_intake_novel.py`, `run_intake_gate.py`,
  `novel_shapes.py`, `build_intake_tables.py`, `make_intake_figures.py`

Results, all under `fgm_solve_campaign/out_intake/`:
* `<shape>_intake.json` and `_intake.npz` the eighteen library shapes: intake fidelity,
  symmetry, spectrum, recommendation, and every mode's kernel
* `gear8_novel.json`, `keyhole_novel.json`, `arrow_novel.json` the novel geometries
* `gear8_maps.npz`, `keyhole_maps.npz` the solved maps and the melt fields at each stop
* `gate_gear8.json` the finite-difference gate, full epsilon sweeps
* `route_equivalence.json`, `route_input_gear8_mask.png` the three-route comparison
* `_tables.md` every table
* `fgm_solve_campaign/logs_intake/*.log` per-run console logs

Figures, **all four viewed before delivery**, in `fgm_solve_campaign/figs_intake/`:
* `fig_intake_routes.png` the three intake routes and their measured disagreement
* `fig_intake_classifier.png` the spectrum and recommendation on all eighteen library shapes,
  with the five measured outcomes marked
* `fig_intake_novel_gear8.png`, `fig_intake_novel_keyhole.png` each novel geometry end to end:
  chi, the anisotropy spectrum, the static and averaged kernels, the 4 bpp deliverable map,
  and the melt region at the deliverable's own stop against the nominal outline

**Wall time, COMPUTED and logged, nothing cut.** Eighteen-shape intake: 222 s of process time
in four pinned single-thread streams, 113 s of wall clock for the longest stream. Novel
end-to-end: gear 612 s, keyhole 891 s. Finite-difference gate on the gear: 695 s. Spectrum-only
passes: 3 to 10 s each. Test suite: 107 s. Figures and tables: under 60 s. **Total about 45
minutes of process time.** One run was discarded and is named: the first keyhole end-to-end run
(1037 s) used a self-intersecting outline, which is the finding of Section 8.1, and none of its
numbers appear in this report. Every stream ran with `fgm_solve_campaign/env1.sh` thread
pinning; without it the 24-angle kernel evaluation was measured at more than 110 s against 6.8 s
pinned, which is the same thread-contention effect `env1.sh` documents.
