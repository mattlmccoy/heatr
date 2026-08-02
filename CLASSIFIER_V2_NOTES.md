# Actuator classifier version 2, and the symmetry-detector fix

**Date:** 2026-08-02. **Scope:** the two named next layers of
`GEOMETRY_GENERALIZATION_REPORT.md`. (1) Make the actuator classifier predict
against the SOLVED static arm rather than against the uniform arm, which is the
fix that report's Section 12 called "the single most valuable next layer".
(2) Root-cause and fix the symmetry detector's one miss, the octagon reported as
ten-fold. A third, cheap robustness item is folded in: polygon intake in either
winding order.

**Nothing was committed. No dissertation file was touched.** `.claude/worktrees/`,
`deck_gifs/`, `studio_handoff/`, `webui/`, `rfam_gui_server.py` and every
objective or solve driver were not read or written. The only modules changed are
`fgm_solve_campaign/adjoint2d/{geometry_actuator,geometry_symmetry,geometry_intake}.py`
and their tests, plus three new drivers under `scripts/analysis/`. No version-1
artifact in `out_intake/` was overwritten; every new result carries a `_v2`
suffix.

**Acronyms, expanded on first use.** RFAM radio-frequency additive
manufacturing. FGM functionally graded material, a spatially varying dopant
saturation map. EQS electro-quasi-static, the low-frequency Maxwell
approximation the two-dimensional solver uses. IoU intersection over union. bpp
bits per pixel. L-BFGS-B limited-memory Broyden-Fletcher-Goldfarb-Shanno with
box constraints. FD finite difference. W/m watts per metre of depth. phi melt
fraction. chi the target indicator.

**Evidence tags.** PROVEN = unit tested red first, or reproduced against a
stored number. COMPUTED = measured from a run in this pass. ASSUMED = a
modelling choice or an inference not measured here.

**Conventions, carried on every number.** Grid 120 throughout;
`SOLVE_ROBUSTNESS_VALIDATION.md` established that grid-120 fidelity does not
transfer to grid 160, so every number here is a property of the method AT GRID
120. Electrical state B, the state the march runs in from the first
`update_interval` tick onwards. Conductivity channel only, the deployable one.
The uniform arm is calibrated to 500 W/m; the anisotropy metric is invariant to
the drive scale, so no calibration can move a recommendation. The anisotropy
metric itself is unchanged from version 1 and is the occupancy-weighted
azimuthal defect defined in `geometry_actuator.azimuthal_anisotropy`, which is a
re-derivation and not the campaign's lost original (that limit stands, see
version 1's Section 5.1).

---

## 1. Verdict, first

| question | verdict |
|---|---|
| **Does version 2 fix the gear8 miss?** | **YES, COMPUTED. gear8 is now MAP_SUFFICES with rotation NOT recommended** (reduction 1.115 against the 1.125 threshold), which is the correct call: its solved static map beat every rotating arm (J 132.36 / IoU 0.8458 against 138.59 / 0.8273). |
| **Does it keep the keyhole and the five measured calls right?** | **YES. 7 of 7 on the whole measured evidence set.** keyhole MODE_SUFFICES via continuous (rotation recommended, correct); square, cross, star MODE_SUFFICES (correct); T_shape and L_shape PHYSICAL_LIMIT with rotation not recommended (correct). |
| **Does the FREE variant suffice, so Studio does not have to wait for a solve?** | **YES, and it is strictly BETTER than the expensive one. COMPUTED: free stand-in 7 of 7, one-forward solved-map variant 4 of 7 and it cannot reach 7 of 7 at ANY threshold.** The free variant is the recommended default. |
| **Symmetry, 16 of 16?** | **YES, PROVEN by test. All sixteen library shapes whose order `shapes.py` defines are recovered, octagon included.** The root cause was the acceptance predicate, not the sampling. |
| **Winding order?** | **PROVEN. Either winding is accepted and the stored ring is normalized to counter-clockwise.** Measured side finding: one library polygon, the cross, already arrives CLOCKWISE from the production domain builder. |

**The honest limit, stated before the tables.** The free variant's separation
margin is **2.0 percent** (keyhole 1.137 against gear8 and L_shape 1.115). That
is thinner than version 1's 8 percent margin on the star and it is the weakest
part of this layer. It widens to 5.0 percent at stand-in magnitude 1.5 and it
DISAPPEARS at magnitude 0.5. Section 4.4 gives the evidence for fixing the
magnitude at 1.0 and why that is not a fit to the outcome.

---

## 2. What changed, file by file

**`fgm_solve_campaign/adjoint2d/geometry_actuator.py`** (added, nothing removed;
every version-1 function and constant is untouched and still exported)

* `prop_inverse_stand_in(Q_static, part_mask, magnitude, baseline)` the
  zero-gradient-solve stand-in for a solved static map. It DELEGATES to the
  production `control.proportional_inverse_map` rather than re-deriving the
  control law.
* `spectrum_v2_from_cfg(cfg, order, solved_static_map=None, prop_magnitude=1.0)`
  measures every mode's residual anisotropy under each available injected map
  and returns them keyed `uniform` / `prop_inverse` / `solved`. The static
  kernel is evaluated once up front to build the stand-in, then each mode's
  kernel is ASSEMBLED once and evaluated against every map, so the kernel
  assembly count is unchanged from version 1 and only the EQS solve count scales
  with the number of maps.
* `classify_v2(a, rotation_recommended)` the four version-2 classes, adding
  **MAP_SUFFICES**, the class version 1 could not express.
* `recommend_v2(spectrum, order, basis="prop_inverse")` the version-2
  recommendation, carrying `basis` and `classifier_version` into the result so
  two bases can never be silently mixed.
* `A2_MODE_SUFFICES`, `A2_PHYSICAL_LIMIT`, `MIN_REDUCTION_V2`,
  `DEFAULT_V2_BASIS`, `PROP_INVERSE_MAGNITUDE`, `PROP_INVERSE_BASELINE`,
  `CALIBRATION_V2` (the seven measured points with their sources).
* `Recommendation` gained two defaulted fields, `basis` and `version`, so
  version-1 call sites are unaffected.

**`fgm_solve_campaign/adjoint2d/geometry_symmetry.py`**

* `group_mismatch(chi, order, ...)` the worst defect over the WHOLE cyclic group
  C_order, `max over k = 1 .. N-1 of mismatch(k * 360 / N)`.
* `analyze` now accepts an order only when the whole group clears the threshold,
  and reports both curves in its diagnostics
  (`tested_orders` = whole group, `tested_orders_generator_only` = the old
  single-angle curve) plus an explicit `acceptance_rule` string.

**`fgm_solve_campaign/adjoint2d/geometry_intake.py`**

* `from_polygon` normalizes a clockwise ring to counter-clockwise and records
  `winding_normalized` in the provenance. Mask and PNG routes set the flag False
  because `geometry_contour` emits counter-clockwise by construction.

**Tests, red first, 20 new** (`test_geometry_symmetry.py` +9,
`test_geometry_actuator.py` +10, `test_geometry_intake.py` +1). Each was
observed failing for the intended reason before the implementation existed:
the octagon test failed with `expected order 8, got 10`; the actuator tests
failed with `AttributeError: module 'adjoint2d.geometry_actuator' has no
attribute 'recommend_v2'` and the others; the winding test failed with
`assert -0.00016 > 0.0` on the stored ring, with the chi equality assertions
already passing (the fill was never winding dependent, only the stored ring
was). **Whole `adjoint2d/tests/` suite: 429 passing, 192 s.** PROVEN.

**New drivers,** `scripts/analysis/`:
`run_intake_classifier_v2.py` (the measurement pass),
`build_classifier_v2_tables.py` (tables and confusion),
`make_classifier_v2_figure.py` (the updated figure).

**Note on one figure path.** `make_intake_figures.py classifier` still writes
the version-1 figure to `figs_intake/fig_intake_classifier.png`, the same path
the new driver writes. Run the version-2 driver last, or alone. The version-1
figure is reproducible from the old driver at any time.

---

## 3. The version-2 measurement

COMPUTED. `run_intake_classifier_v2.py` on all eighteen library shapes plus the
two novel geometries (gear8, keyhole), each pushed through the intake as a bare
vertex list, calibrated, symmetry analysed, then measured. About 14 minutes of
process time across five thread-pinned streams (`env1.sh`). Results in
`out_intake/<name>_v2.json` and `<name>_v2.npz`, tables in
`out_intake/_tables_v2.md`, everything machine readable in
`out_intake/classifier_v2_summary.json`.

**Nothing was cut, and two process facts are recorded.** (1) One parallel launch
of the measurement pass was killed part way through by an interpreter exit after
twelve of the twenty geometries had been written; the six missing geometries
were re-run to completion and no partial result is quoted anywhere. (2) The
version-1 driver `run_intake_library.py` was deliberately NOT re-run, because it
writes `<shape>_intake.json` and would have overwritten version-1 artifacts. The
intake fidelity gate was therefore re-verified separately and non-destructively
(Section 7), which is where the 0 cells differing and 6.2e-14 voltage numbers
come from.

**The two injected maps, and their provenance.**

| variant | injected map | cost before the classifier can run |
|---|---|---|
| `prop_inverse`, the FREE one | the percentile-normalized proportional inverse of THIS geometry's own static kernel, which the version-1 spectrum already computes | **zero** additional gradient solves |
| `solved`, the one-forward one | library shapes: `out_lib/<shape>_maps.npz` key `A1_cont`; novel shapes: `out_intake/<name>_maps.npz` key `static_cont` | one filtered gradient solve, about 40 forward equivalents, measured at 612 to 891 s per geometry in the version-1 pass |

Both variants still cost one extra averaged-kernel evaluation per mode, which is
the report's "one forward per mode", because the averaged kernel is an OPERATOR
in the design map and not a fixed array.

**NAMED SIMPLIFICATION.** The library shapes' solved maps come from the
shape-fidelity campaign, which solved against the BINARY RASTER target, while
the intake convention is the sub-cell area fill. They are therefore solved
static maps for a very slightly different target. They are used because they are
the real artifact the pipeline would warm start from, and re-solving eighteen
shapes was not worth three hours to move a target by the raster-versus-area
delta that `FROZEN_CONVENTIONS_2D.md` measured at -3.26 to +3.23 percent of
area. The two novel geometries carry no such caveat.

---

## 4. The bands, and how they were set

### 4.1 The decision rule

    best mode  = argmin over the non-static modes of A2[mode]
    reduction  = A2[static] / A2[best mode]
    rotate     = (best mode is not static) and (reduction >= MIN_REDUCTION_V2)

    class:  PHYSICAL_LIMIT   if the deciding residual > A2_PHYSICAL_LIMIT
            MAP_SUFFICES     else if not rotate
            MODE_SUFFICES    else if A2[best mode] <= A2_MODE_SUFFICES
            MAP_PLUS_MODE    otherwise

with the deciding residual being `A2[best mode]` when rotation is recommended
and `A2[static]`, what the map alone leaves, when it is not.

### 4.2 The constants

    A2_MODE_SUFFICES  = 0.50      unchanged from version 1
    A2_PHYSICAL_LIMIT = 0.80      unchanged from version 1
    MIN_REDUCTION_V2  = 1.125     was 1.15
    DEFAULT_V2_BASIS  = "prop_inverse"
    PROP_INVERSE_MAGNITUDE = 1.0, PROP_INVERSE_BASELINE = 0.5

The two absolute bands keep their version-1 values, and that is a result rather
than an inheritance: measured on the injected-map numbers, the four rotation
winners' best-mode residuals are 0.256 (square), 0.412 (cross), 0.438 (star) and
0.476 (keyhole), all below 0.50, while the two stranded shapes sit at 1.124
(T_shape) and 1.208 (L_shape), both above 0.80. The gap between 0.50 and 1.083
is a factor of 2.17, wider than version 1's 1.81, and no calibration point sits
inside it. COMPUTED.

`MIN_REDUCTION_V2` is the constant that does the work and it is the one that
moved.

### 4.3 The seven-point evidence set

The version-2 question is "does the actuator beat a SOLVED MAP", so every
ground-truth point must be a measured comparison of a rotating arm against a
solved static arm. All seven are. `CONTINUOUS_ROTATION_REPORT.md` Section 1's
"best static arm" column is the joint campaign's winning angle plus its winning
MAP re-measured on the production engine, not a uniform arm, so those five
already carry the right truth; the two novel geometries add the only
out-of-sample points that exist.

| geometry | measured outcome | source |
|---|---|---|
| square | rotation wins | solved static 25.56 / IoU 0.9804 to rotating 13.93 / 1.0000 |
| cross | rotation wins | 147.03 / 0.8514 to 34.82 / 0.9866 |
| star | rotation wins | 106.28 / 0.7870 to 24.46 / 0.9527 |
| keyhole | rotation wins | solved static 4 bpp 168.35 / 0.8163 to solved continuous 4 bpp 8.08 / 0.9753 |
| T_shape | rotation fails | 522.28 (HORIZON) / 0.5356 to 467.33 / 0.5516, 42.9 percent still unmelted |
| L_shape | rotation fails | 376.94 / 0.6693 to 352.97 / 0.6581, better J and worse IoU, a tie |
| gear8 | rotation fails | solved static 4 bpp 132.36 / 0.8458 BEATS solved continuous 4 bpp 138.59 / 0.8273 |

### 4.4 Why the reduction threshold is 1.125 and the stand-in magnitude is 1.0

COMPUTED. Sorted reduction factors, free stand-in at magnitude 1.0:

| rotation FAILS | T_shape 1.000 | gear8 1.115 | L_shape 1.115 |
|---|---|---|---|

| rotation WINS | keyhole 1.137 | star 1.328 | square 1.424 | cross 1.610 |
|---|---|---|---|---|

Any threshold in the open interval (1.115, 1.137) is 7 of 7. 1.125 is the round
number nearest the geometric midpoint. **Margin 2.0 percent, and that is the
honest limit of this layer.**

Magnitude sensitivity of the free stand-in, MEASURED not assumed, Table V5 of
`_tables_v2.md`:

| magnitude | worst FAILS | best WINS | separates |
|---|---|---|---|
| 0.50 | gear8 **1.416** | keyhole 1.155 | **NO**, the order inverts and gear8 is called for rotation again |
| 1.00 | gear8 / L_shape 1.115 | keyhole 1.137 | yes, margin 2.0 percent |
| 1.50 | L_shape 1.086 | keyhole 1.140 | yes, margin 5.0 percent |

Magnitude 1.5 has the wider margin, and it was NOT chosen, because choosing it
would be fitting the parameter on the outcome. Magnitude 1.0 is fixed a priori
as the full-swing member of the family, the map that uses exactly the whole
[0, 1] box with no free parameter, and the independent evidence for requiring at
least full swing is that every solved static map in the library uses the whole
box (measured in-part minima 0.000 to 0.229, maxima 1.000 on square, cross,
star, T_shape, L_shape, octagon, gear8 and keyhole), while a magnitude-0.5 map
only moves saturation over [0.25, 0.75] and so under-represents the authority a
real solved map has. ASSUMED: that this argument generalizes beyond the eight
maps inspected.

---

## 5. The confusion tables, version 1 against version 2

COMPUTED, `out_intake/classifier_v2_summary.json`, reproduced by
`build_classifier_v2_tables.py`. Both versions are RE-RUN here from the same
measured residual dictionaries, so the version-1 column is not copied forward;
it reproduces the stored version-1 numbers (gear8 static 0.4440 and continuous
0.2280, keyhole reduction 1.185, Spearman -0.794 against the report's -0.79).

**Prediction** = does the classifier recommend rotation. **Truth** = did the
best rotating arm beat the best solved static arm.

### Version 1 (uniform dopant map): 6 of 7

| | truth: rotation wins | truth: rotation fails |
|---|---|---|
| **predicted rotate** | square, cross, star, keyhole (4) | **gear8 (1, FALSE POSITIVE)** |
| **predicted do not rotate** | none (0) | T_shape, L_shape (2) |

### Version 2, FREE proportional-inverse stand-in: 7 of 7

| | truth: rotation wins | truth: rotation fails |
|---|---|---|
| **predicted rotate** | square, cross, star, keyhole (4) | none (0) |
| **predicted do not rotate** | none (0) | gear8, T_shape, L_shape (3) |

### Version 2, ONE FORWARD with the static solved map injected: 4 of 7

| | truth: rotation wins | truth: rotation fails |
|---|---|---|
| **predicted rotate** | square, cross, star (3) | **gear8, L_shape (2, FALSE POSITIVES)** |
| **predicted do not rotate** | **keyhole (1, FALSE NEGATIVE)** | T_shape (1) |

**The one-forward variant is not merely worse at this threshold, it cannot be
rescued by any threshold.** Its sorted reduction factors are

    T_shape 1.000 < keyhole 1.118 < L_shape 1.151 < gear8 1.365
                  < star 1.433 < cross 1.497 < square 1.873

and the keyhole, whose rotation win is the largest in the whole campaign
(J 168.35 to 8.08, IoU 0.8163 to 0.9753), sits BELOW two failures. The best any
single threshold achieves on this variant is 6 of 7.

**Why the expensive variant is the worse signal, and this is the mechanism worth
carrying forward.** A solved static map is optimized against the full transient
objective at its own stop time, so its structure encodes edge compensation and
dose shaping, not just azimuthal flattening. Injecting it into a rotating kernel
produces a field whose anisotropy is dominated by the map's own deliberate
structure. On gear8 the solved map RAISES the static residual from 0.4440 to
0.7059. The proportional inverse, by contrast, does exactly one thing, flatten
the static field, so what survives it is precisely the part of the anisotropy
that no simple map can remove, which is the quantity the decision needs.
COMPUTED on gear8 (0.4440 to 0.3149 under the stand-in, against 0.4440 to 0.7059
under the solved map); ASSUMED as a general mechanism.

**Recommendation for Studio: default to the free variant.** It is 7 of 7, it
costs zero gradient solves, and it removes the ordering constraint that the
static solve must finish before the user can be told whether to buy a turntable.

### Out-of-sample rank correlation, for context only

COMPUTED, eighteen library shapes, Spearman rank correlation between the best
mode's residual and the library campaign's own solved-static 4 bpp IoU:

| injected map | rho | p |
|---|---|---|
| uniform (version 1) | -0.794 | 8.5e-05 |
| prop_inverse (version 2 free) | **-0.812** | 4.2e-05 |
| solved (version 2 one forward) | -0.761 | 2.5e-04 |

This is context and not the version-2 criterion. It says the free variant's
residual ranks static-solve difficulty slightly better than version 1's.

### The thirteen unmeasured shapes

**Thirteen of the twenty rows carry NO ground truth** and are predictions.
Version 2 changes the class on five of them, all in the direction of "do not buy
a turntable": octagon and trapezoid move MODE_SUFFICES to MAP_SUFFICES; triangle
and equilateral_triangle move MAP_PLUS_MODE (with rotation already not
recommended, which was a mislabelled state in version 1) to MAP_SUFFICES;
keyhole moves MAP_PLUS_MODE to MODE_SUFFICES while still recommending rotation.
Full table: `out_intake/_tables_v2.md` Table V2.

---

## 6. The symmetry fix: root cause, not a threshold nudge

### 6.1 What was wrong

The octagon was reported as ten-fold against its defined eight. Three candidate
causes were considered and two were eliminated by the measurement.

* **Sampling density: NOT the cause.** The defect curve is correct at every
  angle; nothing is aliased.
* **Raster staircase harmonics: NOT the cause.** The same behaviour appears on
  the sub-cell area-fill indicator, which has no staircase.
* **The acceptance predicate: THIS is the cause.** `analyze` tested only the
  GENERATOR `360 / N` and then took the largest accepted N. A cyclic group is a
  symmetry of the part only when every one of its elements is, and a
  near-circular part scores a small defect at almost every trial angle, so a
  generator that happens to land near a multiple of the part's TRUE period
  passes by accident.

The octagon's true period is 45 degrees. COMPUTED, the two curves side by side
(`out_intake/octagon_v2.json`, `symmetry.diagnostics`), threshold 0.030:

| tested order N | 7 | 8 | 9 | 10 |
|---|---|---|---|---|
| generator only, `mismatch(360/N)` | 0.0201 | 0.0082 | 0.0169 | 0.0261 |
| generator's offset from a true symmetry | 6.4 deg | exact | 5.0 deg | 9.0 deg |
| **whole group**, `max over k of mismatch(k 360/N)` | **0.0390** | **0.0082** | **0.0394** | **0.0380** |

Orders 7, 9 and 10 pass on the generator and fail on the group. Order 8 passes
both. The old rule took the maximum of {2, 4, 7, 8, 9, 10} and returned 10.

### 6.2 The fix

`analyze` now accepts N only when
`max over k = 1 .. N-1 of mismatch(k * 360 / N) < tol`. The angles are cached,
so the cost is one affine resampling per distinct angle in the union of the
tested groups, tens of 120 by 120 transforms, which is not measurable next to
the mirror scan that was already there. `SYMMETRY_TOL` is UNCHANGED at 0.030;
nothing was nudged.

### 6.3 The rerun

PROVEN by test (`test_rotational_order_is_recovered_on_the_library_shapes`, now
parameterized over all sixteen shapes whose order `shapes.py` defines, and
`test_an_order_is_accepted_only_when_the_WHOLE_group_is_a_symmetry`).

**16 of 16**, was 15 of 16: square 4, hexagon 6, triangle 1,
equilateral_triangle 3, L_shape 1, H_shape 2, T_shape 1, cross 4, diamond 4,
ellipse 2, **octagon 8**, pentagon 5, rectangle 2, star 5, star6 6, trapezoid 1.
The two shapes excluded from the sixteen are the circle, which is flagged
continuously symmetric (order 12, O(2)), and the rounded rectangle, whose
configured 22 by 18 mm box makes 2 correct but whose definition does not fix an
integer. gear8 C8v and keyhole C1v are unchanged.

**Consequence beyond the label.** The octagon's actuator mode set is now
{static, continuous, index2, index4, index8} instead of
{static, continuous, index2, index5, index10}. The index5 and index10 schedules
were schedules whose averaged kernel does NOT carry the part's symmetry, so the
detector was previously offering the octagon two turntable programs that were
wrong for it. Its candidate span is unchanged at 45 degrees.

**The detector's remaining limit is still real and still named:** it cannot see
a symmetry broken by less than roughly 3 percent of the part area, and the
octagon's deciding non-symmetry (72 degrees, 0.0380) clears the 0.030 threshold
by only a factor of 1.27.

---

## 7. Winding order

PROVEN. `from_polygon` accepts either winding and normalizes the stored ring to
counter-clockwise, recording `winding_normalized` in the provenance. The area
fill was never winding dependent (the even-odd rule was already asserted by
`fill_contract.assert_winding_invariance`); what was inconsistent was the STORED
ring and the emitted `polygon_points`, so two conventions were alive downstream.

COMPUTED, all eighteen library polygons pushed through forwards and reversed:
the target indicator and the part mask are **bit identical either way, 18 of
18**, the part mask still matches each shape's stored configuration with **0
cells differing**, and the calibrated drive still reproduces each stored voltage
to **6.2e-14 relative**, so the change is inert to the physics.

**Side finding worth recording.** One library polygon already arrives
CLOCKWISE from the production domain builder: the **cross**, signed area
-2.689e-4. The counter-clockwise convention that `geometry_contour` documents
was therefore already being violated in practice by the campaign's own geometry
source, silently and harmlessly. It is now normalized rather than assumed.

Mask-derived and PNG-derived loops are unaffected: `geometry_contour` emits
counter-clockwise by construction and a clockwise loop from a mask genuinely IS
an interior hole, which is still refused loudly, because
`chi_area.area_fill_union` combines parts with a maximum and a maximum cannot
express a hole.

---

## 8. Artifacts

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

**Changed modules**
* `fgm_solve_campaign/adjoint2d/geometry_actuator.py`
* `fgm_solve_campaign/adjoint2d/geometry_symmetry.py`
* `fgm_solve_campaign/adjoint2d/geometry_intake.py`

**Changed tests** (all red first)
* `fgm_solve_campaign/adjoint2d/tests/test_geometry_actuator.py`
* `fgm_solve_campaign/adjoint2d/tests/test_geometry_symmetry.py`
* `fgm_solve_campaign/adjoint2d/tests/test_geometry_intake.py`

**New drivers**
* `scripts/analysis/run_intake_classifier_v2.py`
* `scripts/analysis/build_classifier_v2_tables.py`
* `scripts/analysis/make_classifier_v2_figure.py`

**Results,** all under `fgm_solve_campaign/out_intake/`, all `_v2` suffixed, no
version-1 file overwritten
* `<name>_v2.json` and `<name>_v2.npz` for the eighteen library shapes plus
  gear8 and keyhole: residual per mode under all three injected maps, the
  magnitude sensitivity, the symmetry report with both acceptance curves, the
  stand-in map, the solved map and every kernel
* `_tables_v2.md` Tables V1 to V6
* `classifier_v2_summary.json` every number machine readable
* `fgm_solve_campaign/logs_intake/v2_s*.log` per-stream console logs

**Figure,** viewed before delivery
* `fgm_solve_campaign/figs_intake/fig_intake_classifier.png`, rewritten. Top
  row: the reduction factor of the seven measured geometries under each of the
  three injected maps, with the threshold, showing that only the free variant
  separates. Bottom row: all twenty geometries under the free stand-in,
  coloured by the version-2 class, with the classes that changed from version 1
  marked.

---

## 9. Proven, computed, assumed

**PROVEN**
* 16 of 16 rotational orders recovered on the library shapes whose order is
  defined, octagon included, by parameterized test, red first.
* An order is accepted only when the whole cyclic group clears the threshold,
  by test on the octagon's own two curves.
* Polygon intake in either winding: identical chi and part mask, ring
  normalized, 18 of 18 library polygons, and the stored-configuration part mask
  still matches to 0 cells with the drive to 6.2e-14 relative.
* The free stand-in delegates to `control.proportional_inverse_map` bit for bit,
  stays inside the box, and holds saturation 1 outside the part.
* Every version-2 calibration point is reproduced by `recommend_v2`, and gear8
  is MAP_SUFFICES, keyhole is rotating, T_shape and L_shape are PHYSICAL_LIMIT.
* 429 tests passing across `adjoint2d/tests/`, 20 of them new and each observed
  red first for the intended reason.
* The version-1 numbers reproduce: gear8 static 0.4440 and continuous 0.2280,
  keyhole reduction 1.185, Spearman -0.794 against the stored -0.79.

**COMPUTED**
* Every anisotropy number in Sections 3 to 5, on twenty geometries, three
  injected maps, three stand-in magnitudes.
* The three confusion tables and the three sorted reduction orderings.
* The octagon's generator-only against whole-group defect curves.

**ASSUMED**
* That the seven measured rotation-versus-solved-map outcomes are the right
  ground truth for the version-2 bands. Four of them are wins and three are
  failures, which is a very small calibration set, and only two of the seven
  (gear8, keyhole) were out of sample for version 1.
* That the library campaign's raster-target solved maps are close enough to
  area-fill-target solved maps to stand in for them in the `solved` variant.
  Since that variant is the one being REJECTED, this assumption can only have
  hurt it, and it would need re-testing before the rejection is called final.
* That the full-swing magnitude argument of Section 4.4 generalizes beyond the
  eight solved maps inspected.
* That the mechanism explanation in Section 5 (the solved map's structure
  dominates the injected-kernel anisotropy) is the right reading. It is measured
  on gear8 and inferred elsewhere.
* Everything version 1 assumed and stated is unchanged, in particular the
  quasi-static averaged kernel, the area-fill target, grid 120 only, and that
  the two-dimensional model over-predicts achievable tuned uniformity against
  hardware by roughly a factor of eight (`ALLISON_LAW_REPLICATION.md` 6.1).

---

## 10. The next layer, named

1. **Re-solve the `solved` variant against the area-fill target on the five
   library calibration shapes.** The rejection of the expensive variant rests on
   maps solved for a slightly different target. It is about three hours of
   compute and it would either confirm the rejection or overturn it.
2. **Widen the seven-point evidence set.** A 2.0 percent separation margin on
   seven points is the weakest number in this layer. The cheapest new points are
   end-to-end runs on geometries whose free-variant reduction lands NEAR 1.125,
   which the twenty-geometry table now identifies by name: trapezoid 1.089,
   octagon 1.031, pentagon 1.174 and equilateral_triangle 1.000 are the
   informative ones, and each costs one novel-style end-to-end pass.
3. **Gate A and Gate B on the keyhole's winning arm**, still outstanding from
   version 1. IoU 0.9753 remains an in-grid grid-120 number and carries no
   SOLVED label.
