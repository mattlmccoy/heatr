# Seam verification pass: dissertation branch `review-edits/ch5-solve-campaign` @ 242758f

**Date:** 2026-08-10
**Method:** read-only. Every dissertation line was read via
`git show 242758f:<path>` from the private repo at
`.../research/dissertation_materials/dissertation`. Nothing in the dissertation repo was
modified, staged, or checked out. No dissertation prose is reproduced here; only labels,
numerals, and line numbers are cited.
**Code-side evidence:** `.../research/binderjet/code/geo-prewarp` (this repo), reports and
raw JSON as cited. JSON re-reads were run under `./.venv312/bin/python`.

Line numbers below are line numbers *in the file as it exists at 242758f* (the git-show
output is the whole file, so they are directly usable with
`git show 242758f:sections/fgm_solve.tex | sed -n 'NNNp'`).

---

## Verdict table

| Seam | Verdict |
|---|---|
| 1. Prismatic scope, one vocabulary, no promised 3-D figure | **CONFIRMED** (one flagged tension, not part of the checklist) |
| 2. Dense-iff negative and `eq:tc_joint` cross-reference, narrow "closes", finished-negative framing | **PARTIAL** (one false gate attribution) |
| 3. Rotating-cross gap attributed to indicator plus stop convention | **CONFIRMED** (reproduced from JSON) |
| 4. Premix appendix numbers versus the two premix reports | **CONFIRMED** |

---

## SEAM 1: prismatic scope. Verdict: CONFIRMED

### 1a. The demonstrated 3-D geometry is stated as prismatic, in both places

`sections/fgm_solve.tex`, subsection `sec:solve_3d_portback` (label at line 1298), paragraph
"What ``three-dimensional'' means here, and what it does not." at lines 1362 to 1378:

* line 1366: the cylinder is named an extruded circle, "a prismatic column whose cross-section
  does not change with height".
* lines 1368 to 1369: the ceiling-constrained case is named a full-height extruded column,
  cross-referenced to `\cref{sec:tc_family}`.
* lines 1371 to 1373: the demanding case is named as the compact z-varying part, "the cone, the
  sphere and the dumbbell of `\cref{app:fgm_3d}`", with the explicit statement that no direct
  volume solve has been run on one.
* line 1378: the blanket reading rule, that every 3-D claim in the chapter is to be read as
  prismatic unless it names one of those geometries.

`chapters/ch_compensation.tex` intro clause, lines 79 to 90, is the same statement in the same
order: solver-is-volumetric at lines 80 to 82, "The geometries demonstrated on it so far are
\emph{prismatic}, extruded columns whose cross-section does not change with height" at lines
82 to 83, the two pointers `sec:solve_3d_portback` and `sec:tc_family` at lines 84 to 85, the
cone/sphere/dumbbell exclusion at lines 85 to 88, and the identical reading rule at lines
89 to 90.

### 1b. One vocabulary shared with `sec:tc_family` and `app:fgm_3d`

* `sections/thermal_ceiling.tex:154`, inside `sec:tc_family` (label at line 146), calls the
  square case "a full-height extruded (prismatic, effectively 2.5-dimensional) column rather
  than a compact three-dimensional shape". Same two-term contrast, same word.
* `appendices/appD_fgm_depth.tex:455` carries `\label{sec:fgm_3d}\label{app:fgm_3d}`. Lines
  460 to 466 use "a prism of constant $x$--$z$ section", the infinite-extrusion limit, and
  "the three-dimensional part whose cross-section \emph{changes} along the build axis".
  Lines 486 to 489 name the four study geometries exactly as the chapter does: cone, sphere,
  dumbbell as the z-varying set, "the cylinder alone is a true extrusion with a
  build-height-independent section".

Note on wording, not a defect: `app:fgm_3d` says "prism" and "true extrusion" rather than the
adjective "prismatic". The concept, the contrast, and the four geometry names are identical
across all three locations, so the vocabulary is shared in substance.

### 1c. No sentence promises a forthcoming 3-D figure

* `sed -n '1296,1470p'` over `fgm_solve.tex` (the whole of `sec:solve_3d_portback` plus the
  port-backs) contains **zero** occurrences of `fig:`, `figure`, `Figure`, or
  `includegraphics`. There is no figure in that subsection and none is promised.
* `ch_compensation.tex` lines 60 to 105 likewise contain zero `fig:` or `figure` tokens.
* A grep for `forthcoming`, `will be shown`, `to be added`, `figure will`, `not yet shown`
  across `fgm_solve.tex`, `ch_compensation.tex`, `thermal_ceiling.tex` returns no
  figure-promise hit. The `pending` and `in-progress` hits are
  `thermal_ceiling.tex:91,104,123,152,173` (a pending shape-family sweep, a pending VNA
  measurement, the pending P-gate) and `ch_compensation.tex:1346,1361,1394,1410,1568`
  (pending patent disclosures). None promises a figure.
* The section instead states the opposite at `fgm_solve.tex:1355` to `1361`: there is no
  publishable 3-D dopant map yet, and no dopant field from that case is shown, printed, or
  carried forward.

### Flagged tension (outside the four checklist items, reported for completeness)

`fgm_solve.tex:1374` and `ch_compensation.tex:87` say the cone, sphere, and dumbbell have been
touched "only by the inverted-mask heuristic, which does not transfer to" them. But
`appD_fgm_depth.tex:606` to `634` reports that the per-voxel inverted-mask map *does* flatten
those three shapes, sphere +54 percent, dumbbell +44 percent, cone +32 percent, on the
`heatr3d` sigma_T metric. The two are reconcilable only if "does not transfer" is read as
"does not transfer under the shape-fidelity objective of this section", which the sentence
does not say. This is a wording risk in the new clause, not a numeric error, and it does not
change the SEAM 1 verdict.

---

## SEAM 2: dense-iff negative and the ceiling joint solve. Verdict: PARTIAL

### 2a. The cross-references exist, in both directions. CONFIRMED

* Ceiling side to dense-iff: `sections/thermal_ceiling.tex:57` is `\label{eq:tc_joint}`; the
  paragraph immediately after, lines 60 to 65, states that the objective is not introduced
  there, that it is the dense-if-and-only-if-in-bounds objective of
  `\cref{sec:solve_dense_iff}` with the same two hinges, the same asymmetric weighting, and
  the same floor, and that the standing of that objective (no shape reached the acceptance
  specification, and adding the drive does not change that) carries across.
* The narrow-"closes" paragraph repeats the pointer at `thermal_ceiling.tex:179`.
* Dense-iff side to the ceiling: `sections/fgm_solve.tex:1199` to `1201` states that the
  ceiling-constrained joint solve of `\cref{sec:comp_thermal_ceiling}` reuses this objective
  with the drive added as a second design variable, that the acceptance gate is unmet there
  too, and that the ceiling section says so where it reports closure.

Minor scope note: the fgm_solve side names the *section* (`sec:comp_thermal_ceiling`), not the
*equation* (`eq:tc_joint`). The round trip is complete at section granularity.

### 2b. "Closes" is defined narrowly at the point of use. CONFIRMED

`thermal_ceiling.tex:175` to `184`, immediately after the sentence at line 172 that uses the
word. The definition given is: the joint problem returns a feasible pair (dopant map $s$,
drive amplitude $a$) that both engines certify as respecting the ceiling at the read state
(lines 176 to 178); it explicitly does *not* mean the objective reaches its acceptance
specification (line 178); the specification is restated inline at lines 179 to 181 as growth
at or below one percent of the part cell count together with at least ninety-five percent of
in-bounds cells at or above the density floor, attributed to `\cref{sec:solve_dense_iff}`;
and lines 181 to 184 state that this is still unmet in simulation on any shape and that the
drive does not move it.

### 2c. The finished-negative framing against the reports, number by number

`fgm_solve.tex:1184` to `1197` is the finished-negative paragraph. Checked against
`DENSE_IFF_INBOUNDS_REPORT.md` and `WOUT_SOLVE_AT_PRICE_REPORT.md`:

| Dissertation claim (file:line) | Source | Match |
|---|---|---|
| 42 arms solved and scored (`fgm_solve.tex:1185-1186`) | `DENSE_IFF_INBOUNDS_REPORT.md:66`, `:609` ("all 42 scored arms of all 7 solve jobs") | yes |
| acceptance gate met by none, `0 of 42` (`:1174-1177`, `:1186-1187`) | report `:66` "**0 of 42 arms pass**"; `:717` | yes |
| best deliverable 6.50 to 16.00 percent growth, 22.6 to 58.5 percent above floor (`:1179-1181`) | report `:67-69` | yes |
| interior optimum on all 42 except three cross arms, no arm stops at first step (`:1141-1144`) | report `:40-42`, `:614-615` | yes |
| **"the finite-difference gate at each arm's own asymmetric stop passed on all forty-two" (`:1187-1188`)** | see below | **NO** |
| no arm violated the standing energy-residual gate (`:1188`) | report `:608-610` | yes |
| production stop rule adopted, assumption closed by follow-up, re-read beats solve-at-price on 2 of 3 shapes (`:1190-1194`) | `WOUT_SOLVE_AT_PRICE_REPORT.md:18,20,22` (square -19.6 percent, hexagon -16.6 percent re-read wins; triangle +20.8 percent solving wins) | yes |
| open item is the actuator, not the measurement, "with conductivity alone at grid 120" (`:1194-1196`) | report `:109`, `:717-724` | yes |

**The discrepancy, stated plainly.** `DENSE_IFF_INBOUNDS_REPORT.md` Section 3
(`:207` to `:249`) says the finite-difference gate was run **before any optimization**, on
**two shapes** (square and L_shape) and four layers, 46 probes total, epsilon swept 1e-3 to
1e-8. Its own limitations list at `:720` reads "Two shapes were gated, five were solved. The
other three inherit the gate", and at `:721-724` "The gate does not reach 1e-6 on most probes
(17 of 46)". The only gate the report evaluates per-arm on all 42 is the **energy-residual**
gate (`:608-610`, "the standing 5 percent gate evaluated at each arm's own ASYM stop passes
on all 42 scored arms of all 7 solve jobs"). The dissertation sentence has attached the
energy-residual gate's per-arm-on-42 scope to the finite-difference gate, and then reports
the energy-residual gate again in the same sentence, so the same fact is doing double duty
and one of its two appearances is false.

This matters because the paragraph's job is to license the negative. The rest of the
paragraph is accurate and the negative itself stands (0 of 42 is confirmed twice over), but
the claim "FD-gated at each arm's own stop, all forty-two" is not supported by the source.
Note also that the same section of the dissertation already states the honest version
correctly at `fgm_solve.tex:1119` to `1129`: the production filtered layer misses the strict
and adopted tiers on the square's wide probes (5.12e-4 random direction, 1.03e-4 smoothed
random direction, against the report's own `:249` table row), and the five-shape results are
read as subgradient-quality descent rather than gradient-verified optimization. The
finished-negative paragraph contradicts that earlier, correct qualification.

**Suggested minimal fix (not applied):** replace the clause with something of the form "the
finite-difference gate was run before any optimization on two shapes and four layers, with the
production filtered layer's wide probes qualified above, and the standing energy-residual gate
evaluated at each arm's own asymmetric stop passed on all forty-two".

Other dense-iff numbers spot-checked and matching (not part of the seam, but adjacent and
worth recording): the exchange-rate sweep at `fgm_solve.tex:1155` to `1161` (square 6.50 to
2.12 percent growth, IoU 0.9366 to 0.9731, density 0.8405 to 0.8135; hexagon 7.09 to 2.76
percent, 0.9301 to 0.9579, 0.8384 to 0.8051) reproduces `DENSE_IFF_INBOUNDS_REPORT.md:77-80`,
`:490-491`, `:526-527` exactly. The WOUT caveat at `fgm_solve.tex:1176-1178` (gradient
direction 3.30e-5, missing the adopted standard by about three) reproduces
`WOUT_SOLVE_AT_PRICE_REPORT.md:136`, `:165`, `:415`. The warm-start bracket 1.6 to 16.1
percent reproduces `WOUT_...:271`; "best map available on all six shape-and-price
combinations" reproduces `WOUT_...:43`; triangle IoU 0.8603 to 0.8490 reproduces
`WOUT_...:49`, `:217-218`.

---

## SEAM 3: the rotating-cross gaps. Verdict: CONFIRMED

### 3a. What the dissertation now says, and where

`sections/fgm_solve.tex`, subsection `sec:solve_rot_holdout` (label at line 907), lines 928 to
942. The explanation given:

* line 928 to 930: the two grid-120 figures differ from the values quoted earlier in the
  section, 0.9866 (indexed cross) and 0.9829 (dwell cross), "and the cause is a reading
  convention rather than a disagreement".
* lines 930 to 932: the earlier numbers are the stored actuator-campaign deliverables, scored
  against the **binary raster** indicator at the raster stop.
* lines 932 to 935: the hold-out numbers are the same stored arms re-executed by the hold-out
  harness and scored against the **sub-cell area-fill** indicator at each arm's **own optimal
  stop**.
* lines 936 to 941: read back at the earlier convention the harness reproduces the stored
  values, 0.9847 against 0.9866 and 0.98292 against 0.9829 with $J$ 34.038 against 34.04;
  therefore the harnesses agree to 0.002 or better at a matched convention, and the visible
  gaps of 0.017 and 0.002 are "the indicator and the stop", inside the 0.019 band.

The harness band is *not* offered as the cause. It is used only as a sizing comparison after
the cause has been named. That is what the seam asked for.

### 3b. Is the explanation true? Reproduced from the raw JSON

Re-read under `./.venv312/bin/python` from
`fgm_solve_campaign/out_rot_holdout/cross_index90.json` and `cross_dwell.json`:

**`cross_index90.json`**
* `stored_120_reference` = `{"report": "CONTINUOUS_ROTATION_REPORT.md Section 7, production
  engine, binary raster chi", "J": 34.8177, "IoU": 0.9866, "arm": "I90_avg4angle"}`. The key
  itself declares the stored value is scored against binary raster chi. This is the
  dissertation's attribution, verbatim from the data file.
* `grids.120.arms.ROT_solved.IoU` = `0.9700374531835206` (rounds to 0.9700) at
  `t_stop_index` 928.
* `grids.120.arms.ROT_solved.IoU_at_raster_stop` = `0.9847328244274809` (rounds to 0.9847).
* `grids.120.chi.construction` = "sub-cell area fill by
  `rfam_eqs_coupled._subpixel_fill_fraction`, union by np.maximum", and the top-level `stop`
  key = "argmin of J over the arm's own trajectory". Both halves of the stated hold-out
  convention are recorded in the file.
* Matched-convention gap: 0.9866 minus 0.9847 = **0.0019**.
* Visible gap: 0.9866 minus 0.9700 = **0.0166**, which the dissertation rounds to 0.017.

**`cross_dwell.json`**
* `stored_120_reference` = `{"report": "DWELL_SCHEDULE_REPORT.md Section 5, time-resolved
  deliverable, binary raster chi", "J": 34.04, "IoU": 0.9829, "arm": "DELIVERABLE"}`.
* `grids.120.arms.ROT_solved.IoU` = `0.9810606060606061` (rounds to 0.9811) at
  `t_stop_index` 944.
* `grids.120.arms.ROT_solved.IoU_at_raster_stop` = `0.9829222011385199` (rounds to 0.98292).
* `grids.120.arms.ROT_solved.J_raster_chi` = `34.03802973866225` (rounds to 34.038) against
  the stored 34.04.
* Matched-convention gap: 0.9829 minus 0.98292 = **-0.00002**.
* Visible gap: 0.9829 minus 0.9811 = **0.0018**, which the dissertation rounds to 0.002.

**The writer's claim of matched-convention harness agreement at or below 0.002 reproduces:**
the two matched-convention residuals are 0.0019 and 0.00002. Both are at or under 0.002. The
$J$ residual on the dwell arm is 0.002 absolute out of 34.04, that is 6e-5 relative.

Also cross-checked in the same files, so the surrounding sentences are not carrying the
verdict on unverified numbers:
* grid-160 falls quoted at `fgm_solve.tex:922-924`: indexed cross
  `grids.160.arms.ROT_solved.IoU` = 0.8052325581395349 (0.8052); dwell cross
  0.8032945736434108 (0.8033). Both match.
* keyhole 0.9684 to 0.9243 pre-fix: `keyhole_cont.json` `grids.120.arms.ROT_solved.IoU` =
  0.9684, `grids.160` = 0.9243. Match.
* the actuator-share split at `fgm_solve.tex:952-957`: `cross_index90.json` grid 160,
  `STATIC_uniform.IoU` = 0.615234375 (0.6152), `ROT_uniform.IoU` = 0.7831541218637993
  (0.7832), `ROT_solved.IoU` = 0.8052325581395349, so the map adds 0.02208 (quoted 0.0220).
  Keyhole grid 160 `ROT_uniform` 0.7648 against `ROT_solved` 0.9243. All match.

### 3c. One qualification on the band comparison

The 0.019 figure is defined at `fgm_solve.tex:80-82` as the spread of *static baselines for
one shape across the three harnesses* `out_joint`, `out_rot`, `out_seq`. Using it as the
yardstick for an *indicator-convention* gap on a rotating arm is a comparison of two different
quantities that happen to be similar in size. The dissertation's causal claim does not depend
on it (the cause is established independently by the matched-convention re-read), so this is a
rhetorical looseness rather than an error. The 0.0166 gap does sit under 0.019 as stated.

---

## SEAM 4: the premix replacement. Verdict: CONFIRMED

The rewritten appendix is `appendices/appD_fgm_depth.tex`, paragraph starting at line 1059
with `\label{app:fgm_premix}` at line 1060, running to line 1122, plus the two retained
figures at lines 1123 to 1149.

### 4a. Number-by-number against `PREMIX_SWEEP_RESULTS.md`

| Dissertation (appD_fgm_depth.tex:line) | Value | Source line | Source value | Match |
|---|---|---|---|---|
| `:1073-1074` mean melt fraction with no premix | 0.999 | `PREMIX_SWEEP_RESULTS.md:21` | 0.999 | yes |
| `:1074` premix range | 0.510 to 0.632 | `:22-25` (0.510, 0.584, 0.610, 0.632) | same | yes |
| `:1074` premix levels | 3.75 to 15 wt% | `:22-25` | same | yes |
| `:1075-1076` peak over mean | 1.147 worsening to 1.251 to 1.288 | `:21-25` (1.147; 1.285, 1.273, 1.258, 1.251) | same, min 1.251 max 1.285 | yes, see note |
| `:1077` power diverted into bed | about 80 percent | `:12` "flips ~80% of the power into the surrounding bed"; bed-absorb column `:22-25` 0.754 to 0.817 | same | yes |
| `:1079-1080` drive rises about 8 percent, 736 to 796 W | 736 to 796 | `:21`, `:25`, and `:31` "(736 -> 796 W, +8%)" | same | yes |
| `:1080-1084` voltage drive, melt fraction 0.519 and 0.655 at peak over mean 1.288 and 1.259 | as quoted | `:44-45` | 0.519/1.288/0.817 and 0.655/1.259/0.754 | yes |
| `:1084` voltage falls about eightfold | eightfold | `:48` "voltage (V) falls ~8x" | same | yes |
| `:1086` parasitic under both drive modes | claim | `:52` "parasitic under BOTH drive modes" | same | yes |

Note on the 1.288: the power-drive table tops out at 1.285 (`:22`); 1.288 is the
**voltage**-drive 3.75 wt% row (`:44`). The dissertation writes the power-drive worsening as
"to between 1.251 and 1.288" at `:1075-1076` and then quotes 1.288 again for the voltage arm
at `:1082-1083`. Read strictly against the power-drive table alone, the upper end of that
first range should be 1.285, not 1.288. The discrepancy is 0.003 in a ratio, it does not
change any sign or conclusion, and 1.288 is a real measured value from the companion sweep in
the same report, but the range as stated does not come from a single table.

### 4b. Number-by-number against `PREMIX_VS_GRADED_RESULTS.md`

| Dissertation (line) | Value | Source | Match |
|---|---|---|---|
| `:1089-1091` graded 1.000 at 1.035 | 1.000 / 1.035 | head-to-head table, graded row | yes |
| `:1090` uniform printed 0.999 at 1.147 | 0.999 / 1.147 | uniform row | yes |
| `:1091` premix 0.632 at 1.251 | 0.632 / 1.251 | premix row | yes |
| `:1091-1093` graded part mean 239.2 C, premix 197.8 C | 239.2 / 197.8 | same table, part mean T column | yes |
| `:1093-1096` 4 bpp indistinguishable at 1.035, 2 bpp erodes to 1.072, melt fraction 1.000 throughout | 1.035 / 1.072 / 1.000 | quantization table (continuous, 4-bpp 15 levels, 2-bpp 4 levels) | yes |
| `:1097-1099` premix not carried as a secondary lever | claim | report conclusion, "a uniform baseline is the wrong lever; grading is the right one" | yes |
| `:1099-1100` sources named | both reports cited by filename | n/a | yes |

The four caveats at `:1107-1117` also track the reports: linear conductivity with no
percolation threshold (`PREMIX_SWEEP_RESULTS.md:64-67`, caveat 4), oversized chamber
maximizing parasitic bed loss (`:60-63`, caveat 3), one shape
(`PREMIX_VS_GRADED_RESULTS.md` caveats, "One shape (jared rectangle)"), and simulation-only
pending the closed-loop protocol.

### 4c. Does any sentence claim premix helps? No

The only favorable-sounding text is the paragraph at `:1101-1117`, which is explicitly framed
as "Why an earlier reading looked favorable" and states that the reading is superseded because
a uniformity metric read at matched melt cannot see a part that fails to fuse at the ceiling.
The retained figure `fig:fgm_premix_bars` (`:1123-1134`) does quote favorable numbers (55.5
percent cylinder, 55.4 percent premix-study square, and FGM-plus-premix as the most uniform),
but its caption carries the bolded instruction at `:1128-1130` to read the panel as the
misleading reading and not as the verdict, and points at the mechanism figure and the text for
the actual verdict. `fig:fgm_premix_mech` (`:1136-1149`) is purely the diversion mechanism
(part-power fraction falling to 0.13 and 0.15, 70 to 87 percent of the energy heating the
bed). The section-opening sentence at `:1068` states outright that the verdict is that it
does not work. No sentence in the appendix asserts a net benefit from premix.

### 4d. Main-body mention exists and points at the appendix. CONFIRMED

`chapters/ch_compensation.tex:93-99`: the uniform powder-bed premix is named in the chapter
intro as tried and ruled out, parasitic because a conductive bed is a parallel load absorbing
the majority of the applied power, so at a matched thermal ceiling the part under-fuses and
its uniformity worsens; grading beats it on both counts; and line 99 states that the
measurements are in `\cref{app:fgm_premix}` and that it is named there so the ruling is
visible without opening the appendix. The label resolves: `app:fgm_premix` is defined at
`appendices/appD_fgm_depth.tex:1060`.

---

## Summary of discrepancies found

1. **SEAM 2, material.** `sections/fgm_solve.tex:1187-1188` claims the finite-difference gate
   passed at each arm's own asymmetric stop on all forty-two arms. The source
   (`DENSE_IFF_INBOUNDS_REPORT.md:207-249`, `:608-610`, `:720-724`) supports that scope only
   for the **energy-residual** gate. The finite-difference gate was run before any
   optimization, on two shapes and four layers, and 17 of 46 probes do not reach 1e-6. The
   sentence also contradicts the correct qualification the same section already makes at
   `:1119-1129`.
2. **SEAM 4, cosmetic.** `appendices/appD_fgm_depth.tex:1075-1076` gives the power-drive
   peak-over-mean worsening as "1.251 to 1.288"; the power-drive table
   (`PREMIX_SWEEP_RESULTS.md:22-25`) tops out at 1.285, and 1.288 is the voltage-drive value
   at `:44`. No conclusion is affected.
3. **SEAM 1, wording risk, not a checklist item.** "the inverted-mask heuristic, which does not
   transfer to them" (`fgm_solve.tex:1374`, `ch_compensation.tex:87`) reads as contradicting
   `appD_fgm_depth.tex:606-634`, where that heuristic reduces sigma_T by 32 to 54 percent on
   exactly those three shapes. The two are consistent only under the shape-fidelity objective,
   which the clause does not name.
4. **SEAM 3, rhetorical.** The 0.019 band is defined for static baselines across three
   harnesses (`fgm_solve.tex:80-82`) and is reused at `:940-941` as a yardstick for an
   indicator-convention gap. The causal claim does not rest on it and reproduces
   independently.
