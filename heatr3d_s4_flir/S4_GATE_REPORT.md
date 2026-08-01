# Gate S4 report — heatr3d vs Allison's historical FLIR data

Date: 2026-08-01. Protocol pre-registered in `README.md`, committed `aaa0cbb`,
before any prediction was scored. All numbers in `results.json`,
`grid_sensitivity.json`, `fit_window_sensitivity.json`; fields in `fields.npz`;
figures in `figs/`.

## Verdict

**GATE S4: NOT PASSED.** Two of the three pre-registered criteria are met and one
is not, and the failure is specific and physical rather than procedural:

| criterion | result |
|---|---|
| M1 pattern correlation, case A (primary anchor) | **PASS** — r = +0.822 (60 % state), +0.690 (95 % state), both >= 0.50 |
| M3 hot-spot location, case A | **PARTIAL** — 0.58 mm at the 60 % state (strong pass), **8.14 mm at the 95 % state (fail, threshold 6 mm)** |
| M1 pattern correlation, case B (independent repeat) | **FAIL** — r = +0.463 at the 60 % state (threshold 0.50); +0.508 at the 95 % state |
| M3 hot-spot location, case B | PASS — 3.40 mm and 3.24 mm |
| M2 discrimination margin (untuned minus tuned) | **PASS** — +0.246 (case A), +0.181 (case B), threshold +0.15 |
| standing gates (energy, clamps, CFL) on every march | **PASS** — see below |

One-line statement of what is and is not proven: **heatr3d reproduces the
mid-ramp corner-lobe structure of a real untuned RF-sintered square to r = 0.82
with a 0.6 mm hot-spot error, and correctly prefers untuned over tuned data, but
it does not reproduce the late-time topology of either untuned run (a centre-hot
plateau in one, narrow diagonal hot bands in the other).**

## Standing gates (every march, all cases, both fit variants)

| case | absorbed power | `energy_residual_frac` | `clamp_bound` | `cfl_violated` | `n_substeps_used` |
|---|---|---|---|---|---|
| A | 15.66 W | +6.9e-16 | False | False | 1 |
| B | 14.10 W | -3.8e-17 | False | False | 1 |
| C | 24.42 W | -6.3e-16 | False | False | 1 |
| A (secondary fit) | 11.27 W | -4.1e-16 | False | False | 1 |
| B (secondary fit) | 15.31 W | ~0 | False | False | 1 |
| C (secondary fit) | 20.11 W | ~0 | False | False | 1 |
| A (coarse grid) | 15.66 W | +8.1e-16 | False | False | 1 |

Residuals are at machine precision (enthalpy scheme), no numerical limiter bound
anywhere, and `dt_s = 0.15 s <= CFL_SAFETY * dt_stable_thermal = 0.169 s` at
h = 0.65 mm, so no substepping was needed. `phase_update="enthalpy"`,
`qrf_gradient="masked"` (the corrected default), `Params` frozen and modified
only through `dataclasses.replace`.

## 1. Anchor-case inventory and usability verdicts

The archive (`JaredFiles/IR Camera Files`, `MANIFEST.csv`) holds **152 decodable
`.seq` recordings**: 83 `verified`, 55 `morphology`-classified, 14 `uncertain`;
119 flagged `clean`. Usability for Gate S4 needs three things at once: (a) part
geometry pinned to the run, (b) dopant distribution known (the forward model
assumes uniform), (c) frames spanning heating. Drive/exposure is NOT required
because the one fitted scalar absorbs it — but its absence is what forces the fit.

| group | n | geometry metadata | dopant | verdict |
|---|---|---|---|---|
| `square/untuned` | 19 | inferred, from `Square/FE.m` (quarter model -> 40 x 40 x 10 mm part in a 62.4 x 62.4 x 20 mm chamber, V0 = 1206 V) | uniform (folder classification) | **USABLE** — the only group meeting all three. Cases A and B drawn from here |
| `square/tuned` | 6 | same footprint | **graded, map not linked to any run** | not an anchor; used as the discrimination control (case C) |
| `rectangle` (Nov-15-2019, n = 10) | 10 | no recovered driver for the rectangle | tuned | NOT USABLE (geometry not pinned) |
| `ring`, `diamond`, `star`, `triangle`, `longhorn` | 28 | dimensions DO exist in the recovered per-shape `FE.m` drivers (e.g. Triangle polygon 21.9 x 38 mm extruded 10 mm; Hollow Cylinder quarter rectangle 10 x 10 mm) | tuned/graded | NOT USABLE as forward anchors: no documented link from a given `.seq` run to a given driver/tuned-sigma map, and the as-printed dopant field is unknown |
| `commissioning_2020-02` | 26 | — | — | NOT USABLE (23 of the archive's `non-heating` runs live here) |
| `campaign_2018_disk_cell` | 18 | different rig (vertical disk cell) | — | NOT USABLE (rig mismatch) |
| `campaign_2019_vertical_samples` | 27 | shape ID open (see its README) | — | NOT USABLE |
| `unclassified` + `anomalies` | 15 | unverified / runaway / abort | — | NOT USABLE |

Minimal metadata that would make the other groups usable, in priority order:
1. A per-run link from the `.seq` file to the COMSOL driver / tuned-sigma map
   actually printed (a single lab-notebook line per run).
2. The as-printed dopant field (or the statement "uniform") per run.
3. Camera standoff and pose (see the unresolved scale discrepancy, §4.5).
4. A recorded generator setpoint and any manual power changes with timestamps.

Cases actually run:

| case | file | class | duration | ambient | measured peak (eps 0.95) |
|---|---|---|---|---|---|
| A | `square/untuned/010320_exp1.seq` | untuned | 615.5 s | 21.7 C | 219.0 C |
| B | `square/untuned/010320_exp5.seq` | untuned | 429.9 s | 24.2 C | 231.8 C |
| C | `square/tuned/031920_exp2.seq` | tuned (control) | 235.5 s | 22.8 C | 214.6 C |

## 2. Prediction configuration

96 x 31 x 96 voxels, h = 0.65 mm (chamber 62.4 x 20.15 x 62.4 mm), part
62 x 15 x 62 voxels = 40.3 x 9.75 x 40.3 mm flush against the y_min electrode
plane, uniform dopant, Allison's material set with `t_pc_c = 185 C`. One EQS
solve (25.7 s) produces a reference field of 25.20 W absorbed; each case's
prediction is that field rescaled by ONE fitted scalar, applied through
`qrf_override` (exact, since `compute_qrf_3d` renormalizes linearly to a total
power target). Time march chained in 40 segments through `T0_override`, verified
**bit-identical** to a single long march by
`test_chained_T0_override_segments_equal_one_long_march` (7/7 tests pass).

## 3. Scores against the pre-registered thresholds

Primary fit (one scalar matched to the measured ROI-mean rise at t = 60 s;
everything below is out of fit), observation plane P (the part's free face):

| case | state | r (M1) | chamfer mm (M3) | argmax mm (M4) | cX meas / pred (M5) |
|---|---|---|---|---|---|
| A | 60 % | **+0.822** | **0.58** | 35.17 | +6.84 / -1.84 |
| A | 95 % | **+0.690** | 8.14 | 10.00 | -0.75 / -7.28 |
| B | 60 % | +0.463 | **3.40** | 8.24 | -0.92 / +5.34 |
| B | 95 % | **+0.508** | **3.24** | 7.45 | +40.62 / -2.74 |
| C (control) | 60 % | -0.424 | 10.36 | 10.92 | -0.31 / +19.26 |
| C (control) | 95 % | -0.413 | 11.02 | 18.66 | +0.05 / +10.56 |

M2 (discrimination, 95 % state, plane P): case A r(untuned) = +0.690 vs
r(tuned) = +0.444 -> margin **+0.246**; case B +0.508 vs +0.327 -> **+0.181**.
Both pass. The uniform-dopant prediction prefers untuned data, so M1 is measuring
pattern and not merely "square".

Non-gating metrics (M6/M7), primary fit:

| case | curve RMS (normalized) | t50 meas / pred | t185 meas (ROI max) | t185 pred (part max / free face) |
|---|---|---|---|---|
| A | 0.063 | 157 / 230 s | 382 s | 220 s / 551 s |
| B | 0.050 | 173 / 193 s | 334 s | 255 s / never |
| C | 0.090 | 83 / 105 s | 224 s | 112 s / never |

The measured melt time sits BETWEEN the predicted interior and free-face times in
every case, which is the quantitative form of the undocumented-viewing-geometry
problem: the model has no single surface that is "the" measured one.

### Fit-window sensitivity (pre-registered secondary variant)

Refitting the same single scalar to the END-of-record rise instead of the 60 s
rise (case A: 11.27 W instead of 15.66 W) moves case A across the M3 threshold:

| case A | r 60 % | r 95 % | chamfer 60 % | chamfer 95 % | curve RMS | t185 pred (part max) |
|---|---|---|---|---|---|---|
| primary fit (60 s), 15.66 W | +0.822 | +0.690 | 0.58 mm | **8.14 mm** | 0.063 | 220 s |
| secondary fit (end), 11.27 W | +0.834 | +0.825 | 0.41 mm | 5.68 mm | 0.105 | 379 s |

Under the secondary fit case A passes M1 and M3 at both states. **The verdict is
therefore fit-window dependent, and the pre-registered PRIMARY fit is the one
reported as the verdict** — cherry-picking the favourable window is exactly what
pre-registration exists to prevent. The direction is understood: the measured
trace ramps fast early and then plateaus (an undocumented manual power step), so
a fit to the early rate over-drives the late field, pushes the prediction deeper
into melt, and flattens the free-face pattern.

Two incidental corroborations: the secondary fit's 11.27 W lands within 5 % of
the independently published 2-D HEATR calibration (11.85 W absorbed,
`ch_validation.tex` sec:val_retro_ir), and at that power the predicted
part-interior time to 185 C is 379 s against 382.5 s measured. Neither is a
free-parameter-free result (the power was fitted to the end temperature), but
both are consistent.

Case B does not move with the fit window (r = 0.432 / 0.522 under the secondary
fit), so its M1 failure at the 60 % state is a real disagreement, not a drive
artifact.

### Grid sensitivity (added; not pre-registered)

Gate S2 (convergence) is not passed, so the pattern claim needs its own guard.
Re-running case A on a 1.5x coarser grid (h = 0.975 mm, 64 x 21 x 64, dt 0.3 s):

- r(coarse pattern vs fine pattern) = **0.989** (60 %), **0.991** (95 %)
- chamfer(coarse vs fine) = 0.046 mm and 0.040 mm
- free-face end temperature 198.5 C vs 195.6 C (1.5 %)
- score against the measurement moves from +0.822/+0.690 to +0.794/+0.636

The scored pattern is grid-robust across this refinement; the S4 result is not a
staircase artifact. (This says nothing about the S2 spread-metric convergence
problem, which concerns std(T) at the conductor corners, not the free-face
pattern.)

## 4. Findings

**4.1 What matches.** At mid-ramp, the measured field of case A is a four-lobed
pattern with hot lobes inset from the corners, cool face midpoints and a cool
centre; heatr3d puts hot lobes at the same four positions with the same cool
centre and cool extreme corners (`figs/case_A_fields.png`, top row): r = +0.822,
hot-set chamfer 0.58 mm. This is the falsifiable part — a single scalar power
cannot place four lobes — and it is met.

**4.2 What does not match: late-time topology.** By the 95 % state the two
untuned runs diverge from the prediction in two different ways. Case A becomes a
broad centre-hot plateau (207 C centre, 112 C corners) while the prediction stays
a flat ring (174 C centre, 184 C ring). Case B develops narrow, bright DIAGONAL
bands crossing at the centre (224 C on the diagonals, cool quadrant interiors) —
the literal "X" the archive names — which the prediction does not produce at all
(`figs/case_B_fields.png`, bottom row). Candidate mechanisms, in the order they
should be tested, all of which are model simplifications rather than data
problems:

1. **Q_rf is frozen.** `heatr3d.run()` solves the EQS once and holds Q_rf for the
   whole march. In reality sigma, eps_r and the geometry all change as the part
   melts and densifies, so the field should re-concentrate as melting proceeds.
   A once-per-N-steps EQS re-solve is the single most direct test.
2. **No sigma(T) or sigma(rho_rel) feedback** (`sigma_temp_coeff`,
   `sigma_density_coeff` exist in the 2-D solver's config, not in heatr3d
   `Params`). Positive feedback of this kind is the natural way to grow narrow
   diagonal channels out of a broad lobed field.
3. **No densification coupling into the EQS** (density evolves, gamma does not).
4. **Uniform-dopant assumption**: the untuned parts were printed, so real dopant
   non-uniformity (and any diagonal print-path signature) is not in the model.

**4.3 Discrimination is real.** The same uniform-dopant prediction correlates
+0.69 with untuned data and +0.44 with tuned data (case A), and +0.51 vs +0.33
(case B). The tuned data are centre-hot and irregular; the tuned-case prediction,
driven at its own fitted power, anticorrelates with them (-0.41). The pattern
metric is therefore not saturated by the shared square footprint.

**4.4 The plane-S (powder cover) hypothesis is falsified by the data.** The
secondary observation plane — the bed surface 10.4 mm of powder above the part —
reaches only 121.6 C (case A) when the measurement reads 210 C, and its
50 %-threshold ROI degenerates to the ENTIRE 62.4 mm domain, so its registration
rectangle is the chamber, not the part. **All plane-S correlations in
`results.json` are therefore void and are not used in any verdict.** The
practical conclusion: whatever the camera saw, it was not a 10 mm powder cover
over the part.

**4.5 An unresolved 1.4x scale discrepancy.** The measured 50 %-threshold hot
rectangle is 52 x 49 px, which at the FLIR E60 + FOL18 optics and the recorded
1.0 m standoff is 72.1 x 67.9 mm. heatr3d's own 50 %-threshold rectangle for its
40.3 mm part is 49.4 mm, i.e. the same operator inflates a known part by +23 %.
After allowing that halo, the measured hot region is still ~1.4x too large for a
40 mm part. Three mutually exclusive explanations, none decidable from the record:
the part was larger than the inferred 40 mm (~57 mm), the standoff was ~0.7 m
rather than the recorded 1.0 m, or the lens/FOV differs from the assumed
25 deg. **Every millimetre-valued metric here (M3, M4) scales with this factor**;
the correlations do not, because registration is part-relative. If the true scale
is 1.4x larger, case A's 95 %-state chamfer becomes 11.4 mm, not 8.1 mm.

**4.6 Fitted absorbed power.** 14-24 W across the three cases with the early-rate
fit, 11-20 W with the end-rise fit — a spread consistent with the vcross survey's
3.5x between-session coupling spread, and bracketing the published 2-D value of
11.85 W. No claim of drive accuracy is made; the power is fitted.

## 5. Honest unknowns in the historical setup

Unchanged from the pre-registration (README section 6), with what this run added:

1. **Drive calibration** — not recorded. Absorbed power is fitted (one scalar per
   case). Quantified: the verdict on case A's M3 flips with the fit window.
2. **Drive stability** — case A's trace plateaus at 2.5-4.5 min, consistent with a
   manual generator step. Left unfitted; it is the direct cause of the
   fit-window sensitivity above.
3. **Emissivity** — 0.95 is the camera default. The eps 0.85 end-member is
   decoded and stored for every case (`measured.e085` in `results.json`); it
   raises measured temperatures one-sidedly (case A peak 219.0 -> 234.9 C,
   end-time 214.1 -> 229.5 C, t185 382.4 -> 359.7 s) and is shown as a band in
   `figs/curves.png`. It does not affect the pattern metrics, which are computed
   on a normalized rise field.

   Decode verification: this run's independent decode of case A reproduces the
   published values of `ch_validation.tex` sec:val_retro_ir exactly — end-time
   T_max 214.1 C (published 214.1) and t185 382.4 s (published 382.5), peak
   219.0 C on a 180-frame sample vs 219.4 C on the full 4503 frames. The measured
   side of this gate is therefore the same data the 2-D anchor used.
4. **Specimen geometry** — inferred from `Square/FE.m`; the same folder's
   `Tune_Conductivity.m` parameterizes a 40 mm CUBE instead. See §4.5: the
   thermography is not consistent with a 40 mm part at the recorded standoff.
5. **Dopant distribution** — uniform vs graded is a folder/morphology
   classification, not a documented per-run fact.
6. **Viewing geometry** — undocumented, and now bounded from two sides: a 10 mm
   powder cover is excluded by §4.4, while the FE.m geometry has the part's top
   face against the electrode plane, which a camera cannot see through. The
   record is internally inconsistent about what was imaged.
7. **Registration** — camera pose and standoff unknown; the part-relative
   registration is derived from the data, so a systematic scale or perspective
   error maps directly into M3/M4 (§4.5) and is not separable.
8. **In-plane orientation** — immaterial by the 4-fold symmetry of the predicted
   field except for a 45 deg ambiguity, which the rectangle fit resolves (fitted
   angles 0.0, 86.5, 0.0 deg for A, B, C).

## 6. Deviations from the pre-registration

| # | deviation | reason |
|---|---|---|
| D1 | scoring window is the interior 90 % of the fitted footprint (`SPAN = 0.90`), not the full `[-0.5, 0.5]^2` | sampling exactly at the footprint edge swings a sample between part (~200 C) and powder (~20 C) on sub-pixel registration error, making the correlation a measure of edge alignment. Applied identically to both fields; the corner lobes are inside the window. Fixed before any score was computed |
| D2 | M3 was pre-registered without naming the matched state; it is evaluated at BOTH states and the strict (both must pass) reading is used | the pre-registration was silent; the strict reading is the conservative one and is the reason case A is reported as not passing |
| D3 | grid-sensitivity run added (not pre-registered) | S2 is not passed, so the pattern claim needed its own refinement guard. Does not change any scored number |
| D4 | the secondary end-of-record fit was pre-registered for case A only; it was also run for B and C | completeness; labelled as beyond the pre-registration and not used in any verdict |
| D5 | M5 (`cX`) disagrees in sign at case A's 60 % state (+6.84 measured vs -1.84 predicted) despite an evident lobe match | operator limitation, reported not repaired: the 5 x 5 block grid's corner cells land on the cold extreme corners, while the actual hot lobes are inset from them. M5 was pre-registered as non-gating |
| D6 | plane-S scores are reported as VOID rather than as numbers | its ROI registration degenerates to the whole chamber (§4.4); reporting a correlation from a failed registration would be a false-green |

No case substitutions, no threshold changes, no post-hoc metric additions to the
verdict rule.

## 7. What would move this gate

In order of expected value:

1. **Re-solve the EQS during the march** (or at least at melt onset) and re-score.
   This is the single most likely explanation of the late-time topology miss and
   it is entirely internal to heatr3d.
2. **Add sigma(T) / sigma(rho_rel) to heatr3d's `Params`** (the 2-D solver has
   both) and test whether positive feedback produces the diagonal banding of case
   B.
3. **Resolve the 1.4x scale question** — a single measured standoff or part
   dimension in Allison's notes would convert M3/M4 from indicative to
   quantitative.
4. Only then widen the case set; more runs of the same undocumented kind will not
   change what is knowable.

## 8. Reproducing this

```bash
cd geo-prewarp
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m pytest \
    heatr3d_s4_flir/test_s4_flir.py -q          # 7 tests, incl. chaining equivalence
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -u \
    heatr3d_s4_flir/s4_run.py                    # EQS solve + 3 cases -> results.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -u \
    heatr3d_s4_flir/s4_extras.py                 # grid sensitivity + figures
```

`cache/` is gitignored and rebuilt automatically on first run; the rebuild was
verified **bit-identical** (`np.array_equal`) to the Q_rf field these scores were
computed from. Pin the BLAS thread count: with it unset, thread oversubscription
turned a 26 s EQS solve into an 18 min one on this machine (load average 258).

Read-only inputs, untouched by this work: `heatr3d.py`, `extract_flir_seq.py`,
`triage_flir_archive.py`, `flir_archive_paths.py`, the `JaredFiles` archive, and
`dissertation_materials/`.
