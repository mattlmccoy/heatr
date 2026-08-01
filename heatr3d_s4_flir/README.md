# Gate S4 — heatr3d vs Jared Allison's historical FLIR data

**PRE-REGISTRATION.** This file is written and committed BEFORE any prediction is
scored. It fixes the cases, the registration rule, the metrics, and the pass
thresholds. Anything that changes after the first score is recorded as a
deviation in `S4_GATE_REPORT.md`, not edited away here.

Spec: `docs/superpowers/specs/2026-07-30-heatr3d-graduation-design.md`, Gate S4:
"Score heatr3d predicted top-surface temperature fields against Jared Allison's
decoded FLIR sequences ... Deliverable: quantitative agreement report (pattern
correlation, peak location error, timing), with an honest account of unknowns in
the historical setup (drive calibration, emissivity, exact geometry)."

## 1. What is being tested

heatr3d is a 3-D EQS + enthalpy thermal solver. Against a historical IR record in
which the drive power, the emissivity, the camera-to-part registration and even
the specimen geometry are all inferred rather than documented, exactly one thing
is falsifiable with zero free parameters: **the SPATIAL PATTERN of the predicted
surface temperature field**, because a single scalar drive power (the only thing
fitted) cannot move a pattern.

So the design is:

1. Fit ONE scalar (`Params.power_density_w_per_m3`, which sets total absorbed
   power) to the measured early-time mean heating RATE of the part ROI. Nothing
   else is tuned. All material properties, geometry, and BCs come from Allison's
   own recovered COMSOL driver (`Square/FE.m`) and the canonical heatr3d Params.
2. Score the resulting spatial pattern, the hot-spot location, and the heating
   curve shape, all out of fit.
3. Run a DISCRIMINATION control: the same single prediction is scored against a
   *tuned* (graded-dopant) square from the same rig. A uniform-dopant prediction
   must match the untuned data BETTER than the tuned data. If it matches both
   equally, the pattern metric is not measuring physics, it is measuring "square".

## 2. Cases (fixed here, before running)

Geometry for all three: the specimen inferred in `ch_validation.tex`
(sec:val_retro_ir) from Allison's recovered `Square/FE.m` quarter model
(`BLK1` chamber `0.0312 x 0.0312 x 0.02` at `(0,0,-0.01)`, part `blk1`
`0.02 x 0.02 x 0.01` at `(0,0,0)`, symmetry planes x=0 and y=0, `V0 = 1206 V`),
which mirrors to a **40 x 40 x 10 mm uniformly doped square in a
62.4 x 62.4 x 20 mm chamber**, the part flush against one electrode plane with
10 mm of powder on the other side. Marked INFERRED, not documented (§6).

| case | file | class | role | why usable |
|---|---|---|---|---|
| A | `square/untuned/010320_exp1.seq` | untuned square | primary anchor | `verified` classification, 4503 frames / 615.5 s spanning ambient to melt to peak, the canonical run already decoded and deep-dived; the run the 2-D HEATR calibration used |
| B | `square/untuned/010320_exp5.seq` | untuned square | independent repeat | same session, same inferred geometry, `verified`, 429.9 s, reaches 185 C; tests whether the pattern agreement survives a re-fit of the drive on a different run |
| C | `square/tuned/031920_exp2.seq` | tuned (graded dopant) square | NEGATIVE control | same footprint, `verified`, but the dopant field is graded and undocumented, so the uniform-dopant prediction SHOULD fit it worse |

Case C is deliberately not an anchor: its dopant map is unknown, so it cannot
validate a forward prediction. It is used only as the discrimination control.

Not used as anchors, and why (full inventory in `S4_GATE_REPORT.md`):
non-square geometries (ring/diamond/star/triangle/longhorn) have no recovered
driver pinning their in-plane dimensions to the run; the 2018/2019 vertical-cell
campaigns are a different rig; `commissioning_2020-02` is mostly non-heating;
`unclassified/` has unverified geometry.

## 3. Prediction protocol (heatr3d, unmodified)

- `heatr3d.run()` with `phase_update="enthalpy"` and the corrected default
  `qrf_gradient="masked"`. `Params` is frozen; every change via
  `dataclasses.replace`.
- Domain: cubic voxels, `h = 0.65 mm`; mask shape `(nx, ny, nz) = (96, 31, 96)`
  = 62.4 x 20.15 x 62.4 mm, y = electrode/field axis (heatr3d convention:
  Dirichlet plates at y_min/y_max, Neumann on x and z, convection at y_max).
  n <= 96 per the S1 solver-validity cap.
- Part: 62 x 15 x 62 voxels = 40.3 x 9.75 x 40.3 mm, centred in x and z, flush
  against the y_min electrode plane (as in `FE.m`), leaving 10.4 mm of powder
  between the part's free face and the y_max plane.
- Material set: Allison's `FE.m` values, which are already the heatr3d canonical
  set (doped solid k 0.10 / rho 460 / cp 2500; liquid 0.26 / 1010 / 3279;
  powder 0.197 / 490 / 1072; L = 96.7 kJ/kg; eps_r doped 20, virgin 2), with
  `t_pc_c = 185 C` (Allison's value, NOT the heatr3d default 180) and
  `ambient_c = preheat_c =` the measured per-run ambient.
- Time march: `dt_s` at or below `CFL_SAFETY * dt_stable_thermal` (0.169 s at
  this h) so `n_substeps_used == 1`; `phi_target = 1.5` so the march never
  early-stops at melt onset and runs the full record length. The trajectory is
  chained in segments via `T0_override` to capture intermediate fields; the
  chaining is verified to be bit-identical to one long run before use (test).
- Standing gates on every march: `|energy_residual_frac| <= 1e-2`,
  `clamp_bound == False`, `cfl_violated == False`. Any march failing these is
  reported as a failed march, not scored.

### The one fitted scalar

`power_density_w_per_m3` is fitted per case by matching the predicted part-ROI
mean temperature RISE at `t_fit = 60 s` (pre-melt, where rise is very nearly
linear in absorbed power) to the measured ROI-mean rise at the same time. One
scalar, one target number, solved by secant iteration to <1 % in the matched
rise. Everything after 60 s — curve shape, melt timing, peak, and the entire
spatial pattern — is out of fit.

A secondary fit variant (`fit_full`: match the ROI-mean rise at the END of the
record instead) is run for case A only, to expose how much the conclusions depend
on the fit window. Both are reported.

### Observation planes (both reported; the viewing geometry is undocumented)

- **Plane P (primary)**: the part's free face — the outermost part voxel layer on
  the powder side (y index 14). This is "the top-surface temperature field" in
  the sense of the spec: the part surface as it would be seen if directly viewed.
- **Plane S (secondary)**: the domain's y_max plane, i.e. the bed surface 10.4 mm
  of powder above the part. This is what a camera looking at an undisturbed
  powder bed would actually see.

Plane P is the pre-registered primary because the measured peaks (214-230 C at
melt) are part-body temperatures; if plane S is far colder than the measurement,
that is itself a reportable finding about the viewing geometry, not a physics
failure.

## 4. Registration (measured pixels -> part-relative coordinates)

The camera pose is undocumented, so registration must be derived from the data by
an operator applied IDENTICALLY to the measured and predicted fields:

1. Reference frame = first frame whose 3x3-median-filtered global T_max rise
   reaches 60 % of the run's total rise (the mid-ramp criterion already used by
   `triage_flir_archive` and `deepdive_flir_shortlist`).
2. Threshold at `ambient + 0.5*(Tmax_ref - ambient)`; take the largest connected
   component (drops electrode/chamber glow).
3. Fit the minimum-area rotated rectangle to that component. Its centre, angle,
   and mean side length define the mapping of the part footprint onto the unit
   square `u, v in [-0.5, 0.5]`; 1 unit = 40 mm.
4. Resample both fields onto a common 64 x 64 part-relative grid over
   `[-0.5, 0.5]^2` and apply the same Gaussian smoothing (sigma = 1 cell) to
   both, so the operator's blur bias is common-mode.

The SAME steps 1-4 are applied to the predicted field. Because both rectangles
come from the same threshold rule, the thermal-halo inflation that would
otherwise bias the measured scale outward is largely common-mode. The
independent camera-optics scale (E60 + FOL18, 25 deg HFOV at the recorded 1.0 m
standoff => 1.386 mm/px) is reported as a cross-check on the fitted scale, not
used for registration.

Rotation ambiguity: for a square part in a parallel-plate field the predicted
top-face pattern is 4-fold symmetric, so the correspondence of the two lateral
axes is immaterial; only the 45 deg ambiguity matters and it is fixed by the
rectangle fit.

## 5. Metrics and PRE-REGISTERED pass thresholds

All metrics are computed on the common 64 x 64 part-relative grid, over the
footprint, on the temperature RISE field `T - T_ambient` normalized by its own
ROI mean (so any residual scale error cancels), at MATCHED thermal state — the
frame/time where the measured and predicted ROI-mean rise are equal fractions of
their own final rise (matched at 60 % and at 95 %).

| # | metric | definition | pass | strong |
|---|---|---|---|---|
| M1 | pattern correlation `r` | Pearson r of the two normalized rise fields over the footprint, at the 60 % and 95 % matched states | `r >= 0.50` at BOTH states | `r >= 0.70` |
| M2 | discrimination margin | `r(pred, untuned case) - r(pred, tuned case C)` at the 95 % state | `>= 0.15` | `>= 0.30` |
| M3 | hot-spot location error | Chamfer distance (mean nearest-neighbour distance, measured hot set -> predicted hot set) between the top-10 % pixel sets, in mm | `<= 6 mm` | `<= 3 mm` |
| M4 | hot-spot peak offset | distance between the two argmax locations after smoothing, in mm | reported | — |
| M5 | corner contrast `cX` | corner-minus-edge-mid contrast on a 5x5 block grid of the footprint (the `triage_flir_archive` X-pattern discriminator), computed identically on both | SIGN agreement | — |
| M6 | heating-curve shape | RMS difference of the normalized rise curves `theta(t) = (T_mean(t)-T_amb)/(T_mean(t_end)-T_amb)` | reported, NOT gating | — |
| M7 | melt timing | measured vs predicted time to 185 C (ROI max), and time to 50 % of final rise | reported, NOT gating | — |

**Threshold justification.**
- M1 `r >= 0.50`: the two fields share a footprint, so a floor of positive
  correlation is trivially achievable; the real discriminator is M2. 0.50 is set
  where a wrong-topology prediction (corner-hot vs centre-hot on the same square)
  would fail: on synthetic corner-X vs centre-hot fields over the same footprint
  the correlation is strongly negative. 0.70 is called "strong" because the
  effective spatial degrees of freedom here are tens, not thousands (the fields
  are smooth), so r must be large to mean anything.
- M2 `>= 0.15`: the between-run spread of the untuned corner contrast in the
  archive (`cX` +24 to -14 C across nominally identical untuned squares) is the
  natural noise scale; 0.15 in r is roughly the correlation change that spread
  produces, so a margin at or above it is not explainable by run-to-run noise
  alone.
- M3 `<= 6 mm` = 15 % of the 40 mm part width, and about 3x the registration
  uncertainty budget: 1.39 mm/px optical scale, +/-2 px rectangle-fit
  uncertainty from the thermal halo, plus one 0.65 mm prediction voxel.
- M6/M7 are NOT gating: the record contains an undocumented manual generator
  power step (documented in `ch_validation.tex` caveat (i)) that a constant-power
  simulation cannot reproduce, and the absolute drive calibration is unknown.
  Gating on them would be gating on an unknown, not on the model.

**Gate S4 verdict rule (per case).** PASS iff M1 passes at both matched states
AND M3 passes. The gate as a whole PASSES iff case A passes, case B passes, and
M2 passes. Any other outcome is reported with the specific metric that failed;
"we could not honestly evaluate it" is an allowed and preferred outcome over a
forced comparison.

## 6. Honest unknowns (fixed here, before results)

These are properties of the historical record, not of the model, and none of them
is fixable with more computation:

1. **Drive calibration.** Neither the delivered RF power nor the matching-network
   efficiency is recorded. `V0 = 1206 V` in `FE.m` is a model input, not a
   measurement. This is why exactly one scalar power is fitted and why absolute
   temperature agreement is not scored.
2. **Drive stability.** The vcross coupling survey found a 3.5x between-session
   spread in per-run coupling, and case A's own trace shows a 2.5-4.5 min plateau
   consistent with a manual power step. Constant-power prediction cannot
   reproduce that; it is left unfitted.
3. **Emissivity.** 0.95 is the camera default, not a measured powder value.
   Lower emissivity raises the reconstructed temperature (one-sided band), so all
   measured temperatures are reported at eps = 0.95 and eps = 0.85.
4. **Specimen geometry.** Inferred from `FE.m`, and the recovered driver family is
   not unambiguous: the same folder's `Tune_Conductivity.m` parameterizes a 40 mm
   CUBE. The 40 x 40 x 10 mm slab is adopted, marked inferred.
5. **Dopant distribution.** Uniform vs graded is inferred from the folder
   classification (morphology-based), not documented per run.
6. **Viewing geometry.** Undocumented: whether the top electrode was removed,
   whether the part surface or a powder cover was imaged, and the camera axis
   relative to the field axis. Hence two observation planes are reported.
7. **Registration.** Camera pose, standoff (1.0 m is the camera's own setting,
   not a measured distance), and any obliquity are unknown; the part-relative
   registration of §4 is derived from the data, so a systematic scale/perspective
   error maps into M3/M4 and is not separable.
8. **Absolute in-plane orientation** of the part relative to the electrodes is
   not recorded; the 4-fold symmetry of the predicted field makes this immaterial
   except for a 45 deg rotation, which the rectangle fit resolves.

## 7. Files

| file | contents |
|---|---|
| `README.md` | this pre-registration |
| `s4_flir_lib.py` | measured-field extraction, registration, metric operators |
| `s4_run.py` | driver: prediction marches, power fit, scoring, JSON output |
| `test_s4_flir.py` | red-green tests: chaining equivalence, registration and metric operators on synthetic fields |
| `results.json` | all numbers, machine-readable |
| `S4_GATE_REPORT.md` | verdict per case, gate status, honest-unknowns section |
| `figs/` | measured vs predicted field panels, curves |
