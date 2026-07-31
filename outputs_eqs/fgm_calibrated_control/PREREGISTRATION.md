# Pre-registration: gain-calibrated one-shot FGM control (2-D)

Frozen 2026-07-30, BEFORE any new solve was launched. Written to answer P0 / Q1 of
`FGM_INVERSE_DESIGN_ASSESSMENT.md` Section 6 ("the missing control").

Acronyms, expanded on first use: FGM = functionally graded material (a spatially varying
dopant saturation map). EQS = electro-quasi-static. sigma_T = the 2-D uniformity metric
`ui_rms_part(t) * (T_bar_part(t) - 23 C)`, always quoted with its read state.

## The question

The production one-shot proportional inverse map has one free scalar, `magnitude` (the gain).
It is currently either hand-set or picked best-of-four in-sample on the SAME metric that is
then reported. The assessment diagnoses every "compensation made it worse" result as a GAIN
failure rather than a map-shape failure. This campaign builds the missing control: same map
shape, gain set by a one-dimensional line search on the real forward solve, SELECTED ON A
HOLD-OUT read state.

## Frozen protocol

- **Engine.** `rfam_eqs_coupled.py` (the trusted 2-D EQS + thermal engine).
- **Configuration.** Identical to `outputs_eqs/geometry_dual_readstate/run_sweep.py`:
  per-shape base config, voltage drive (`enforce_generator_power: false`) at the per-shape
  calibrated `v_cal` stored in `outputs_eqs/geometry_dual_readstate/<shape>.json`,
  grid 120 x 120, as-sized geometry, horizon 1500 steps (750 s).
- **Map construction.** `fgm_generator.generate_fgm(baseline_run, bpp=4, proxy_field="T_phi90",
  invert=True, magnitude=m, baseline_saturation=0.5, dead_band=0.05)`. PROPORTIONAL variant
  (`use_delta_correction` off). Only `m` varies.
- **Design variable.** `m` (the scalar gain), domain `[0.05, 2.50]`. The cap 2.50 is chosen
  because `sat` is clipped to `[0, 1]` about `baseline_saturation = 0.5`, so beyond roughly
  `m = 2` the map is already essentially two-level and further gain changes little.

## Fit metric and hold-out metric (frozen before any solve)

- **Fit metric (what the line search minimizes):** heating-peak sigma_T, i.e. the maximum of
  sigma_T over all steps with `phi_bar < 0.90`.
- **Hold-out metric (what is reported, never used to choose the gain):** melt-onset sigma_T,
  read at the first `phi_bar >= 0.90` crossing.

The two read states are physically distinct (the assessment and `HEATR_STANDARD_PARAMETERS.md`
Section 3 record a 2.24x compression between them for the square), so the melt-onset number is
a genuine hold-out for a gain chosen on the heating peak.

## Feasibility constraint (uses the EXISTENCE of the hold-out state, never its value)

A candidate gain is FEASIBLE only if its run crosses `phi_bar >= 0.90` inside the 1500-step
horizon. A map that never melts is not a usable part. This constraint is declared here, before
any solve, and it reads only `melt_reached`, never `melt_onset_sigma_T`. Both the constrained
selection and the unconstrained (fit-only) selection are reported for every shape.

If NO candidate is feasible for a shape, that is reported as NOT_REACHED. A final-step fallback
is never substituted for melt-onset.

## Budget

Up to 8 forward-solve evaluations of the gain per shape, of which the 4 already stored in
`outputs_eqs/geometry_dual_readstate/runs/<shape>/fgm_m*` (m = 0.30, 0.50, 0.70, 0.85) are
reused as exact cache hits (same config, same map construction). Cache validity is verified by
re-running one stored point and comparing sigma_T. New solves per shape are therefore at most 4,
and both counts are reported.

## Arms compared, per shape, at BOTH read states

- **(a) uniform baseline**: no FGM map. Reused from `geometry_dual_readstate`.
- **(b) best-of-four in-sample**: the stored 4-point grid winner, selected on melt-onset
  sigma_T, which is the metric it is then reported on. Reused, not re-run.
- **(c) calibrated gain**: this campaign: gain chosen on heating-peak sigma_T among all
  evaluated gains, reported at melt-onset.

## Verdict criterion (frozen)

Calibration WINS on a shape if arm (c)'s melt-onset sigma_T is lower than arm (b)'s.
Calibration RESCUES a shape if arm (b) is worse than arm (a) (harmful) and arm (c) is not.
Every shape attempted is reported, win or lose. Shapes not attempted are named.

## Shape set

Required: square, circle, hexagon (reconciled reference shapes), triangle, L_shape (the
one-shot-harmful class; L_shape historically never reaches melt with FGM). Extended to more of
the 19-shape standardized library only if per-solve timing allows; coverage is stated
explicitly and never silently truncated.
