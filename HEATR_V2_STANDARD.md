# HEATR 2-D v2.0 standard: the frozen decisions on one page

Engine: `rfam_eqs_coupled.py`, `ENGINE_VERSION = "2.0.0"` ("HEATR 2D v2.0").
Every decision below cites the report that established it. Pinned run
parameters (drive mode, grid, read states, gates) live in
`HEATR_STANDARD_PARAMETERS.md`; this page does not restate them.
Versioning policy and the capability list: `CHANGELOG_ENGINE.md`.

Acronyms: FGM = functionally graded material (spatially varying dopant
saturation map). bpp = bits per pixel. IoU = intersection over union.
MMA = method of moving asymptotes. EQS = electro-quasi-static.

## Production solve recipe (the printing deliverable)

Source: `FROZEN_CONVENTIONS_2D.md` sections 1, 6, 7; `MULTISTART_REPORT.md`;
`TOPOPT_REPORT.md`; implemented in `scripts/solve_fgm.py` and the GUI
`fgm_solve` mode (`SOLVE_INTEGRATION_NOTES.md`).

- **Filter only, 1.0 mm physical radius.** Normalized-convolution Gaussian
  over the part, radius held as a LENGTH and converted per grid. The filter,
  not the projection, is what makes solves grid-robust (blur cost +153
  percent to +9.9 percent, MULTISTART_REPORT section 6). The radius is frozen
  but was not swept (FROZEN_CONVENTIONS_2D section 9 item 1).
- **Full-depth SINGLE cold start.** Budget-split multi-start costs depth and
  loses on 12 of 18 shapes at the 40 forward-equivalent budget
  (MULTISTART_REPORT section 4).
- **Warm-start exception.** Warm start ONLY where a strong historical mask
  exists, injecting the FILTERED mask (`ms_solve.build_starts`).
- **4 bpp deliverable.** The delivered arm is always the continuous map
  quantized to 4 bits per pixel inside the part and re-run through the real
  forward, never the continuous map alone (FROZEN_CONVENTIONS_2D section 6).
- **Conductivity (sigma) only.** The deployable actuator channel. The
  permittivity channel is MODEL-ONLY pending the material question (does
  carbon-black loading move eps_r?); its 17-of-18 census is not quotable as
  deployable (EPS_CHANNEL_REPORT; FROZEN_CONVENTIONS_2D section 7).

## Dwell (turntable programs): composable per shape class

Source: `DWELL_SCHEDULE_REPORT.md`; engine execution
`ENGINE_DWELL_SUPPORT_NOTES.md` (commit 2408329).

- Cross-class (rotation-symmetric, multi-lobe): dwell helps; the cross
  reached in-grid IoU 0.9829 at grid 120 with a solved map plus an
  asymmetric 20 s-cycle turntable program, engine-verified.
- T_shape / L_shape: partial / no benefit. T parks 73.5 percent at 90
  degrees for its best-ever 0.644 (honest partial); L collapses to a static
  135-degree orientation (rotational actuation exhausted).
- Star: null; rotation alone reaches 0.953 and the dopant map is inert
  (CONTINUOUS_ROTATION_REPORT).
- For any NEW asymmetric rotating study, switch `corotate_eps_geometry` ON:
  the dielectric ghost costs 8 to 21 percentage points of J on asymmetric
  parts and is zero only for rotation-invariant shapes
  (ENGINE_DWELL_SUPPORT_NOTES section 5.2). The GUI defaults it ON in
  program mode; the engine default stays OFF so no shipped number moves
  silently.

## Retired and demoted

- **Temporal power scheduling: RETIRED.** On top of dwells it is worth
  +0.004 IoU (DWELL_SCHEDULE_REPORT; OVERNIGHT_REPORT_3.md item 10). Its one
  real effect (cross re-heat) is subsumed by the dwell program.
- **Heaviside projection: retired from the production recipe.** The
  projection tax is physics and parameterization, not optimizer mechanics:
  with MMA at a matched 40 forward-equivalent budget it loses in grid on six
  of six shapes, and the budget that closes part of the gap gives back the
  Gate B robustness and the near-binary maps (MMA_RETEST_REPORT.md sections
  1 and 8). **MMA itself is RETAINED in the toolbox**
  (`fgm_solve_campaign/adjoint2d/mma.py`) for future constrained objectives,
  where the dual restores gradient-magnitude sensitivity.

## Quoting rules (every number, every time)

Source: `FROZEN_CONVENTIONS_2D.md` sections 2, 3, 6; `MMA_RETEST_REPORT.md`
conventions block.

1. **Three baselines**: quote a solve against the uniform arm, the best
   stored historical mask, and the SOLVED threshold (absolute IoU >= 0.95),
   not against any single one of them.
2. **Grid qualifier**: every fidelity number carries its grid. Grid-120
   fidelity does not transfer to 160 unaided (SOLVE_ROBUSTNESS_VALIDATION);
   no SOLVED label without the Gate A grid hold-out and Gate B sub-filter
   blur, both passing.
3. **Dose-match limit**: arms are NOT dose matched (measured spread 316 to
   500 W per metre); state it wherever arms are compared. Historical masks
   were scored in the permittivity-co-varying channel, so comparisons
   against them are actuator-mismatched and must say so.

## Standing checks on every run

- **Melt-onset fallback is LOUD**: a run that never reaches the phi = 0.90
  melt-onset read state says so (`sigma_T_melt_reached = false` in
  summary.json; the GUI renders "melt not reached", never a silent
  final-state number).
- **Energy-residual gate**: |residual| / integrated dose <= 5 percent at the
  read state, reported on every run (`energy_residual_frac_final`; run cards
  flag values above 5 percent).
- **Grid ceiling**: 120 validated, 160 stable, at or above 200 is the
  melt-onset instability regime, never 240 (`HEATR_STANDARD_PARAMETERS.md`;
  the GUI warns at 200).
- **Dual read-state reporting**: heating-peak AND melt-onset sigma_T,
  stamped by the engine into every summary.json and shown on run cards
  (`outputs_eqs/geometry_dual_readstate/GEOMETRY_DUAL_READSTATE.md`).

## What v2.0 does not settle

The permittivity deployability question (a VNA dielectric sweep vs
carbon-black loading decides it), the T/L geometry frontier, the filter
radius sweep, and any experimental validation (the model over-predicts
tuned uniformity by roughly 8x against hardware, ALLISON_LAW_REPLICATION.md
section 6.1).
