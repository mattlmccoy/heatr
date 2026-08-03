# HEATR 2-D engine changelog

The engine is `rfam_eqs_coupled.py`. The version constant is
`rfam_eqs_coupled.ENGINE_VERSION` (single source of truth; the GUI server and
every run's `summary.json` / `used_config.yaml` echo read it from there).

## Versioning policy

- **MAJOR**: physics or convention changes (anything that moves an existing
  number: discretization, material laws, read-state definitions, gate
  thresholds).
- **MINOR**: new modes or actuators added behind flags or new config keys,
  with flag-off bit identity to the previous release.
- **PATCH**: bug fixes and reporting additions that do not change any
  simulated field.

Acronyms: FGM = functionally graded material (spatially varying dopant
saturation map). EQS = electro-quasi-static. GUI = graphical user interface.
bpp = bits per pixel. IoU = intersection over union.
MMA = method of moving asymptotes.

## v2.1.0 (2026-08-03) - the adopted solve-mode decisions, wired

MINOR per policy: new read states and optimizer modes behind new `fgm_solve`
config keys, with the previous behaviour reachable by flag. The engine's
forward physics, discretization and material laws are UNTOUCHED; no
non-solve-mode run changes at all.

**BEHAVIOUR CHANGE, stated plainly.** A default `fgm_solve` run now reads its
deliverable at a LATER stop than a v2.0.x run of the same configuration, so its
reported stop time, melt-region objective J, intersection over union, growth,
under-melt and the production-verification engine run all move. Set
`fgm_solve.stop_rule: j_phi` to restore the v2.0.x read state. Nothing else in
the delivered map changes: the MAP is still solved on the melt-region
objective.

- **The dense-if-and-only-if-in-bounds stop rule is the default read state**
  (`fgm_solve.stop_rule`, default `j_asym`, legacy value `j_phi`). The stop is
  the argmin over the arm's own stored trajectory of

      J_asym(s, t) = [ w_out * sum over the BED of phi(x, t)^2
                     + w_in  * sum over the PART of h(rho_rel(x, t))^2 ] / n_part

  at the report's recommended production values `w_out = 2.0`
  (`fgm_solve.w_out`) and density floor 0.85 relative density
  (`fgm_solve.density_floor_rho_rel`). The SOLVE still optimizes the
  melt-region objective for the MAP, which is the recorded verdict: reading the
  same melt-solved map at its own asymmetric argmin improved J_asym by 4.3 to
  36.1 percent, while RE-SOLVING under the asymmetric objective on top of that
  changed it by only +17.5, +6.5, +5.3, +0.1 and -15.6 percent and lost on one
  shape of five (DENSE_IFF_INBOUNDS_REPORT.md Sections 5.1, 6 and 7). Both
  stops and both objective values are recorded in every `results.json`
  (`read_state`, and `stop` inside each arm), so v2.0.x comparisons stay
  possible. New module `fgm_solve_campaign/adjoint2d/stop_rule.py`.
- **The method of moving asymptotes is the optimizer default on constrained
  objectives** (`fgm_solve.optimizer`, default `auto`; `lbfgsb` and `mma`
  override it). The policy is per objective class: the constrained
  (hinge / asymmetric) class defaults to the method of moving asymptotes, which
  beat L-BFGS-B on 4 of 5 shapes at a matched 40 forward-equivalents
  (DENSE_IFF_INBOUNDS_REPORT.md Section 8); the smooth melt-region class keeps
  L-BFGS-B, where the same comparison was 4 of 6 with two catastrophic failures
  (MMA_RETEST_REPORT.md Section 1 item 3). Because `solve_fgm` drives the MAP
  with the melt-region objective, the resolved default there is still L-BFGS-B
  and the map-solving path is unchanged; the policy is what makes that a stated
  decision rather than a hard-coded one. New module
  `fgm_solve_campaign/adjoint2d/optimizer_policy.py`.
- **The 1/|g0| objective rescale is applied at solve start on all solves**
  (`fgm_solve.objective_rescale`, default true). The objective and its gradient
  are divided by the Euclidean norm of the gradient at the start point, once. It
  is a pure reparameterization by a positive constant, so it cannot move a
  minimizer, cannot rotate a descent direction and cannot reorder two candidate
  designs; reported J values stay raw and only the optimizer sees the scaled
  pair. It fixes the upper-rail stall class, in which an objective magnitude far
  below 1 turns L-BFGS-B's relative convergence test into an absolute one and
  the solve returns the start point (confirmed in the two-dimensional gear8
  solve and independently in the three-dimensional port lane's Phase C). New
  module `fgm_solve_campaign/adjoint2d/objective_scale.py`.
- **Resolve-at-deployment-grid guidance** (`--resolve-native`). A stored dopant
  map whose solve grid differs from the configuration's grid is now REFUSED as a
  warm start and the run goes cold with a loud warning; `--resolve-native`
  acknowledges the mismatch and re-solves at the requested grid using the
  resampled map as a START only, with the design filter radius held in
  millimetres rather than in cells. Evidence: a keyhole map transferred from
  grid 120 to grid 160 read intersection over union 0.9447 while the same shape
  solved natively at 160 read 0.9716 and returned to the solved class
  (HOLDOUT_FOLLOWUP_REPORT.md Verdict 2); the filter width must stay a physical
  length or a finer solve buys fidelity with finer features (Section 3.1).
  Documented in SOLVE_MODE_USAGE.md.
- **Consequence of the new read state, recorded because it changes cost.**
  Under `stop_rule: j_asym` every SCORING forward runs the full horizon: the
  shape-fidelity early stop truncates the march 250 stored steps after the melt
  argmin, and the asymmetric argmin sits later than the melt argmin on every arm
  the report measured. The SOLVE-LOOP forwards keep the early stop, so the
  budget accounting is unchanged; the uniform reference costs one extra forward.

## v2.0.1 (2026-08-02) - solve-mode production-verify suite

PATCH per policy: reporting additions, no simulated field changed.

- Every default `fgm_solve` run now ends with a production verification
  pass: a real `rfam_eqs_coupled.py` re-run of the delivered 4 bpp map at
  the solve's optimal stop, emitting the complete standard per-run figure
  set into `production_verify/` and recording solve-vs-production deltas in
  `results.json` (`--skip-verify` opts out, recorded). Landed after the
  2.0.0 stamp; this entry corrects the version record.
- Not engine-versioned (GUI layer, tracked by `API_GENERATION`): the
  stale-server handshake banner and the legacy-marking dropdown pass.

## v2.0.0 (2026-08-01) - the standardized solve-era engine

v1.x is the pre-solve era: everything before the shape-fidelity solve
integration. v2.0.0 names the engine as it stands after the 2026-07/08
solve workstream and freezes its conventions.

Capabilities standardized in this release, with the commits that landed them:

- **Shape-fidelity FGM solve mode** as a first-class HEATR mode (command line
  `scripts/solve_fgm.py` + GUI mode `fgm_solve`): gradient-solved 4 bpp
  printable dopant map, production recipe = filtered full-depth single start,
  1.0 mm physical filter radius, no Heaviside projection, conductivity-only
  channel, map format contract-tested against the real injection loader.
  Commits `4deff91` (integration), `b04e356` (topology-optimization
  parameterization + FROZEN_CONVENTIONS_2D.md), `eaf3aec` (MMA retest: the
  projection tax is physics, not optimizer mechanics; MMA retained in the
  toolbox, projection stays optional).
- **Turntable program mode** with dwell support: arbitrary ordered
  (angle, duration) hold programs (`turntable.program` /
  `turntable.program_json`), dopant co-rotation (default ON in program mode),
  and opt-in relative-permittivity re-rasterization
  (`corotate_eps_geometry`, the measured dielectric-ghost fix: 8 to 21
  percentage points of the shape objective J on asymmetric parts). Fixed-step
  turntable mode proven bit-identical to the pre-edit engine. Commits
  `2408329` (engine support), `2cc1548` (asymmetric dwell campaign that
  produced the machine-readable turntable programs).
- **Loud melt-onset fallback**: runs that never reach the phi = 0.90
  melt-onset read state say so explicitly instead of silently reporting a
  final-state read (see `test_heatr3d_melt_fallback.py` for the 3-D
  counterpart of the convention).
- **Energy-residual gate conventions**: the standing 5-percent-of-integrated-
  dose residual gate is a reported, first-class quantity
  (`energy_residual_frac_final` in every summary).
- **Dual read-state sigma_T stamped into every run**: heating-peak
  (max sigma_T over phi_bar < 0.90) and melt-onset (first phi_bar >= 0.90)
  are computed by the engine and written to `summary.json`
  (`sigma_T_heating_peak_c`, `sigma_T_melt_onset_c`), matching the
  campaign extractor `outputs_eqs/geometry_dual_readstate/dual_readstate.py`
  (agreement pinned by `test_engine_version.py`).
- **Version stamping**: `ENGINE_VERSION` / `ENGINE_VERSION_NAME` constants;
  every `summary.json` and the `used_config.yaml` echo carry
  `engine_version`. Runs without the key are pre-v2. GUI restoration and
  standard-default promotion of the FGM surfaces: commit `0d1f6a1`.

Retired or demoted in the same workstream (decisions recorded in
`HEATR_V2_STANDARD.md`): temporal power scheduling (retired; +0.004 IoU on
top of dwells), Heaviside projection (optional, not the production default),
budget-split multi-start (single filtered cold start is the recipe).
