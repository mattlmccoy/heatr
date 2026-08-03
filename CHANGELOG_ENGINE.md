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
