# FGM Solve mode: usage

FGM = functionally graded material. This document covers the shape-fidelity SOLVE
integrated into HEATR 2-D as a first-class FGM creation mode (overnight queue
2026-08-01 item 7). The solve produces a gradient-solved, 4 bits per pixel (bpp)
printable dopant map by minimizing the melt-region shape-fidelity objective
J = sum over the domain of (phi - chi)^2 through the real forward model.

Production recipe (frozen; MULTISTART_REPORT.md and TOPOPT_REPORT.md verdicts):
filtered full-depth single start; filter radius 1.0 mm as a physical length;
NO smoothed-Heaviside projection by default (beta = 0); warm start only where a
strong historical mask exists; conductivity-only actuator channel; 4 bpp output
through the production quantizer.

## Run from the command line

```
./.venv312/bin/python scripts/solve_fgm.py \
    --config outputs_eqs/fgm_calibrated_control/configs/<shape>_m*.yaml \
    --output-dir outputs_eqs/runs/<shape>/fgm_solve/<run_name> \
    [--budget 40] [--skip-verify]
```

Use a per-shape calibrated configuration (the calibrated voltage drive is part
of the frozen drive convention). The budget is in forward-equivalents; 40 is
the campaign standard. Below 20 the run labels itself a smoke-test class run.

By default the run ends with a production verification pass: the real engine
(rfam_eqs_coupled.py, the v2.0.0 production march) re-simulates the delivered
4 bpp map at the solve's optimal stop and emits the entire standard per-run
figure suite into `production_verify/`. `--skip-verify` turns that pass off
for fast iterations only; a run without it has no standard figure suite.

Optional `fgm_solve` block in the configuration yaml (all keys optional):

```yaml
fgm_solve:
  budget_forward_equivalents: 40.0   # campaign standard
  filter_radius_mm: 1.0              # physical length, frozen value
  warm_start: auto                   # auto | cold | <path to stored map npz>
  eps_channel_model_only: false      # keep false; see deployability below
  bpp: 4
  # --- v2.1.0 ---
  stop_rule: j_asym                  # j_asym (default) | j_phi (v2.0.x)
  w_out: 2.0                         # out-of-bounds price of the stop rule
  density_floor_rho_rel: 0.85        # in-bounds relative-density floor
  optimizer: auto                    # auto | lbfgsb | mma
  objective_rescale: true            # 1/|g0| at solve start
```

## The read state (v2.1.0 behaviour change)

**A default run now delivers at a LATER stop than v2.0.x.** The stop is the
argmin over the arm's own stored trajectory of the dense-if-and-only-if-in-bounds
objective

    J_asym(s, t) = [ w_out * sum over the BED of phi(x, t)^2
                   + w_in  * sum over the PART of h(rho_rel(x, t))^2 ] / n_part

at `w_out = 2.0` and a density floor of 0.85 relative density, the values
DENSE_IFF_INBOUNDS_REPORT.md Sections 6 and 7 recommend. Set
`stop_rule: j_phi` to restore the v2.0.x melt-region read state exactly.

**The MAP is still solved on the melt-region objective.** Only the stop
changed. That split is the recorded verdict: reading the same melt-solved map at
its own asymmetric argmin improved J_asym by 4.3 to 36.1 percent, while
re-solving under the asymmetric objective on top of that changed it by only
+17.5, +6.5, +5.3, +0.1 and -15.6 percent and lost on one shape of five.

Both stops and both objective values are recorded on every run, so a v2.0.x
comparison never needs a re-run: `results.json` carries `read_state` at the top
level and a `stop` block inside every arm, each with `j_phi_stop` and
`j_asym_stop` sub-records (index, time, both objectives at that index, the
horizon flag and the asymmetric guard flags).

One cost consequence, recorded because it is real: under `j_asym` every SCORING
forward runs the full horizon, because the shape-fidelity early stop truncates
the march 250 stored steps after the melt argmin and the asymmetric argmin sits
later than that argmin. The solve-loop forwards keep the early stop, so the
budget accounting is unchanged; the uniform reference costs one extra forward.

## The optimizer (v2.1.0)

`optimizer: auto` applies a per-objective-class policy: the constrained
(hinge / asymmetric) class defaults to the method of moving asymptotes, which
beat L-BFGS-B (limited-memory Broyden-Fletcher-Goldfarb-Shanno with box
constraints) on 4 of 5 shapes at a matched 40 forward-equivalents; the smooth
melt-region class keeps L-BFGS-B, where the same comparison was 4 of 6 with two
catastrophic failures. Because this entry point drives the MAP with the
melt-region objective, `auto` resolves to L-BFGS-B here. `optimizer: mma` or
`optimizer: lbfgsb` overrides the policy in either direction, and the resolved
choice with its source (`policy` or `config`) is logged and written into
`results.json` under `recipe.optimizer`.

## The objective rescale (v2.1.0)

`objective_rescale: true` divides the objective and its gradient by the
Euclidean norm of the gradient at the start point, once, at solve start. It is a
pure reparameterization by a positive constant: it cannot move a minimizer,
rotate a descent direction or reorder two candidate designs, and every J the run
reports stays RAW. It fixes the upper-rail stall class, where an objective
magnitude far below 1 turns the optimizer's relative convergence test into an
absolute one and the solve returns its own start point. The applied factor,
`|g0|` and the stated reason are recorded in `results.json` under
`recipe.objective_rescale`.

## Grid mismatch and `--resolve-native` (v2.1.0)

Maps are grid entangled, even filtered and even under rotation: a keyhole map
transferred from grid 120 to grid 160 read intersection over union 0.9447, while
the same shape SOLVED natively at 160 read 0.9716 and returned to the solved
class (HOLDOUT_FOLLOWUP_REPORT.md Verdict 2).

So when a stored map's solve grid differs from the grid the configuration asks
for, the run REFUSES that map as a warm start, starts cold, and says so loudly.
Passing `--resolve-native` acknowledges the mismatch and re-solves at the
requested grid, using the resampled map as a START only, with the design filter
radius held in millimetres (a physical length) rather than in cells. Holding the
cell count instead would shrink the design length with the grid and let a finer
solve buy fidelity with finer features. The decision, both grids, the radius and
the warning text are recorded in `results.json` under
`recipe.start.grid_transfer`.

## Run from the graphical user interface

Operation tab, Mode select, "FGM Solve (shape fidelity)". Pick the shape, keep
the defaults (budget 40, filter radius 1.0 mm, warm start Auto), and Launch
Run. The job card shows progress denominated in forward-solve counts, for
example "Solve 7/15 gradient evals, 18.3/40 forward-equivalents, J=..., IoU=...".
The existing integral, proportional, hybrid and import FGM modes are untouched;
this mode is an addition, not a replacement.

## Outputs (in the run directory under outputs_eqs)

* `fgm_<run>_solve_4bpp.npz`: the printable map in the production
  `fgm_generator.py` npz format (level_map at printer resolution, sat_map at
  simulation resolution, x_mm, y_mm, width_mm, height_mm, bpp, n_levels, and
  the standard metadata keys). It injects directly through
  `fgm_feedback.saturation_map_npz`, feeds the Results-tab "Re-simulate with
  FGM" button, and converts through `fgm_to_rip`.
* `results.json`: J (against the grid-independent area-fill target),
  `J_raster_chi` (the comparable reading against the binary raster target),
  IoU against the binary part mask, growth (`bed_melt_pct_of_part`), under
  melt (`part_under_melt_pct`), stop time with the horizon flag, energy-gate
  reading, solves spent, the warm-start provenance, and the recipe block.
* `fgm_<run>_solve_4bpp_preview.png` and `fgm_<run>_solve_4bpp_meteor_import.png`:
  the standard FGM map figure pair every other FGM creation mode ships
  (the `fgm_generator.py` convention). Preview: white = max ink, physical top
  at the image top. Meteor import: the exact pixel inversion, black = max
  ink, importable directly into the Meteor raster image processor.
* `solve_map_melt.png`: delivered map and melted region at the stop (the
  solve's own quick check figure).
* `solve_maps.npz`: continuous and quantized maps plus part mask and target.
* `production_verify_config.yaml` and `production_verify.log`: the generated
  engine configuration for the verification pass and its full log.
* `production_verify/`: a REAL `rfam_eqs_coupled.py` run (engine v2.0.0) of
  the delivered map at the solve's optimal stop, injected through
  `fgm_feedback.sat_map_npz_direct`. It contains the entire standard per-run
  figure suite, exactly what the Results tab renders: `electric_fields.png`,
  `thermal_fields_final.png`, `rf_summary_v5.png`, `paper_style_report.png`,
  `validation_report.png`, `time_series.png`, `time_series.json`,
  `density_evolution.gif`, `electric_field_evolution.gif`,
  `thermal_evolution.gif`, `fields.npz`, `summary.json`, `used_config.yaml`
  and `report_manifest.json`. The pass also scores the production run's final
  temperature field with the solve's own objective and writes the deltas into
  `results.json` under `production_verify` (`dJ_rel`, `dIoU`,
  `agrees_within_1_percent`); disagreement beyond 1 percent is flagged loudly
  in the log. Absent only when `--skip-verify` was passed, in which case
  `results.json` records the skip.

## Grid qualifier (applies to every number the solve reports)

Every fidelity number carries its grid. The solve reports at the grid of its
configuration and runs no grid hold-out. FROZEN_CONVENTIONS_2D.md section 8,
verbatim:

> **Gate A, grid hold-out.** Solve at grid 120, score at grid 160, with the
> drive voltage recalibrated at 160 so the uniform arm absorbs 500 W per metre,
> and with the target chi REBUILT from the geometry at 160.

> **A map does not get a SOLVED label unless Gate A and Gate B both pass.** On
> this pass no map earned that label under all forms; see `TOPOPT_REPORT.md`
> Section 1.

A map from this mode is therefore a deliverable candidate at its own grid, not
a SOLVED-labelled map, until the gates are run separately.

## Deployability caveats

FROZEN_CONVENTIONS_2D.md section 7, verbatim:

> **Conductivity only is the DEPLOYABLE channel** and is what every adjoint arm
> in this and every previous pass actuates: `eps_covary=False`.

> **The permittivity channel is MODEL ONLY, pending the material question.**
> [...] Until it is settled whether the real binder co-varies permittivity,
> **every permittivity-channel census is a model result with no deployment
> path** (`EPS_CHANNEL_REPORT.md` Section 11).

The `eps_channel_model_only` flag exists for research runs only; leaving it
false is the deployable configuration, and results produced with it true are
labelled model-only in results.json.

And FROZEN_CONVENTIONS_2D.md section 9 item 7, verbatim:

> **Any experimental validation.** These are two-dimensional model results.
> `ALLISON_LAW_REPLICATION.md` Section 6.1 records that the model over-predicts
> achievable tuned uniformity by roughly a factor of eight against hardware.

## Shape guidance

The solve beats the best stored historical mask on 13 of 18 library shapes and
reaches IoU >= 0.95 on 7 (SHAPE_LIBRARY_SOLVE_REPORT). L_shape, T_shape, cross,
star and rectangle are limited by geometry rather than by the dopant map; use
the Orientation Optimizer or Turntable modes for those.
