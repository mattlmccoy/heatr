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
    [--budget 40]
```

Use a per-shape calibrated configuration (the calibrated voltage drive is part
of the frozen drive convention). The budget is in forward-equivalents; 40 is
the campaign standard. Below 20 the run labels itself a smoke-test class run.

Optional `fgm_solve` block in the configuration yaml (all keys optional):

```yaml
fgm_solve:
  budget_forward_equivalents: 40.0   # campaign standard
  filter_radius_mm: 1.0              # physical length, frozen value
  warm_start: auto                   # auto | cold | <path to stored map npz>
  eps_channel_model_only: false      # keep false; see deployability below
  bpp: 4
```

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
* `solve_map_melt.png`: delivered map and melted region at the stop.
* `solve_maps.npz`: continuous and quantized maps plus part mask and target.

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
