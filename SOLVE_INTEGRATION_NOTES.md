# Solve integration notes (overnight queue 2026-08-01, item 7)

Verdict: the shape-fidelity SOLVE is integrated into HEATR 2-D as a first-class
FGM (functionally graded material) creation mode, end to end: a config-driven
command-line entry point running the production recipe, the production npz map
format proven bit-exact against the real engine loader, a graphical user
interface mode that launches it as a job with forward-solve-count progress, and
a run directory the Results tab browses. Nothing existing was removed or
changed in behavior; the integral, proportional, hybrid and import FGM modes
are untouched. Nothing is committed; all changes sit in the working tree for
review.

Production recipe implemented (per the MULTISTART_REPORT.md and
TOPOPT_REPORT.md verdicts, frozen in FROZEN_CONVENTIONS_2D.md): FILTERED
full-depth SINGLE start; 1.0 mm PHYSICAL filter radius
(`topopt.sigma_cells_for` conversion per grid); NO smoothed-Heaviside
projection by default (beta = 0, at which `topopt.design_to_map` is
bit-identical to the plain filter); warm start only where a strong historical
mask exists (library HIST_best, injected filtered, the `ms_solve.build_starts`
convention); conductivity-only channel (`eps_covary=False`; the permittivity
channel sits behind `eps_channel_model_only`, default off, labelled model-only
when used); 4 bits per pixel (bpp) deliverable through the production
quantizer `printability.quantize_in_part`, re-run through the real forward.

## What was added (files and line counts)

New files:

* `scripts/solve_fgm.py` (555 lines). The entry point. Imports
  `fgm_solve_campaign/adjoint2d` as a library (topopt, topopt_objective,
  chi_area, adjoint, forward, printability, control, ms_solve, multistart,
  verify_hist, pins); nothing was copied or forked. Contains the
  `fgm_solve` config block parser (frozen dataclass `FgmSolveConfig`), the
  production npz emitter `emit_production_npz`, the warm-start resolver, the
  filtered beta = 0 L-BFGS-B solve loop with a StopIteration budget guard, and
  the map+melt check figure.
* `test_solve_fgm.py` (172 lines, 14 tests). Red-first test-driven
  development: the config-parse tests were written first and failed with
  `ModuleNotFoundError: scripts.solve_fgm`, then the parser was written to
  green; the emitter tests were added and failed with
  `ImportError: cannot import name 'emit_production_npz'`, then the emitter
  was written to green. Final state: 14 passed.
* `SOLVE_MODE_USAGE.md` (108 lines). Command-line and interface usage, output
  inventory, and the grid qualifier and deployability caveats quoted verbatim
  from FROZEN_CONVENTIONS_2D.md sections 7, 8 and 9 (as the queue required;
  HEATR_STANDARD_PARAMETERS.md was not touched).
* `SOLVE_INTEGRATION_NOTES.md` (this file).

Changed files (all changes in these three files are from this pass; the GUI P0
restoration was already committed as 0d1f6a1):

* `rfam_gui_server.py` (+164/-2). `_launch_fgm_solve_mode` (after
  `_launch_fgm_resimulate_mode`, whose config-patch-then-`_run_command` shape
  it reuses; the fresh-start job pattern is the `fgm_iterate` one restored in
  GUI P0, handler at rfam_gui_server.py:2181, endpoint style at the
  `/api/tools/fgm-iterate` branch); dispatch `elif mode == "fgm_solve"` in
  `_job_worker`; new allowlisted endpoint `/api/tools/fgm-solve`; and an
  optional `line_cb` parameter on `_run_command` (default None, every existing
  caller unchanged) that tails the job log inside the existing 0.2 s poll loop
  so the entry point's SOLVE_PROGRESS lines become job-card progress.
* `webui/static/index.html` (+36). Mode option "FGM Solve (shape fidelity)"
  in the mode select and a `data-mode="fgm_solve"` section: budget, filter
  radius, warm start, plus the honesty affordances from HEATR_GUI_AUDIT.md
  section 3.1 (per-shape geometry-limited note naming L_shape, T_shape,
  cross, star, rectangle and pointing at Orientation/Turntable; the grid
  qualifier; the smoke-class warning for reduced budgets).
* `webui/static/app.js` (+29). `buildPayload` branch for `fgm_solve`, a
  dedicated-endpoint launch branch in `launchRun` (the fgm_iterate pattern),
  and a `modeTag` entry so auto output names read `<shape>_fgmsolve_<date>`.

Map format contract (citations): the emitter reproduces the
`np.savez_compressed` key set of `fgm_generator.py:679-696` exactly
(level_map, sat_map, x_mm, y_mm, width_mm, height_mm, bpp, n_levels,
magnitude, baseline_saturation, dead_band, proxy_field, invert, dpi), with
magnitude 1.0 and baseline 0.5 so the injection loader applies no rescale
(rfam_eqs_coupled.py:383-388). The loader itself is
`rfam_eqs_coupled._FgmFeedback.from_config`, rfam_eqs_coupled.py:366-380.

## Verification evidence

1. **Unit tests (red-green).** `./.venv312/bin/python -m pytest
   test_solve_fgm.py -q` gives 14 passed. Includes the real-data contract
   test: the emitted npz is read by the REAL production loader
   (`rfam_eqs_coupled._FgmFeedback.from_config`, not a copy), decodes to the
   expected printer round trip to 1e-6, and stays within one printer level
   quantum of the delivered map deep inside the part.
2. **Command-line end to end (smoke class, labelled).** Square, budget 8
   forward-equivalents:
   `outputs_eqs/runs/square/fgm_solve/square_solve_smoke_20260801/`. Log shows
   the recipe banner (filter 1.00 mm = 1.983 cells at grid 120, beta 0,
   conductivity only), the smoke-class self-label, warm start AUTO picking the
   library HIST_best, SOLVE_PROGRESS lines, and the deliverable score
   (SOLVE_4bpp J 159.26, IoU 0.8375 at grid 120, stop 750 s HORIZON,
   under-melt 16.25 percent). These are smoke-budget numbers, not quality
   numbers, and the run says so itself. The figure was viewed personally.
3. **Real-engine injection.** The emitted npz was injected through
   `fgm_feedback.saturation_map_npz` into a real `rfam_eqs_coupled.py` run
   (short 120-step config): loader line
   "[fgm_feedback] loaded fgm_square_solve_smoke_20260801_solve_4bpp.npz,
   sat_inside=[0.267, 0.667] mean=0.480", exit code 0. Decoded engine
   saturation vs the delivered map inside the part: max abs difference 0.0
   (bit-exact round trip).
4. **Interface end to end (live server, port 8085).** Mode list served as
   [single, sweep, optimizer, turntable, orientation_optimizer,
   placement_optimizer, shell_sweep, fgm_solve, fgm_iterate, fgm_import,
   prewarp]; the fgm_solve section becomes visible with defaults budget 40,
   radius 1.0, warm start auto; auto output name "square_fgmsolve_20260801".
   Clicking Launch Run created job 20260801-142245-94003c, which ran to
   completed (42 s at budget 4) with the run landing at
   `outputs_eqs/runs/square/fgm_solve/square_fgmsolve_20260801/`. A second
   launch through the raw endpoint captured the live log-tail progress label:
   "Solve 1/1 gradient evals, 2.8/4 forward-equivalents, J=197.2,
   IoU=0.8528" at progress 95 percent. `/api/results` lists the fgm_solve run
   directories, and the Operation page's Live Figures panel picked up
   `solve_map_melt.png` on its own. Screenshots checked personally; no em
   dashes render in the new section.
5. **No-regression checks.** `ast.parse` on rfam_gui_server.py, `node --check`
   on app.js. `_run_command`'s new parameter defaults to None so every
   existing caller is byte-for-byte behavior-identical.

Cleanup: the second interface verification run (square_fgmsolve_progresscheck)
was moved to ~/.Trash. Kept as evidence: the command-line smoke run, the
interface run square_fgmsolve_20260801, and the job logs in
`outputs_eqs/_logs/`. A verification server instance is still listening on
port 8085 (started by this pass; quit it from the interface header or kill the
`rfam_gui_server.py` process when done reviewing).

## Deferred (out of scope, per the audit and the queue)

* The P1 three-intent method chooser ("Quick look" / "Deliverable map
  (SOLVE)" / "Legacy comparison") presented in BOTH FGM creation places, with
  presets, dual-read cards and drive guidance (HEATR_GUI_AUDIT.md sections
  3.1 and 4). This pass adds the solve as a separate mode option; the chooser
  redesign stays deferred.
* The solve dashboard analogous to the convergence dashboard (J sparkline in
  the job card, IoU-vs-historical panel, "Re-simulate with this map" button on
  a solve card). The map already feeds the existing Results-tab re-simulate
  and RIP paths by virtue of the production npz format.
* Grid hold-out (Gate A) and sub-filter-radius perturbation (Gate B) as
  automatic post-solve gates; today they remain campaign scripts
  (`adjoint2d/topopt_robust.py`), and no SOLVED label is claimed by this mode.
* Batch library solves stay a command-line campaign; the interface queues one
  shape per job.
* gt_logo is not solvable (its geometry rasterizer needs cv2, absent from
  .venv312; `library_solve.GT_LOGO_SKIP_REASON`); the interface will return
  the missing-calibrated-config or build error for it.

## Addendum (2026-08-01): production verification pass closes the reporting gap

**The gap.** Every other run mode emits the full standard per-run figure suite
(electric_fields.png, thermal_fields_final.png, rf_summary_v5.png,
paper_style_report.png, validation_report.png, time_series.png, the three
evolution gifs, fields.npz, summary.json), which is exactly what the Results
tab renders. A solve run emitted only the single map+melt check figure, so
Matt's ellipse run through engine v2.0.0 produced one figure instead of the
expected suite, and no standard FGM map figure artifacts.

**The fix (scripts/solve_fgm.py only).** After the 4 bits per pixel
deliverable map is written, the entry point now:

1. Emits the standard FGM (functionally graded material) map figure pair the
   other creation modes ship: `<stem>_preview.png` (white = max ink) and
   `<stem>_meteor_import.png` (pixel inversion for the Meteor raster image
   processor), reproducing the fgm_generator.py:702-744 convention including
   the vertical flip (`emit_map_pngs`).
2. Runs a PRODUCTION VERIFICATION PASS by default: it builds an engine config
   from the solve's own config following the showcase precedent
   (`outputs_eqs/fgm_solve_showcase/triangle_solved_A1_4bpp_stop270.yaml`):
   `fgm_feedback.sat_map_npz_direct` pointing at the deliverable npz (whose
   `sat_map` key is the quantized map at simulation resolution, exactly what
   the direct loader reads, rfam_eqs_coupled.py:390-418), `thermal.n_steps`
   set to the solve's optimal stop (round(t_stop_s / dt_s); the showcase ran
   540 = 270 s / 0.5 s), and the `fgm_solve` block stripped
   (`build_production_verify_config`). It then invokes rfam_eqs_coupled.py as
   a subprocess into `<output-dir>/production_verify/`, asserts the complete
   standard suite exists (`PRODUCTION_SUITE_FILES`, captured from the
   showcase run), scores the production run's final temperature field with
   the solve's own objective functions, and records the deltas in
   results.json under `production_verify` (dJ_rel, dIoU,
   agrees_within_1_percent); disagreement beyond 1 percent is flagged loudly
   in the log (`run_production_verify`).
3. Adds `--skip-verify` for fast iterations. Default is the full pass, so a
   default solve run now lands with the entire standard figure suite from the
   REAL v2.0.0 engine, and the pass doubles as the end-to-end integration
   check on every run.

Files changed in this pass: `scripts/solve_fgm.py` (the three functions above
plus wiring and the flag), `test_solve_fgm.py` (9 new tests), and
`SOLVE_MODE_USAGE.md` (output inventory and the flag). rfam_gui_server.py and
webui/static/ were NOT touched (another agent owns them right now); the
graphical user interface picks the new artifacts up automatically because
they land under the run directory the Results tab already browses.

**Verification.**

1. Red-green test-driven development: the 9 new tests (config builder, map
   figure pair, suite inventory constant) were written first and failed with
   ImportError on the three new names; after implementation
   `./.venv312/bin/python -m pytest test_solve_fgm.py -q` gives 23 passed.
2. End-to-end smoke on Matt's exact case, the ellipse
   (`outputs_eqs/fgm_calibrated_control/configs/ellipse_m0p0500.yaml`,
   budget 4, smoke class and self-labelled as such):
   `outputs_eqs/runs/ellipse/fgm_solve/ellipse_verify_smoke_20260801/`. The
   run directory contains the deliverable npz, both map figure PNGs,
   solve_map_melt.png, solve_maps.npz, results.json,
   production_verify_config.yaml, production_verify.log, and
   `production_verify/` with all 14 standard-suite files, stamped
   engine_version 2.0.0 in used_config.yaml. Solve deliverable:
   SOLVE_4bpp J 31.48, IoU 0.9223 at grid 120, stop 427.5 s.
3. Solve versus production deltas (results.json `production_verify`):
   J 31.4755324 (production) vs 31.4755336 (solve), relative difference
   3.87e-08; IoU 0.9222798 vs 0.9222798, delta 0.0. In the 1e-9 to 1e-4
   class the prior gates established; agrees_within_1_percent true.
4. Figures viewed personally: solve_map_melt.png, the map preview PNG, the
   production thermal_fields_final.png and rf_summary_v5.png all show the
   same ellipse melt region and consistent stop time (428 s).
