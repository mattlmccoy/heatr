# HEATR v2 rollout notes (2026-08-01)

Directive: standardize and VERSION the HEATR 2-D engine and surface it in the
graphical user interface (GUI). Nothing committed; all changes sit in the
working tree for review. `.claude/worktrees/` and
`fgm_solve_campaign/adjoint2d/seq_dwell*.py` untouched. No dissertation file
touched. Test-driven development (TDD) red-first for every pure-logic change;
the GUI (not unit-testable) was verified live in the browser and over the
HTTP API, stated per check below.

Acronyms: FGM = functionally graded material (spatially varying dopant
saturation map). bpp = bits per pixel. IoU = intersection over union.

## What changed, by file

### Engine versioning (rfam_eqs_coupled.py)

- `ENGINE_VERSION = "2.0.0"` and `ENGINE_VERSION_NAME = "HEATR 2D v2.0"`
  constants with the v2.0 changelog comment and the versioning-policy note,
  rfam_eqs_coupled.py:63-90 (right after R_GAS). One constant pair, single
  source of truth; Matt can rename the string there.
- `stamped_config_echo(cfg)` (rfam_eqs_coupled.py:92-97): deep-copy config
  echo carrying engine_version, used by `save_outputs` for used_config.yaml
  (rfam_eqs_coupled.py:4415-4417).
- `dual_read_state_from_hist(hist)` (rfam_eqs_coupled.py:100-139): engine-side
  dual read-state sigma_T (heating-peak, melt-onset), pinned against the
  campaign extractor `outputs_eqs/geometry_dual_readstate/dual_readstate.py`
  by test. Melt never reached => `sigma_T_melt_reached: false` and null
  melt-onset (the loud fallback, never a silent final-state read).
- Every `run_sim` summary now carries: engine_version, engine_version_name
  (summary head, rfam_eqs_coupled.py:3775-3777), the four sigma_T dual
  read-state keys, and `energy_residual_frac_final` (|final residual| /
  final integrated dose; rfam_eqs_coupled.py:3856-3864). `save_outputs` also
  setdefaults the version so every summary path is stamped
  (rfam_eqs_coupled.py:4418-4420).
- Tests: `test_engine_version.py` (5 tests, red observed first:
  AttributeError on ENGINE_VERSION). Includes a real tiny `run_sim`
  (grid 40, 12 steps) asserting the stamped summary.

### CHANGELOG_ENGINE.md (new, repo root)

v2.0.0 entry naming the standardized capabilities with commit refs
(0d1f6a1, 4deff91, 2408329, b04e356, eaf3aec, 2cc1548; all six resolved in
git log) and the versioning policy (minor = new modes/actuators, patch =
fixes, major = physics/convention changes).

### Per-shape standards JSON (new)

- `scripts/analysis/build_fgm_shape_standards.py` generates
  `webui/static/fgm_shape_standards.json` (19 shapes). Sources, all read
  from the real campaign artifacts, never re-typed: v_cal from
  `outputs_eqs/geometry_dual_readstate/<shape>.json`; quick-look gain +
  verdict parsed from the FGM_WINDOW_RESELECTION.md winners table (the
  campaign's record of that reselection); solve class from
  `fgm_solve_campaign/out_lib/<shape>.json` verdict block, with
  MATCHED = census SOLVED that does not beat the stored historical mask
  (square, rounded_rect); cross dwell upgrade IoU read from
  `out_dwell/cross_deliverable_moves_engine_program_gate.json` (0.9829);
  star rotation upgrade from `out_rot/star_headline.json` (0.953).
- Tests: `test_fgm_shape_standards.py` (6 tests, red first:
  ModuleNotFoundError). Data-contract pins: square v_cal 2428.1732..., cross
  gain 1.1855, class mapping incl. MATCHED, gt_logo nullable (not run,
  cv2 absent).

### GUI server (rfam_gui_server.py)

- `_engine_version_info()` (above `_summary_excerpt`): reads the constants
  from rfam_eqs_coupled once; no hardcoded duplicate.
- `GET /api/engine-version` (do_GET, next to /api/ping) and engine_version
  merged into `GET /api/meta`.
- `GET /api/turntable-programs`: lists executable dwell programs
  (`fgm_solve_campaign/out_dwell/*_turntable_*.json`, gate results excluded).
- `_summary_excerpt` extension: engine_version passthrough (absent = pre-v2),
  sigma_T_heating_peak_c / sigma_T_melt_onset_c / sigma_T_melt_reached, and
  energy_err_pct (v2 fraction key, with a pre-v2 fallback from the raw
  residual/dose J-per-m pair). This closes the audit finding that the run
  cards' err gauge could never render.
- `_configure_turntable` program mode: payload `turntable_program_json`
  (repo-relative, must exist, path-traversal guarded) becomes
  `turntable.program_json` with `corotate_dopant` and
  `corotate_eps_geometry` both defaulting ON in program mode (the dielectric
  ghost costs 8 to 21 percentage points of J on asymmetric parts,
  ENGINE_DWELL_SUPPORT_NOTES.md section 5.2; engine-side default stays OFF
  so no non-GUI number moves). Legacy fixed-step payloads produce the exact
  pre-change config block (pinned by test).
- Tests: `test_gui_server_v2.py` (8 tests, red first: KeyError /
  missing-branch failures observed; 2 legacy-behavior tests were green
  before the change by design).

### Front end (webui/static/)

Not unit-testable; verified live (evidence below). Promote, never remove:
no control was deleted anywhere.

- index.html: header engine-version badge (id engineVersionBadge); per-shape
  standard-parameter row under the Shape select (button + info line);
  three-intent FGM method chooser shown for the FGM modes (Quick look /
  Deliverable map SOLVE, default-highlighted RECOMMENDED / Legacy
  comparison iterate) with the per-shape hint line; grid >= 200 warning
  under grid_nx/ny; drive-mode guidance line above Advanced Parameters;
  turntable "Dwell program" advanced panel (program select + path input +
  both co-rotation flags, eps co-rotation tooltip citing the dielectric
  ghost). results.html: same header badge.
- app.js: `_setEngineVersionBadge` (fed from /api/meta),
  `_loadShapeStandards` + per-shape hint/info updaters,
  `_applyShapeStandard` (grid 120, enforce_generator_power false, V_cal
  voltage, T_phi90 proxy), `_updateGridWarning`, `_loadTurntablePrograms`,
  chooser wiring (SOLVE/iterate cards switch the mode select; quick-look
  card explains the Results-tab path with this shape's calibrated gain);
  turntable buildPayload sends program + co-rotation keys only when a
  program is chosen.
- results.js: engine-version chip on every run card (green v-chip from the
  run's stored engine_version; absent renders a gray "pre-v2" chip); metric
  line now shows sigma_T@peak, sigma_T@melt (or "melt not reached"), and
  the energy residual percent with a warning above 5 percent; the
  Generate FGM magnitude prompt prefills the window-reselected calibrated
  gain for the run's shape from fgm_shape_standards.json.

### Standard doc

`HEATR_V2_STANDARD.md` (new, repo root): production solve recipe, dwell
composability per shape class, power scheduling retired, projection retired
(MMA retained in the toolbox), quoting rules, standing checks; every
decision cites its source report. `HEATR_STANDARD_PARAMETERS.md` linked,
not edited.

## Verification evidence

1. Unit tests, red observed first for every new behavior:
   `./.venv312/bin/python -m pytest test_engine_version.py
   test_gui_server_v2.py test_fgm_shape_standards.py -q` = 19 passed.
   Neighbor regression: `test_min_T_part_history.py test_turntable_program.py
   test_solve_fgm.py` = 31 passed (includes the turntable bit-identity
   guard, which re-runs the fixed-step engine).
2. Syntax gates: `ast.parse` on rfam_gui_server.py and rfam_eqs_coupled.py
   (via test import), `node --check` on app.js and results.js.
3. Live server (port 8091, fresh instance): /api/engine-version returns
   {"engine_version": "2.0.0", "engine_version_name": "HEATR 2D v2.0"};
   /api/meta carries the same; /api/turntable-programs lists the six
   L_shape/T_shape/cross deliverable + control programs.
4. Browser (screenshots viewed personally while the pane was visible):
   header badge "engine v2.0.0" green pill; the standard-parameter row
   showing "V_cal 2428.2 V, grid 120, voltage drive, T_phi90, 4 bpp" for
   square. The Browser pane later reported visibilityState hidden (pane
   closed on the user side; screenshots black), so the remaining checks
   were DOM-verified in the live page: chooser visible in fgm_solve mode;
   cross hint reads "NOT-RESCUED (deliverable IoU 0.675 at grid 120) ...
   dwell upgrade ... IoU 0.9829"; Standard parameters click set grid 120,
   V_cal 2815.4, enforce false, proxy T_phi90; grid warning turns on at
   240; program select populated. Console: zero errors on Operation and
   Results pages.
5. Run cards: /api/results-runview shows 764 cards, all historical cards
   render the "pre-v2" chip, 543 of 764 now carry energy_err_pct from the
   pre-v2 fallback (the err gauge was previously never fed). Fresh smoke
   run: single mode, square, grid 40, 0.2 min, output
   `outputs_eqs/runs/square/single/baseline/square_v2smoke_tinybudget_20260801`
   (labeled tiny-budget; not a quality run). On disk: summary.json has
   engine_version 2.0.0 / "HEATR 2D v2.0", sigma_T_heating_peak_c 5.791,
   sigma_T_melt_reached false with null melt-onset (the loud fallback:
   0.2 min never melts), energy_residual_frac_final 1.3e-13;
   used_config.yaml echoes engine_version 2.0.0. The card excerpt from
   /api/results-runview carries all of those, and the rendered Results page
   shows the green v2.0.0 chip with metric line "sigma_T@peak 5.79 C,
   melt not reached, res 0.00%" (DOM-verified; zero console errors).
6. Existing modes untouched: fgm_iterate fresh launch accepted
   (job 20260801-203746-747cca queued) and then cancelled cleanly;
   fixed-step turntable payload produces the exact legacy config block
   (unit-pinned); `_run_command`, solve mode, and all other launchers
   unmodified in this pass.

## Deferred (named, not silently dropped)

- Solve-completion dashboard (J sparkline, IoU vs historical panel) from
  HEATR_GUI_AUDIT.md section 3.2: the solve job already streams
  forward-solve progress; the dashboard remains open.
- Server-side named preset YAMLs (audit P1 item 6): the per-shape standard
  button covers the standard-parameter intent; full preset files deferred.
- Pre-v2 dual read-state backfill: pre-v2 cards show only the energy
  residual; computing sigma_T for old runs needs a time_series.json read
  per run (cost on the Dropbox-backed tree) and is left to the backfill
  tooling.
- HEATR-3D magnitude field and near-continuous turntable preset chip
  (audit P2 items 11-12).
- gt_logo standards row is nullable (cv2 absent from .venv312).

## Cleanup

- Verification server on port 8091 left running for review (quit from the
  GUI header or kill the rfam_gui_server.py process).
- The smoke run and the cancelled launch-check job are labeled and cheap;
  move to ~/.Trash after review if unwanted.
- Planning file: heatr_v2_rollout_plan.md (throwaway, delete after review).
