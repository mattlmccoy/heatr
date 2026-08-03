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

## Addendum (2026-08-01, launch-fix + legacy-marking pass)

Appended by the follow-up session; the sections above are unchanged. Full
detail in GUI_LAUNCH_FIX_NOTES.md at the repo root.

1. "Launch Run queues nothing" root-caused as stale-server skew, not a
   code defect: a server process predating the solve-mode integration was
   serving old python routes underneath the new static files, so promoted
   launches posted to routes the old process lacked. Fresh server: all
   launch modes queue. Guard added: integer API_GENERATION handshake
   (rfam_gui_server.py constant, served in /api/meta and
   /api/engine-version; EXPECTED_API_GENERATION pin plus a red
   stale-server banner in app.js). Tests: test_api_generation.py
   (3 tests, red first).
2. Legacy-marking pass over the Operation-tab dropdowns per Matt's
   directive (promote the newer standards, delete nothing): run mode,
   FGM method chooser card order, proxy field, correction mode, drive
   mode (enforce_generator_power), turntable dwell-program select, and
   import bits per pixel now list v2-standard options first with
   optgroup labels "v2 standard" / "legacy" and " (legacy)" suffixes on
   superseded options. All values, handlers, and the per-shape standards
   JSON hints untouched; defaults unchanged except fgm_import bpp which
   now defaults to the 4 bpp standard. Tests:
   test_gui_legacy_marking.py (6 tests, red first). Combined suite after
   both passes: 59 passed; zero console errors live.

---

## Addendum (2026-08-03, v2.1.0 wiring pass)

Appended by the follow-up session; every section above is unchanged. This pass
wired the three decisions that CHANGELOG_ENGINE.md had been carrying as
"Planned v2.1.0 (adopted by verdict, NOT yet wired into production paths)",
added the resolve-at-deployment-grid guidance alongside them, and bumped
`ENGINE_VERSION` to 2.1.0. Test-driven development red-first on every pure-logic
change; the two paths that are not unit-testable (a full solve run, the
production-verification engine subprocess) were verified by a real end-to-end
run, quoted below.

Acronyms, expanded here because this file is read on its own: J_phi = the
melt-region shape-fidelity objective. J_asym = the asymmetric
dense-if-and-only-if-in-bounds objective. MMA = the method of moving asymptotes
(Svanberg 1987). L-BFGS-B = limited-memory Broyden-Fletcher-Goldfarb-Shanno
with box constraints. IoU = intersection over union. bpp = bits per pixel.
g0 = the objective gradient at the solve's start point.

Evidence tags: PROVEN = unit-tested red-first, or reproduced exactly against a
stored run. COMPUTED = measured from a real run in this pass. ASSUMED = a
choice carried forward, not measured here.

### What changed, by file

**New solver modules** (`fgm_solve_campaign/adjoint2d/`):

- `stop_rule.py`: the read state. `validate_stop_rule`, `select_stop` (pure),
  `dual_stop` (both argmins over one stored trajectory plus both cross-read
  values). Production constants `W_OUT_PRODUCTION = 2.0`,
  `W_IN_PRODUCTION = 1.0`, `FLOOR_RHO_REL_PRODUCTION = 0.85`, each with the
  report section that set it named in the docstring.
- `optimizer_policy.py`: `objective_class` and `resolve_optimizer`. Constrained
  (hinge / asymmetric) class defaults to MMA, smooth melt-region class to
  L-BFGS-B, an explicit configuration value always wins, and the source of the
  choice (`policy` or `config`) is returned so the run record can carry it.
- `objective_scale.py`: `ObjectiveRescale.from_start_gradient(...).apply(J, g)`,
  the 1/|g0| rescale with its guards (zero or non-finite start gradient falls
  back to the identity with a stated reason rather than producing an infinity).

**`scripts/solve_fgm.py`**:

- `FgmSolveConfig` gains `stop_rule`, `w_out`, `density_floor_rho_rel`,
  `optimizer`, `objective_rescale`, all validated, plus a read-only
  `map_objective` property fixed at `j_phi`. There is deliberately NO config key
  that switches the map driver: the verdict is that the melt objective drives
  the MAP and the asymmetric objective owns the STOP, and a run must not be able
  to become a different experiment through a config key.
- `plan_grid_transfer` and `stored_map_grid`: the resolve-at-deployment-grid
  guidance, plus the `--resolve-native` command-line flag.
- `run_solve`: reads the deliverable at the selected stop
  (`metrics_at_selected_stop`), branches to `topopt_stage.StageRunner` with an
  `mma.MMA` state when the resolved optimizer is MMA, and wraps the objective in
  the rescale. Rows keep RAW J; only the optimizer sees the scaled pair.
- `results.json` gains `engine_version`, `recipe.stop_rule`,
  `recipe.stop_rule_w_out`, `recipe.stop_rule_density_floor_rho_rel`,
  `recipe.map_objective`, `recipe.optimizer`, `recipe.objective_rescale`,
  `recipe.scoring_forward_full_horizon`, `recipe.start.grid_transfer`, a
  top-level `read_state`, and a `stop` block inside every arm.

**`rfam_eqs_coupled.py`**: `ENGINE_VERSION = "2.1.0"`,
`ENGINE_VERSION_NAME = "HEATR 2D v2.1"`, and the version comment block rewritten
from "PLANNED" to what was actually wired. No other line changed; the engine's
forward physics, discretization and material laws are untouched.

**Docs**: `CHANGELOG_ENGINE.md` (the "Planned v2.1.0" section replaced by a real
v2.1.0 entry), `SOLVE_MODE_USAGE.md` (four new sections: the read state, the
optimizer, the objective rescale, and grid mismatch with `--resolve-native`).

### One finding that came out of the wiring, and it changes cost

COMPUTED. The shape-fidelity early stop truncates the march 250 stored steps
after the MELT argmin, and the asymmetric argmin sits LATER than the melt argmin
on every arm DENSE_IFF_INBOUNDS_REPORT.md measured (43 to 160 stored steps at
w_out = 1). A truncated trajectory can therefore pin the asymmetric argmin at
the truncation instead of at the physics. Under `stop_rule: j_asym` every
SCORING forward now runs the full horizon; the SOLVE-LOOP forwards keep the
early stop, so the budget accounting is bit-identical to v2.0.x and the only
extra cost is one forward for the uniform reference. A stop that lands on the
last stored step of an early-stopped trajectory is flagged loudly as a
truncation rather than an argmin.

### Test gate

PROVEN. 37 new tests, every one observed failing first with
`ImportError: cannot import name '<module>' from 'adjoint2d'` or with the named
attribute missing:

- `adjoint2d/tests/test_stop_rule.py`, 13 tests: the frozen production
  constants, both selections, the always-recorded dual stop, the signed stop
  gap, the guard flags, and against a synthetic trajectory the two argmins, the
  two cross-read values, the legacy reproduction and the monotonicity that a
  harder out-of-bounds price never stops later.
- `adjoint2d/tests/test_objective_scale.py`, 12 tests: the factor, the three
  guards, the immutable record, the reparameterization properties (same
  constant on J and g, direction preserved to roundoff with exact signs,
  ordering of two candidates preserved), the smooth-case regression, and the
  rail-stall case.
- `adjoint2d/tests/test_optimizer_policy.py`, 12 tests: classification, both
  per-class defaults, the override in both directions, `auto` equals no
  override, and a stated non-empty reason on every resolution.
- `test_solve_fgm.py`: 12 new v2.1.0 config-key tests, 8 new grid-transfer and
  stored-map-grid tests, and the slow legacy-reproduction test.
- `test_engine_version.py`: the version pin moved to 2.1.0, plus a new test that
  makes it impossible for the changelog to still say "Planned v2.1.0" while the
  constant already reads 2.1.0.

Suites: `adjoint2d/tests` **483 passed, 1 failed** (the failure is
`test_eqs_assembly.py::test_assembled_solve_is_bit_identical_to_production`,
PRE-EXISTING and already recorded in HOLDOUT_FOLLOWUP_REPORT.md Section 2;
re-confirmed here by reverting `rfam_eqs_coupled.py` to its committed state and
watching it fail with the identical value). Root `test_*.py` **261 passed, 1
failed, 25 errors**; the failure is
`test_turntable_program.py::test_fixed_step_turntable_run_is_bit_identical_to_the_pre_edit_engine`
(T drifts 1.56e-12 against the stored baseline) and it too reproduces
BYTE-FOR-BYTE with `rfam_eqs_coupled.py` reverted, so it is pre-existing and not
this pass; the 25 errors are the documented `fixture 'run_dir' not found`
collection errors that `pytest.ini` records as deliberately left visible.

### The end-to-end run under the new defaults

COMPUTED. `scripts/solve_fgm.py --config
outputs_eqs/fgm_calibrated_control/configs/ellipse_m0p0500.yaml --budget 40`,
grid 120, conductivity-only channel, 4 bpp deliverable, production verification
pass ON. Output:
`outputs_eqs/runs/ellipse/fgm_solve/ellipse_v210_wiring_20260803/`. 17 gradient
evaluations, 702 s wall.

**The dual stop of the delivered 4 bpp map:**

| read state | index | stop, s | J_phi there | J_asym there | at horizon |
|---|---|---|---|---|---|
| j_phi (the v2.0.x rule) | 1012 | 506.5 | 11.4372 | 0.15501 | no |
| **j_asym (the v2.1.0 default, DELIVERED)** | **1039** | **520.0** | **14.1027** | **0.14455** | no |

Gap **+27 stored steps, that is +13.5 s**. Reading at the asymmetric argmin buys
**6.7 percent of J_asym** (0.15501 to 0.14455) and pays **23.3 percent of J_phi**
(11.44 to 14.10), which is exactly the trade the objective was built to express
and exactly why the two numbers are both recorded rather than one replacing the
other.

Deliverable at the delivered stop: **IoU 0.9634, growth 2.69 percent of the part
cell count, under-melt 1.08 percent**, against the uniform control's J 232.62 and
IoU 0.6957 at its own stop of 237.5 s.

**The three wirings, visible in that run's own record.** `engine_version`
"2.1.0"; `recipe.stop_rule` "j_asym" with `stop_rule_w_out` 2.0 and
`stop_rule_density_floor_rho_rel` 0.85; `recipe.map_objective` "j_phi";
`recipe.optimizer` resolved to **lbfgsb with source "policy"** (the smooth class
default, logged with its measured reason); `recipe.objective_rescale`
**applied true, |g0| = 14.5187, factor 0.0688766**;
`recipe.scoring_forward_full_horizon` true; `recipe.start.grid_transfer.action`
"use_map_at_its_native_grid" (map and configuration both grid 120).

**The production verification pass agrees, on the new stop.** The real engine
re-ran the delivered map for n_steps 1040 (520.0 s / 0.5 s), emitted the
complete standard figure suite, and scored **J 14.1027 against the solve's
14.1027 (2.97e-07 relative) and IoU 0.9634 against 0.9634 (delta exactly 0)**,
stamped **engine v2.1.0**. The print-package fragment emitted and passes
`scripts.package_fragment.validate_fragment` with zero problems, as do the
fragments of the two alternate-branch smoke runs.

### The flag-off identity claim, proven twice

PROVEN, and the claim is stated exactly rather than inflated.

1. **Unit level, against a stored v2.0.1 run.** The delivered 4 bpp map of
   `outputs_eqs/runs/ellipse/fgm_solve/ellipse_fragment_smoke_20260802`
   (`production_verify.engine_version` reads 2.0.1) was re-marched with the
   current code and read under `stop_rule: j_phi`. **Stop index 854, stop time
   427.5 s and IoU 0.9222797927461139 reproduce EXACTLY.** J reproduces to
   1.1e-13 relative, and the reason is measured rather than assumed:
   `solve_maps.npz` archives the map in float32, which perturbs saturation by up
   to 2.8e-08; feeding the archived float32 map straight in reproduces J to
   7.4e-10 relative, and restoring the exact 16 printable levels (k / 15, which
   is what the run itself marched) closes it to 1.1e-13, that is the
   accumulation noise of a 1500-step double-precision march.
2. **End to end.** A full `scripts/solve_fgm.py` run of the same configuration
   under `stop_rule: j_phi`, `optimizer: mma`, `objective_rescale: false`
   delivered **J 31.48, IoU 0.9223, stop 427.5 s**, matching that stored v2.0.1
   run to every printed digit.

### The MMA branch, exercised and reported honestly

COMPUTED. `optimizer: mma` at budget 14 on the ellipse ran 6 design updates
through `topopt_stage.StageRunner` and did NOT improve on its warm start
(best J 31.00 at evaluation 1; the follow-ups read 64.67, 43.12, 83.66, 44.22,
65.47). That is the expected behaviour of the policy rather than a defect: the
melt-region objective is the SMOOTH class, where the measurement says L-BFGS-B
stays the default, and MMA without the globally convergent inner loop is not a
descent method. The branch is wired and works; it is simply not the default
here.

### What is proven, computed and assumed

**PROVEN**: every pure-logic behaviour listed under the test gate; that the
1/|g0| rescale cannot move the minimizer of a smooth box-constrained case
(same converged design to 1e-9 per variable) while un-stalling the rail case
(0 iterations and the design still on the rail without it, the exact minimizer
with it); that `stop_rule: j_phi` reproduces a stored v2.0.1 run's stop exactly.

**COMPUTED**: every number in the end-to-end run section; the pre-existing
status of both failing tests, re-measured by reverting the one engine file this
pass touched.

**ASSUMED, and how it bites**: (1) `w_out = 2.0` is calibrated on a READ-STATE
trade curve that re-read maps SOLVED at w_out = 1
(DENSE_IFF_INBOUNDS_REPORT.md Section 6 caveat and Section 11 assumption 1); a
map solved at w_out = 2 has still not been run, which is precisely why this pass
changes the STOP and not the MAP. (2) The floor of 0.85 relative density was
swept on ONE shape (the hexagon). (3) The out-of-bounds term is charged at the
read state rather than as a running maximum over time, although bed fusing is
irreversible; that remains the one known modelling error in the objective.
(4) Everything here is the two-dimensional engine at grid 120, and no number
transfers to another grid without the hold-out gate.

## Addendum (2026-08-03, deferred-list closure + schedule co-solve commissioning)

Appended by the follow-up session; the sections above are unchanged. This
pass CLEARS the deferred list of the 2026-08-01 rollout and makes schedule
co-solves commissionable from the graphical user interface (GUI).
Test-driven development red-first on every pure-logic change (each new test
file observed failing with ModuleNotFoundError before its module existed);
front-end changes verified live in the browser with screenshots viewed
personally. Promote never remove: no control, route, or option was deleted.

### Deferred item (a): solve-progress dashboard

- rfam_gui_server.py: `_solve_progress_fields` / `_schedule_progress_fields`
  parse the machine-readable progress lines (SOLVE_PROGRESS from
  scripts/solve_fgm.py, SCHEDULE_PROGRESS from the new wrapper);
  `_record_solve_progress` stores structured fields on the job
  (`job["solve_progress"]`: latest forward-equivalents spent/budget, current
  J, running best J, running best intersection over union (IoU), capped
  trace, `SOLVE_TRACE_CAP` 400), serialized by /api/jobs.
- webui/static/app.js: `_buildSolveProgressPanel` renders the panel on
  fgm_solve and schedule_cosolve job cards: labeled numbers (FE spent/budget
  with a fill bar, J now, J best, IoU best) plus a log-scale J sparkline
  once the trace has two points.
- Verified live: a labeled tiny-budget square fgm_solve
  (square_dashboard_smoke_20260803, budget 5, smoke class) rendered
  "FE 2.8 / 5, J now 168.11, J best 168.11, IoU best 0.8287" on its job
  card (screenshot viewed personally).

### Deferred item (b): server-side preset YAML files

- scripts/build_shape_presets.py regenerates presets/<shape>.yaml (19 files)
  from webui/static/fgm_shape_standards.json (single source; never edit the
  YAML by hand). `rfam_gui_server._load_shape_presets` reads the directory;
  GET /api/presets serves it in the exact structure the front end already
  consumed. app.js `_loadShapeStandards` now fetches /api/presets FIRST and
  falls back to the static JSON on a pre-preset server.
- Tests: test_presets_yaml.py (6, red first), including a drift pin that
  fails when presets/ and the standards JSON disagree.
- Verified live: /api/presets 200 with source "presets_yaml"; the square
  standard-parameter hint (V_cal 2428.2 V) rendered from the YAML-backed
  payload; the page's network log shows it loading /api/presets.

### Deferred item (c): pre-v2 sigma_T backfill

- scripts/backfill_sigma_t.py walks run directories, computes the dual
  read-state sigma_T from stored time_series.json via the engine's own
  `dual_read_state_from_hist` (data contract probed first: time_series.json
  carries ui_rms_part / mean_T_part_c / mean_phi_part), and stamps
  summary.json with `sigma_T_backfilled: true`. Never overwrites an existing
  sigma field (byte-identical skip), skips and counts unrecoverable runs,
  prunes _archive, honors --dry-run / --subset / --limit.
- POST /api/tools/backfill-sigma-t queues it as a job (payload dry_run
  defaults TRUE); the counts land on the job card and in
  `job["backfill_counts"]`.
- Tests: test_backfill_sigma_t.py (6, red first).
- Real runs: full dry-run census scanned=1396, backfillable=1376,
  already_present=3, no_time_series=17, unrecoverable=0, read_errors=0.
  Real subset run on runs/square/single/baseline: written=3,
  already_present=1; the stamped no_rotation summary now carries
  sigma_T_heating_peak_c 11.047 with melt-onset null and
  sigma_T_melt_reached false (the loud fallback), and the Results card
  excerpt renders it. The full-tree real pass (1376 writes on the
  Dropbox-backed tree) is deliberately left for Matt to trigger:
  `./.venv312/bin/python scripts/backfill_sigma_t.py` or the endpoint with
  {"dry_run": false}.

### Deferred item (d): near-continuous turntable preset chip

- webui/static/index.html + app.js: a chip in the Turntable section fills
  the fixed-step fields with the stored near-continuous class (15 degrees
  per step, 48 events = two full turns, interval auto-spaced; source
  configs/diamond_tt_15deg_48rot_nearcont.yaml). The tooltip names the
  15-degree rotation-remap energy caveat (CONTINUOUS_ROTATION_REPORT.md
  section 6.1). The dwell-program select is untouched. Verified live:
  clicking set 15 / 48 and the info line appeared.

### Schedule co-solve commissioning

- New GUI mode "Schedule co-solve (map + turntable program)" in the
  v2-standard optgroup: schedule mode (indexed N positions / asymmetric
  dwell / sequential), indexed position count (2 to 24), budget in
  forward-equivalents, optional label. POST /api/tools/schedule-cosolve
  validates the payload with the same argv builder the job uses (400 on a
  bad shape/mode/budget), then queues mode schedule_cosolve.
- NEW wrapper scripts/solve_schedule.py shells the existing campaign
  drivers read-only: indexed -> run_rot_avg_solve (matched N-position
  averaged kernel), asym_dwell -> run_dwell_solve, sequential ->
  run_seq_arms (L_shape/T_shape only; its screen inputs are copied into the
  run folder so the campaign originals are never touched). Each driver's
  output-directory constant is redirected into the run's campaign_raw/;
  no campaign artifact is overwritten and no driver file was edited.
- Outputs land in outputs_eqs/runs/<shape>/schedule_cosolve/<id>/:
  summary.json (engine version stamped, run_type schedule_cosolve, smoke
  class label under budget 20), schedule_maps.npz (the co-solved 4 bits per
  pixel map), turntable_program_deliverable.json plus
  turntable_program_equal_dwell_control.json (dwell campaign format),
  j_trace.json, fig_schedule_cosolve.png (map + melt-versus-nominal +
  objective trace). The Results tab picks the folder up;
  `_detect_run_type_from_summary` now honors an explicit run_type key
  (pre-v2 classification unchanged, pinned by the existing suites).
- Honest labeling, stamped into every summary: schedules execute on the
  part-frame march at solve time; the engine turntable program mode
  (rfam_eqs_coupled tt_program_mode) is the execution path for
  verification. The Results tab shows a "Verify on engine" button on
  schedule_cosolve cards that runs the emitted deliverable program through
  POST /api/run mode=turntable with both co-rotation flags on.
- Tests: test_solve_schedule.py (12, red first): wrapper argument building
  and cross-field validation, the budget-to-evaluations convention
  (40 forward-equivalents = 16 evaluations), the driver-log progress watch,
  the SCHEDULE_PROGRESS round trip, and the server-side builder rejections.
  The wrapper module imports without the heavy solver stack.
- API_GENERATION bumped 20260801 -> 20260803 together with the app.js pin
  (route additions; test_api_generation.py green).

### Verification evidence

1. Suites owned by this pass: 97 passed
   (test_presets_yaml.py, test_backfill_sigma_t.py, test_solve_schedule.py,
   test_api_generation.py, test_gui_server_v2.py, test_gui_legacy_marking.py,
   test_engine_version.py, test_fgm_shape_standards.py,
   test_min_T_part_history.py, test_solve_fgm.py), every new behavior
   observed red first; `node --check` clean on app.js and results.js.
2. Fresh server (heatr-gui-workbench, port 8090): /api/engine-version
   carries api_generation 20260803; no stale banner; zero console errors on
   the Operation AND Results pages after all changes.
3. End-to-end schedule co-solve commissioned from the GUI: cross, indexed 4
   positions, budget 5 forward-equivalents, labeled smoke
   (outputs_eqs/runs/cross/schedule_cosolve/cross_sched_cosolve_smoke_20260803,
   job 20260803-114117-dd42a1). Result: J 94.34 against uniform 128.71,
   IoU 0.8935 against uniform 0.8768, warm start won, recommended stop
   370.5 s, wall about 34 minutes. SMOKE CLASS: budget 5 is not a quality
   solve and the summary labels itself; the quality run is 40
   forward-equivalents. The live dashboard showed evals 4/4, FE 10/10
   (2 evaluations per solve stage at this budget), J and best-IoU running
   values during the run.
4. Program validity: the emitted deliverable program compiles through the
   engine's own `parse_turntable_program` (151 events, strictly increasing
   sentinels, correct incremental deltas); moves carry
   {position_deg, dwell_s, move_at_s} exactly as the dwell campaign format.
5. Engine-verify follow-up exercised through the REAL button (prompt
   stubbed to 0.5 minutes): job 20260803-114452-86ac57 completed; its
   used_config.yaml shows turntable.program_json pointing at the co-solve's
   emitted program with corotate_dopant and corotate_eps_geometry true, and
   the run landed in
   runs/cross/turntable/baseline/cross_sched_cosolve_smoke_20260803_engineverify.
6. Legacy modes untouched: a single-mode launch
   (square_legacy_launch_smoke_20260803, 0.2 min) queued through /api/run,
   ran to completion, and created its run directory. No legacy route,
   handler, or dropdown changed in this pass.
7. Viewed personally: the solve dashboard panel screenshot and
   fig_schedule_cosolve.png for both the standalone wrapper smoke and the
   GUI-commissioned smoke.

### Known conditions

- rfam_eqs_coupled.py is mid-edit by the parallel engine lane (v2.1.0 in
  the working tree). Its bit-identity guard
  (test_turntable_program.py::test_fixed_step_turntable_run_is_bit_identical...)
  fails at 1.6e-12 temperature drift against the stored pre-edit baseline.
  That file is outside this pass's ownership and was not touched here.
- The /api/heatr3d/* routes and webui/static/heatr3d.html were not touched
  (owned by the 3-D tab rebuild lane).

### Cleanup

- The workbench verification server on port 8090 is left running for
  review; the hub-managed instance on 8080 was never touched and needs a
  restart (or hub relaunch) to pick up the new routes before the page stops
  showing the stale-server banner there.
- Labeled smoke artifacts kept as evidence, cheap to trash after review:
  runs/square/single/baseline/square_legacy_launch_smoke_20260803,
  runs/square/fgm_solve/square_dashboard_smoke_20260803,
  runs/cross/schedule_cosolve/cross_sched_cosolve_smoke_20260803, and
  runs/cross/turntable/baseline/cross_sched_cosolve_smoke_20260803_engineverify.
