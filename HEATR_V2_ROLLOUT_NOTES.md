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
