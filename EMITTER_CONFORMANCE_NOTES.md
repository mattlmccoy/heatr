# Print-Package Fragment Emitter: Conformance to Frozen Schema 2.0.0

Date: 2026-08-02. Lane: 2-D solve (geo-prewarp).
Contract: docs/superpowers/specs/2026-08-02-grade-and-print-design.md,
section 7b, at commit e8600d1 ("PACKAGE FORMAT SCHEMA 2.0.0: FROZEN
2026-08-03 ... any further change is a schema version bump, not an edit").
All spec line numbers below are as shown by
`git show e8600d1:docs/superpowers/specs/2026-08-02-grade-and-print-design.md`.

## What was built

Every shape-fidelity solve run (`scripts/solve_fgm.py`) now emits
`package_fragment.json` alongside its existing outputs. The fragment is the
2-D lane's contribution that the Studio Phase 4 emitter folds into a full
schema 2.0.0 print package; it is not itself a package manifest.

Files:

- `scripts/package_fragment.py` (new): builder, validator, emitter.
- `scripts/solve_fgm.py` (edited): calls `package_fragment.emit_fragment`
  after the final results.json write; new optional `--intake-v2-json` flag.
- `test_package_fragment.py` (new): 25 tests, red-first, spec line numbers
  cited in comments; real-data fixtures captured from the existing ellipse
  smoke run (`outputs_eqs/runs/ellipse/fgm_solve/ellipse_verify_smoke_20260801`).

## Fragment schema and the frozen-field mapping

Top level of `package_fragment.json`:

| Fragment field | Frozen 7b field | Spec lines | Source of the value |
|---|---|---|---|
| `fragment_kind`, `target_schema_version` ("2.0.0"), `created_utc`, `emitted_by`, `source_run` | fragment bookkeeping (folding metadata for the Phase 4 emitter) | 240-242 | the run itself |
| `engine_versions` | manifest `engine_versions` | 243-247 | `{"heatr_2d": rfam_eqs_coupled.ENGINE_VERSION}`; the constant is READ at emit time (`rfam_eqs_coupled.py:86`, currently "2.0.1"), never hardcoded |
| `plan.classifier` | optional `plan` block: classifier recommendation, advisory disclaimer, classifier version | 250-253 | when `--intake-v2-json` names a geometry intake actuator classifier version 2 record: `geometry_actuator.recommend_v2` on the stored residual spectrum (`fgm_solve_campaign/out_intake/<shape>_v2.json`), `Recommendation.as_json()` (carries `classifier_version` 2) plus the `ADVISORY_DISCLAIMER` from `studio_handoff/print_package.py`. When the run did NOT come through intake (the default for a command-line solve): `{"present": false, "reason": ...}`, the explicit absent-with-reason record |
| `plan.expected_outcomes` | expected outcomes (J and IoU, intersection over union) with the grid qualifier | 253 | the deliverable arm of results.json (`arms[deliverable_arm]`: J, J_raster_chi, IoU), the grid read from the map npz `sat_map` shape, and the run's own `grid_qualifier` string verbatim; the budget note is carried when the run is smoke-test class |
| `correction_provenance.engine` | engine value | 254-256 | "heatr_2d_solve" for this entry point (the adjoint-solve channel); `engine_for_method` also maps "proportional" to "heatr_2d_proportional" for the calibrated-inverse channel |
| `correction_provenance.artifacts` | artifact ids/hashes | 257 | sha256 of the deliverable map npz bytes, plus the run's results.json |
| `correction_provenance.trust_badge` | trust badge string | 257 | built by `trust_badge()`: deliverable candidate at its own grid only, no SOLVED label (Gate A grid hold-out and Gate B sub-filter blur not run by this entry point), 2-D model result, sim-only (FROZEN_CONVENTIONS_2D.md sections 8 and 9) |
| `correction_provenance.transfer` | THREE-state transfer record | 257-261 | `{"state": "transfer_not_applicable", "statement": ...}`. Our printer rasters are native to the production path: the printer-resolution level map comes directly from the fgm_generator dots-per-inch resample (fgm_generator.py:588-606, 720 DPI dots per inch, Meteor native); there is no volume-to-grid transfer step and no dopant-mass move to measure. Stated explicitly because absence never implies it |
| `power_settings` | EXACTLY ONE of power_density_w_per_m3 or the 2-D voltage block | 264-266 | the voltage block from the run configuration's `electric` section (`voltage_v`, `voltage_mode`), `rf_mode` "constant" retained, with the HEATR v2 standard note. The validator rejects a fragment carrying both channels or neither |
| `production_verify` | real record or explicit not-run | 236-237, 274-276 | when the verify pass ran: `{"run": true}` plus the full record from results.json (engine version stamp of the actual verification run, J/IoU deltas, `agrees_within_1_percent`, suite file list). When `--skip-verify` was used: `{"run": false, "statement": ...}` including the recorded reason. A missing record is treated as not run and says so in the statement |

`validate_fragment` encodes these rules as machine checks and
`emit_fragment` refuses to write an invalid fragment.

## Evidence

- Data contract probed before building: section 7b read at commit e8600d1;
  `rfam_eqs_coupled.py:86` (`ENGINE_VERSION = "2.0.1"`); the calibrated
  config's `electric.voltage_v` / `electric.voltage_mode`; the real
  `fgm_solve_campaign/out_intake/ellipse_v2.json` residual spectrum;
  `geometry_actuator.py:238-247` (`Recommendation.as_json`, version 2 at
  line 600); the real ellipse smoke results.json.
- Red-first: `test_package_fragment.py` was written and run before
  `scripts/package_fragment.py` existed; it failed with
  `ImportError: cannot import name 'package_fragment' from 'scripts'`,
  then went green with no test edits.
- Test result: 48 passed (25 new fragment tests plus the 23 existing
  solve_fgm tests, unchanged).
- Real-data fixtures: the fragment tests build from the captured
  `ellipse_verify_smoke_20260801` artifacts, not invented dicts; the
  artifact hash test recomputes the map npz sha256 from the file bytes.
- End-to-end: a fresh labeled smoke run
  (`outputs_eqs/runs/ellipse/fgm_solve/ellipse_fragment_smoke_20260802`,
  budget 4 forward-equivalents, verify pass ON) emitted
  `package_fragment.json`; `validate_fragment` on the emitted file returns
  no problems. Field inventory in the final report of this session.

## Limits

- The fragment covers the five blocks this lane owns. The Studio Phase 4
  emitter owns the rest of the 2.0.0 manifest (part/intake verdict,
  densify_summary, raster, turntable, scheduler, files).
- The classifier part of the plan block is present only when the caller
  passes `--intake-v2-json`; a command-line solve does not come through
  the geometry intake path, and the fragment says so explicitly rather
  than borrowing a per-shape record that this run never consulted.
- A budget-4 run is smoke-test class, not a quality solve; the fragment
  carries that budget note inside `plan.expected_outcomes`.
