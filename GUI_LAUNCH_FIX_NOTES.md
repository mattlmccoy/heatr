# GUI launch fix notes (2026-08-01)

## Root cause, one sentence

Launch Run stopped queueing because a long-running rfam_gui_server.py
process predating the solve-mode integration (commit 4deff91) was still
serving stale python routes underneath the new static files it reads from
disk per request, so the promoted launch paths posted to routes the old
process did not have; there is no defect in the current code, and a fresh
server queues every mode.

## Evidence for the root cause

1. Reproduction against current code failed to fail: with the current
   server process and a freshly loaded page, all five launch paths were
   exercised live in the browser (single, fgm_solve, fgm_iterate, prewarp
   via Launch Run; fgm_import has its own button and validation path).
   Every one returned HTTP 202, appeared in the queue user interface, and
   the single-mode job created its run directory
   (outputs_eqs/runs/square/single/experimental/square_single_20260801)
   and stepped normally before being cancelled. Zero console errors on the
   Operation page throughout.
2. The skew mechanism is concrete: `git show` confirms commit 0d1f6a1 (the
   P0 restoration the reporter's server likely dated from) has zero
   references to the /api/tools/fgm-solve route, while the current
   webui/static/app.js posts fgm_solve launches there. Static files are
   read from disk per request, python routes are frozen in the running
   process, so old process plus new page equals launches that 4xx with no
   job created (no job log is written for a rejected launch, matching the
   absence of any failed-launch logs in outputs_eqs/_logs/).
3. Timeline corroboration: the GUI server is supervised by the Research
   Toolkit Hub (research-toolkit/hub/app.py, port 5050), which relaunched
   rfam_gui_server.py fresh at 21:34 on 2026-08-01; immediately afterward a
   real user launch (POST /api/tools/fgm-solve at 21:35:49, job
   ellipse_fgmsolve_20260801) queued and ran successfully. The symptom did
   not survive the restart.

Note on interpreters: the hub launches the server with the system Python
3.14 framework, not ./.venv312. Both interpreters run the server and the
engine today; this is recorded here as an observation, not changed.

## Fix

Since the failure mode is process hygiene rather than a code defect, the
minimal durable fix is a server/page handshake that makes the skew
announce itself instead of silently breaking launches:

- rfam_gui_server.py: new integer constant `API_GENERATION = 20260801`
  (bump on any route or launch-payload shape change), served by both
  /api/meta and /api/engine-version via `_engine_version_info()`.
- webui/static/app.js: `EXPECTED_API_GENERATION = 20260801` pinned next to
  a `_checkApiGeneration(meta)` call in `loadMeta()`. On mismatch or
  absence (any pre-handshake server) a red sticky banner appears:
  "Stale server process: ... Restart rfam_gui_server.py, then reload this
  page."
- Test-driven development, red first: test_api_generation.py (3 tests)
  failed with AttributeError before the constant existed, then passed.
  The client/server agreement is itself pinned by a test that parses
  app.js for the expected value.
- Browser check: `_checkApiGeneration({api_generation: 123})` rendered the
  banner with the correct text; a matching generation removed it; the live
  page against the current server shows no banner.

## Legacy-pass inventory (Task 2)

Directive: promote the v2 standards, delete nothing, keep legacy fully
functional, defaults stay the v2 standards. Optgroup labels "v2 standard"
and "legacy" used wherever the widget is a select; a middle group holds
options that are neither. Contract pinned red-first by
test_gui_legacy_marking.py (6 tests, all failed before the markup change,
all pass after). Live page verification: every reordered dropdown was read
back from the rendered document object model with zero console errors.

Every dropdown touched, before and after order:

1. Run mode (`select#mode`, webui/static/index.html)
   - Before: single, sweep, optimizer, turntable, orientation_optimizer,
     placement_optimizer, shell_sweep, fgm_solve, fgm_iterate, fgm_import,
     prewarp (flat list).
   - After: optgroup "v2 standard": fgm_solve (deliverable map), single,
     turntable (dwell program capable); optgroup "studies and tools":
     sweep, optimizer, orientation_optimizer, placement_optimizer,
     shell_sweep, fgm_import, prewarp; optgroup "legacy": fgm_iterate
     "(legacy)". Default remains Single Exposure via an explicit selected
     attribute (fgm_solve now sits first in the list without becoming the
     landing default).

2. Functionally graded material (FGM) method chooser
   (`#fgmMethodChooser` cards, index.html)
   - Before: Quick look, Deliverable map (SOLVE, RECOMMENDED), Legacy /
     comparison (iterate).
   - After: Deliverable map (SOLVE, RECOMMENDED) first, Quick look second,
     Legacy / comparison (iterate) last. No card removed; behavior wiring
     in app.js untouched.

3. Proxy field (`select#fgmIterProxy`, index.html)
   - Before: T_phi90 (selected), T, Qrf, rho_rel, Thorough,
     Regime-adaptive (flat list).
   - After: optgroup "v2 standard": T_phi90 (selected, default unchanged);
     optgroup "research alternates": T, rho_rel, Thorough,
     Regime-adaptive; optgroup "legacy": Qrf "(legacy)".

4. Correction mode (`select#fgmIterCorrMode`, index.html)
   - Before: integral (selected, "recommended"), proportional, hybrid.
   - After: optgroup "v2 standard": proportional (calibrated quick-look
     class); optgroup "research alternates": hybrid; optgroup "legacy":
     integral "(legacy)", which KEEPS the selected attribute because it is
     the legacy-comparison mode's own historical default (changing the
     default inside the legacy path would alter its A/B meaning).

5. Drive mode (`select#advEnforceGen`, enforce_generator_power,
   index.html)
   - Before: Use base config, true, false (bare labels).
   - After: Use base config (default unchanged), then
     "false - voltage drive with per-shape V_cal (v2 standard)", then
     "true - generator power scaling (legacy)". Values unchanged.

6. Turntable dwell program (`select#turntableProgramSelect`, index.html
   plus `_loadTurntablePrograms` in app.js)
   - Before: static "none (fixed-step mode)" first, program options
     appended after it by app.js.
   - After: app.js inserts the fetched programs inside an optgroup
     "v2 standard (dwell programs)" ABOVE the static option, which is
     relabeled "none - fixed-step rotation (legacy)" and keeps its
     selected attribute (no program is ever auto-picked). A field-help
     line states the v2 standard for asymmetric parts. Verified live: the
     optgroup lists the six deliverable/control programs plus the two
     square programs, and the select value defaults to the empty string.

7. Import bits per pixel (`select#fgmImportBpp`, index.html plus the
   app.js fallback)
   - Before: 2 bpp selected first, 4 bpp second; app.js fallback "2".
   - After: "4 bpp (16 levels, v2 standard)" selected first,
     "2 bpp (4 levels, printer-constrained)" second; app.js fallback "4".
     This closes the one remaining 2 bpp default (the P0 pass had already
     moved fgm_iterate and the Results-tab prompts to 4 bpp).

Left alone on purpose (no legacy axis): fgmIterTimeSource (manual is not
legacy), fgmIterBpp (4 bpp already selected; 2 bpp is a printer
constraint), freqProfile (13.56 MHz is untested, not superseded),
fgmSolveWarmStart, shape/shell/physics selects. The per-shape standards
JSON (webui/static/fgm_shape_standards.json) and its hint wiring are
untouched.

## Verification summary

- Test suite: 59 passed
  (test_api_generation.py, test_gui_legacy_marking.py,
  test_gui_server_v2.py, test_fgm_shape_standards.py,
  test_engine_version.py, test_min_T_part_history.py,
  test_turntable_program.py, test_solve_fgm.py), plus `node --check` on
  app.js and results.js.
- Fresh server: the hub-relaunched 21:34 instance postdates every python
  edit in this pass; /api/meta serves api_generation 20260801 and the page
  shows no stale banner.
- Default-path launch end to end: job 20260801-214135-88763f (single,
  square, 0.2 min) queued via the Launch Run button, rendered in the
  Active Jobs panel (screenshot captured; queue position 1 behind a real
  user solve job), then cancelled cleanly. Run-directory creation for the
  identical /api/run path was demonstrated by the earlier smoke
  (square_single_20260801, directory created, solver stepping, cancelled).
- Legacy launch smoke: fgm_iterate (the "(legacy)"-marked mode) job
  20260801-214207-ba739c queued through the same button and was cancelled,
  proving legacy stays fully functional.
- Zero console errors on the Operation page after all changes.
- The real user job running during verification
  (ellipse_fgmsolve_20260801) was never paused, cancelled, or restarted,
  and the server was deliberately NOT restarted to protect it.

## Changed files (all uncommitted, for review)

- rfam_gui_server.py (API_GENERATION constant, _engine_version_info
  carries it)
- webui/static/app.js (EXPECTED_API_GENERATION + _checkApiGeneration
  banner; dwell-program optgroup insertion; import bpp fallback 4)
- webui/static/index.html (the seven dropdown/chooser changes above)
- test_api_generation.py (new)
- test_gui_legacy_marking.py (new)
- HEATR_V2_ROLLOUT_NOTES.md (addendum section appended)
- GUI_LAUNCH_FIX_NOTES.md (this file, new)

Untouched: rfam_eqs_coupled.py, results.js, all engine and campaign code,
.claude/worktrees/, deck_gifs/.

## Cleanup

- Smoke jobs cancelled (20260801-214135-88763f, 20260801-214207-ba739c);
  their queue entries are cheap metadata. The earlier reproduction jobs
  (21:24 to 21:27) were cancelled the same way; the square_single_20260801
  run directory and its log remain as evidence and can be moved to
  ~/.Trash after review.
- No verification server was started by this pass; the only running
  server is the hub-managed instance on port 8080 serving the user's live
  job.
