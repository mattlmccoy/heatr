# GUI P0 Restoration Notes

**Date:** 2026-07-31. **Scope:** P0 items 1-4 of `HEATR_GUI_AUDIT.md` (restore the
functionally graded material (FGM) launch surface dropped by commit `8946538`, restore the
preset controls, fix stale Generate FGM defaults, remove stale prewarp-disabled strings).
The P1 method chooser was explicitly out of scope. Nothing is committed; all changes are
uncommitted in the working tree for review.

Acronyms, expanded on first use: GUI = graphical user interface. FGM = functionally graded
material. bpp = bits per pixel. API = application programming interface.

---

## 1. Split of work: prior (interrupted) session vs this session

A previous agent session performed the P0 edits and died before verifying or documenting
them. This session found the edits already complete, verified them live, and wrote this
report. Exact split:

**Done by the prior session (found already in the working tree, all uncommitted):**

1. `webui/static/index.html`: re-merged the FGM form blocks from
   `git show 8946538^:webui/static/index.html` into the current file.
   - Mode options `fgm_iterate` ("FGM Iterative Optimize") and `fgm_import`
     ("Import FGM PNG") restored to the mode select (index.html:126-127), with the
     `prewarp` mode that commit 8946538 added preserved (merge, not revert).
   - The full FGM iterate section (index.html:680 onward) and FGM import section
     (index.html:963 onward). Identifier check: all 40 `fgmIter*` / `fgmImport*` /
     `preset*` element ids match the pre-8946538 file exactly (diff of sorted id lists:
     identical). Current count on the live page: 27 `fgmIter*` controls, 9 `fgmImport*`
     controls.
   - Preset controls restored: `presetSelect`, `presetSaveBtn`, `presetLoadBtn`,
     `presetDeleteBtn` (index.html:111-113 area).
   - Also restored `navJobBadge` (index.html:93) and `expCalcChip` (index.html:176),
     the two other elements the audit listed as orphaned.
   - Restored form defaults were adapted to the empirical standard, not just copied:
     correction mode `integral`, magnitude 0.7, 4 bpp, proxy `T_phi90`, n = 12,
     time source `optimizer`. These match the backend defaults at
     `rfam_gui_server.py:2223-2237` and `HEATR_STANDARD_PARAMETERS.md` section 4.
2. `webui/static/results.js` (~2293-2325): the three Generate FGM prompts rewritten.
   bpp default "2" changed to "4" with the campaign rationale; proxy prompt now lists
   `T_phi90` first, recommends it, and defaults to it (previously `Qrf` recommended and
   `T_phi90` not even listed); magnitude default 0.5 changed to 0.7 with the per-shape
   tuned ladder (0.3 / 0.5 / 0.7 / 0.85) in the help text.
3. `rfam_gui_server.py` `/api/tools/generate-fgm` endpoint defaults (now at ~6245):
   `bpp` 2 to 4, `proxy_field` default `T_phi90` (new key), `magnitude` 1.0 to 0.7, with
   the doc comment updated to cite `HEATR_STANDARD_PARAMETERS.md`.
4. Stale strings removed (audit P0 item 4): "Prewarp scripts are intentionally excluded."
   (old index.html:668) and the `main()` startup print "Prewarp flows are disabled in
   this interface." (old rfam_gui_server.py:6998). Verified absent by grep.
5. The prior session also left the GUI server running (pid 27577, port 8080, started
   after its last file edit, so it serves the current code).

**Done by this session:**

- Assessment of the prior state (id-level diff against the pre-8946538 file, handler
  cross-check, git diff review) to confirm nothing was half-merged.
- Cross-check of every `byId("...")` reference in `webui/static/app.js` (221 unique ids)
  against the current index.html: one miss, `fgmIterSourceRun` (app.js:884), which never
  existed in the old file either and is guarded by `if (!sel) return;`. No markup change
  needed; no handler signatures drifted, so no adaptation edits were required.
- The full live verification below, including one real launch smoke test.
- Cleanup of all verification artifacts (moved to `~/.Trash`, listed in section 3).
- This report.

**Untouched by both sessions (pre-existing uncommitted work, preserved as required):**

- The Results-tab performance pass in `rfam_gui_server.py` (`_scan_run_files` single-walk
  scan, cached manifest `_manifest_for_listing`, archive-pruned walk, hero-thumbnail
  prewarm) and the deferred image-list changes in `webui/static/results.js`
  (`ensureImages`, lazy gallery). These remain in the diff exactly as found.
- `rfam_eqs_coupled.py` modifications (min_T_part history, `stop_after_phi_bar` early
  stop). These belong to a different workstream (see `test_min_T_part_history.py`) and
  were not touched.
- `webui/static/app.js`: not modified by either session; the restored markup reconnects
  the existing handlers (payload builder app.js:721-790, presets app.js:1977-2040).

---

## 2. Verification evidence (live server, port 8080)

The user interface work is not unit-testable and no JavaScript test runner exists in this
repository (no package.json), so the concrete verification gate below substitutes for
red-green tests, per the audit's own suggestion of a live-launch smoke test.

1. **Page loads, no console errors.** Operation and Results pages both loaded; the
   browser console error log was empty on both.
2. **Restored modes and form present.** Live page inspection returned mode options
   `[single, sweep, optimizer, turntable, orientation_optimizer, placement_optimizer,
   shell_sweep, fgm_iterate, fgm_import, prewarp]`; 27 `fgmIter*` and 9 `fgmImport*`
   controls in the document; all four preset controls present.
3. **FGM iterate form defaults (live values after selecting the mode):**
   `corrMode=integral, magnitude=0.7, bpp=4, proxy=T_phi90, nIter=12,
   timeSource=optimizer`. All three fgm_iterate mode sections became visible.
4. **FGM import section** becomes visible on selecting Import FGM PNG, with all nine
   controls including the file input `fgmImportFile`.
5. **Real launch smoke test (well-formed request).** Clicking Launch Run in fgm_iterate
   mode launched job `20260731-232654-b6031f`: the server accepted the payload, created
   `outputs_eqs/runs/square/fgm_iterate/square_run_20260731/`, generated
   `configs/_gui_generated/20260731-232654-b6031f_..._opt_probe_optimizer.yaml`, and
   started the FGM optimizer probe (job card showed "FGM opt probe: running to 6.00 min
   ceiling", progress "0/12", matching the n = 12 default). The job log shows the forward
   solver stepping normally (HEATR_PROGRESS lines to step 45 of 720). The job was then
   cancelled through `/api/job/control` (log ends "FGM optimizer probe exited with code
   -15", the termination signal) so no compute was wasted.
6. **Preset save/load round-trip.** With the name prompt stubbed: Save created
   "p0_verify_preset" in the dropdown and in localStorage key `heatr_presets`;
   `fgmIterMagnitude` was then mutated to 0.3 and Load restored it to the saved 0.85.
   The test preset was removed afterward.
7. **Generate FGM prompt defaults (Results tab, live capture).** Clicking a run card's
   FGM button with `window.prompt` instrumented captured, in order:
   bpp prompt default "4"; proxy prompt "Proxy field (T_phi90 | Qrf | T | rho_rel):"
   default "T_phi90"; magnitude prompt "Gradient magnitude (per-shape tuned; campaign
   values 0.3 / 0.5 / 0.7 / 0.85):" default "0.7". The third prompt was cancelled so no
   request fired.
8. **Endpoint defaults end to end with real data.** POST `/api/tools/generate-fgm` with
   only `{"output_dir": "outputs_eqs/fgm_dosecheck/square_baseline_voltage"}` returned
   `ok=true, bpp=4, proxy_field=T_phi90, magnitude=0.7` and wrote a valid 4 bpp map
   (13 unique levels, 1715x1715). A control probe against an older run whose fields.npz
   predates the `T_phi90` field failed with "proxy_field='T_phi90' not found", further
   confirming the default is now `T_phi90` (older runs must pick `Qrf` in the prompt).

---

## 3. Cleanup performed

All verification artifacts were moved to `~/.Trash` (not deleted): the smoke-test run
directory `square_run_20260731`, its generated optimizer config yaml, the three
generate-fgm outputs (npz, json, preview png) plus one `_meteor_import.png` side file in
`outputs_eqs/fgm_dosecheck/square_baseline_voltage/`, and the localStorage test preset.
The job log `outputs_eqs/_logs/gui_job_20260731-232654-b6031f.log` was left in place as
evidence. The server was left running as found.

## 4. Changed and unchanged files

Changed (all uncommitted, ready for review; edits by the prior session, verified here):
- `webui/static/index.html` (FGM iterate + import sections, mode options, presets,
  navJobBadge, expCalcChip, stale footer line removed)
- `webui/static/results.js` (Generate FGM prompts; also carries the pre-existing
  performance pass)
- `rfam_gui_server.py` (generate-fgm endpoint defaults, stale startup print removed;
  also carries the pre-existing performance pass)
- `GUI_P0_RESTORATION_NOTES.md` (this file, new, by this session)

Unchanged: `webui/static/app.js`, `fgm_generator.py`, `rfam_eqs_coupled.py` (other
workstream), everything under `.claude/worktrees/` (never touched).

## 5. Deferred items (out of P0 scope)

- P1 FGM Method chooser (audit section 3.1) and solve-mode integration.
- P1 standard-parameter server-side presets, drive-mode promotion, dual read-state and
  energy-residual display in run cards.
- P2 guardrails: grid > 200 warning, turntable near-continuous preset, HEATR-3D
  magnitude field, pre-launch FGM suitability hint.
- Older runs lack the `T_phi90` field in fields.npz, so the new default fails on them
  with a clear error message; a graceful fallback offer (retry with `Qrf`) would be a
  small P2 usability follow-up.
