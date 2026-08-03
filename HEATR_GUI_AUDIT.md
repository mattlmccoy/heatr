# HEATR GUI Audit: alignment with current practice, promotion plan

**Date:** 2026-07-31. **Scope:** audit only, no code changed. Audited **from source only**
(server `rfam_gui_server.py`, front end `webui/static/`, git history); the server was not
started live because other agents are active in this tree and the finding set was fully
determined by source inspection. All paths relative to the repo root
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/`.

Acronyms, expanded on first use: GUI = graphical user interface. FGM = functionally graded
material (a spatially varying dopant saturation map). EQS = electro-quasi-static.
bpp = bits per pixel. ILT = inverse-lithography-technology-style level-set boundary prewarp.
GA = genetic algorithm. OC-TO = optimality-criteria topology optimization.
EBC = energy-budget-conserving exposure rescale. DSC = differential scanning calorimetry.
IoU = intersection over union. STL = stereolithography mesh file format.

---

## Verdict

**PARTIALLY ALIGNED, with one outright regression.** The server backend supports every
workflow the project runs today, including the full FGM iterate machinery with integral,
proportional, hybrid, and OC-TO modes. But the Operation tab lost its entire FGM launch
surface in commit `8946538` ("recover dropped April recovered local work"): the mode
options `fgm_iterate` and `fgm_import`, the roughly 30-control FGM form section, and the
Save/Load Preset controls were all dropped from `webui/static/index.html` while
`webui/static/app.js` kept the (now dead) handlers. Today FGM is reachable only through
the Results tab, and the one FGM entry point a user will actually find there (the
"Generate FGM" button) prompts with defaults that contradict the project standard on all
three axes: it suggests 2 bpp instead of 4, proxy `Qrf` instead of `T_phi90`, and
magnitude 0.5 flat instead of the per-shape tuned values. Separately, the pinned standard
parameters of `HEATR_STANDARD_PARAMETERS.md` (voltage drive for FGM comparisons, dual
read-state reporting, the grid ceiling, the energy-residual gate) are not presented,
defaulted, or displayed anywhere in the interface. Nothing needs to be removed; the fix is
restoration plus promotion.

---

## 1. Inventory: what the GUI exposes today

### 1.1 Server run modes (job dispatch `rfam_gui_server.py:4176-4196`)

| Mode string | Launcher | Front-end entry point today |
|---|---|---|
| `single` | `_launch_single_mode`, rfam_gui_server.py:1872 | Operation tab mode select, webui/static/index.html:110 |
| `sweep` (exposure sweep) | `_launch_sweep_mode`, rfam_gui_server.py:3564 | index.html:111, sweep minutes field index.html:167-171 |
| `optimizer` (exposure t\*) | via `_launch_single_mode` config, rfam_gui_server.py:1892 | index.html:112, phi snapshots / ceiling / highlight index.html:173-183 |
| `turntable` | via config, rfam_gui_server.py:1894 | index.html:113, rotation deg / events / interval index.html:185-196 |
| `orientation_optimizer` | via config, rfam_gui_server.py:1888 | index.html:114, full grid+refine controls index.html:198-245 |
| `placement_optimizer` | via config, rfam_gui_server.py:1890 | index.html:115, GA controls index.html:247-330 |
| `shell_sweep` | `_launch_shell_sweep_mode`, rfam_gui_server.py:3612 | index.html:116, thicknesses/shapes index.html:332-342 |
| `prewarp` (ILT boundary) | `_launch_prewarp_mode`, rfam_gui_server.py:1792 | index.html:117, controls index.html:344-373 |
| `fgm_resimulate` | `_launch_fgm_resimulate_mode`, rfam_gui_server.py:1920 | Results tab only: "Re-simulate with FGM" button, webui/static/results.js:163-217 |
| `fgm_iterate` | `_launch_fgm_iterate_mode`, rfam_gui_server.py:2180 | **NO fresh-start entry.** Only "Continue FGM" from an existing run, results.js:1489 -> `/api/tools/fgm-continue` (rfam_gui_server.py:6505). app.js:721-790 still builds the payload for a mode option that no longer exists in index.html |
| `fgm_gradient_descent` | `_launch_fgm_gradient_descent`, rfam_gui_server.py:3822 | **No front-end caller at all** (endpoint rfam_gui_server.py:6474; only a prose mention in results.js:3365) |
| `antennae_calibrate` / `antennae_size_sweep` | rfam_gui_server.py:3709 / 3763 | Antenna Workshop modal, index.html:692-791 |
| `backfill_reports` | `_run_backfill_reports_job`, rfam_gui_server.py:4888 | Results tab backfill buttons, results.js:2273-2286 |

### 1.2 FGM options exposed vs supported

| FGM capability | Backend | Front end |
|---|---|---|
| Correction mode integral / proportional / hybrid (OC-TO phase 2) | payload `use_delta_correction`, `move_limit`, `sensitivity_filter_sigma`, `use_hybrid`, rfam_gui_server.py:2267-2305 | Lost. The `fgmIterCorrMode` select existed pre-8946538 (old index.html:795-800, section 617-830) with integral as recommended default; current index.html has none of it |
| Magnitude, decay, floor, dead band, momentum | rfam_gui_server.py:2229-2241 | Lost (old `fgmIterMagnitude` etc.) |
| 2 vs 4 bpp | rfam_gui_server.py:2233; generate endpoint rfam_gui_server.py:6212 | Results tab prompt, **default "2"**, results.js:2293-2296. Old Operation form defaulted to 4 bpp with rationale (old index.html:777-793) |
| Proxy field incl. `T_phi90`, thorough schedule, regime-adaptive | rfam_gui_server.py:2234, 2244-2262 | Results tab prompt lists only `Qrf | T | rho_rel` and recommends Qrf, results.js:2303-2310. `T_phi90` (the standard) is not even offered in the prompt text, though the endpoint accepts it |
| EBC exposure rescale (`use_ebc`, clamps) | rfam_gui_server.py:2327-2330 | Never had a control (payload-only) |
| Overprint cold zone, perturbation, stagnation, regression threshold | rfam_gui_server.py:2264-2292 | Lost with the form |
| Saturation-map `.npz` direct injection | `fgm_feedback` block via `_launch_fgm_resimulate_mode`, rfam_gui_server.py:1993-2005 | Exposed: "Re-simulate with FGM" in the FGM preview modal, results.js:192-217, and "Use this FGM" in the convergence dashboard, results.js:698-700 |
| Import external FGM PNG | endpoint `/api/tools/import-fgm-png`, rfam_gui_server.py:6275 | Lost (`fgmImport*` controls existed pre-8946538, old index.html:96) |
| FGM-to-RIP layer export | `/api/tools/fgm-to-rip`, rfam_gui_server.py:6595 | Exposed, results.js:170-180, 232-251 |
| FGM convergence dashboard (per-iter sigma_T, thumbnails, best-iter reuse) | `/api/convergence/`, rfam_gui_server.py:5657 | Exposed and good, results.js:268-845 |
| Uniform-shape gate | `MIN_SIGMA_T_FOR_FGM = 4.0` C, rfam_gui_server.py:2798 | Server-side only; user gets it as an abort message, not pre-launch guidance |

### 1.3 Physics and drive controls (Operation tab)

- Frequency profile select with calibrated-27.12 vs untested-13.56 warning: index.html:552-558. Good, already promoted.
- Drive: `voltage_v` index.html:563, `generator_power_w` index.html:564,
  `enforce_generator_power` tri-state index.html:565-571, transfer efficiency index.html:572.
  All buried in the collapsed "Advanced Parameters" panel (index.html:543).
- Grid `grid_nx/ny` index.html:546-547, no upper bound, no melt-instability warning.
- Physics Model panel (DSC-calibrated PA12 vs baseline) with full phase, densification,
  crystallization parameters: index.html:403-541.
- Materials (virgin, powder, doped incl. `sigma_profile`, temperature and density
  coefficients): index.html:582-646.
- Geometry size override with lock-aspect: index.html:124-149. Shell geometry: index.html:383-401.
- YAML Inspector (effective merged config preview): index.html:652-664. Good reproducibility aid.
- Base config override select: index.html:151-155.

### 1.4 Results tab and HEATR-3D tab

- Run cards, starring, backfill, profile/fields viewers, run-type detection:
  rfam_gui_server.py:5643-5924, results.js throughout.
- **Results-tab performance pass is already done and uncommitted in this tree**: single
  `os.walk` scan `_scan_run_files` (rfam_gui_server.py:1460, see `git diff rfam_gui_server.py`)
  plus results.js pagination edits (54 lines changed). Do not redo.
- HEATR-3D page: shape/STL select webui/static/heatr3d.html:140-165, grid n 32-64
  heatr3d.html:172, FGM mode `none | melt | density` heatr3d.html:174-177, densify toggle
  heatr3d.html:182, exposure heatr3d.html:186, stop rho heatr3d.html:188. The job payload
  (webui/static/heatr3d.js:101) sends **no `magnitude`**, although `heatr3d_job.py` accepts
  it; 3-D FGM magnitude is therefore fixed at its default from the page.

### 1.5 Dropped by commit 8946538 (still alive in app.js, dead in index.html)

Element-identifier diff of `git show 8946538^:webui/static/index.html` vs current: all 24
`fgmIter*` controls, all 9 `fgmImport*` controls, `presetSaveBtn` / `presetLoadBtn` /
`presetDeleteBtn` / `presetSelect` (handlers still present at app.js:1977-2040, localStorage
based), `navJobBadge`, `expCalcChip`. The same commit added the prewarp mode section, so a
plain revert is not the fix; the FGM and preset blocks must be re-merged into the current file.

---

## 2. Gap analysis against current practice

Evidence base: `HEATR_STANDARD_PARAMETERS.md` (pinned defaults, section 4, lines 78-102),
`outputs_eqs/geometry_dual_readstate/GEOMETRY_DUAL_READSTATE.md` (the standardized campaign:
voltage drive, grid 120, one-shot proportional FGM at 4 bpp, proxy `T_phi90`, per-shape
magnitude from {0.3, 0.5, 0.7, 0.85}, dual read states, energy-residual gate),
`outputs_eqs/fgm_calibrated_control/`, `configs/diamond_tt_15deg_48rot_nearcont.yaml`,
`scripts/analysis/` drivers.

| # | Capability the project actually runs | Status | Evidence |
|---|---|---|---|
| 1 | FGM iterate, integral mode, magnitude 0.7, 4 bpp, `T_phi90`, n=12 (the empirically-best default) | **RUNNABLE ONLY FROM CLI / API** for a fresh start; Results-tab "Continue FGM" only from an existing run | Backend defaults are correct (rfam_gui_server.py:2223-2237) but the launch form is gone (section 1.5) |
| 2 | One-shot proportional inverse with calibrated per-shape gain (the dual-readstate campaign method) | EXPOSED but **mis-defaulted**: prompt pushes 2 bpp / Qrf / 0.5 | results.js:2293-2325 vs GEOMETRY_DUAL_READSTATE.md line 3 (4 bpp, T_phi90, m in {0.3..0.85}); endpoint defaults also stale (bpp=2, Qrf, mag 1.0, rfam_gui_server.py:6169-6173) |
| 3 | Saturation-map npz direct injection (fgm_feedback resimulate) | EXPOSED (Results tab) | results.js:192-217; rfam_gui_server.py:1920-2027 |
| 4 | 2 vs 4 bpp selection | EXPOSED, wrong default (2) in the only surviving control | results.js:2294; standard says "bpp stated (2 or 4)" with 4 the campaign norm, HEATR_STANDARD_PARAMETERS.md:87 |
| 5 | Drive mode: voltage drive for FGM comparisons, enforced power for absolute dose | EXPOSED but buried and unguided | index.html:563-571 in collapsed Advanced panel; the rule that enforced-power renormalization "cancels the mechanism under test" (HEATR_STANDARD_PARAMETERS.md:84) appears nowhere in the UI; per-shape calibrated voltages (V_cal 2428-3688 V, GEOMETRY_DUAL_READSTATE.md G1 table) have no preset |
| 6 | Dual read states (heating-peak AND melt-onset sigma_T) | **NOT RUNNABLE / NOT DISPLAYED**: `_summary_excerpt` surfaces only max/mean T, phi, rho, t (rfam_gui_server.py:5052-5086); no heating-peak extraction exists server-side; the campaign used the standalone `outputs_eqs/geometry_dual_readstate/dual_readstate.py` extractor | HEATR_STANDARD_PARAMETERS.md:61-74 mandates reporting both |
| 7 | Energy-residual standing gate on every solve | NOT DISPLAYED in run cards or job completion | HEATR_STANDARD_PARAMETERS.md:86; field exists in summary.json (`energy_balance_residual_final_J_per_m`) but `_summary_excerpt` drops it |
| 8 | Grid ceiling (never 240 for melt-state work; 120 validated, 160 stable) | EXPOSED without guardrail | index.html:546-547 has `min="16"`, no max, no warning; HEATR_STANDARD_PARAMETERS.md:83 |
| 9 | Turntable, incl. near-continuous practice (15 deg x 48 events) | EXPOSED; fields accept the practice values but default is 90 deg x 1 and there is no near-continuous preset | index.html:185-196; configs/diamond_tt_15deg_48rot_nearcont.yaml |
| 10 | Orientation optimizer | EXPOSED, aligned | index.html:198-245 |
| 11 | Standard-parameter presets ("intended GUI defaults", the doc's own words) | **NOT RUNNABLE from GUI**: preset UI removed; nothing encodes the standard | HEATR_STANDARD_PARAMETERS.md:9-10; app.js:1977-2040 orphaned |
| 12 | Import external FGM PNG | RUNNABLE ONLY FROM API (endpoint alive, form dead) | rfam_gui_server.py:6275; old index.html:96 |
| 13 | FGM gradient descent (inner-loop) | RUNNABLE ONLY FROM API | rfam_gui_server.py:3822, 6474; no js caller |
| 14 | Per-node conductivity law / two-sided actuation (current branch `feat/pernode-twosided-tuning`) | NOT RUNNABLE from GUI (research-stage, CLI drivers `scripts/analysis/run_pernode_square.py`) | acceptable for now; list under de-emphasized future work |
| 15 | Dose-control / drive-audit analyses | CLI only (`scripts/analysis/dose_control_2d.py`, `drive_audit.py`) | acceptable: analysis passes, not run launches |
| 16 | Run-output browsing performance | EXPOSED, already optimized (uncommitted) | git diff rfam_gui_server.py `_scan_run_files`; do not redo |
| 17 | heatr3d FGM magnitude | NOT SETTABLE from the 3-D page | heatr3d.js:101 payload omits `magnitude`; `heatr3d_job.py` supports it |
| 18 | Shape-fidelity SOLVE (new, about to land) | NOT YET IN GUI (expected) | see section 3 |

---

## 3. New-feature alignment: presenting the FGM method choice

Source for the solve: `.claude/worktrees/agent-a02efc1141ba69c58/SHAPE_LIBRARY_SOLVE_REPORT.md`.
What it produces: a per-cell dopant map from a gradient solve of the shape-fidelity
objective J = sum (phi - part mask)^2, quantized to a **4 bpp printable map in the
production `fgm_generator.py` convention**, re-verified through the real forward solver,
about **40 forward solves per shape** (L-BFGS-B, 15 gradient evaluations typical), with an
optional second printing pass triggered when more than 15 percent of the part is left
unmelted (worth it on L_shape and T_shape). It beats the best stored historical mask on
13 of 18 shapes and reaches absolute IoU >= 0.95 on 7; it flags T/L/cross/star as needing
an orientation or turntable actuator instead.

### 3.1 Proposed presentation: one "FGM Method" chooser, three named intents

Replace the current scattered entry points with a single method radio group that appears
in BOTH places a user creates an FGM (the restored Operation-tab FGM mode, and the
Results-tab per-run FGM button), phrased by intent, not by algorithm:

1. **"Quick look" (seconds, 0 extra solves)** = proportional inverse, calibrated gain.
   Defaults: proxy `T_phi90`, 4 bpp, baseline 0.5, dead band 0.05, magnitude pre-filled
   per shape from the dual-readstate calibration table {0.3, 0.5, 0.7, 0.85} keyed by the
   baseline sigma_0 bands already documented in the dropped help text (old
   index.html:711-730). Caption: "instant map from this run's fields; good for previews
   and printer tests, not the deliverable."
2. **"Deliverable map (SOLVE)" (about 40 forward solves, tens of minutes)** = the
   shape-fidelity solve, marked **Recommended for printing** and made the visually
   primary option. Caption states what the user gets: "gradient-solved 4 bpp printable
   map, re-verified through the forward model; best known method on 13 of 18 library
   shapes." Two honesty affordances, both grounded in the report: (a) a per-shape
   fitness note pulled from the census classes (SOLVED / IMPROVED / NOT RESCUED), so
   selecting L_shape, T_shape, cross, star, or rectangle shows "this shape is
   limited by geometry, consider Orientation Optimizer or Turntable instead", mirroring
   the report's actuator finding; (b) the double-pass option surfaced as a checkbox
   ("second printing pass, helps L and T shapes only").
3. **"Legacy / comparison (iterative)" (n forward solves)** = the existing fgm_iterate
   integral mode with its current correct defaults (magnitude 0.7, decay 1.0, 4 bpp,
   `T_phi90`, n=12, rfam_gui_server.py:2223-2237), labelled as the method used by the
   historical campaigns and for A/B comparison against the solve. Proportional-multi,
   hybrid, OC-TO, thorough, EBC, perturbation and the rest stay exactly where they are
   but inside a collapsed "Research options" panel under this choice. Nothing removed.

The stale generate-fgm endpoint defaults (rfam_gui_server.py:6169-6173) and prompt strings
(results.js:2293-2325) should be updated to the quick-look defaults above so that even the
raw button gives the standard answer.

### 3.2 Surfacing multi-solve progress

The job plumbing already supports this: `_set_job_progress` with `progress_label`
(rfam_gui_server.py:1368) drives the job card, and fgm_iterate already streams per-iteration
convergence rows into the card (app.js:1454, `convergence_iters`). Propose the solve job
reuse the same contract:

- Progress denominated in **forward-solve equivalents** ("solve 23 / ~40"), since gradient
  evaluations cost two forwards; the report already accounts in forward-equivalents.
- A live J sparkline in the job card, reusing the existing sparkline helper
  (app.js:1294), replacing sigma_T with J and marking the current best J-stop time.
- On completion, a solve dashboard analogous to the convergence dashboard
  (results.js:268): final map thumbnail, J and IoU vs uniform and vs best stored mask,
  the energy-residual gate value, the horizon flag if the J minimum sat on the last
  stored step (the report's `(H)` convention), and the same "Re-simulate with this map"
  and "Send to RIP" buttons the FGM preview modal already has (results.js:163-180).
- Queueing: one job per shape; batch-library solves stay a CLI campaign for now.

---

## 4. Usability pass: promote, de-emphasize, fix labels

### Promote (all grounded in an observed control)

- **Restore and promote the FGM mode into the Operation mode select** (was old
  index.html:95, "FGM Iterative Optimize"); rename per section 3.1 so method choice is
  the first thing seen. This is the single feature the project runs most and the GUI
  currently cannot start fresh.
- **Ship the standard-parameter presets as first-class named presets**, not just the
  restored localStorage Save/Load (app.js:1977-2040): "Standard 2-D FGM comparison"
  (voltage drive, enforce off, grid 120, optimizer t\* at phi 0.90, dual read-state report)
  and "Standard absolute-dose" (enforce_generator_power true, 500 W). Source of truth:
  HEATR_STANDARD_PARAMETERS.md section 4. Server-side preset YAMLs in `configs/` keep it
  out of the browser store.
- **Promote drive mode out of the Advanced panel** to a visible two-option control
  ("Voltage drive: for FGM and uniformity comparisons" / "Enforced generator power: for
  absolute dose"), keeping the raw fields (index.html:563-572) in Advanced. Add the
  one-line warning from HEATR_STANDARD_PARAMETERS.md:84.
- **Show the two read states and the gate on every run card**: extend `_summary_excerpt`
  (rfam_gui_server.py:5052) with melt-onset sigma_T, heating-peak sigma_T (port the
  extraction from `outputs_eqs/geometry_dual_readstate/dual_readstate.py`), and
  `|energy residual| / dose` with a red badge above 5 percent.
- **Turntable near-continuous preset button** ("15 deg x 48, near-continuous") next to
  the existing fields (index.html:185-196), matching configs/diamond_tt_15deg_48rot_nearcont.yaml.
- **Pre-launch FGM suitability hint**: the server already refuses baselines below
  sigma_T 4.0 C (rfam_gui_server.py:2798) but only after iter-0; echo the geometry
  guidance from the dropped help text (old index.html:633-635) next to the method chooser.

### De-emphasize, never remove

- Proportional-multi-iteration, hybrid, OC-TO, thorough proxy rotation, regime-adaptive,
  momentum, perturbation (rfam_gui_server.py:2244-2305): collapse under "Research
  options"; the dropped help text itself says proportional beyond iter-1 hits the glass
  ceiling.
- `fgm_gradient_descent` (rfam_gui_server.py:3822): keep the endpoint, no new UI until it
  earns a use case.
- Shell sweep and Antenna Workshop: functional, niche; keep as-is.
- Prewarp (ILT) mode: keep, but its footer line "Prewarp scripts are intentionally
  excluded" (index.html:668) now contradicts the prewarp mode present at index.html:117.
  Delete or reword the footer line.

### Mislabeled / confusing, with fixes

- index.html:668 vs index.html:117 contradiction above.
- Results.js Generate FGM prompt calls Qrf "recommended" (results.js:2305): flip to
  `T_phi90` and 4 bpp per GEOMETRY_DUAL_READSTATE.md line 3.
- `main()` startup print "Prewarp flows are disabled in this interface"
  (rfam_gui_server.py:6998) is also stale.
- Grid inputs (index.html:546-547) need a max/warning at 200 per the melt-onset
  instability (HEATR_STANDARD_PARAMETERS.md:83, "NEVER 240").
- HEATR-3D page cannot set FGM magnitude (heatr3d.js:101); add the field, defaulting to
  1.0 with the caveat that magnitude 1.0 is the aggressive 3-D value only
  (HEATR_STANDARD_PARAMETERS.md:87).

---

## 5. Prioritized change list

**P0 (restore what regressed; hours each)**
1. Re-merge the FGM iterate + FGM import form sections and mode options from
   `git show 8946538^:webui/static/index.html` (lines 95-96, 617-830) into the current
   index.html, keeping the prewarp section that commit added. app.js handlers already
   exist; verification is a live-launch smoke test of one 2-iteration circle run.
   Effort: about half a day including testing.
2. Re-merge the preset controls (`presetSelect`, `presetSaveBtn/LoadBtn/DeleteBtn`) that
   the same commit dropped; handlers at app.js:1977-2040 are intact. Effort: 1-2 hours.
3. Fix the Generate FGM prompt and endpoint defaults to the standard (4 bpp, `T_phi90`
   offered and recommended, magnitude help citing the per-shape table): results.js:2293-2325,
   rfam_gui_server.py:6169-6173. Effort: 1-2 hours.
4. Remove the two stale "prewarp disabled" strings (index.html:668,
   rfam_gui_server.py:6998). Effort: minutes.

**P1 (align with the standard and the incoming solve; a day or two each)**
5. FGM Method chooser per section 3.1, wrapping the restored form; solve mode wired to
   the integration when it lands, with the per-shape SOLVED/IMPROVED/NOT-RESCUED hints
   and double-pass checkbox. Effort: 1-2 days UI plus the solve job launcher.
6. Standard-parameter server presets (voltage-drive FGM-comparison and enforced-power
   absolute-dose) surfaced as one-click chips above the form. Effort: about a day.
7. Dual read-state sigma_T plus energy-residual gate in `_summary_excerpt` and the run
   cards; port the heating-peak extraction from
   outputs_eqs/geometry_dual_readstate/dual_readstate.py. Effort: about a day (touches
   solver summary emission or a post-read of time_series.json; keep it read-side to
   avoid solver changes).
8. Solve progress surfacing per section 3.2 (job-card forward-solve counter, J sparkline,
   completion dashboard). Effort: 1-2 days, mostly reuse of the convergence dashboard.
9. Promote drive mode to a visible toggle with the sign-fidelity warning. Effort: half a day.

**P2 (guardrails and polish; hours each)**
10. Grid > 200 warning on advGridNx/Ny (index.html:546-547).
11. Turntable near-continuous preset button (index.html:185-196).
12. HEATR-3D magnitude field (heatr3d.html:174-188, heatr3d.js:101).
13. Pre-launch FGM suitability hint (sigma_0 gate echo, rfam_gui_server.py:2798).
14. Decide the fate of the API-only `fgm_gradient_descent` and document it as such in
    the Theory or Getting Started page rather than leaving it invisible.

**Explicitly out of scope / do not redo:** the results-tab performance pass (uncommitted
`_scan_run_files` in rfam_gui_server.py plus results.js pagination) is done; per-node
two-sided tuning stays CLI-research; no feature is removed anywhere in this plan.
