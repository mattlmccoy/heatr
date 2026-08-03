# Grade and Print: Full Feature Set for RFAM Print Studio

Date: 2026-08-02
Status: APPROVED by Matt 2026-08-03 (given directly in the Studio session:
"approved, start phases 1-3"; also relayed via the 2-D solve session:
"I approve the grade and print spec, freeze the package format").
PACKAGE FORMAT SCHEMA 2.0.0: FROZEN 2026-08-03. Both freeze gates satisfied:
Matt's approval + the 2-D lane's recorded sign-off on section 7b as amended
(their APPROVE WITH THREE CHANGES verdict, folded in: conditional
engine_versions keys, optional plan block, exactly-one power settings rule,
three-state transfer record, 2-D correction engine values). Emitters in both
lanes conform to 7b as written at this commit; any further change is a schema
version bump, not an edit.
Owner: Matt McCoy
Builds on: docs/superpowers/specs/2026-07-31-solve-port-3d-design.md (Studio
integration thin and badge-gated), studio_handoff/ (2-D lane's manifest seed),
solve3d/PHASE_C_REPORT.md (first solved 3-D map), the existing RFAM Print
Studio Grade tab (software/meteor/tools/grade_server.py, static/grade.js).

## 1. Goal (Matt's words)

Make Grade and Print "a real tool with real simulation outputs and graded
output results": "the densification volume that we can look at before
correction, and then obviously after correction and layer wise."

Pipeline: import STL -> uncorrected 3-D densification volume -> correction
(solved 3-D map where one exists, 2.5-D per-slice grading otherwise, engine
always labeled) -> after-correction volume, side by side with difference view
-> layer-wise corrected dopant maps + predicted per-layer density -> versioned
print package -> MetPrint hot folders, with mandatory heatr3d verification of
the final package before send.

## 2. Current state (verified by reading the code)

- Grade tab today: upload (/slice_upload) -> analyze (meteor_bridge.
  analysis_polygons -> slice table + routing) -> verify (subprocess
  stl_compensation_tool/pipeline.py, 2.5-D HEATR per representative slice) ->
  send (slicer + meteor_bridge.grade_tiff_stack -> graded 4 bpp TIFFs +
  job_info.json into the hot folder). grade_server.py is thin glue; grade.js
  owns the UI with a MeteorViewport 3-D view, master z-scrub, and a
  single-plane FGM layer overlay.
- MeteorViewport (static/viewport.js) has mesh loading, one clip plane
  (setClipY), and one textured overlay plane (showLayerOverlay). It has NO
  volume rendering (no 3-D textures, no raymarching).
- No 3-D engine is wired anywhere in the Studio. Verification is 2.5-D only.
- Intake checks today: the web path has NONE. slicer_cli.py:75-76 refuses
  non-watertight meshes in the offline CLI only; /slice_upload ->
  slicer.get_mesh_info returns triangle count + bbox and nothing else;
  holes are display-only flags (largest loop modeled).
- heatr3d run API (verified against heatr3d.py Result, heatr3d_job.py):
  drive = Params.power_density_w_per_m3; run(grid, part, p, phi_target=0.90,
  densify=...) marches to phi_target (densify=False) or the full horizon
  (densify=True); Result exposes rho_final (the density VOLUME, densify runs
  only), T_phi90, T_final, phi_final, phi_hist (scalar mean melt fraction per
  step), sigma_T, T_max_c, reached, energy_residual_frac, clamp_bound.
  There is NO per-timestep field snapshot mechanism.
- Solved 3-D artifact (opened, real): solve3d/results/phase_c_map_solve_
  filter_only_asymmetric_scaled.npz = DG0 cell data on the FEM mesh
  (v_raw, s_map, centroids (105191,3), volumes), solved_label true,
  sim-only trust. It is NOT a voxel grid; a transfer step is required.
- 2.5-D dopant volume (opened, real): dopant_volume.npz = sat
  (n_layers, 120, 120) float32 with sat=1.0 outside the part
  (1.0 = unmodulated), part_mask, z_mm, area_mm2, method (string per layer),
  gain, chamber_m.
- TIFF conventions (meteor_bridge.py:33-50, meteor_rip.py:233-262): Meteor
  TIFFs are WhiteIsZero, black = max ink, bpp in {1,2,4}, LZW; grading
  multiplies levels and preserves zeros; production config is bpp 4, dpi 720.

## 3. Constraints carried into every feature

- ENGINE LABELS EVERYWHERE. Three engines can produce numbers on this
  surface: "heatr3d native" (the fast in-tool 3-D planner), "solve3d solved"
  (dolfinx, solved_label maps), "HEATR 2.5-D per-slice" (the deployable
  grading path). Every card, viewer, manifest block, and report row names its
  engine. Blending engines silently is a construction-time error.
- TRUST BADGES. Every 3-D number carries its graduation-gate badge (per the
  approved graduation spec): today that is "S1 passed within validity domain,
  S4 not passed, sim-only" for heatr3d native and "solved, sim-only gates"
  for the Phase C cylinder map. Badge text comes from one shared table, not
  per-page strings.
- Grid ceilings ENFORCED IN THE UI: n <= 96 full physics, default n = 64 for
  interactive runs. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 on every solver
  subprocess.
- Continuous fields render with real gradient colormaps (T, rho); NEVER
  saturated indicator fields. The phi = 0.9 melt front is drawn as a contour
  overlay on top, per the deck-figure lesson.
- No hand-coded shape knowledge anywhere; every feature works on any imported
  geometry.
- heatr3d.py and heatr3d_s2/ are frozen while the S2 campaign runs; all 3-D
  orchestration goes through a NEW runner module in geo-prewarp that imports
  heatr3d as-is and replicates heatr3d_job.main's render wiring (_field_meta
  -> _render_slices + _render_summary_plots) with the enthalpy-standard
  Params, never by calling heatr3d_job.main (it hardcodes apparent_cp).
- Studio accent #7EC8E3. No em dashes in any user-facing or spec text.

## 4. Feature 1: Import and intake refusal rules

New intake gate, server side, that runs at Grade-tab mesh load (not inside
/slice_upload, which the RIP/Slicer tabs share and must keep tolerating
imperfect meshes the way Meteor's slicer does today).

- Endpoint: POST /grade/intake {job_id, filename}. Loads the mesh with
  trimesh in the geo-prewarp venv (subprocess, same pattern as verify) and
  returns a verdict record.
- REFUSED loudly, with informative errors naming the defect and the count:
  - non-watertight (trimesh is_watertight false): "REFUSED: mesh is not
    watertight (N open edges). Repair the mesh; grading a leaking volume
    would model geometry that does not exist."
  - self-intersecting (triangle-triangle intersection test): "REFUSED: mesh
    self-intersects (N intersecting face pairs)." (The 2-D lane's intake
    discarded even-odd fills for exactly this failure; we refuse rather than
    guess.)
  - through-holes in slices (interior loops): REFUSED until hole-aware fill
    exists, matching the port spec's rule: "holes REFUSED rather than
    silently filled until hole-aware fill is built and tested." This
    replaces today's display-only "largest loop modeled" behavior FOR THE
    GRADE PATH; the Slicer tab is untouched.
- The verdict is stored in grade state; every downstream Grade endpoint
  refuses to run on a mesh whose intake verdict is not "accepted" (no
  bypass flag).
- UI: refusal renders as a prominent error card (red border, defect name,
  count, remediation hint), not a toast.

## 5. Feature 2: Uncorrected densification volume (heatr3d native)

The "before" picture: a native heatr3d densify=True run on the imported part.

Orchestration:
- New module geo-prewarp/studio3d/runner.py (name final at implementation):
  voxelizes the accepted mesh onto H.Grid(n), builds Params with the
  enthalpy standard (phase_update="enthalpy"), drive =
  power_density_w_per_m3 default, runs densify=True, and writes the
  heatr3d_job-standard artifact set (fields.npz with part, T_phi90,
  phi_final, rho_final, sat, h; fieldmeta.json; slices/*.png;
  plots/*.png; results.json) into uploads/<job>/grade/heatr3d/uncorrected/.
  Voxelization uses the mesh itself (trimesh voxelized fill), no shape
  presets.
- Grade server gains POST /grade/densify {job_id, arm: "uncorrected",
  n: 32..96} running the module as a background subprocess with the same
  progress-log pattern verify uses. UI slider for n is clamped to 96 and
  defaults to 64; the request validator rejects n > 96 server side too.
- Standing gates surfaced, not buried: reached, energy_residual_frac
  (want |.| < 1e-2), clamp_bound, T_max_c vs the 250 C ceiling. A run that
  fails a standing gate renders its numbers greyed with the gate failure
  named; it is not hidden and not trusted.

Viewer (the load-bearing UI work):
- 3-D density volume viewer with z-scrub: the master z-scrub drives BOTH the
  mesh clip plane and a density slice plane textured into the viewport
  (reusing the proven showLayerOverlay plane path), plus a larger 2-D slice
  panel beside it.
- Slice rendering server side (PIL, same no-matplotlib constraint as
  grade_server's existing renders): rho and T colored with real perceptual
  gradients, outside-part transparent, per-field global vmin/vmax shared
  across arms so before/after are comparable, and the phi = 0.9 front
  overlaid as a contour line. Endpoint: /grade/vol3d/<job>/<arm>/<field>/
  <axis>/<idx>.png with vmin/vmax pinned by a /grade/vol3d_meta endpoint.
- Cross-sections: XY, XZ, YZ slice axes selectable (the npz is (n,n,n);
  slicing any axis is cheap server side).
- Time scrubbing, stated honestly: heatr3d has no per-timestep volume
  snapshots and heatr3d.py is frozen during S2. NOW: the melt-progression
  curve (phi_hist x dt) is the time axis, scrubbable, with the melt-onset
  marker; the volume fields are end-state (rho_final) and melt-onset
  (T_phi90). AFTER S2 UNFREEZES: a small opt-in snapshot hook
  (Params.snapshot_interval_s writing rho/T volumes) is the engine ask that
  upgrades the same scrubber to true volume time-scrub. The UI labels the
  time axis "mean melt fraction over time" until then; it never implies the
  volume itself is time-resolved.
- Badge: "heatr3d native, S1 passed within validity domain, sim-only" from
  the shared badge table.

## 6. Features 3 and 4: Correction and the after-correction view

Correction sources (engine registry, never blended):
- "solve3d solved": where a solved artifact exists for the imported
  geometry. Today that is exactly one: the Phase C cylinder map
  (solved_label true). Badge "solved, sim-only gates". The Studio matches
  imported geometry to solved artifacts by geometry hash recorded in the
  artifact registry, never by shape heuristics; no hash match, no solved map.
- "HEATR 2.5-D per-slice": the deployable path for arbitrary parts, i.e.
  the existing analyze -> verify -> dopant_volume.npz pipeline, unchanged.
- The correction card shows WHICH engine produced the active dopant volume,
  with its badge, in the card header, in every export, and in the manifest.

Support-aware transfer (the Phase C staircase lesson, mandatory):
- Any transfer of a map between supports (DG0 cells -> voxel grid, voxel
  grid -> conforming cells, solve grid -> printer raster) EXTENDS the field
  beyond its support BEFORE interpolating: nearest-inside-support extension
  (or equivalent), then interpolation, then re-mask to the target support.
  Phase C measured plain trilinear-with-zeros thinning the rim map by 7.15%
  (vs 0.065% for a support-aware path); that mechanism must never reach the
  printer. Acceptance test: total in-part dopant moves < 2% across the
  transfer, the same threshold Phase C pre-registered, asserted in code.
- Concretely for the cylinder map: s_map at centroids (DG0) -> voxel sat
  volume on the heatr3d grid via nearest-cell extension + volume-weighted
  averaging, gated by the < 2% dopant-mass check.

After-correction run and views:
- POST /grade/densify {arm: "corrected"} runs the SAME heatr3d native
  densify=True configuration with the active correction applied as the sat
  volume (heatr3d's sat input path, as heatr3d_job wires it). Same gates,
  same artifact set, written to .../heatr3d/corrected/.
- Side-by-side viewer: uncorrected and corrected in the same slice viewer,
  same colormap, same pinned vmin/vmax, one shared z/axis scrub.
- Difference view: server-rendered signed difference (corrected minus
  uncorrected) for rho and T with a diverging colormap centered at zero,
  plus summary deltas (mean part density, sigma_T as diagnostic only,
  out-of-part melt volume).
- The corrected arm's card repeats the correction engine label: a corrected
  volume is "heatr3d native forward on [solve3d solved map]" or
  "heatr3d native forward on [2.5-D per-slice map]", never just "corrected".

## 7. Feature 5: Layer-wise output and the versioned print package

### 7a. Layer-wise views

- Per-layer corrected dopant map: the existing dopant browser, upgraded to
  show the active correction (either engine) at the analysis layers AND at
  print layers (the printer's layer_height_mm), with the engine label.
- Predicted per-layer density: horizontal z-profile of mean rho_final per
  layer (uncorrected and corrected overlaid), plus per-layer slice images
  from the corrected volume. Both read from the densify artifacts; nothing
  is re-simulated for display.

### 7b. Package format spec (SEED: studio_handoff SCHEMA_VERSION 1.0.0; this
section goes to the 2-D session for review before freeze)

One package per planned print, directory + zip, emitted into packages/ at
the geo-prewarp root (interim scheduler drop folder, uploader blocked on
hardware). Naming: pkg_<part>_<YYYYMMDD-HHMMSS>.

Kept from the 2-D lane's draft (their two best ideas, preserved verbatim in
spirit):
- turntable_program.json OR turntable_static.json, NEVER silently absent;
  static is an explicit record.
- production_verify block: a real verification record or an explicit
  not-run statement; "run": false is never implied by absence.

Contents:
- manifest.json (SCHEMA_VERSION 2.0.0, since the layer-wise 3-D content is
  a breaking extension of their 1.0.0 draft; the three 2-D-lane review
  changes of 2026-08-03 are folded in below):
  - schema_version, created_utc, engine_versions (plural). Keys are
    CONDITIONALLY REQUIRED by source_route, not always-required: a 3-D
    Studio package carries heatr3d version stamp, solve3d artifact id,
    stl_compensation_tool rev; a 2-D-native package may carry only
    {"heatr_2d": <rfam_eqs_coupled ENGINE_VERSION>}.
  - part: name, source_geometry_sha256 (STL bytes), source_route, intake
    verdict record (the refusal gate result that admitted this mesh)
  - plan (OPTIONAL block, restored from the 1.0.0 draft for
    Import-and-Plan provenance): classifier recommendation + its advisory
    disclaimer, classifier version, expected outcomes (J/IoU) with the
    grid qualifier. 3-D packages may omit the whole block.
  - correction_provenance: engine ("solve3d_solved" | "heatr_25d_perslice"
    | "heatr_2d_solve" | "heatr_2d_proportional" | "none"; the last two
    are the 2-D lane's adjoint-solve and calibrated-inverse channels),
    artifact ids/hashes, trust badge string, transfer record with THREE
    explicit states: measured-and-passed (< 2% dopant-mass move),
    measured-and-failed, or transfer_not_applicable (for rasters native to
    the production path with no volume-to-grid transfer step, e.g. the 2-D
    fgm_generator dpi resample). Absence NEVER implies not-applicable.
  - densify_summary: uncorrected and corrected arm summaries (mean part
    rho, sigma_T diagnostic, T_max_c, standing gates, n, engine label each)
  - power_settings: EXACTLY ONE of power_density_w_per_m3 (3-D arms) or
    the 2-D voltage block (rf_mode "constant" retained), enforced by the
    validator; neither, or silently both, is invalid
  - raster block: dpi (720), bpp, level_map shape, convention string
    (WhiteIsZero, loader inverse), per-layer file list
  - turntable block: mode program|static, file, plus advisory: true and the
    z-coupling caveat string (the recommendation is per-build, dose-weighted
    aggregation default with worst-layer flag, and stays ADVISORY because
    z-coupling is real; heatr3d verification of the chosen program is the
    authority)
  - production_verify: run true/false; when true, the heatr3d verification
    summary of THIS package's rasters (see 7c); when false, the explicit
    statement. A package with run: false cannot be sent to the hot folder.
  - scheduler block, files list with sha256 per file (both kept from 1.0.0)
- Per-layer rasters at PRINTER DPI, one per print layer, emitted through
  the fgm_generator resample path (fgm_generator.py dpi=720 native,
  resample :588-606; 1715-class level maps at the 60 mm chamber), 4 bpp
  production default. NEVER solve-grid rasters in a package. Transfer from
  the dopant volume to printer rasters is support-aware (section 6).
- map volume npz (the corrected dopant volume actually used), preview PNGs.
- turntable_program.json | turntable_static.json, production_verify_
  summary.json as above.

Interim MetPrint contract (until the scheduler exists): the send step writes
the graded TIFF job + job_info.json into the hot folder as today, with
turntable_program.json (or the static record) side by side in the job
folder, and job_info.json gains the package id + manifest hash.

### 7c. Feature 6: mandatory verification before send

- "Send to hot folder" is DISABLED until a heatr3d verification of the
  FINAL package rasters (re-loaded from the emitted TIFFs/level maps, not
  from the in-memory volume, so what is verified is what prints) has run
  and its standing gates pass. The verification is a densify=True corrected
  arm re-run at the declared n with the package's rasters transferred back
  to the solver grid support-aware.
- The turntable recommendation remains advisory everywhere it appears;
  verification of the chosen program is what gates send.

## 8. Non-goals

- No scheduler/uploader (hardware blocked); packages/ drop folder only.
- No 3-D turntable simulation (heatr3d has no rotation machinery; the
  co-rotation design requirement stands for whoever builds it).
- No eps_r channel exposure in the Studio (law uncalibrated until M2).
- No hole-aware fill (refused instead, feature 1).
- No dissertation edits. No changes to heatr3d.py or heatr3d_s2/ while S2
  runs; the snapshot hook is queued as a post-S2 engine ask.
- No new claims about print physics: everything 3-D stays sim-only badged.

## 9. Hardening items found while reading (P1, bundled into this work)

- /mesh_file/<job_id>/... and /grade/asset/... do not sanitize job_id
  (path traversal). Fix with the same regex used for filenames.
- DEFAULT_CONFIG in web_server.py says bpp 2 while config.json and grading
  default to bpp 4; a machine without config.json silently grades at the
  wrong bit depth. Align the default and log the effective value into
  job_info.json (it already records bpp; the default itself is the bug).

## 10. Phasing (each phase red-first TDD; UI phases browser-verified with
screenshots viewed personally)

- Phase 1: intake gate (trimesh checks subprocess + endpoint + refusal UI).
  Pure logic (verdict rules) unit-tested red-first; UI browser-verified.
- Phase 2: studio3d runner + /grade/densify uncorrected + volume slice
  viewer (meta endpoint, slice PNGs with contour overlay, z/axis scrub,
  gates surfaced, badge). n=64 smoke run on a small STL as the real-data
  integration gate.
- Phase 3: correction registry + support-aware transfer (the < 2% mass-move
  gate is the acceptance test) + corrected densify arm + side-by-side +
  difference view.
- Phase 4: layer-wise views + package emitter (manifest 2.0.0, printer-DPI
  rasters, turntable + verify records) + mandatory pre-send verification
  wiring + hot-folder interim contract.
- Phase 5: hardening items + end-to-end run on a real STL with the full
  artifact set, screenshots of every view.

Freeze order: Matt approves this spec; section 7 goes to the 2-D session;
their review lands; the package format freezes; only then does the Phase 4
emitter get built against the frozen schema (Phases 1-3 do not depend on
the freeze and may start on approval).
