# heatr3d Workbench: the 3-D simulation operator surface

Date: 2026-08-02
Status: stage-1 design, awaiting Matt's review (no build until approved)
Owner: 3-D / graduation lane
Related: docs/superpowers/specs/2026-07-30-heatr3d-graduation-design.md (badge
requirement), docs/superpowers/specs/2026-07-20-heatr3d-usability-design.md
(the prior HEATR-3D tab spec; F0-F6 shipped, this supersedes its scope),
solve3d/PHASE_C_REPORT.md (first SOLVED map), HEATR_STANDARD_PARAMETERS.md.

## 1. Goal

Matt: "make the heatr3d gui much much better with all the workflow needed that
would make sense for running the known 3d library and testing densification
and or creating FGM solved maps" - streamlined the way the HEATR 2-D process
is. The workbench is the heatr3d OPERATOR surface: run the 14-shape library or
an uploaded STL under the standard parameter set, study densification in a
real volumetric viewer, view FGM solved maps against uniform baselines, and
compare runs on SHAPE metrics - with a trust badge on every displayed number.

Out of scope by ownership: heatr3d_s2/ (S2 campaign runs there), heatr3d.py
(read-only; canonical-sync gated), software/meteor/tools/ (Grade-and-Print
Studio session). The Studio remains the product surface; the workbench is the
research/verification instrument that feeds it.

## 2. Current state (verified 2026-08-02, citations)

The existing surface is the HEATR-3D tab of the 2-D HEATR web GUI:

- Server: rfam_gui_server.py (plain http.server, 7310 lines, owned by the 2-D
  lane as their operator tool). The heatr3d block is lines 5584-5719 + routes
  at 5759-5817 and 6231-6240. Runs spawn heatr3d_job.py as a detached
  subprocess under a pinned venv (numpy<2.2); job state in the in-memory dict
  _H3D_JOBS with an on-disk fallback; no queue, no concurrency cap, no cancel,
  no OMP/OPENBLAS thread pinning on spawn (the S4 campaign measured 26 s ->
  18 min from oversubscription).
- Frontend: webui/static/heatr3d.html (273 lines) + heatr3d.js (389 lines),
  three.js 0.160 via CDN importmap, no build step. Has: parametric shapes
  (sphere/cone/cylinder/dumbbell only), STL upload (base64-in-JSON, no
  validation), grid n up to 64 (default 40), FGM mode none/melt/density,
  densify checkbox, preview + run, 1.5 s status polling of a 4-step progress
  bar (PROGRESS 5/30/90/100 lines only), surface-voxel InstancedMesh viewer
  (color: uniform or sat), post-sinter warp toggle, per-z pre-rendered slice
  PNGs with field select + play, 7 summary plots, grouped gauge metrics, a
  shape-grouped past-run picker, and sigma_T compare bars (commits eb6f/26bab).
- Job wrapper: heatr3d_job.py (488 lines). Writes per run dir
  (outputs_eqs/_heatr3d/<12hex>/): config.json, geometry.json, results.json,
  summary.json (run_type heatr3d), fieldmeta.json, fields.npz (part, T_phi90,
  phi_final, Qrf, rho_final, sat, h - FINAL-STATE volumes only), slices/*.png,
  plots/*.png, preview.png, warped_geometry.json (densify runs). Captured from
  a real completed run (8a16b9459f84); 113 run dirs on disk, only 15 complete.
  KNOWN DEFECT: main() hardcodes H.Params() = apparent_cp legacy, not the
  enthalpy standard (heatr3d_job.py:418; memory note heatr3d-run-api-facts).
- Engine facts (heatr3d.py, READ-ONLY): run(grid, part, p, sat, max_time_s,
  phi_target=0.90, densify, stop_mean_rho, ..., qrf_gradient="masked",
  t_start_s, T0_override) -> Result. Drive is Params.power_density_w_per_m3
  (no voltage calibration). densify=False stops at the phi_target crossing;
  densify=True marches the FULL exposure. Result carries final-state volumes
  (T_phi90, phi_final, Qrf, rho_final, T_final), scalar phi_hist (one entry
  per dt_s step), the standing gates energy_residual_frac / clamp_bound /
  cfl_violated / reached, and n_eqs_solves. verbose=True prints a line every
  200 iterations (heatr3d.py:1237) - the only in-march signal available
  without touching the solver. Ceilings: n<=96 full physics, n<=128 EQS-only
  (EQS-01 guard). Params(phase_update="enthalpy") is the standard;
  eqs_update_interval_s is the coupling master switch (default 0 = legacy).
- Shape library (branch feat/shape-library-3d): shape_library_3d/ - 14
  curated watertight STLs equal-volume to V* = 4188.79 mm^3 (10 Tier-1
  physics, 2 Tier-2 sphere probes, 2 Tier-3 rejection fixtures). Public API
  SHAPES / iter_parts() / load_part_stl(); validate_part_mesh() raises typed
  NonWatertightMeshError / ZeroVolumeError (planarity checked FIRST because an
  open cylinder reports nonzero signed volume); voxelize.stl_to_mask() is the
  heatr3d Grid bridge (mm -> m, containment test, Tier-3 rejected before any
  voxel work). NOTE: validate_part_mesh does NOT yet check self-intersection.
- Solved maps: solve3d/results/ has the first SOLVED artifact (cylinder,
  phase_c_map_solve_filter_only_asymmetric_scaled.npz on the dolfinx mesh,
  solved_label=true, gates in phase_c_gate.json, scores in
  phase_c_baselines.json / phase_c_solves.json). PHASE_C_REPORT.md: "No
  Studio badge earned yet" - sim-only. The dropped inversion arm proved the
  staircase-support transfer hazard (voxel<->FEM map transfer moved in-part
  dopant 7.15%, over the 2% pre-registered bound).
- Shape metrics: solve3d/shape_metrics.py has iou() (empty-vs-empty = NaN,
  never 1.0), out_of_part_fraction(), in_part_melt_fraction(),
  symmetric_surface_distance_mm() - the voxel port for run scoring reuses
  these definitions.

## 3. Decision: a new heatr3d_workbench/ app (not more surgery on the tab)

RECOMMENDED: build `heatr3d_workbench/` as a standalone app - its own small
stdlib http server (same pattern as rfam_gui_server.py: BaseHTTPRequestHandler,
POST allowlist, no framework, no build step) on its own port, its own static/
frontend, reading and writing the SAME run store (outputs_eqs/_heatr3d/) so
every historical run appears. The existing HEATR-3D tab keeps working
untouched; it gains only a link ("open the 3-D Workbench") if the 2-D lane
wants one.

Why not evolve the tab in place:

1. Ownership and contention. rfam_gui_server.py and webui/ are the 2-D lane's
   operator tool (memory: studio-ownership-and-handoff), and two other agents
   are active in this tree. The workbench needs server-side work (queue,
   metrics, badges, snapshots, library endpoints) that would mean sustained
   editing inside their 7310-line file. A separate app is zero-contention and
   cleanly this lane's own.
2. Scale mismatch. The five workflows below need roughly 15 endpoints, a job
   queue, a badge registry, and a much larger frontend. That is an app, not a
   tab. The current tab's single-page layout (360 px form + one viewport)
   cannot hold a library gallery, a study viewer, a solved-map screen, and a
   comparison screen without becoming the index.html mode-select monster the
   2-D page already is.
3. The run executor must change anyway: heatr3d_job.py hardcodes legacy
   Params() and 4-step progress. Memory (heatr3d-run-api-facts) already
   directs new render wiring to REPLICATE heatr3d_job's functions, not call
   its main(). A fresh workbench_job.py entry point that imports heatr3d and
   reuses/extends the proven artifact writers is the recorded-correct path.

What is reused, not rewritten: the run-dir artifact contract (section 2), the
slice/plot pre-render approach (server stays numpy-light), the subprocess
isolation under the pinned venv, three.js viewer machinery, and the shared CSS
(styles.css / design-b.css / hub-theme.css) for visual continuity.

## 4. The app: four screens plus a persistent run rail

Layout: left nav with four screens - LIBRARY, STUDY (viewer), SOLVED MAPS,
COMPARE - plus a persistent bottom/side RUN RAIL showing the queue (running,
queued, recent; live progress; every row jumps to STUDY). No em dashes in UI
text. The word surrogate never appears.

### 4.1 LIBRARY - pick, validate, launch

- Shape gallery: the 14 shape_library_3d shapes as cards (name, tier chip,
  rf_characteristic one-liner, thumbnail from a pre-rendered STL snapshot,
  V* volume). Tier-3 fixtures shown greyed with their rejection reason - they
  exist in the gallery precisely to demonstrate the refusal path.
- STL upload: multipart or base64 (small meshes), then INTAKE GATE before any
  run is allowed:
  * validate_part_mesh (planarity -> watertight -> zero volume), surfaced as a
    loud red refusal card quoting the typed error (naked edge count, extents).
  * NEW self-intersection gate: trimesh is_volume + winding consistency +
    a manifold3d round-trip check; refusal is loud and typed like the others.
    TDD with a constructed self-intersecting fixture (red first).
  * Holes = non-watertight = refused (same gate). No silent repair.
  * Chamber fit check (60 mm) with the mm-vs-m unit trap named in the message.
  Accepted STLs are stored once under a workbench library dir with a content
  hash, so re-runs never re-upload.
- Parameters panel, standard-set defaults with ceilings ENFORCED in the UI
  and revalidated server-side:
  * grid n: 32 / 48 / 64 (default) / 96, labeled "96 = full-physics ceiling";
    128 selectable only in an explicit "EQS-only diagnostic" mode that
    disables the thermal march options. Nothing above 128 is offered.
  * phase_update: enthalpy (DEFAULT - fixes the heatr3d_job legacy defect),
    legacy apparent_cp behind an "advanced" reveal for reproduction runs.
  * drive: power_density_w_per_m3 with the 10 W-into-V* default and a
    plain-language explainer (no voltage calibration exists in 3-D).
  * densify toggle, exposure_s, stop_mean_rho, FGM mode (none / melt /
    density / solved-map when one is loaded), fgm magnitude (currently
    hidden by the old UI even though the job supports it), coupling knobs
    (eqs_update_interval_s + sigma coeffs) behind "advanced", qrf_gradient
    masked default with legacy reveal.
  * A visible cost estimate before launch, anchored on MEASURED numbers
    (solve3d/results/parity_tolerances.json, PHASE_A_REPORT.md): extruded
    circle melt-onset ~147 s at n=64 (EQS 37 s + march 109 s) and ~558 s at
    n=96; square n=96 ~18.5 min; n=48 densify smoke runs 544-773 s (densify
    marches the FULL exposure). Presented as a labeled estimate, never a
    promise.
  * Voxelization report shown at preview: voxel volume vs V* from
    voxel_volume_report - the lattice loses ~49% of its volume at n=64
    (struts ~2 cells), and that staircase deficit must be visible before
    launch, never corrected away silently.
- Launch = enqueue (section 5). "Run the whole library" button enqueues all
  12 loadable shapes as a named campaign with shared parameters.

### 4.2 STUDY - the densification-volume viewer (centerpiece)

One screen per run (deep-linkable ?run=<id>), three synchronized regions:

- 3-D VOLUME VIEW (three.js): the surface-voxel shell plus an axis-aligned
  CUTTING PLANE. The visible cut face shows the selected field as a true
  color-mapped surface (textured from the same slice PNGs; the shell outside
  the cut renders dimmed). Warp toggle (nominal vs post-sinter geometry,
  displacement-colored) carried over.
- SLICE PANEL: field selector (rho / T / phi / Qrf / sat), axis selector
  (z / y / x - the current viewer is z-only; x/y slices are pre-rendered the
  same way), layer scrub slider synchronized with the 3-D cutting plane,
  play/pause, colorbar with min/max and units.
- TIME PANEL: melt-progression curve (phi_bar vs t, from phi_hist), the new
  live march series (section 5), and the TIME SCRUB (below).

Field rendering rules (the deck-figure failure is the design driver):
- Continuous fields ALWAYS render with real gradients on a per-field locked
  colormap per the visualization standard: T = inferno, rho = the diverging
  powder-gray -> melt-purple -> dense-gold density map, sat/dopant = viridis,
  Qrf = viridis. DPI 180 for all newly rendered artifacts.
- phi is NEVER shown as a saturated indicator field. The default "melt front"
  presentation is: color by T or rho, with the phi = 0.9 front overlaid as a
  contour line drawn from the ORIGINAL (unsmoothed) phi array; phi = 0.5 as a
  secondary dashed contour. A raw-phi view remains available in the field
  list but renders as a gradient with the front contours, not a binary blob.
- Outside-part voxels are transparent/hatched, never colormapped zeros
  (already the convention in _render_slices; kept).
- Explicit empty states: rho on a non-densify run says "not computed
  (densify was off)", never a blank.

TIME SCRUBBING - honest tiering, because fields.npz holds final-state volumes
only and heatr3d.py cannot be modified right now:

- Tier 1 (all runs, v1): scalar time scrub. The time slider moves a cursor on
  phi_bar(t) and the live march series (T_max(t), energy residual(t) where
  parsed); volumes shown remain the final/melt-onset states, clearly labeled
  "final state" so the cursor is never mistaken for volume playback.
- Tier 2 (densify=False runs, v1): true volume snapshots via CHAINED
  SEGMENTED MARCHES in workbench_job: run() is called K times (K ~ 8-12
  segments) with t_start_s advancing and T0_override carrying the temperature
  state across segments (both are existing run() parameters; the coupling
  schedule is on absolute time by design). Each segment boundary saves a
  T/phi snapshot volume (float16 npz, snapshots/) and pre-renders its mid-z
  and cut-plane slices. TDD gate: a chained K-segment march must reproduce
  the single-march final T field within a stated tolerance (and bit-identical
  when coupling is off and segment boundaries align with dt_s steps) before
  the feature ships.
- Tier 3 (densify=True runs): run() has no rho/phase state injection, so
  chained segments CANNOT carry densification state and would be silently
  wrong - explicitly not built. Two candidate paths, decided by Matt
  (open question 2): (a) an additive snapshot callback in heatr3d.py once
  the file is editable again (default-inert, canonical-sync gated), or
  (b) an opt-in "checkpoint replay" that re-runs the march to each of K
  increasing max_time_s values (cost K x full run, labeled as such, n<=64
  only). Until then densify runs get Tier 1 plus final volumes.

SHAPE METRICS strip (headline, per workflow 4 requirements) on every run:
melt-region IoU vs nominal at phi>=0.8 and phi>=0.9, out-of-bounds melt
fraction (melted volume outside the part / part volume - the hard side of
"dense iff in-bounds"), in-part melt fraction, front distance in mm. Computed
by a voxel port of solve3d/shape_metrics.py definitions (iou with the
NaN-on-empty rule, out_of_part_fraction, symmetric_surface_distance_mm) in
workbench post-processing, TDD red-first, fixtures captured from the real
8a16b9459f84 run. The nominal reference is the run's part mask; for STL runs
a finer-grid reference mask is also computed and both numbers shown (open
question 3). The existing `dice` from heatr3d.sinter_metrics stays reported
but is DEMOTED: sintered is defined inside the part mask there, so Dice can
never see out-of-bounds spill - the IoU + out-of-bounds pair replaces it as
headline. sigma_T^3D appears BELOW the shape strip as a diagnostic bar
only, always written sigma_T^3D, with the standard non-comparability note.

Standing-gate banner on every run view: energy_residual_frac vs the 1e-2
gate, clamp_bound, cfl_violated, reached / MELT_ONSET_FALLBACK, T_max vs the
250 C ceiling. Any tripped gate renders as a red banner ABOVE the metrics -
a run that failed its gates cannot show quietly healthy numbers.

### 4.3 SOLVED MAPS - FGM solve artifacts vs uniform baseline

v1 is a truthful READ surface over solve3d/results/ (nothing re-simulated):

- Solved-map cards discovered from solve3d/results/phase_c_*.json/npz (and
  future phase_e artifacts): shape, arm name, status (converged /
  budget_exhausted / DROPPED / NOT_RUN rendered distinctly - NOT_RUN is never
  blank), solved_label chip, and the deviation flag for recorded-deviation
  arms (the scaled-start arm carries "recorded deviation: 1/|g0| rescale").
- The decisive comparison table printed FROM the JSON (phase_c_baselines /
  phase_c_solves / phase_c_gate), never transcribed: J asymmetric / symmetric,
  out-of-bounds and in-bounds components, melt fractions, sigma_T^3D
  diagnostic, both meshes, both weightings from the re-read. Gate results
  (hold-out, smoothing) with PASS/FAIL chips and their bands.
- Map format fact (captured): phase_c_map_*.npz holds v_raw, s_map,
  centroids (105191, 3), volumes - a DG0 field on the dolfinx tetrahedral
  mesh, NOT a voxel grid; the voxel-native format is
  phase_c_inversion_map.npz (sat (64,64,64) + part + h).
- Map visualization: the solved dopant field rendered as slice images in its
  NATIVE space by a small solve3d-env render script invoked offline (the
  workbench serves the PNGs). The map is NOT resampled onto the heatr3d voxel
  grid for display-as-truth: Phase C measured the staircase-support transfer
  hazard at 7.15% dopant shift, so any voxel resample shown is labeled
  "display resample, not the solved map" if we show one at all.
- Badges: every solved-map number carries the sim-only badge ("design method
  certified on the dolfinx forward; no S-gate earned" per PHASE_C_REPORT).
- Predicted-outcome-vs-uniform view = the scored fields already in the
  artifacts (uniform arm vs solved arm), presented side by side with the
  phi-front contour convention of 4.2.
- Explicitly deferred: pushing a solved map through the heatr3d forward from
  the workbench (crosses the transfer hazard and would imply a parity that
  Phase D drive reconciliation has not established). Listed as future work
  gated on Phase D/E, not a hidden button.

### 4.4 COMPARE - two runs side by side

- Pick two runs (default: same shape, baseline vs FGM; the picker groups by
  shape like the current optgroup picker and warns when shapes differ).
- HEADLINE = the SHAPE METRICS strip for both runs with deltas: IoU at 0.8 /
  0.9, out-of-bounds melt fraction, in-part melt fraction, front distance.
  Green/red deltas keyed to "dense iff in-bounds" (out-of-bounds down = good;
  IoU up = good).
- Side-by-side synchronized slice viewers (same field, same layer, same
  colormap range across both runs - shared colorbar per the visualization
  standard for like-for-like panels).
- sigma_T^3D as a small diagnostic bar pair at the bottom (the existing
  compare-bars idea, demoted from headline), plus t90, T_max, energy gate
  status per run.
- Config diff table (n, phase_update, drive, coupling, fgm, densify) so a
  comparison never hides a confound.

## 5. Run orchestration

- workbench_job.py (new, in heatr3d_workbench/): imports heatr3d, builds
  parts via shape_library_3d.voxelize.stl_to_mask (library/STL) or
  H.make_geometry (parametric), applies the standard Params
  (phase_update="enthalpy" default), runs with verbose=True, and writes the
  SAME artifact contract as heatr3d_job (section 2) plus: snapshots/ (Tier 2),
  shape_metrics.json, march_series.json, badges.json. It reuses heatr3d_job's
  proven writer functions by import where possible rather than copying.
- Environment: every spawn sets OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  (the missing pin that cost 26 s -> 18 min in S4). n=64 is the development
  smoke default.
- Queue: a small on-disk job queue (queue.json + per-run state files, not an
  in-memory dict) with max 1 concurrent solve by default (configurable to 2).
  Server restart re-adopts running PIDs or marks them stale-failed loudly.
  Cancel = SIGTERM with the run dir marked cancelled (distinct from failed).
- Live progress: workbench_job prints structured lines
  (MARCH t=<s> phi=<f> Tmax=<C> resid=<f>) on a fixed stride by parsing its
  own march via the verbose hook it controls (it wraps run() per segment in
  Tier 2, and for single-march runs derives progress from t/max_time using
  run(verbose=True) stdout captured in job.log). The server tails job.log
  into /api/wb/status (progress %, current phase, live series). Frontend
  polls at 1.5 s (proven pattern; SSE not needed at this rate).
- Campaigns: a named batch (e.g. "library sweep n=64 densify") is a queue
  group; the rail shows per-run rows and a campaign summary row; COMPARE can
  pull any two members.
- Runs remain visible to the existing Results browser (summary.json contract
  preserved, run_type heatr3d unchanged).

## 6. Trust badges (graduation spec section 3, made concrete)

- Single source of truth: heatr3d_workbench/badges.py + a versioned
  badge_registry.json mapping quantity class -> current gate level, with the
  gate evidence path (e.g. s1-gate-report.md) recorded per entry. The
  registry is updated by hand when a gate report lands - never inferred.
- Current truthful levels (from the gate reports as of 2026-08-02):
  * thermal/phase scalars and volumes: "S1 verified (validity domain n<=96)"
  * EQS-derived quantities (Qrf, sigma_T^3D): "S1 verified, EQS-02 corrected
    drive" with the exploratory caveat chip (S2 convergence pending)
  * all heatr3d numbers: exploratory-tier styling until S2 passes
  * shrinkage/warp outputs: exploratory (no P2)
  * solved maps: "sim-only, no S-gate earned"
  * S4 chip where FLIR-relevant quantities appear: "S4 not passed"
- Rendering: a small badge chip next to EVERY displayed number group (metric
  strips, gauges, compare tables, solved-map tables), hover = one-sentence
  meaning + link to the gate report. Per-run flags (energy gate, clamp,
  MELT_ONSET_FALLBACK, cfl) render as run-level banner state, distinct from
  the tool-level gate badges.
- The badge system is TDD-able logic (registry lookup, per-run flag
  derivation from results.json) and is tested red-first.

## 7. Verification plan (stage 2 discipline)

- TDD red-first for all pure logic: intake gates (incl. the new
  self-intersection fixture), voxel shape metrics (fixtures captured from
  8a16b9459f84, not invented), chained-segment equivalence gate, badge
  derivation, queue state machine, cost estimator, march-line parser.
- Data-contract rule: every artifact the workbench reads was captured from a
  real run/solve artifact in this spec (section 2); fixtures come from those
  files. Unknown/absent never renders as healthy (NaN IoU, NOT_RUN,
  not-computed states are explicit).
- UI: browser verification with screenshots at every feature gate; every
  rendered figure viewed personally (view-figure-renders-personally); n=64
  smoke runs during development; DPI 180 and locked colormaps; no em dashes
  in UI text.
- One end-to-end real-data run per workflow before calling it done (library
  shape run, STL refusal + acceptance, densify study, solved-map screen
  against solve3d/results, two-run compare).

## 8. Build order (stage 2, after approval)

1. Skeleton app + run store read (LIBRARY list, run rail over existing runs,
   STUDY read-only on existing artifacts) - proves the surface on real data.
2. workbench_job + queue + live march series + enthalpy-default launch path.
3. STL intake gates (incl. self-intersection) + library gallery + campaigns.
4. STUDY viewer: cutting plane, x/y/z slices, front-contour rendering, Tier 1
   + Tier 2 time scrub, shape-metrics strip, gate banners, badges.
5. COMPARE screen.
6. SOLVED MAPS screen.
Each step independently shippable; commits with explicit paths only.

## 9. Open questions for Matt

1. Surface: confirm the new heatr3d_workbench/ app (section 3) over further
   surgery on the shared HEATR-3D tab. Port suggestion: 8081.
2. Densify time scrubbing (4.2 Tier 3): (a) wait for an additive snapshot
   hook in heatr3d.py (canonical-sync gated, after S2 frees the file), or
   (b) also ship opt-in checkpoint replay (K x cost, n<=64) in the interim?
3. Shape-metric nominal for STL runs: run-grid part mask only, or also the
   finer-grid STL reference mask (two labeled numbers)? Parametric/library
   shapes could additionally use analytic masks where they exist.
4. sigma_T compare bars: agreed to demote to diagnostic-only placement?
5. shape_library_3d lives on branch feat/shape-library-3d - merge it into
   feat/pernode-twosided-tuning before stage 2, or should the workbench
   vendor the STL/meta artifacts until the merge lands?
6. Badge wording: is the proposed "S1 verified (validity domain n<=96)" +
   exploratory chip the right public phrasing until S2, or should every
   heatr3d number carry the plain "exploratory" word until S2 passes?
