# Task Plan: 3a - wire the ceiling-restoration into the producer (autonomous, 2026-08-19 night)

## Goal
Make studio_solve's producer emit a GRADED, ceiling-respecting, SENDABLE dopant map
for coarse-meshable parts - not just a shape-only map that busts the ceiling and
falls back to uniform. Wire the B3/B4 ceiling-coupled AL restoration (targeting
T_eff = 250 - reserve) into the producer's shaped solve, BEHIND AN OPT-IN FLAG
(default OFF = current shape-only behavior preserved), so nothing changes for
other lanes until Matt approves flipping the default.

## Context / why
- Cube milestone (this session): producer emits drive 0.58x, runs a SHAPE-ONLY
  L-BFGS-B solve (box [0,1], J_asymmetric), does NOT run the B3/B4 restoration, so
  the graded map relocates the peak UP (260.97 C, over 250) -> is_sendable=FALSE ->
  Studio benefit gate reverts to UNIFORM. Ships, but ungraded. THIS is gap 3a.
- The B3/B4 restoration machinery EXISTS (stage_b3.al_objective_and_grad + the
  restoration shift). run_tamper_rescue.py is the template that injects it (build
  an ALCase + drive the AL loop). The cube B4 (hand-run) proved a graded cube
  ships: 235.08 dolfinx / 246.9 heatr3d, both < 250.
- ENVELOPE COST IS NOT A BLOCKER HERE: the envelope (J_shape melt-onset) is
  expensive only on FINE meshes (Tamper, n_sub~458). On COARSE parts (cube 9183
  nodes, CFL-stable, n_sub=1) the full B3/B4 solve is tractable (~87 min, Matt's
  "path A ~1hr/upload"). The Tamper/fine path is OUT OF SCOPE (Phase 8/research).

## Constraints (autonomous safety)
- OPT-IN flag, default OFF. Existing producer behavior byte-preserved for other lanes.
- TDD. ASCII only. Commit linearly to feat/pernode-twosided-tuning, only my files.
- ONE heavy verification solve max (the cube through the new mode). No fine-mesh/Tamper.
- Nothing pushed to remote. Leave everything ready for Matt's review.

## Phases
- [ ] P0: Investigate studio_solve's exact solve+emission path (where s_best is
      produced; how uniform_rec/solved_rec/gates/emission work). Confirm the
      injection point.
- [ ] P1: Spec (docs/superpowers/specs/) - the producer restoration mode, opt-in.
- [ ] P2: Plan (TDD tasks).
- [ ] P3: Build TDD - a `ceiling_restore` solve mode in studio_solve behind a flag;
      reuses stage_b3 AL + the recommended drive; emits the graded map + its
      standing-gate peak. Unit-test the wiring with a stubbed solve (no heavy).
- [ ] P4: Verify - ONE heavy cube re-solve through the new mode; confirm the graded
      map lands at ~T_eff (235-ish dolfinx) and its own standing gate is under 250
      (sendable on the producer's own engine). Record numbers.
- [ ] P5: Report + memory + hand-off note for Matt's review.

## Status: COMPLETE for the night (P0-P3 + knob done; P4 characterized not full-run; P5 report delivered).
- P3 mode built + verified: e377fed. P3b grade-mesh knob built + verified: f3e41a8. Both TDD, opt-in, legacy byte-identical, regressions green.
- P4 (heavy verify) NOT run as a full graded-melt solve - characterized instead
  (see findings). A meaningful run needs a cube-like part (melts within the 0.26-
  0.66x drive-candidate range) + --ceiling-restore --grade-node-density SOLVE/28
  --grade-lc0 SOLVE*3 (~1h). My solid block was too bulky (melts 1.5-2x, drive-
  limited for the producer) - a poor demo part.
- DRIVE APPLICATION VERIFIED: producer _ceiling_restore_al_loop applies the drive
  correctly (dataclasses.replace(ForwardParams, power_density) -> build_case, so tc
  carries the drive). Corrected sweep (rebuild tc per drive): 1.0x/180.6 1.5x/214.1
  2.0x/250.4 3.0x/325.5 C - properly drive-dependent, forward 5s each on the coarse
  mesh. My first sweep was a probe bug (build_al_case_from_tc is provenance-only).
- LATENT BUG FOUND (Tamper lane, pre-existing, flag for Matt): run_tamper_rescue.
  build_tamper_al_case builds tc via rt.build_case WITHOUT the drive, then passes
  power_density to build_al_case_from_tc (provenance-only) -> the Tamper rescue's
  "drive 1.2x" was actually rt.build_case's DEFAULT drive. Didn't matter yet (the
  rescue NaN'd, then was envelope-bound) but would mislabel/misfire a real run. Fix
  = build_tamper_al_case must build tc with a drive-scaled ForwardParams (like the
  producer does). NOT fixed tonight (Tamper is deferred; small change).

## Decisions / findings
- P0: injection point confirmed. solve_extruded calls run_solve (shape-only L-BFGS-B
  on J_asymmetric); studio_solve does NOT import stage_b3/b4. The restoration needs
  the drive selected FIRST (AL targets T_eff at that drive), so ceiling_restore is a
  REORDERED flow (drive -> fine-mesh guard -> AL restore -> emit), built as an opt-in
  path, legacy shape-only path byte-identical (default off).
- Envelope is cheap on coarse meshes (n_sub=1) so the full B3/B4 solve is tractable
  for the cube (~87 min); fine parts guarded out (n_sub>1 -> honest-null).
- Spec: docs/superpowers/specs/2026-08-19-producer-ceiling-restoration-design.md
- Memory updated: phase-e-adjoint-cfl-instability.md (envelope bottleneck + 3a pivot).

## P3/P4 findings (autonomous verification)
- P3 UNIT BUILD DONE + verified: commit e377fed. ceiling_restore mode (opt-in,
  default off, legacy byte-identical), DRY build_al_case_from_tc helper,
  run_tamper_rescue byte-identical, fine-mesh guard via da._explicit_n_sub, 46-51
  tests green. Solid, reviewable.
- P4 VERIFICATION - IMPORTANT SCOPING FINDING (the "coarse=tractable" premise is
  QUESTIONABLE): the producer's SOLVE_NODE_DENSITY makes EVERY part a LARGE solve
  (tube -> 38k in-part nodes / block -> 37.6k nodes, ~206k design vars),
  regardless of part size. The producer marches at dt=0.05 (MAX_TIME_S=500 ->
  10000 steps; restore_march defaults to drive_max_time_s=3000 -> 60000 steps).
  solve_extruded does NOT expose the reduced-budget knobs, so the PUBLIC
  ceiling_restore runs the FULL config (72 evals x 60k-step marches x 40k nodes x
  the coupled envelope) = computationally INFEASIBLE (worse than the Tamper).
- Fine-mesh guard PASSES on these meshes (n_sub=1 at dt=0.05, CFL-stable) - so
  they are not "fine" by CFL, they are just LARGE + long-march. The compute wall
  is mesh-size x march-length x envelope, NOT CFL instability.
- STRATEGIC: graded shipping is compute-blocked on BOTH paths (coarse-producer AND
  fine-Tamper) by the same root - the ceiling-coupled AL solve is too expensive at
  production config. The real investment is solve PERFORMANCE (coarser grading
  mesh? larger dt? reduced budget exposed as a knob?), not more wiring.
- MEASURED per-eval cost (the actionable resolution):
  * Production mesh (n_design=206053): one AL eval = 942s (~15.7 min) even at a
    TINY config (march=100, env=300s). Full config INFEASIBLE.
  * COARSENED grading mesh (n_design=7146, ~B4-cube scale, SOLVE_NODE_DENSITY/28,
    lc0*3): one AL eval = 45s (21x faster), n_sub=1. Full 72-eval extrapolation
    ~0.9h -> TRACTABLE (matches the hand-run B4 cube ~87 min at 9k nodes).
  * ROOT CAUSE: the producer's SOLVE_NODE_DENSITY over-resolves (~22x) for the
    SMOOTH dopant-grading AL. The dopant map is low-frequency, so a coarse grading
    mesh is physically fine (B4 proved it at 9k). ALSO: restore_march_time_s
    defaults to drive_max_time_s=3000 -> 60000 steps at dt=0.05 (a second lever).
- ACTIONABLE FIX (for Matt): expose two knobs on the producer's ceiling_restore -
  a COARSER grading mesh (grade_node_density ~ B4 scale) and a shorter/bounded
  restore march - so the ceiling solve runs in ~1h instead of never. Neither is a
  correctness change; the map is smooth. This turns 3a from "built but unrunnable"
  into "built + runnable" with a config exposure, no new physics.
- Demonstration (bac8vof7x): B3/B4 ceiling-restore on the coarse block, uniform vs
  restored KS, graded + under-250 check.

## NIGHT 2 (2026-08-24 autonomous): the definitive cube-like verification
- SELF-CORRECTION: the morning "~0.9h full solve" was extrapolated from a 45s eval
  that used a SHORT march (n_steps=200) + short envelope (300s). The FULL
  ceiling_restore config uses restore_march_time_s=drive_max_time_s=3000 -> 60000
  steps at dt=0.05, envelope 1800s. So the coarse MESH (7k vs 206k, 21x) is only
  ONE lever; MARCH LENGTH is a second. Must re-measure a realistic melting eval
  (march ~long-enough-to-densify + real envelope) before claiming tractability /
  launching. Do NOT trust the 45s number for the full run.
- Plan: (1) compact cube melt-drive check (in 0.26-0.66x range) [bq7ki9e8v];
  (2) MEASURE one realistic al eval (melting march + envelope) at coarse mesh;
  (3) if tractable (~minutes/eval), launch full/reduced ceiling_restore on the
  cube, watched; (4) assess graded-under-250; (5) report.

## NIGHT 2 RESULT (2026-08-24): demo-part hunt CLOSED, mode stands built+tractable
- Realistic-melt eval MEASURED at coarse grading mesh (n_design=7277, SOLVE/28,
  lc0*3): one full densifying AL eval (march 10000 steps/500s + envelope 1200s) =
  115s -> full ceiling_restore ~2.3h, reduced ~11-23min. TRACTABLE confirmed. The
  night-1 "45s/0.9h" was a short-march artifact (self-corrected); 115s is the real
  realistic-march number.
- DEMO-PART HUNT (3 candidates, all unsuitable - the honest gap):
  * Bulky solid block: cooks to 524C at 2.0x/500s, rho only 0.776. Drive-limited,
    poor coupling (y-centered bulk). NOT a demo part.
  * Compact centered cube: stayed cold (152C at 0.9x). Under-driven. NOT a demo.
  * Full-y-spanning slab [c,:,c] (physics-motivated: spans the y-electrode gap ->
    strong coupling) [bkk5kxt9z]: couples better (gradual 175->246C over 0.30-
    0.66x) BUT under-dense throughout - at 0.66x it is already at 246.5C (~ceiling)
    with rho only 0.648. Cannot reach rho>0.9 without busting 250. Drive-limited in
    a 500s march.
- ROOT of the gap: the ceiling_restore DEMO needs a part that HOT-SPOTS (uniform
  map busts 250 at a drive where the part still needs the heat to densify) so that
  REDISTRIBUTING dopant is what pulls the peak under 250. A uniform slab heats too
  evenly to show a grading benefit; a bulky block just won't couple. The cube
  milestone that DID show it (session cube: drive 0.58x, uniform relocates peak to
  260.97C > 250 -> is_sendable FALSE -> reverts uniform) is the right part shape,
  but its part_npz is NOT checked into this repo.
- VERDICT: 3a mode is BUILT, TDD-tested, committed (e377fed + f3e41a8), opt-in,
  legacy byte-identical, and TRACTABLE at coarse grading mesh (115s/eval). The one
  thing not producible autonomously is the definitive graded-under-250 SCREENSHOT,
  because it needs a real hot-spotting milestone geometry that is not in-repo. That
  is a ~15-30min run for Matt on an actual cube/part upload with
  --ceiling-restore --grade-node-density SOLVE/28 --grade-lc0 SOLVE*3.

## Errors
- Two probe typos self-corrected: shape_gen->studio_geom import; _explicit_n_sub
  returns a (n_sub, ratio) TUPLE (mis-compared to 1 first, corrected).
