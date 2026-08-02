# STUDIO ALPHA plan (phase 1 of the single-tool vision)

Status file for this session. Rules: TDD red-first for pure logic; no commits;
no em dashes; never the word beginning with "surro"; expand acronyms.

## Decisions (locked before code)
- Production DPI = 720 (Meteor native; fgm_generator.py:70 default `dpi: int = 720`;
  resample path fgm_generator.py:588-606). At grid 120 over the 60 mm chamber the
  level_map is 1715 x 1715, matching the stored production artifacts
  (SHAPE_LIBRARY_SOLVE_REPORT.md: "five ACTUAL stored 1715 x 1715 level_map artifacts").
- Packages land in top-level `packages/`; that directory IS the scheduler drop
  folder for now (real uploader waits on hardware). Naming:
  `packages/pkg_<part>_<YYYYMMDD-HHMMSS>/` plus a sibling `.zip`.
- The plan-run driver reuses run_intake_novel.py machinery (score_static,
  solve_static, solve_mode imported, not copied).
- Production verify: static best arm uses the real engine via fgm_feedback
  sat_map_npz_direct (solve_fgm convention); rotating best arm uses engine
  turntable PROGRAM mode with corotate_dopant + corotate_eps_geometry ON
  (ENGINE_DWELL_SUPPORT_NOTES.md section 5.2). Agreement thresholds: 1 percent
  static, 5 percent rotating (quasi-static march vs engine gap is real physics,
  measured -1.16 percent class on unequal dwell).
- GUI: "Import & Plan" section at the TOP of the Operation tab (v2 placement).
  Analyze = blocking subprocess (seconds). Run recommended plan = new job mode
  `import_plan` through the existing queue. Customize = drops intake into the
  existing fgm_solve form via an imported-geometry token the launcher accepts.

## Phases
- [x] P0 read the required docs + code recon
- [ ] P1 print_package.py: manifest schema + emitter, TDD red first,
      real-data PNG round-trip contract test (keyhole_maps.npz fixture)
- [ ] P2 studio_plan.py pure logic (plan card, difficulty class, advisory text), TDD
- [ ] P3 scripts/analysis/intake_analyze.py CLI (geometry file -> plan JSON)
- [ ] P4 scripts/analysis/run_import_plan.py (recommended plan -> package)
- [ ] P5 server endpoints + job mode, TDD on pure parts (test_gui_studio_alpha.py)
- [ ] P6 front end Import & Plan section; live browser verify
- [ ] P7 verification gate: fresh server, keyhole through GUI, smoke budget end
      to end, package inventory, legacy launch smoke, zero console errors
- [ ] P8 STUDIO_ALPHA_NOTES.md + final report

## Verification gate checklist (from the directive)
- [ ] plan card renders with the stored keyhole recommendation (MAP_PLUS_MODE via
      continuous, calibrated 2630.33 V class, C1v)
- [ ] package lands in packages/ with every manifest field populated
- [ ] raster at production DPI (1715 px at grid 120)
- [ ] run visible in Results tab
- [ ] zero console errors
- [ ] legacy launch smoke (existing mode untouched)
- [ ] view plan card + key package artifacts personally
