# Studio handoff (2-D lane -> Studio/port lane, 2026-08-02)

Transplant material per the confirmed ownership split: the Studio product
surface (package format, scheduler, orchestration) is OWNED BY THE STUDIO
LANE; these files are input for their package-format spec, not a competing
implementation. Authored by a stopped agent that had completed exactly one
red-green unit before the split was enforced.

- print_package.py + test_print_package.py: manifest schema draft
  (SCHEMA_VERSION 1.0.0), validate_manifest, and the PRODUCTION-DPI raster
  requirements with citations (720 DPI Meteor native, fgm_generator.py:70;
  resample path :588-606; 1715x1715 level_map class at grid 120 / 60 mm).
  The DPI emission requirement is the one piece of real engineering here:
  the intake/solve route must emit at printer DPI, not solve grid.
- studio_plan.py + test_studio_plan.py: plan-card logic sketch (untested
  beyond red stubs; take or discard freely).
- studio_alpha_plan.md: the stopped agent's own plan file, for context.

Interim contract until the Studio lane's versioned format lands: MetPrint
hot-folder job_info.json for print jobs + tt_program (angle, duration) JSON
for turntable programs, side by side.
