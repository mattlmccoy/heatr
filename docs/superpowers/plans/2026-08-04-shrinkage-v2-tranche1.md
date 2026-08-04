# Shrinkage v2 Tranche 1: L0 in solve3d, Arbitrary-STL Tet Meshing, CG/AMG Spike

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** This lane's first tranche of the APPROVED shrinkage-prewarp v2 spec (docs/superpowers/specs/2026-08-04-shrinkage-prewarp-v2-design.md): Level 0 pre-compensation in the solve3d intake, arbitrary-STL tet meshing (unblocks L2 and the Studio's direct-solve rung), and the CG/AMG iterative-EQS spike (compute item 1). Studio-side L0/L1 belongs to the Studio lane and is NOT in this plan.

**Authorship note:** executing agent is the Phases A-E author; tasks give files, red-first tests, and numeric gates.

**Ground rules:** frozen conventions stand; every gate JSON-quoted; no widening; atomic solve3d/-only commits (+ this plan); no push without protocol; dissertation_materials READ-ONLY; OMP/OPENBLAS=1; compute-scheduling convention for anything heavy (announce, load<20, one heavy slot).

### Task 1: Level 0 pre-compensation in solve3d intake

- [ ] Red tests (solve3d/tests/test_precomp.py): (a) affine map applies xy_scale=1/(1-s_xy), z_scale=1/(1-s_z_mat) to geometry BEFORE chi/mesh, exact on analytic vertices; (b) DOUBLE-COUNTING GUARD: with densify on, the pre-scale must NOT include consolidation (assert the config schema separates s_xy/s_z_mat from any densification parameter, and a run with s_*=0 is bit-identical to today); (c) coefficients read from config with defaults from SHRINKAGE_COEFFICIENTS_MEMO.md (s_xy=0.030, s_z_mat=0.020) and recorded in run provenance with the uncertainty band.
- [ ] Implement solve3d/precomp.py + intake wiring. Green. Commit.

### Task 2: Arbitrary-STL tet meshing

- [ ] Red tests (solve3d/tests/test_stl_mesh.py): watertight library STL (pyramid) meshes to a conforming tet mesh whose volume matches the analytic value within a stated tolerance; chi volume-fill on the STL mesh passes the shared fill contract; non-watertight and self-intersecting fixtures are REFUSED loudly (reuse the Tier-3 library fixtures); the Phase A circle anchor meshed via the STL path reproduces the OCC-construction mesh results within the Phase A same-engine bands (the honest equivalence gate).
- [ ] Implement (gmsh or the spike env's meshing route - justify choice; record versions). Green. Commit. Then notify: this widens the Studio's direct_solve gate from is_extrusion to has_solve_mesh (their side flips it).

### Task 3: CG/AMG iterative EQS spike

- [ ] Pre-register (solve3d/results/cgamg_protocol.json): solver config (PETSc CG + AMG per D1's precedent), acceptance = solution matches the direct solve within a MEASURED tolerance (report the direct path's own rtol floor first), speedup measured at Phase A coarse AND mid meshes, adjoint solve gets the same treatment (A^H system).
- [ ] Red test: iterative-vs-direct field agreement gate + FD re-gate of one Phase B probe with the iterative path in the loop (gradient correctness must survive the solver swap).
- [ ] Implement behind a flag (default = direct, bit-identical off-path). Benchmark honestly (load-checked). Record speedups in solve3d/results/cgamg_results.json. Commit. Recommendation to Matt for default flip if gates pass.

### Task 4: Report

- [ ] TRANCHE1_REPORT.md in solve3d/: gates table, meshing route + versions, speedup table, what is NOT covered (L1/L2/L3, Metal port, CUDA), next-tranche recommendation. Commit.
