# heatr3d Graduation: Trusted Volumetric Grading and Shrinkage for RFAM Print Studio

Date: 2026-07-30
Status: approved design (brainstorm with Matt, this date)
Owner: Matt McCoy

## 1. Problem and goal

RFAM Print Studio (the renamed Meteor RIP and Slice tool plus the Grade tab)
currently plans dopant grading with the validated HEATR 2.5-D stacked-slice
pipeline. That decomposition is approximately correct for in-plane map shape
and explicitly co-designs the build-axis dose (z-gain), but it cannot design
grading truly volumetrically and it cannot predict how the full part
densifies, compacts, and distorts when melted. Those capabilities exist only
in heatr3d, which is currently untrusted (exploratory) for documented
reasons:

1. A coupled melt-onset instability at grid >= 200 (phi jumps 0.019 to 0.953
   in one step; energy residual ~1e5 x dose). Currently avoided by a grid
   cap, not understood.
2. The reported sigma_T = std(T) is not grid-converged (conductor-corner
   field singularity: finer grid gives higher std).
3. No formal calibration against measurement. HEATR 2-D/2.5-D has COMSOL and
   FLIR anchoring; heatr3d has neither.

Goal: make heatr3d real and trustworthy, so the Studio can (a) design dopant
fields volumetrically and (b) predict full-part densification and shrinkage
and eventually apply geometry pre-compensation automatically.

North star (Matt, restated 2026-07-30 after the S1 gate attempt): a FAST and
ACCURATE 3-D simulation of densification in the RF heating field, plus
inverse grading: given known conditioning, SOLVE for the dopant field whose
outcome is the intended printed geometry, so parts print well instead of
forming an arbitrary overheated shape. Forward fidelity is the foundation;
inverse design is the destination. This strengthens the adjoint criterion in
the D1 engine decision.

Decision context (user answers during brainstorm):
- Design target user tier: successors after graduation operate the tool
  (tier C usability), but the physics core comes first.
- Trust bar for shrinkage compensation: print-validated (option C) as the
  end state.
- Constraint: the RF apparatus is not currently operational. No new physical
  testing is possible yet. The near-term program is therefore maximum
  simulation-side confidence, with the physical campaign designed and parked.
- Available validation instruments once the rig runs: dimensional (calipers,
  flatbed scans via rfam-web, 3-D scan or CT), top-surface FLIR only,
  density (Archimedes or sectioning), mechanical tests.
- Dissertation relationship: ideally a full chapter arc (A); acceptable to
  straddle (C) with the simulation-side gates in the dissertation and the
  physical campaign designed and ready.

## 2. Approach: a trust ladder with explicit gates

heatr3d earns trust through ordered gates. Simulation-side gates (S1-S4) are
executable now with no lab time. Physical gates (P1-P2) are fully specified
but parked until the RF apparatus runs. The Studio exposes the current gate
level on every heatr3d-derived number, so tool capability and evidence level
can never silently diverge.

Rejected alternatives:
- Fix-first (pour effort into heatr3d numerics, then validate at the end):
  back-loads all evidence; instability may require deep surgery; worst shape
  for a defense timeline.
- Hybrid-by-construction (2.5-D thermal everywhere plus a small new
  mechanical settle model; heatr3d only as cross-check): cheapest, but
  abandons true 3-D EM physics (vertical field topology, top and bottom face
  coupling), which is the reason heatr3d exists. Its settle-model idea is
  held in reserve as a P2 component if heatr3d mechanics prove weak.

### Gate S1: numerical integrity (no lab time)

- Root-cause the melt-onset instability. Deliverable: a written mechanism
  (which coupling term, why grid-dependent), a fix or a principled stability
  criterion, and a regression test that runs the previously blowing-up case
  clean. Avoidance by grid cap is explicitly not a pass.
- Analytic and manufactured-solution benchmarks:
  - Thermal plus phase core: 1-D Stefan problem (analytic front position),
    slab and sphere conduction transients.
  - EQS core: parallel-plate field, dielectric or conductive sphere in a
    uniform field (analytic), charge and current conservation checks.
- Standing energy-conservation gate on every heatr3d solve (|residual| /
  integrated dose), mirroring the 2.5-D standard, printed in every run
  summary.

Pass criteria: instability reproduced, explained, and fixed with regression
test; analytic benchmarks within stated tolerances (set during
implementation planning, recorded in the test suite); conservation gate
wired into all entry points.

Status 2026-07-30: first campaign complete, GATE NOT PASSED (see
docs/superpowers/specs/s1-gate-report.md). The latent-skip defect was real
and is fixed (enthalpy update), but the documented grid >= 200 blow-up is a
powder-bed conduction CFL violation (THM-03, unstable for n > 179 at
dt = 0.05 s), and n = 200 cannot currently be solved at all (EQS-01: ILU
memory failure falls through a bare except into a segfaulting direct solve).
S1b split: strategy-independent fixes now (EQS-01 hard MemoryError with size
estimate; THM-03 stability assertion with the correct powder alpha plus
auto-substepping); the scalable large-N EQS investment waits for D1.

### Decision point D1 (before S2): FEM engine spike (dolfinx)

Approved addition (Matt, 2026-07-30). The voxel grid, not the Python
implementation, is heatr3d's structural weakness: staircase corners produce
the non-converging field concentration that pollutes the 3-D spread metric,
and the compaction-to-distortion mechanics (Gate P2's core) is what
hand-rolled voxel code does worst and structural FEM does best.

Reordered before S2 (Matt, 2026-07-30): the S1 gate attempt found that
heatr3d currently cannot solve above n~96 at all (EQS-01 segfault), so S2
is blocked regardless, the spike needs no rig, and its outcome decides how
much large-N investment heatr3d's own EQS deserves before S2 is designed.
Prerequisite: the S1b strategy-independent fixes (below). Time-boxed one
week:

- Reproduce the S3 extrusion-anchor case in FEniCSx/dolfinx on a
  geometry-conforming tetrahedral mesh from the same STL: complex-valued EQS
  solve, enthalpy-based thermal-phase march, same material parameters.
- Compare against heatr3d and the COMSOL anchor: field agreement, corner
  behavior under refinement, wall-clock cost, and implementation effort.
- Decision output, recorded in a short report: adopt dolfinx as (a) the
  high-fidelity cross-check engine and the P2 mechanics engine (heatr3d
  remains the fast in-tool planner), (b) mechanics engine only, or (c) not
  adopted (voxel cost acceptable). CalculiX remains the narrower fallback
  for role (b) if dolfinx tooling disappoints.

Rationale for dolfinx over alternatives: open source, Python-facing (lab
succession), complex-number support for EQS, unstructured meshes eliminate
the staircase-corner artifact class, and its adjoint ecosystem opens the
path to gradient-based volumetric FGM inverse design later. Rewrites in
non-Python stacks are rejected at this stage: trust is the scarce resource
and a rewrite restarts the trust ladder.

### Gate S2: convergence

- Grid and timestep refinement studies on 3-4 canonical geometries (sphere,
  extruded square, extruded L, one tapered solid).
- Replace raw std(T) as the headline 3-D uniformity metric with a metric
  that converges under refinement (candidates: quantile-trimmed spread,
  diffusion-regularized spread, or volume-weighted spread excluding the
  singular first boundary cell; chosen empirically during the study).
  The metric change is documented and the old metric remains available for
  comparison with historical numbers. The existing non-comparability rule
  (2-D ui_rms*(T_bar-23) vs any 3-D spread metric) stays in force.
- Deliverable: documented convergence bands (value plus band at the working
  grid) for the headline metrics; the working grid chosen from the study,
  not asserted.
- ABSORBED from the S4 re-score (Matt, 2026-08-01): the coupled-march
  densification question. The S4 coupling study proved sigma_density_coeff
  is inert under the pre-registered densify=False march; S2 adds at least
  one densify=True march with the in-march EQS re-solve enabled
  (eqs_update_interval_s > 0) to exercise the density-coupling term and
  characterize its effect on the convergence of the headline metrics. Any
  re-scoring of S4 FLIR cases under that coupling requires its own explicit
  re-registration; S2 itself only characterizes the term.

### Gate S3: cross-anchors (the strongest available substitutes for the rig)

- Extrusion anchor vs the validated 2.5-D stack: an extruded 2-D shape
  solved in heatr3d must reproduce, at mid-height, the validated 2.5-D
  cross-section behavior: calibrated drive (V_cal), part-mean heating curve,
  melt timing (t90), and the in-plane Q_rf pattern, within stated
  tolerances. Run for at least: circle (the d=20 anchor), square, and one
  reentrant shape.
- COMSOL 3-D anchor: compare heatr3d EQS fields (V, |E|, Q_rf) against the
  Tuned_Sigma.mph volumetric exports already in the repo, on the reference
  geometry. Extend exports from COMSOL if needed.
- Densification-law consistency: where geometry permits (tall extrusions),
  the per-voxel densification trajectory must match the 2-D law applied to
  the corresponding cross-section history.

### Gate S4: historical-data anchor (real physics, zero lab time)

- Score heatr3d predicted top-surface temperature fields against Jared
  Allison's decoded FLIR sequences (extract_flir_seq.py pipeline and the
  existing experimental comparison scripts). This is the same anchoring move
  that made 2-D trusted, applied to the 3-D tool using data that already
  exists.
- Deliverable: quantitative agreement report (pattern correlation, peak
  location error, timing), with an honest account of unknowns in the
  historical setup (drive calibration, emissivity, exact geometry).

### Gates P1-P2: physical (designed now, executed when the rig runs)

- P1 thermal: FLIR top-surface calibration prints on 2-3 geometries;
  calibrates the 3-D thermal field like the 2-D calibration did.
- P2 shrinkage: CT or 3-D scan plus Archimedes density on printed parts vs
  the predicted compaction field. Density pins the middle of the
  thermal -> densification -> distortion chain so a shape miss can be
  attributed to the right link. Passing P2 unlocks automatic prewarp in the
  Studio.
- Deliverable now: a written experiment plan for P1-P2 (geometries, sample
  counts, measurement protocol, acceptance thresholds) produced with the
  rfam-experimentalist workflow, so the campaign starts the day the rig is
  up. Dissertation fallback C cites this plan as designed-and-ready.

## 3. Studio integration: capability tiers bound to gates

Every heatr3d-derived quantity in RFAM Print Studio carries a visible badge
of the highest gate it has passed (for example "S3-anchored" or "P
pending"). Concretely:

- After S1-S2: heatr3d results appear in the Grade tab as clearly badged
  exploratory-plus (numerically verified) visualizations: volumetric T and
  phi fields, per-voxel density.
- After S3-S4: volumetric grading design and full-part shrinkage or warpage
  overlays ship as advisory (shown, not acted on). The 3-D view gains a
  predicted-distortion overlay (exaggerated displacement rendering).
- After P1: 3-D thermal numbers lose their caveat badge.
- After P2: the "apply prewarp compensation" action unlocks. Until then the
  button exists, visibly locked, stating exactly which gate it awaits.

The badge system is the tier-C safety mechanism: successors can see what is
trustworthy without knowing the history.

## 4. Dissertation cut

- All S-gates passed: a verification-and-validation chapter arc for the 3-D
  process model (instability root-caused, converged metrics, anchored to the
  validated 2-D tool, COMSOL, and historical experimental data), plus the
  designed physical campaign. This stands alone even if the rig stays down.
- Rig recovers before the deadline: execute P1 (and one P2 case if time
  permits) to upgrade toward the ideal full arc.
- Every gate produces defensible artifacts independently; the cut point is
  a deadline decision, not a design decision.

## 5. Usability track (separate sub-project, sequenced second)

Tier-C usability work proceeds independently of the physics core and gets
its own spec when it starts. Scope reserved: Windows and install story for
the print PC, presets and locked-down defaults, onboarding documentation,
per-print provenance records (config, gate levels, calibration state at
print time), and self-diagnostics. Not in scope for the physics-core plan.

## 6. Out of scope (this design)

- Multi-part bed grading interactions (existing multipart studies remain
  preliminary).
- Orientation and turntable design automation (routing to them stays).
- GH Pages hosted front-end (architecture already supports it; deferred).
- Any claim about dopant edge width or print registration (unmeasured until
  the rig runs).

## 7. Execution notes

- S1 work is solver-core engineering: use the computational-solver-engineer
  workflow with FD or analytic gates on every change; every fix lands with
  its regression test (red first).
- S2-S4 are study campaigns: driver scripts with resumable runs, JSON
  artifacts, and honest gate reports, following the established
  dual-readstate campaign pattern.
- The P1-P2 experiment plan is authored with the rfam-experimentalist
  workflow against the instrument list above.
- Tool-trust and metric rules from HEATR_STANDARD_PARAMETERS.md remain in
  force throughout; nothing in this program relaxes the 2-D vs 3-D metric
  non-comparability rule.
