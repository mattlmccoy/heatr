# Shrinkage and Prewarp v2: Solving for the Measured Part

Date: 2026-08-04
Status: DRAFT per Matt's direction of 2026-08-04 (levels approved in
discussion; this document is the written spec for his review).
Owner: Matt McCoy

## 1. Goal and the core distinction

Make the FGM solve target the part you MEASURE after printing, not the
melt field. Two separate physical effects, never conflated (Matt's
distinction, recorded):

- MATERIAL SHRINKAGE: the 1-3% class XY (and Z) contraction of nylon 12
  itself on melt/recrystallization, as published for SLS. Anisotropic,
  roughly affine at part scale. NOT currently in any model here.
- DENSIFICATION CONSOLIDATION: the powder-to-solid volume change
  (rho 0.55 -> ~1.0, ~45% volumetric, expressed dominantly in Z as
  vertical collapse). The density FIELD is marched by heatr3d; the
  GEOMETRY does not move in-model. The Studio's studio3d densified gauge
  (bed-suspended, plate-resting) is the current post-hoc geometry
  estimate.

Compensation must enter BEFORE or INSIDE the solve: the objective is
geometry-nonlinear, so post-scaling a solved map is wrong (Matt,
recorded).

## 2. Levels

### Level 0: affine pre-compensation (build first)

- Anisotropic inverse scale applied to the STL BEFORE chi construction
  and voxelization: xy_scale = 1/(1 - s_xy), z_scale = 1/(1 - s_z_mat),
  with s_* the MATERIAL shrinkage coefficients only (consolidation is
  the model's job, not the pre-scale's - double-counting guard is a
  named test).
- Coefficients from SLS PA12 literature (Matt: "plenty of papers...
  we can absolutely use these values"). Deliverable
  SHRINKAGE_COEFFICIENTS_MEMO.md: published values table with citations,
  chosen defaults + stated selection rule, uncertainty range. Config-
  driven (never hardcoded), recorded in run provenance, uncertainty
  carried as a labeled band on any dimensional claim.
- Applies to: Studio pipeline (before chi), solve3d campaigns, workbench
  STL runs. Off by default until the memo lands; then default ON with
  the coefficients displayed.

### Level 1: prewarp fixed-point loop (build second)

- Loop: solve dopant -> forward with densify -> predict deformed
  geometry via the collapse-kinematics gauge (studio3d's densified
  gauge, promoted to a shared library function with tests) -> compare to
  nominal -> warp the INPUT geometry by the inverse displacement ->
  re-solve. 2-3 iterations budget; convergence = displacement update
  below a pre-registered floor; non-convergence reported, never hidden.
- This is the ILT-style geometry prewarp of the original charter, now
  with a gated solver inside the loop.

### Level 2: differentiable shrinkage in the objective (the research core)

- Extend J: compare the PREDICTED FINAL SHAPE S(rho_final(s)) against
  nominal chi, with S the collapse-kinematics map (differentiable,
  simple per-column integral form first). Adjoint carries
  d(final shape)/d(dopant) through the existing Phase B machinery plus
  one new VJP layer (the kinematics map). FD-gated per the frozen
  protocol; mutation tests (dropped-kinematics-VJP mutant must fail).
- Requires the rho co-state (Phase B deliberately matched the forward's
  fixed-rho mode; this level ports the densify march into solve3d's
  forward + adjoint - the largest engineering item at this level).
- Result: the solve directly minimizes deviation of the measured-part
  prediction from nominal - melt fidelity becomes an intermediate, not
  the target.

### Level 3: full sintering mechanics (ASAP per Matt, staged honestly)

- L3a FORWARD: a viscous-sintering mechanics module (Skorohod-Olevsky
  viscous sintering class: sintering stress + temperature-dependent
  shear/bulk viscosity, quasi-static momentum balance marched with the
  thermal solve), on the solve3d FEM stack. Its own S1-class gate suite:
  analytic benchmarks (free sintering of a sphere, gravity slump limit),
  energy/volume audits, convergence bands, literature anchoring; then
  S3-class comparison against the collapse-kinematics gauge and any
  measured parts.
- L3b ADJOINT: differentiate the mechanics march (same checkpointed
  reverse-march pattern as Phase B) so the full model sits inside the
  solve. Only after L3a passes its gates.
- Trust: L3 predictions carry exploratory badges until P-gates exist;
  L2's simple kinematics remains the deployable default until L3 beats
  it against measurement.

## 3. Compute (Matt's direction, 2026-08-04)

- Platform now: M2 MacBook Pro. Order of work: (1) CG/AMG iterative EQS
  on CPU (D1 precedent 229-431x; S1-class gates, this lane); (2)
  Metal/MPS march kernel as an engine_speed-pattern parallel module
  (tolerance-floor gates - GPU floats forfeit bit-identity, disclosed);
  (3) CUDA bifurcation later as a parallel backend behind the same gate
  suite, when hardware exists. The march/EQS gains compound with L2/L3,
  whose solves are the long-pole runs Matt named.

## 4. Sequencing and ownership

L0 (Studio + solve3d intake; coefficients memo first) -> L1 (Studio loop
using the shared gauge) -> L2 (solve3d, this lane, after the arbitrary-
STL tet-meshing increment) -> L3a -> L3b. Compute items (1) and (2)
proceed in parallel with L0/L1. Symmetry projector increment (already
queued) slots before or alongside L2 - it reduces L2/L3 solve cost.

## 5. Out of scope

Multi-part beds; thermal warpage of the surrounding CAKE; printer
placement compensation; any dissertation edit from this lane.

## 6. Open questions for Matt

1. L0 coefficient selection rule when literature values disagree:
   midpoint with band, or conservative end?
2. L2 target read: final shape at end-of-exposure or after a cooldown
   segment (recrystallization shrinkage happens on cooling - literature
   memo will inform whether cooldown must be modeled for L2 or only L3)?
3. L3a validation data: any measured RFAM part dimensions from prior
   prints available before the rig returns?
