# Direct-Solve 3-D Port: Solved Volumetric Dopant Fields for RFAM Print Studio

Date: 2026-07-31
Status: APPROVED by Matt 2026-08-01 (updated same date with the 2-D
robustness results, SOLVE_ROBUSTNESS_VALIDATION.md; the former [PENDING-2D]
decisions are evidence-based and RESOLVED below). Phase A may start.
Companion decision, same date: the untested sigma_density_coeff /
densify=True coupled-march question from the S4 re-score is ABSORBED INTO
S2 scope (graduation ladder), not re-registered as an S4 extension.
Owner: Matt McCoy

## 1. Goal

Port the 2-D directly-solved FGM creation method to 3-D on the graduated
solver stack, so the Studio can produce SOLVED per-layer dopant masks: given
a part geometry and known conditioning, solve for the volumetric dopant
field whose predicted outcome is the intended printed shape, then slice it
through the existing dopant_volume -> graded-TIFF path. This is the
north-star inverse-design destination of the graduation spec
(docs/superpowers/specs/2026-07-30-heatr3d-graduation-design.md).

Why now (evidence from both lanes, 2026-07-31):
- 2-D lane: solved maps beat the inversion-heuristic family (13/18 oracle /
  16/18 standardized), and ideal maps are structurally OUTSIDE that family
  (R^2 0.01-0.12 vs the proportional-inverse proxy).
- 3-D lane: the cylinder null (FGM_BENEFIT_RERUN.md) - on corrected fields
  the inversion rule yields +0.3% (nothing) where a real optimum plausibly
  exists. The heuristic runs out of signal exactly when the field is honest.
- The hard 3-D ingredient is already proven: the D1 spike's gated adjoint
  (dJ/dsigma over 105,191 DG0 dofs at 0.2% of a forward; worst per-dof FD
  error 7.4e-5; mutation-tested) in dolfinx.

## 2. What is being ported (ingredient map)

From the 2-D lane (paths in the review of 2026-07-31; canonical artifacts
in their worktree adjoint2d/ and the root FGM_*/SHAPE_* reports):

| ingredient | 2-D status | 3-D port action |
|---|---|---|
| Objective J = whole-domain sum (phi - chi)^2, chi = part indicator | proven, subgradient character measured | volume integral; chi must be MESH-INDEPENDENT (signed-distance or sub-cell fill from the STL, never a binary raster on the solve grid). RESOLVED 2026-08-01: the 2-D grid hold-out showed rim solutions tune themselves to the solve grid's boundary rasterization (SOLVED class empties at 160: square IoU 0.9816 -> 0.7767; circle ranking FLIPS) |
| Envelope stop-time (t_stop = argmin, no dt*/ds term) | verified exactly (0.0 rel diff) | dimension-free; re-verify with the same S1-vs-S2 exact-agreement gate in 3-D |
| Steady EQS adjoint | 2-D proven | ALREADY DONE in 3-D (D1 Task 5, dolfinx) |
| Transient thermal-phase adjoint (coupled T, rho reverse march, clip VJPs) | 2-D proven | LARGEST PORT ITEM: implement in dolfinx with the enthalpy forward; Griewank-style checkpointing (store state at intervals, recompute segments) - the 2-D store-everything approach does not fit 3-D memory |
| FD/subgradient gate protocol (single-cell + multi-cell + random probes, eps sweeps, paired-difference estimator, subgradient labeling) | proven; measured failure mechanisms | port verbatim as the acceptance protocol for every gradient layer |
| Actuator: conductivity channel | proven | matches D1 dJ/dsigma |
| Actuator: permittivity channel | LANDED in 2-D (commit 890dc22): census 13/18 -> 17/18 vs best stored masks. DEPLOYABILITY CAVEAT (2026-08-01): rfam_eqs_coupled.py:290-292 asserts the real binder's eps_r is FIXED at 20 with only sigma varying - until Matt settles the material question (VNA dielectric vs carbon-black loading), conductivity-only is the deployable channel and eps results are MODEL-ONLY | add dJ/d-eps_r in dolfinx (cheap: complex-symmetric A^H = conj(A) reuse) as a MODEL-ONLY channel, badged as such; do not present eps-channel maps as printable until the material question is settled - the 2-D evidence says this channel decides shapes (their cross: J 343 with eps vs 1036 without) |
| Optimizer: L-BFGS-B on subgradients, box constraints | proven with known stalls | same, plus multi-start and scaled first step (their rectangle lesson); iteration-based budget accounting (not wall clock) |
| Drive convention | 2-D pins power-enforcement OFF | RECONCILE: one convention for the 3-D solve, chosen with the 2-D lane, before any cross-lane map comparison; D1 proved differentiating through the renormalization is tractable (and that freezing it is wrong by up to 150% per dof) |
| Regularization / rim structure | CONFIRMED load-bearing rim sculpture: one-cell blur costs +37% to +892% J on 5/6 shapes (rectangle, the stalled map, IMPROVES); the shapes losing most to blur lose most to the grid change - one mechanism, two witnesses. TOPOPT ablation (TOPOPT_REPORT.md, 2026-08-01) REALLOCATES CREDIT: grid transfer is bought by the PHYSICAL-radius filter + area-fill chi (4/6 improve-or-hold at 160; circle IoU 0.9616), while the Heaviside projection is what passes the sub-radius-sensitivity gate (152.7% -> 1.4% square) but COSTS in-grid fidelity (beta=0 control IoU 1.0000 vs 0.8734 projected). No 2-D map passes both gates in all forms; nothing labelled SOLVED there yet | UPDATED 2026-08-01: filter (frozen 2-D value: 1.0 mm physical radius, solver-convergence justified) + grid-independent area-fill chi are MANDATORY from Phase A of the solve layers. Projection sharpness (eta=0.5, beta continuation 1/2/4/8/16 in 2-D) is a TUNABLE fidelity-vs-robustness trade, not fixed-on; the 3-D solve reports both a low-beta and a continued-beta arm against the Phase C gates. Part of the residual grid-160 gap is FORWARD discretization (uniform arm alone moves 0.076 IoU) - Phase A forward parity quantifies how much of that gap is closable at all |

## 3. Architecture

New module directory `solve3d/` at geo-prewarp root (same isolation pattern
as heatr3d_d1_spike/, which it imports from and eventually absorbs):

- `solve3d/forward.py` - dolfinx forward: complex EQS (from the spike's
  eqs_common conventions, corrected-Q only; legacy Q is forbidden here by
  construction) + enthalpy thermal-phase march (ported semantics from
  heatr3d's phase_update="enthalpy", solved on the FEM mesh).
- `solve3d/adjoint.py` - steady EQS adjoint (lift from spike adjoint_core)
  + the new transient reverse march with checkpointing.
- `solve3d/objective.py` - shape-fidelity J, chi construction from STL,
  stop-time envelope handling, MANDATORY filter + Heaviside projection regularization (resolved 2026-08-01).
- `solve3d/gates.py` - the ported FD/subgradient protocol + energy gate +
  the S1-style standing prints.
- `solve3d/solve.py` - optimizer loop, budgets, multi-start, artifacts
  (results JSON per solve, same no-transcription rule as D1).
- Studio integration LAST and thin: a solved volumetric field resamples
  onto the dopant_volume.npz grid and flows through the EXISTING slice ->
  graded-TIFF -> Hot Folder path unchanged. It enters the Studio behind the
  gate-badge system: "solved (S-gates only)" until P-gates exist.

Environment: the spike env (heatr3d_d1_spike/env) is reused as-is,
including jit_fix.py (the Dropbox-path shim). Promotion to a first-class
environment happens only when solve3d leaves prototype status.

## 4. Trust coupling (what this port does NOT do)

The port certifies the DESIGN METHOD; the trust ladder certifies the
PHYSICS. A solved map inherits the forward model's gate level and the
Studio badges it accordingly. Ladder work (S1 completion, S2, S3, S4,
P1/P2) proceeds in parallel and is prerequisite for printing solved maps
with confidence. The solve must always run on corrected/dolfinx fields;
building J on legacy heatr3d Q is a construction-time error, not a config
option.

## 5. Phases and gates (each phase = its own implementation plan)

- Phase A: forward parity. dolfinx thermal-phase march reproduces heatr3d's
  enthalpy march on the extrusion-anchor cases (t90, sigma_T, heating curve
  within stated tolerances) and passes the S1 analytic benchmarks (latent
  plateau, Fourier decay). Gate: numeric, recorded JSON.
- Phase B: transient adjoint. FD/subgradient protocol passes on the coupled
  march (the 2-D acceptance thresholds, ported); checkpointing validated
  (gradient identical vs store-everything on a small case); cost target:
  gradient <= ~2 forward-equivalents (2-D achieved 1.4-1.7).
- Phase C: single-shape 3-D solve. Extruded circle and the CYLINDER NULL
  case: does the solve find the uniformity the inversion rule cannot? This
  is the port's decisive demonstration. Compare against uniform, the
  corrected-design inversion map, and the 2-D solved map extruded (where
  meaningful). Budgets and baselines pre-registered.
  ACCEPTANCE now includes (2026-08-01, from the 2-D robustness lesson):
  a built-in mesh hold-out (solve on mesh A, score on refined mesh B) and
  a smoothing-robustness check (J insensitive to sub-filter-radius
  perturbation) - a 3-D map is not called solved unless it survives both.
  The 2-D result predicts these pass BY CONSTRUCTION with filtering on;
  the gate verifies the prediction.
  OBJECTIVE REFINEMENT (Matt, 2026-08-01): the goal is dense IF AND ONLY IF
  in-bounds - out-of-bounds melt (bed growth) is the hard penalty side;
  in-bounds under-density is a SOFT trade with a floor near 80-90% density.
  The symmetric (phi-chi)^2 does not encode this; the Phase C objective
  uses asymmetric weighting (or a hinge on in-bounds phi below the floor),
  with sigma_T kept as a reported flatness diagnostic, never the objective.
- Phase D: eps_r channel + drive reconciliation + the 2-D lane's frozen
  conventions folded in (their conventions doc still owed; drive lesson
  from the robustness report: pinned voltage vs dose-matched changed
  margins <= 4.5 points and NO rankings in 2-D - reassuring, but the 3-D
  solve still standardizes one convention).
- Phase E: library campaign (the 3-D analog of their 18-shape library) and
  Studio integration behind badges.

## 6. Out of scope

- Turntable/rotation (design requirement on record: part and sat must
  co-rotate from the first commit, with a regression test - the 2-D lane's
  co-rotation defect must be impossible here by construction).
- Mechanics/shrinkage (P2 lane; the solved map feeds it, not vice versa).
- Multi-part beds; temporal power scheduling as a design variable (their
  temporal work may promote this later).
- Any dissertation edits.

## 7. Inputs awaited before Phase A starts

1. RECEIVED 2026-08-01: SOLVE_ROBUSTNESS_VALIDATION.md - regularization
   and chi decisions folded in above. Headline kept honest: absolute
   fidelity does NOT transfer across grids in 2-D (unregularized); the
   solved-vs-uniform win DOES transfer 6/6; rankings vs historical 5/6
   (circle flips). The port's filtering-by-construction is the designed
   answer, verified by the Phase C acceptance gates.
2. RECEIVED IN FULL 2026-08-01: FROZEN_CONVENTIONS_2D.md at geo-prewarp
   root (commit b04e356; companion TOPOPT_REPORT.md). Final frozen values:
   filter radius 1.0 mm PHYSICAL (solver-convergence justified; printer
   dopant edge scale is finer than any solve grid so not binding);
   smoothed-Heaviside eta=0.5, beta continuation 1/2/4/8/16; chi =
   grid-independent sub-cell AREA FILL (analytic circle test). Projection
   status per the ablation: TUNABLE trade (see ingredient table), not
   fixed-on. Earlier partial delivery, kept for the record:
   - Drive per arm: voltage-driven, per-shape calibrated v_cal
     (geometry_dual_readstate campaign), enforce_generator_power=False;
     absorbed power reported per arm; dose NOT matched (stated limit).
   - Outside-part saturation = 1.0 (adjoint2d/control.py:39-61,
     deliberate and load-bearing; historical masks mix outside=1/asstored,
     solve arms are outside=1.0).
   - FD-gate checklist: L0 bit-identity vs production before any gradient;
     central-difference gates at max-sensitivity cell + random cell +
     random direction; 1e-5 subgradient standard with the measured
     evaluation floor reported; filter/projection TRANSPOSE checked
     against the adjoint exactly; flag-off bit-identity regression per
     new channel.
   - Optimizer/budget: L-BFGS-B, box [0,1], filtered full-depth SINGLE
     start at 40 forward-equivalents per shape (multi-start splitting the
     budget costs depth, 7/18 vs 9/18); warm-start only where a strong
     historical mask exists; adjoints counted at measured
     forward-equivalent cost.
   - NEW WITNESS for the physical-radius requirement: their 1.5-2 CELL
     filter at grid 120 cuts the square's one-cell-blur cost +152.7% ->
     +9.9% (smoothing robustness fixed in-grid) but grid-160 transfer
     recovers only IoU 0.8063 vs 0.7767 unfiltered (0.968 in-grid) - a
     cell-count radius does NOT fix grid transfer. Physical length stands.
   - STILL OWED: physical filter radius value + beta continuation
     schedule (their topology pass); their full-fix grid hold-out rerun
     is that pass's acceptance test.
3. RECEIVED 2026-08-01: Matt's approval of this updated spec.
4. Sequencing note: the S4 field-coupling finding (frozen Q_rf) is being
   fixed in heatr3d now; the solve3d forward inherits whatever coupling
   law lands, so Phase A parity targets the coupled forward, not the
   frozen one.
