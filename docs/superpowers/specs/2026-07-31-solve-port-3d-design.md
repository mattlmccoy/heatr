# Direct-Solve 3-D Port: Solved Volumetric Dopant Fields for RFAM Print Studio

Date: 2026-07-31
Status: DRAFT, approved-to-draft by Matt; PROVISIONAL pending two inputs from
the 2-D solve workstream (their grid hold-out 120->160 and rim-perturbation
tests, deliverable SOLVE_ROBUSTNESS_VALIDATION.md). Sections marked
[PENDING-2D] update when those land.
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
| Objective J = whole-domain sum (phi - chi)^2, chi = part indicator | proven, subgradient character measured | volume integral; chi from the voxelized/meshed STL. [PENDING-2D: sub-voxel fill convention - adopt whatever their robustness tests validate] |
| Envelope stop-time (t_stop = argmin, no dt*/ds term) | verified exactly (0.0 rel diff) | dimension-free; re-verify with the same S1-vs-S2 exact-agreement gate in 3-D |
| Steady EQS adjoint | 2-D proven | ALREADY DONE in 3-D (D1 Task 5, dolfinx) |
| Transient thermal-phase adjoint (coupled T, rho reverse march, clip VJPs) | 2-D proven | LARGEST PORT ITEM: implement in dolfinx with the enthalpy forward; Griewank-style checkpointing (store state at intervals, recompute segments) - the 2-D store-everything approach does not fit 3-D memory |
| FD/subgradient gate protocol (single-cell + multi-cell + random probes, eps sweeps, paired-difference estimator, subgradient labeling) | proven; measured failure mechanisms | port verbatim as the acceptance protocol for every gradient layer |
| Actuator: conductivity channel | proven | matches D1 dJ/dsigma |
| Actuator: permittivity channel | MISSING in 2-D (their #1 gap, queued) | add dJ/d-eps_r in dolfinx (cheap: complex-symmetric A^H = conj(A) reuse); do not declare the 3-D solve complete without it - the 2-D evidence says this channel decides shapes (their cross: J 343 with eps vs 1036 without) |
| Optimizer: L-BFGS-B on subgradients, box constraints | proven with known stalls | same, plus multi-start and scaled first step (their rectangle lesson); iteration-based budget accounting (not wall clock) |
| Drive convention | 2-D pins power-enforcement OFF | RECONCILE: one convention for the 3-D solve, chosen with the 2-D lane, before any cross-lane map comparison; D1 proved differentiating through the renormalization is tractable (and that freezing it is wrong by up to 150% per dof) |
| Regularization / rim structure | none in 2-D; rim structure possibly breakpoint sculpture | [PENDING-2D: if their rim-perturbation test shows J collapses under 1-2 cell smoothing, the 3-D objective ADDS a smoothness/TV term and a filter+projection parameterization from day one; if J is insensitive, port without] |

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
  stop-time envelope handling, optional regularization [PENDING-2D].
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
- Phase D: eps_r channel + drive reconciliation + the 2-D lane's validated
  conventions folded in [PENDING-2D].
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

1. SOLVE_ROBUSTNESS_VALIDATION.md (grid hold-out + rim perturbation) - sets
   the regularization decision and the chi convention.
2. The 2-D lane's frozen recording of: exact objective functional, FD-gate
   checklist, optimizer settings and budget rule, outside-part saturation
   convention, drive convention per arm (requested 2026-07-31).
3. Matt's approval of this spec once 1-2 are folded in.
