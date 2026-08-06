# Thermal-Ceiling-Constrained Joint Solve: Dopant, Drive, and Schedule

Date: 2026-08-06
Status: DRAFT for Matt's review. Objective expansion GREENLIT in principle
2026-08-06 ("Greenlight, spec it first"); no build until this spec is
approved. Interface sections (Stage A power_settings, Stage C tt_program
extension) route to the Studio lane before any schema freeze.
Owner: Matt McCoy

## 1. The problem, and why the dopant alone cannot solve it

Measured (Studio heatr3d densify verify, n=64, end-state reads): a
shape-optimal dopant FGM beats uniform on shape metrics yet still cooks
over the 250 C ceiling at full densification. Pyramid FGM end-state peak
281.7 C (uniform 277.3 C); cube ~265 C. Both over, both unsendable.

Root cause is conservation, not a solver defect: at fixed total absorbed
power the dopant map changes WHERE heat goes, not HOW MUCH. It can
relocate the peak, not lower its magnitude. The measured lever is drive
backoff (pyramid: 0.55x drive reaches 243 C UNDER ceiling at rho 0.98,
at ~3.2x exposure), and the durable lever is a dwell/rotation schedule
that spreads the same energy over time and angle to hold the peak down
with less bed leakage than a slow uniform burn.

Consequence for the objective: the ceiling CANNOT be a constraint on the
dopant-only problem (infeasible above ~0.7x drive). It must be satisfied
by the DRIVE and/or SCHEDULE actuator, with the dopant optimizing shape
WITHIN the feasible envelope.

## 2. The read-state / densify prerequisite (couples to shrinkage-v2 L2)

The ceiling violation is an end-state quantity: the peak occurs at the
rho=0.98 densification stop (T_final), not at the melt-onset envelope the
current Phase C/E solve reads. To gate on it, the solve3d FORWARD must
march densify to rho_target so the trajectory reaches that peak. That is
exactly shrinkage-v2 Level 2 (port the densify march into solve3d's
forward + adjoint, the rho co-state). THEREFORE: this workstream pulls L2
forward as its prerequisite; the ceiling constraint is not buildable
without the densify forward in the solve. The Studio already records the
target quantity (gates.T_end_max_C on the true T_final peak,
T_ceiling_ok); the solve consumes that definition so both lanes read one
peak.

## 3. Formulation

Decision variables (staged in): s (per-cell dopant saturation, existing
conductivity channel); a (drive scalar, uniform power-density multiplier);
theta (dwell/rotation schedule: ordered segments {angle, duration, drive});
t_stop handled as the densification stop (march to rho_target).

Minimize  J_shape(s, a, theta)   [asymmetric dense-iff-in-bounds, read at
                                  the densified end-state per L2]
subject to
  g1: peak-T constraint  max_{t,x} T(x,t) <= T_ceiling
  g2: densification       rho_final >= rho_target  (structural: the march
                          stops at rho_target, so g2 is met by construction)

Peak-T handling: max over space AND time is nonsmooth. Aggregate with a
KS / p-norm smooth surrogate of the peak for the gradient, and ALWAYS
report the true max alongside (never present the smooth surrogate as the
physical peak - a false-green class). Verify the surrogate tracks the true
peak within a stated band before trusting the constrained solve.

Feasibility structure: a (and later theta) make g1 satisfiable; s shapes
within. The ceiling is nearly independent of the dopant map (conservation),
which the implementation exploits (Stage A below).

## 4. Staging (each stage ships value; adjoint grows one actuator at a time)

### Stage A - dopant + drive scalar (immediate product value)

Because the ceiling is nearly dopant-independent: (1) a cheap drive
continuation/bisection on a fixed (uniform) map finds the largest a whose
end-state peak <= ceiling; (2) one full dopant shape-solve runs at that a.
Output: the recommended per-part drive plus the shape-optimal map under
ceiling. This is the "system tells the user what power to run" feature,
delivered within the frozen 2.0.0 schema (populates
power_settings.power_density_w_per_m3). Gate: the recommended (drive, map)
verified by heatr3d exactly as the map is today; is_sendable unchanged.

Adjoint scope: none new beyond L2 (drive is a scalar found by forward
bisection, not adjoint). Cheapest stage.

### Stage B - joint drive + exposure/stop-time

Add stop-time / exposure as a joint variable with drive (the energy-time
tradeoff the 0.55x/3.2x-exposure data quantifies). The envelope stop-time
adjoint (Phase B B4) already exists; this couples it with drive under the
ceiling. Output: recommended (drive, exposure).

### Stage C - dwell/rotation schedule adjoint (the durable version, L3-class)

theta segments {angle, duration, drive} become design variables; the
adjoint carries schedule sensitivity (the 2-D dwell-duration adjoint is
the precedent - port with the co-rotation-of-ALL-material-rasters
requirement from the first commit, the dielectric-ghost lesson, and a
regression test on a non-rotation-invariant shape). This beats the energy
penalty of pure backoff. Output: a real advisory turntable+power program.

## 5. Interface (routes to the Studio lane before freeze)

- Stage A: NO schema change. Solve populates
  power_settings.power_density_w_per_m3 (the field currently hardcoded
  1.5915e6) with the recommended per-part drive. Stays 2.0.0.
- Stage C: SINGLE source of truth. The Studio turntable block is
  static-intent-only today ({mode, file, requested_intent, note},
  package.py:156-206); mode "program" is allowed but no program-segment
  schema exists. So Stage C DEFINES the turntable program-segment schema
  for the first time as {angle, duration, drive} (it does not extend a
  legacy segment - cleaner, no migration). Scalar drive (A) is the
  degenerate single-segment/constant-drive case of the same object, so A
  and C share one contract and the turntable and power blocks cannot
  disagree. 2.1.0 bump under the cross-lane freeze protocol when C lands.
  THIRD-REPRESENTATION GUARD: the dwell-schedule planner / 2-D lane intake
  API may already carry an (angle, duration) turntable schema; the package
  program-segment schema MUST match it (extended with per-segment drive),
  not mint a third representation of one physical schedule. Reconcile with
  the 2-D lane before freezing the Stage C schema.
- End-state peak: consume gates.T_end_max_C / T_ceiling_ok from the
  Studio runner contract (true T_final peak, not melt-onset).

## 6. Gates / acceptance (per stage, pre-registered)

- KS-surrogate-vs-true-peak tracking band, stated before the constrained
  solve is trusted.
- Every new actuator's gradient FD-gated per the frozen protocol; mutation
  tests (a dropped ceiling-penalty VJP, a dropped schedule VJP, must fail).
- The constrained optimum's end-state peak <= ceiling on a mesh hold-out,
  not just the solve mesh.
- Recommended drive/schedule reproduces its claimed peak under heatr3d
  verify (cross-engine check, the Studio's existing gate).
- Honest-null clause: if a part is drive-limited such that no feasible
  (a, theta) both meets the ceiling and reaches rho_target at acceptable
  exposure, the solve reports that (the part cannot be made under this
  ceiling/chamber) rather than shipping an over-ceiling or under-dense map.

## 7. Out of scope

Multi-part beds; the physical rig's realizable power/gap (P-gate); eps_r
channel (Phase D, pending M2); any dissertation edit; the full sintering
mechanics of shrinkage-v2 L3a (separate, though L2 is shared).

## 8. Sequencing and dependencies

Prerequisite: shrinkage-v2 L2 (densify in solve3d forward+adjoint) - this
workstream pulls it forward. Order: L2 -> Stage A -> B -> C. Stage A can
begin the moment L2's densify forward lands (Stage A needs the forward,
not the rho adjoint; the rho adjoint is needed at B/C). Compute-schedule
convention governs all heavy runs.

## 9. Open questions for Matt

1. T_ceiling value: fixed 250 C, or a per-material config with a margin
   band (the ink/percolation work may move the usable ceiling)?
   [Studio: 250.0 hardcoded in runner today; per-material is a small
   runner change - free on their side.]
2. rho_target: 0.98 (the Studio's densification stop) or a configurable
   floor - does the dense-iff-in-bounds 80-90% density trade apply to the
   ceiling problem too (accept lower density to stay under ceiling)?
   [Studio: runner already takes stop_mean_rho, so configurable is free.]
3. Stage A drive-continuation objective: largest drive under ceiling
   (fastest print) vs a drive that leaves headroom (robustness) - product
   preference.
