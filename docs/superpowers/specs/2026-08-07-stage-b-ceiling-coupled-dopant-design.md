# Stage B: Ceiling-Coupled Dopant Solve (density co-state adjoint)

Date: 2026-08-07
Status: APPROVED by Matt 2026-08-07 (design confirmed: penalty-first, fixed 0.40x
drive, KS-peak hinge at the densify end-state, rho+T co-state as the new gradient
path).
Parent spec: docs/superpowers/specs/2026-08-06-thermal-ceiling-joint-solve-design.md
(APPROVED). This instantiates the "rho co-state / density adjoint" that the parent
placed at Stage B+.
Owner: Matt McCoy

## Why this exists (the phase-2 finding that triggered it)

Stage A phase 2 (solve3d/results/stage_a_phase2_square.json): the dopant
shape-solve at a FIXED 0.40x drive converged well (J -60.9%, mean melt 84%,
sigma_T 16.8, solve-mesh peak 227.99 C) but the mesh HOLD-OUT ceiling gate REFUSED
it -- true end-state peak 251.15 C on a finer mesh, 1.15 C OVER the 250 C ceiling,
is_shippable=FALSE.

Root cause, and it is a real finding, not a bug: the shape-optimal dopant
RELOCATED the peak ~+11 C above the uniform 0.40x map (uniform ~240 C class,
shaped 251 C on the fine mesh). So the Stage A assumption "the ceiling is nearly
dopant-independent" -- which justified treating the ceiling as a dopant-free
FORWARD gate with no ceiling gradient on the dopant -- has a real LIMIT. At this
drive the dopant is NOT ceiling-neutral. To keep the peak under ceiling by
construction, the dopant solve needs a GRADIENT of the end-state peak. That
gradient must back-propagate through the densify march (the peak is an end-state
quantity), which is the density (rho) co-state adjoint the parent spec deferred.

## Formulation

Decision variable: s (per-cell dopant saturation, the existing conductivity
channel). Drive FIXED at 0.40x = power_density 636619.7723675814 W/m^3 (the Phase 1
certified value; joint drive is B4/later).

Minimize over s:

    J(s) = J_shape(s) + mu * [ ( That_peak(s) - T_ceiling )_+ ]^2

- J_shape(s): the existing asymmetric dense-iff-in-bounds objective, read at the
  MELT-ONSET ENVELOPE (argmin over the trajectory). UNCHANGED from Stage A phase 2.
- That_peak(s): the SMOOTH KS aggregate of the peak temperature (ceiling.peak_temp,
  the volume-weighted mean log-sum-exp, a LOWER bound approaching the true max from
  below), read at the DENSIFY END-STATE (T_final at rho_target = 0.98).
- (x)_+ = max(0, x): hinge. mu: penalty weight, ramped by continuation.
- T_ceiling = 250.0 C from solve3d/thermal_config.json (the shared source).

False-green guard (unchanged, load-bearing): the TRUE trajectory peak, never
That_peak, is what ceiling_status / is_shippable read. The KS aggregate is used
ONLY inside the gradient. It approaches the true max from BELOW, so a solve that
drives That_peak to the ceiling may still leave the true peak slightly over -- the
mesh HOLD-OUT gate remains the arbiter of is_shippable, exactly as in phase 2.

## The core new machinery: the rho + T co-state adjoint

The end-state peak occurs at the densification stop, so dThat_peak/ds requires the
adjoint to run THROUGH the densify march (not just the melt-onset envelope). Two
coupled co-states, integrated backward over the densify time-stepping:

- lambda_T (temperature co-state): already exists for the melt-onset read; extended
  to carry the sensitivity of That_peak at the densify end-state.
- lambda_rho (density co-state): NEW. densify_rate depends on (T, rho) and rho
  feeds back into the density-dependent material properties, so the peak's
  sensitivity to s propagates partly through the rho evolution. lambda_rho carries
  that path.

At each backward step the two co-states update from the linearized forward
(d densify_rate/dT, d densify_rate/drho, d properties/drho, d Qrf/d sigma(s)), and
accumulate dThat_peak/ds onto the design. This is the L2 adjoint second half.

Read-state coupling: the solve runs the full transient ONCE and takes TWO reads --
J_shape at the melt-onset envelope (argmin, ~t=900 s at 0.40x) and That_peak at the
densify end-state (rho_target, ~t=1000 s). dJ_shape/ds uses the existing
melt-onset envelope adjoint; dThat_peak/ds uses the new rho+T co-state; the total
gradient is their sum plus the mu-hinge chain rule.

## Staging (each ships something; the adjoint grows one piece at a time)

### B1 - the density co-state, ISOLATED and FD-gated (the hard, risky piece)
Build That_peak(s) forward + its rho+T co-state adjoint. FD-gate dThat_peak/ds
ALONE on a coarse case against central finite differences, at the frozen tolerance,
NO widening. Mutation test: dropping the lambda_rho term must FAIL the FD gate
(proves the density co-state is load-bearing, not decorative). This is the gate
that de-risks everything downstream; B2 does not start until B1 is green.

### B2 - the penalty solve at fixed 0.40x
Assemble J(s) = J_shape + mu*hinge^2. FD-gate the COMBINED dJ/ds. Solve with
L-BFGS-B (frozen conventions: 1/|g0| rescale, density filter + tanh projection,
beta-continuation) with mu-continuation. Acceptance: the solved map's end-state
true peak <= 250 C on the mesh HOLD-OUT (the phase-2 failure fixed by
construction), best shape achievable under that constraint. Then route (shaped map,
0.40x drive, chamber tag) to the Studio heatr3d cross-engine verify -> is_sendable.

HONEST-NULL clause (mandatory, mirrors Stage A select_from_sweep): 0.40x may be
infeasible for EVERY dopant on the hold-out -- if even the ceiling-optimal
(peak-minimizing) dopant leaves the true hold-out peak > 250 C, B2 reports
`no_feasible_dopant_at_this_drive` with the achieved min-peak as evidence, and does
NOT ship an over-ceiling map. That is a real, reportable result meaning the drive
itself is too high (escalate to B4 joint drive or a drive backoff), NOT a failure of
the machinery. First check to run in B2, before trusting any shaped result: compute
the uniform-map true peak at 0.40x on the SAME fine hold-out mesh -- phase 1's
240.1 C was on the coarser drive-sweep mesh, so the uniform hold-out peak is
currently unknown and bounds what any dopant can achieve.

### B3 - augmented Lagrangian (later)
Upgrade the penalty to an augmented Lagrangian (multiplier updates) so peak <=
ceiling holds at convergence by construction, not just approached via mu.

### B4 - joint drive (later)
Add the drive scalar back as a joint variable under the coupled ceiling (the drive
drops where the dopant cannot hold the peak). This reconnects to the parent spec's
Stage A/B drive actuator.

HEADROOM DRIVE POLICY (answers parent spec Q3; confirmed with the Studio lane
2026-08-07). The phase-2 +11 C dopant peak-relocation shows that selecting the
drive as "largest feasible on the UNIFORM map" is too aggressive: a drive with no
headroom on uniform becomes infeasible once the shape-optimal dopant concentrates
the peak. So the drive-selection margin (Stage A select_from_sweep, and B4 joint
drive) must cover the MEASURED dopant peak-relocation (~+11 C here), NOT just solver
/ mesh error. Concretely: feasible-drive := true hold-out peak of the UNIFORM map
<= T_ceiling - Delta_dopant, where Delta_dopant is the measured (or bounded)
shape-optimal peak relocation at that drive. The Stage A drive backoff is this
policy applied by hand; B4 makes it the selection rule. This is a policy the parent
Stage A spec should adopt for its drive actuator.

## Gates / acceptance (pre-registered, per the frozen protocol)

- B1: dThat_peak/ds matches central FD within the frozen tolerance on a coarse
  case; the interior read state is valid (densify end-state reached, rho_target
  hit); mutation (drop lambda_rho) fails the gate. Artifact: a B1 FD-gate JSON.
- B2: combined dJ/ds FD-gated; the mu-continuation converges; the solved map's TRUE
  end-state peak <= 250 C on a mesh HOLD-OUT (not the solve mesh); is_shippable
  computed on the true peak; is_sendable via the Studio cross-engine verify.
- No threshold widening; every new gradient FD-gated before use; the true peak
  (never That_peak) is what any ship/send gate reads (false-green structurally
  unexpressible).

## Honest expectations and limits

- At fixed 0.40x, forcing the peak under 250 C COSTS shape quality vs the phase-2
  unconstrained (infeasible) map. The B2 result is the BEST FEASIBLE shape at 0.40x,
  which may be meaningfully worse than the infeasible phase-2 shape. That is the
  honest price of feasibility; if it is too compromised, B4 (joint drive) or a drive
  backoff is the escape, but we will have the principled machinery either way.
- The KS aggregate approaches the true max from below, so the penalty alone does not
  guarantee the true peak is under ceiling; the hold-out gate stays the arbiter
  until B3 (augmented Lagrangian) tightens it.
- Densify-forward coupling convention: the Stage A hold-out ran the densify gate with
  EQS-thermal coupling OFF (a Stage A convention). B1/B2 must state whether the
  density adjoint is derived with coupling on or off and keep the forward it gates
  against consistent (a coupled adjoint FD-gated against an uncoupled forward would
  be a silent mismatch).

## Out of scope (named so silence is not agreement)

Joint drive (B4), exposure/stop-time (parent Stage B original), the dwell/rotation
schedule adjoint (parent Stage C), the eps_r channel (Phase D), multi-part beds, the
physical rig P-gate, any dissertation edit. B1/B2 are dopant-only at fixed drive.

## Files (anticipated; the plan will pin exact paths)

- solve3d/ceiling.py: That_peak already present; add its VJP (dThat_peak/d end-state
  T, then into the rho+T co-state).
- solve3d/adjoint.py or a new solve3d/density_adjoint.py: the rho+T co-state through
  the densify march (the new L2 second half).
- solve3d/forward.py: expose the linearization terms (d densify_rate/dT,
  d densify_rate/drho, d properties/drho) the adjoint needs.
- solve3d/stage_b.py: the penalty objective assembly + solve driver at fixed 0.40x.
- solve3d/tests/: FD-gate tests for B1 (dThat_peak/ds) and B2 (combined), mutation
  tests, red-first.
