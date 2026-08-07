# Thermal-Ceiling Stage A Report

Date: 2026-08-06
Branch: feat/pernode-twosided-tuning
Spec: docs/superpowers/specs/2026-08-06-thermal-ceiling-joint-solve-design.md (APPROVED)
Plan: docs/superpowers/plans/2026-08-06-thermal-ceiling-stage-a.md
Config memo: POLYMER_AM_DENSITY_THERMAL_MEMO.md (commit c25eb5c)

## Conclusion

Stage A delivers the densify forward, the ceiling observable, and best-part
drive selection. The dopant relocates the peak but cannot lower its magnitude
(conservation), so the degradation ceiling is satisfied by the DRIVE, not the
dopant; the dopant shapes within the feasible envelope. Tasks 0 through 3 are
implemented, tested, and gated green. Task 4 phase 1 (drive selection on a real
part at the physical rho_target 0.98) is DONE and decisive: the square needs a
0.40x drive backoff to reach full density under the 250 C ceiling; every higher
drive cooks over, and best-part coincides with coolest (no tradeoff). Task 4
phase 2 (the dopant shape-solve at that drive) is queued behind the Tamper solve
per the compute convention; its scope reading is now confirmed by both lanes.
Task 5 output contract is built, routed to the Studio lane, and now carries a
real recommended drive; no schema change (stays 2.0.0).

## Config values (all cited to the memo, c25eb5c)

| Quantity | Value | Role |
|---|---|---|
| degradation ceiling | 250.0 C | the CONSTRAINT (max over t,x of T <= this) |
| warning band start | 235.0 C | 10-20 C solver margin |
| melt onset | 185.0 C | COMPLETENESS (min in-part peak must exceed) |
| rho_target floor | 0.90 | mechanical knee; refuse below |
| rho_target good | 0.95 | working norm |
| rho_target practical ideal | 0.98 | the densify stop target |
| rho_target ideal | 1.0 | normalization only, not attainable |

Pre-registered before any solve code in `solve3d/results/stage_a_preregistration.json`.

## Task 1: densify forward (L2 forward half)

The degradation-ceiling peak is an end-state quantity: it occurs at the
densification stop, not at the melt-onset envelope the Phase C/E solve reads. So
the solve3d forward now marches densify to a target mean relative density.

- `forward.densify_rate` is a bit-for-bit port of `heatr3d.densify_rate`
  (solid-state Arrhenius creep + liquid viscous-capillary flow, gated by
  porosity), pinned exactly by `test_densify_rate_matches_heatr3d_bit_for_bit`.
- `march_enthalpy` gains `densify` / `stop_mean_rho`, guarded behind
  `if densify:` so the OFF path is bit-identical to the Phase A forward. rho_rel
  evolves nodally, density-dependent properties recompute per step, the march
  stops at the target mean density, and the TRUE trajectory peak of the in-part
  temperature is tracked (the ceiling quantity, never an end-state snapshot).
- Off-path bit-identity: the existing forward suite stays green, plus
  `test_zero_rate_densify_reproduces_constant_rho_march` (with the densification
  rate forced to zero the densify march reproduces the constant-rho march, so
  the only thing densify changes is via the density update).

Equivalence gate vs heatr3d's own densify march on the extruded circle, judged
against a MEASURED cross-family band (both engines' n40-vs-n64 grid self-spread,
combined by the Phase A rule 1.5 x (heatr3d_spread + solve3d_spread)). The peak
is a max, so its grid spread is the widest; a borrowed melt-onset std band would
be the wrong yardstick, which is why the band is measured, not assumed.

```json
{
 "all_pass": true,
 "part_mean_rho":    {"rel": 0.00019, "tol": 0.00110},
 "T_end_max_c":      {"rel": 0.04154, "tol": 0.07006},
 "part_mean_T_end_c":{"rel": 0.00284, "tol": 0.00584},
 "band": {"heatr3d_T_end_max_spread": 0.04279, "solve3d_T_end_max_spread": 0.00392}
}
```

Mutation check: dropping the liquid viscous-capillary term from the density rate
fails all three checks decisively (rho off 19.1%, peak off 121%, mean-T off
152%), so the gate is discriminating and the dropped term is load-bearing.
Artifacts: `densify_parity_tolerances.json`, `densify_equivalence.json`,
`densify_equivalence_mutation.json`.

## Task 2: ceiling observable (KS peak + TRUE max)

`solve3d/ceiling.py`. The peak-temperature constraint is a max over space and
time and is nonsmooth; a gradient needs a smooth aggregate. `peak_temp` returns
the volume-weighted KS aggregate (mean log-sum-exp, a LOWER bound on the max
that approaches from below) AND the true max.

False-green guard, by construction: `ceiling_status` reads the verdict off the
TRUE max only and has no argument for the smooth aggregate, so the false-green
class (a smooth value under the ceiling while the true peak is over it) cannot
be expressed. `test_ceiling_ok_is_on_true_max_not_the_ks_proxy` builds exactly
that field (bulk 240 C, one spike 254 C, ceiling 250 C) and confirms the verdict
is NOT ok while the smooth aggregate sits under the ceiling.

Melt completeness (min per-node trajectory peak vs onset) is reported
separately from the ceiling; the forward returns a per-node running-peak field
for it.

Tracking band (`ceiling_ks_band.json`), pre-registered at 5%:

```json
{"all_pass": true, "synthetic_gap_rel": 0.0222, "real_march_gap_rel": 0.0089,
 "T_ceiling_ok": false, "ceiling_read_on": "true_trajectory_max"}
```

## Task 3: best-part drive selection (not speed)

`solve3d/stage_a.py`. The drive is chosen to MAXIMIZE part quality (density
completeness toward rho_target plus fused-body shape fidelity) subject to the
degradation ceiling, not to be the fastest under-ceiling drive (Matt: get good
parts first, speed later). On a quality tie the cooler drive wins for margin.
The sweep runs a uniform map because the ceiling is nearly dopant-independent.

`select_from_sweep` is pure logic, tested by five cases: over-ceiling drives
rejected, best-part-not-fastest, cooler tie-break, the honest null, and clipped
scoring. Honest null (mandatory): when no feasible drive reaches the density
floor the verdict is `cannot_make_under_ceiling` with the best feasible density
reported as evidence only, never as a shippable map.

Short end-to-end demonstration on the extruded circle (`stage_a_drive_sweep.json`),
one sweep read at two floors:

```
a=2.0  peak 230.4 C  feasible
a=4.0  peak 244.8 C  feasible
a=6.0  peak 254.4 C  OVER ceiling -> rejected
a=8.0  peak 262.1 C  OVER ceiling -> rejected

at the pre-registered 0.90 floor : cannot_make_under_ceiling (honest null)
at a reachable demo floor (0.60)  : ok -> chose a=2.0 (cooler of two feasible tied drives)
```

The honest null fires faithfully because the short demo densifies only to 0.66,
below the 0.90 floor; the physical rho_target=0.98 selection on the deliverable
part is the Task 4 heavy run.

## Task 5: output contract + shared thermal config

Two Studio-lane corrections (verified against their package/runner code) are
folded in:

1. `stage_a.recommended_power_settings(a)` writes ONLY the one field Studio
   hardcodes (studio3d/package.py:193):

   ```
   power_settings.power_density_w_per_m3 = a_chosen * 1.5915e6
   ```

   a = 1.0 is exactly the current Studio baseline. rf_mode is deliberately NOT
   written: 'constant' is the only value until schedules exist, so it carries
   zero information and would break the genuinely-zero-schema-change property.
   rf_mode becomes a real discriminator (constant vs schedule) at the Stage C
   2.1.0 bump alongside the turntable {angle, duration, drive} extension. Stays
   truly 2.0.0, no new keys. Pinned by
   `test_recommended_power_settings_writes_only_the_one_field`.

2. The per-material T_config is a SHARED single source of truth, not hygiene:
   the Studio heatr3d verify is the cross-engine gate on the recommended drive
   (is_sendable requires it), and that check is only meaningful if BOTH lanes
   read the IDENTICAL ceiling. Shipped at a stable path:

   PATH: `solve3d/thermal_config.json` (schema_version 1.0, like the
   shrinkage_precomp.json precedent).
   KEY SCHEMA (the keys the Studio runner should read to replace its two
   hardcoded 250.0 literals):
   ```json
   {"schema_version": "1.0", "material": "PA12",
    "T_ceiling_C": 250.0, "T_warning_C": 235.0, "T_melt_onset_C": 185.0,
    "rho_target": {"floor": 0.90, "good": 0.95, "practical_ideal": 0.98,
                   "ideal_normalization": 1.0}}
   ```
   solve3d/ceiling.py callers and solve3d/stage_a.py read it (T_ceiling_C ==
   gates.T_CEILING_C == the Studio runner's 250.0, asserted by
   `test_shared_thermal_config_is_the_single_ceiling_source`). Until the Studio
   runner is wired to this file (their side, TDD), its hardcoded 250.0 matches
   T_ceiling_C so there is no drift today.

The field write and the shared-config path/schema were routed to the Studio lane
session ("Make Grade and Print real in RFAM Print Studio") for confirmation.

## Task 4: dopant shape-solve at the chosen drive

### Phase 1 (DONE 2026-08-06): drive selection on a real part at rho_target 0.98

`solve3d/stage_a_launch.py`, detached (pid 16498), checkpointed per drive,
completed clean (verdict `ok`, not honest-null). Sweep on the square anchor
(5600 in-part nodes), five drives to mean rho 0.98 under the shared-config
250 C degradation ceiling. Result (`solve3d/results/stage_a_task4_square.json`):

```
 drive   power_density      true peak   rho    shape IoU   exposure   feasible
 0.40x   636,620 W/m^3       240.1 C    0.980    0.7047     1336.6 s     YES
 0.55x   875,352 W/m^3       255.8 C    0.980    0.7009      944.4 s     over
 0.70x  1,114,085 W/m^3      268.2 C    0.980    0.6991      732.4 s     over
 0.85x  1,352,817 W/m^3      278.1 C    0.980    0.6968      600.2 s     over
 1.00x  1,591,549 W/m^3      286.4 C    0.980    0.6956      510.0 s     over
```

Reading:
- The ceiling BITES on a real deliverable part. To reach full density (0.98),
  only 0.40x baseline keeps the square's true peak under 250 C (240.1 C, a 10 C
  margin). Every higher drive densifies to 0.98 sooner but cooks over the
  degradation onset. n_feasible = 1.
- This is the conservation argument made concrete: the dopant cannot rescue an
  over-ceiling drive (it relocates the peak, not its magnitude); the DRIVE is
  the lever, and here it must back off to 40%.
- Best-part and coolest COINCIDE (no tradeoff): shape IoU is monotone in cooler
  drive (0.7047 at 0.40x vs 0.6956 at 1.00x), so the slow, cool bake wins on
  BOTH feasibility and shape. The quality score Q picks 0.40x on merit, not just
  by the cooler-tie-break.
- Energy residual ~1e-13 across all five (conservation held), no rho clamp.
- Recommended output (2.0.0, one field): power_density_w_per_m3 = 636,620
  (0.40x baseline), with provenance recorded in `recommended_power_settings`.

This is the "system tells the user what power to run for the best part under the
ceiling" feature working end-to-end on a real part. It also updates the spec's
worked example: the square at 1.0x reaches 286.4 C at rho 0.98, well over both
the pyramid (281.7 C) and cube (~265 C) full-density peaks quoted in spec S1.

### Phase 2 (FD-GATED + LAUNCHED 2026-08-07): dopant shape-solve at 0.40x

Scope reading CONFIRMED (by this lane AND the Studio lane, cross-session
2026-08-06): optimize the dopant shape with the existing melt-onset adjoint at
the chosen 0.40x drive, and evaluate the degradation ceiling at the densify
END-STATE as a FORWARD gate (the ceiling is nearly dopant-independent, so the
dopant needs no ceiling gradient in Stage A; the drive handles the ceiling; the
rho co-state is Stage B+). No open scope question remains.

Driver: `solve3d/stage_a_phase2.py` (tests `test_stage_a_phase2.py`, 7 green).
The ONLY substantive change vs a default-drive Phase-E square solve is the drive
`ForwardParams(power_density_w_per_m3=636619.77)`, read from
`stage_a_task4_square.json` so it cannot drift; the asymmetric envelope
objective, the design_chain (filter + tanh projection), the 1/|g0| rescale and
per-eval checkpointing are the frozen conventions reused from run_tamper.

PRE-LAUNCH FD GATE (the cardinal rule; `stage_a_phase2_fd_gate.json`), coarse
square at the 0.40x drive, frozen tolerances, NO widening:

```
 A map-space adjoint dJ/ds     worst rel 3.71e-08   4/4 pass 1e-6
 B filter transpose identity        rel 2.46e-16    < 1e-10
 C composite dJ/dv (solve grad) worst rel 2.94e-08   4/4 pass 1e-6
```

Interior envelope argmin (t=863 s) + 56 live melt-window nodes; filter mean 4.99
neighbours (genuinely participates); the strongest-signal probes are the
cleanest (1e-9..1e-11), the signature of a correct gradient, not a floor
artifact. Solve-mesh sanity (`stage_a_phase2_solve_mesh_sanity.json`) on the
real 5600-in-part-node / 28977-design-cell case: interior argmin (18086/36000,
t=904 s), first |g|=4.47e-8 finite+nonzero; cost 355 s/eval.

RUNNING (detached, `caffeinate -i`, checkpointed/resumable): budget 12,
`ckpt_phase2_square.npz`, status `stage_a_phase2_square_status.json`, log
`stage_a_phase2_square.log`. ETA ~71 min (budget 12). Monitor:
`python -m solve3d.phase_e.track_solve solve3d/results/ckpt_phase2_square.npz <pid> 12`.
On completion run_solve writes `stage_a_phase2_square.json` (solved map +
recommended drive in the Stage A shape) after the end-state ceiling HOLD-OUT
gate (finer 0.060/64 mesh, solved dopant transferred nearest-neighbour); the
map `map_phase2_square.npz`. is_shippable = FD-gated gradient AND end-state peak
<= 250 C on the hold-out.

PHASE 2 RESULT (DONE 2026-08-07, `stage_a_phase2_square.json`): the shape-solve
CONVERGED WELL but the mesh HOLD-OUT ceiling gate REFUSED it -- an honest,
important outcome.

- Convergence (solve mesh): J_asymmetric 2.371e-6 (eval 1) -> 9.280e-7 (eval 12),
  -60.9%, monotone, gradients finite (~1.6e-8), energy residual 3e-13, no clamp,
  no CFL violation. Quality: part_mean_phi 0.836 (84% melt, vs Tamper's 34%),
  sigma_T 16.76 C, solve-mesh peak 227.99 C (under 250). Map: mean 0.822, min
  0.020, max 1.0 (dopant carved in places).
- HOLD-OUT gate (the arbiter, finer mesh 9868 in-part nodes, lc0 0.9375 mm):
  true end-state peak 251.15 C vs ceiling 250.0 -> margin -1.15 C -> feasible
  FALSE -> **is_shippable = FALSE**. rho 0.98 reached, energy residual 3e-13.

WHY THIS MATTERS (a real finding, not a bug): on the SOLVE mesh the shaped map
peaked 227.99 C, comfortably under; on the properly-resolved HOLD-OUT mesh it
peaks 251.15 C, 1.15 C (0.46%) OVER. Two effects compound: (1) the finer mesh
resolves a sharper peak the solve mesh under-resolved, and (2) the shape-optimal
dopant RELOCATED the peak upward relative to the uniform 0.40x map (uniform was
~240 C class) -- so "the ceiling is nearly dopant-independent" has a real LIMIT:
here the dopant moved the peak ~+11 C, enough to cross. The hold-out gate did
exactly its job -- it refused a map that a solve-mesh-only read (227.99 C) would
have shipped as a false-green over-ceiling part.

CONSEQUENCE / FIX (next step, not this run): back the drive off below 0.40x to
leave margin for the dopant's peak relocation, then re-solve + re-gate on the
hold-out; and/or (deeper, Stage B+) the ceiling may need coupling to the dopant
after all, since the dopant is NOT ceiling-neutral at this drive. The drive
selection (Phase 1, coarse mesh) was slightly optimistic; the honest feasible
drive for the SHAPED map, hold-out-resolved, is below 0.40x. NOT routed to the
Studio cross-engine verify: is_shippable is already FALSE on our own hold-out, so
asking heatr3d to confirm a known-over-ceiling map would be dishonest; the verify
comes after the drive-backoff re-solve passes the hold-out.

Cross-engine verify of the DRIVE recommendation (uniform map at 0.40x): DONE
and CONFIRMED (Studio lane heatr3d, 2026-08-06). Same geometry constructor, n=64,
power_density 636,620 W/m^3, densify to mean rho 0.98, ceiling read from the same
shared solve3d/thermal_config.json:

```
 quantity              solve3d (dolfinx FEM)   heatr3d (voxel FDM)
 end-state peak (C)    240.1                   244.3   (+4.2 C, 1.7%)
 reaches rho 0.98      yes                     yes (0.980)
 under 250 C ceiling   yes                     yes
```

heatr3d's 244.3 C is inside the cross-family band [223.3, 256.9] (7% of 240.1)
and under the ceiling; heatr3d energy audit clean (in 20578 J, stored 19982,
loss 596, residual +0.0000, phi 1.000). Two independent engines agree on the
densification END-STATE peak to 1.7%, both under the ceiling, from the same
ceiling file. The 0.40x drive backoff is a cross-engine fact; is_sendable holds
for the recommended (drive, uniform map). Evidence artifact (Studio lane):
`solve3d/results/verify_square_stageA_heatr3d.json`. The shaped-map verify
follows phase 2.

## Not covered by Stage A (named so silence is not read as agreement)

- Stage B: joint drive + exposure/stop-time (the energy-time tradeoff the
  0.55x / 3.2x-exposure data quantifies).
- Stage C: the dwell/rotation schedule adjoint and the turntable
  program-segment schema (the durable actuator; needs the co-rotation and
  dielectric-ghost lessons and the third-representation guard).
- The rho co-state / density adjoint (needed at B/C; Stage A uses the densify
  FORWARD only, drive found by forward bisection).
- eps_r channel (Phase D), multi-part beds, the physical rig P-gate, any
  dissertation edit.

## Files

Code: `solve3d/forward.py` (densify branch + densify_rate), `solve3d/densify_forward.py`,
`solve3d/ceiling.py`, `solve3d/stage_a.py`.
Tests: `solve3d/tests/test_densify_forward.py`, `test_ceiling.py`, `test_stage_a.py`.
Artifacts: `solve3d/results/stage_a_preregistration.json`,
`densify_parity_tolerances.json`, `densify_equivalence.json`,
`densify_equivalence_mutation.json`, `ceiling_ks_band.json`,
`stage_a_drive_sweep.json`.
