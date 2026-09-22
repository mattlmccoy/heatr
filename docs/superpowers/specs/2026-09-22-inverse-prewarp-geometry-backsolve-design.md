# Green-Geometry Pre-Warp Backsolve (Z densification compensation)

**Date:** 2026-09-22
**Status:** Design — awaiting review
**Owner lane:** geo-prewarp (this lane); consumes the shared MetPrint staging tools

## 1. Motivation

The 3-D dopant backsolve optimizes an FGM dopant map so the part fuses to the
nominal shape **in the fixed (un-shrunk) frame**: its objective is
`IoU(fused region, nominal part)` on the solve nodes
(`solve3d/stage_a.py:_shape_iou_of_march` — fused = per-node peak T ≥ melt onset,
scored against `part_peak_mask`). It rewards the right region fusing uniformly to
full density and penalizes bed-melt / over-fusion. It does **not** reposition
material for the physical densification collapse.

Separately, as the powder consolidates from its green packed state
(ρ_green ≈ 0.55) toward full density, mass conservation removes volume; with the
loose powder laterally constraining the footprint, most of that collapse is in
the build (Z) direction (`heatr3d.shrinkage_factors`, xy_frac = 0.04; dissertation
`sec:fgm_shrinkage`). Measured on the actual solved parts:

| part | ρ̄_final | layer_multiplier (green/final Z) | warp_std_pct |
|------|---------|----------------------------------|--------------|
| cube (16.12 mm) | 0.987 | 1.707 | 2.2 |
| pyramid (23.25 mm) | 0.935 | 1.583 | 9.5 |

Two effects the dopant solve leaves on the table: (a) the **bulk** Z collapse
(~1.6–1.7× — the green must be printed taller), and (b) the **residual**
column-to-column warp the dopant cannot flatten (boundary heat-loss keeps edges
cooler; the dissertation reports the FGM only reduces warpage ~12%, it does not
remove it). This mechanism pre-distorts the **green** geometry so that after the
Z collapse the part lands on the nominal boundary — owning both (a) and (b) in one
per-column green-height field.

**Not in scope / kept separate:** nylon-12 **material** shrinkage
(melt/recrystallization) is a distinct, small, affine factor in
`studio3d/precomp.py` with an explicit double-counting guard. This mechanism
carries **densification consolidation only**; the two compose and must not overlap.

## 2. Approach

Chosen after ruling out a one-shot inversion (that is the discredited "invert the
map" pattern and ignores that the shrink field depends on the geometry it is
solving for). Instead: **solve the green geometry the same way we solve the
dopant** — a march-in-the-loop, error-driven backsolve against the shape-fidelity
objective. This is a **sequential outer loop** wrapped around the *unchanged*
dopant solve.

```
g_0 = nominal part as green, pre-scaled by the bulk layer_multiplier   (warm start)
repeat:
  march  F(g_k)  with the FIXED solved dopant  ->  rho_final, per-node peak T
  H_m(x,y) = Sum_z lambda_z(x,y,z) * h          (measured dense column height)
  H_t(x,y) = target dense column height from the nominal part
  H_green_{k+1}(x,y) = H_green_k(x,y) * H_t(x,y) / max(H_m(x,y), eps)   (multiplicative)
  converged when  max |H_t - H_m| / H_t  <  tol   (default 1%)   or k = k_max
output g* = pre-warped green volume (warped mask + dopant resampled to it)
```

- **Design variable:** a 2-D per-column green-height field `H(x,y)`. Its mean is
  the bulk Z factor; its variation is the warp correction — one field owns both.
- **Full per-column z-remap:** as `H(x,y)` changes, the dopant and occupancy are
  resampled along z per column (monotonic remap; generalizes the uniform
  `resample_z` already in the staging bridge).
- **Dopant held fixed** across outer iterations (solved once on the nominal part;
  it is robust to the small warp). Optional single re-solve at the end. Flagged as
  the assumption to revisit.
- **Forward model / gain:** `heatr3d.run(densify=True)` +
  `heatr3d.shrinkage_analysis` / `_warped_centers` (the existing forward warp). The
  update is quasi-Newton (forward gain ≈ λ_z), so ~2–4 iterations are expected
  given the small warp.

## 3. Toggle & revertibility (required)

This is an **additive, default-OFF** mechanism. With it off, the pipeline is
byte-for-byte the current dopant-only path.

- New geometry-prewarp module is a **separate file**; nothing in the dopant solve
  or the staging bridge changes behavior unless prewarp is explicitly requested.
- A single switch (`--prewarp` / `prewarp=False` default) selects whether staging
  consumes the **nominal** spec (current behavior) or the **pre-warped green**
  spec this mechanism emits.
- The pre-warped green volume is written as its **own** spec npz (e.g.
  `*_prewarped_green_spec.npz`); the nominal spec is never overwritten. Reverting =
  stage the nominal spec (or drop the flag).
- `job_info.json` records a `prewarp` provenance block (enabled bool, iters,
  converged, tol, final max/mean column error, warp_std before/after, source
  densify run). Absence and off are distinguishable, never confusable.

## 4. Interfaces & data flow

New module (proposed): `solve3d/geom_prewarp.py` (lives beside the forward-warp
code; runs in the geo-prewarp venv that already imports heatr3d).

```
inputs:
  - nominal part (voxel mask or STL) + physical bbox
  - solved dopant map (the SOLVE_cont saturation, e.g. densify_inputs/*_solve_spec.npz)
  - Params p (carries rho_green, xy_frac)
forward (per iter):
  - heatr3d.run(grid, green_mask_k, p, sat=dopant_k, densify=True) -> Result
  - shrinkage_analysis(Result, p, h) -> H_final(x,y), lambda_z, warp_std_pct
update:
  - H_green_{k+1} = H_green_k * H_t / max(H_m, eps); rebuild green mask + resample dopant
output:
  - <part>_prewarped_green_spec.npz  { SOLVE_cont, part_mask, proxy_field="solve", + prewarp provenance }
  - a convergence record (iters, errors, warp before/after)
```

**Staging is untouched.** `stage_3d` already consumes a `proxy_field="solve"` spec
(mask + dopant) and slices it per layer at the machine layer height; it simply
receives the pre-warped green spec instead of the nominal one when prewarp is on.
The `--z-densification` manual scalar becomes redundant when prewarp is on (the
green heights are baked into the mask) and remains only as the off-path override.

## 5. Verification (TDD; the in-sim round-trip is the objective)

- **Unit — update rule.** On a synthetic λ_z field with a known per-column
  compaction, one/two multiplicative updates recover the analytic green height
  (deterministic, no solver).
- **Unit — z-remap.** The per-column monotonic remap preserves dopant ordering and
  conserves dopant mass within tolerance; zero dopant outside the mask.
- **Unit — toggle.** prewarp=False returns the nominal spec unchanged
  (byte-identical staging inputs); provenance says `enabled:false`.
- **Integration — round-trip on the pyramid** (worst warp, 9.5%). Run the loop,
  then march the converged green and confirm: per-column height error < tol AND
  `warp_std_pct` drops materially vs the uncompensated baseline. This is the
  acceptance gate (no hardware needed). Cube (2.2%) is the easy-convergence check.
- **Regression.** `test_stage.py` stays green; the dopant-only staged jobs are
  unchanged when prewarp is off.

## 6. Compute & convergence

- Each outer iteration = **one densify march** (dopant frozen). Grid n = 48
  (matches the existing marches); n = 64 optional for tighter warp stats.
- Expect 2–4 iterations; `k_max` default 5, `tol` default 1% max column error.
- One heavy solve at a time; checkpoint each iteration (green mask + errors) so a
  crash resumes. Cube likely converges in 1–2; pyramid more.

## 7. Assumptions & risks

- **Monotonic parts only** (cross-section does not fold back on z), so a per-column
  height field is sufficient. Overhangs/undercuts (multiple green→dense segments
  per column) are explicit future work.
- **Dopant frozen** during the loop — assumes the grade is insensitive to the small
  warp. Mitigation: optional end re-solve; the round-trip gate would expose it.
- **xy_frac = 0.04 and ρ_green = 0.55 are uncalibrated** (model parameters, per the
  dissertation). The mechanism is only as trustworthy as the shrink law; results
  are simulation-scoped until the P1 sinter measurement. Provenance carries this.
- **Grid resolution** limits the warped-mask edge fidelity (voxel slicing). A crisp
  warped-STL path (marching cubes) is a later option, not in this spec.
- **Fixed-point may not monotonically converge** if the gain is mis-estimated;
  cap iterations and keep the best-error iterate.

## 8. Deliverables

1. `solve3d/geom_prewarp.py` — the outer-loop backsolve + green-volume emitter.
2. Tests (unit + pyramid round-trip integration).
3. Re-staged cube + pyramid as pre-warped green prints (prewarp ON), alongside the
   nominal dopant-only jobs (prewarp OFF) for comparison.
4. A short results note: warp_std before/after, iters to converge, per-part.

## 9. Out of scope

- Overhang/undercut geometries; XY (in-plane) pre-warp; material (nylon-12)
  shrinkage (that stays in `precomp.py`); joint dopant+geometry optimization;
  crisp warped-STL meshing; any hardware print.
