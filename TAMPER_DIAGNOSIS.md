# Tamper Job Diagnosis: what the Studio actually ran, and why it lost to uniform

Date: 2026-08-04. Diagnosis only; no product code or job artifacts were modified.

Job: "Part Studio 1 - Tamper.stl", Grade and Print job `feb850ec`, run 2026-08-04 08:49 to 09:11.
Artifacts: `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/software/meteor/tools/uploads/feb850ec/grade/`

## Conclusion (one paragraph)

Matt's hypothesis is correct. The volumetric correction was NOT produced by the Phase E
solve3d stack (the engine behind the pyramid/cube results). It was produced by
`heatr3d_native_inversion`, a legacy proportional-inverse heuristic (`heatr3d.make_fgm` on
the `rho_final` proxy at `magnitude=1.0`), reached as the LAST rung of the Studio's
fallback chain after (1) no fresh direct solve existed, (2) no registry match existed, and
(3) the good 2.5-D per-slice map was correctly REFUSED at the transfer gate (it moved
"37.37" percent of in-part dopant, over the "2" percent Phase C staircase threshold).
Because the BEFORE arm ends nearly fully dense ("rho_final_mean" = "0.98"), the inverse
rule normalized against a saturated field and emitted a map that ZEROES the dopant in
"79" percent of the part (in-part sat mean "0.0707"; "7972" of "10076" voxels exactly
"0.0"). The corrected arm therefore printed a mostly UNDOPED part with small fully doped
pockets at the under-dense skin: heating time nearly doubled ("t_phi90" = "1348" s vs
"789" s), the run hit the "1500" s cap before reaching the density stop, peak temperature
blew through the ceiling ("T_max" = "448.4" C vs ceiling "250" C; uncorrected "288.0" C),
and every outcome metric got worse than uniform ("sigma_T" "53.179" vs "28.545";
"warp_std_pct" "11.041" vs "6.371"; "rho_final_std" "0.1046" vs "0.0596"). No optimizer
ever ran on this part: zero gradient evaluations, no objective, no budget, no gates.

## 1. What engine ran (evidence)

`grade/heatr3d/correction_provenance.json` (verbatim fields):

```json
{
  "engine": "heatr3d_native_inversion",
  "trust_badge": "heatr3d native inversion | legacy heuristic (the direct solve is the primary generator) | sim-only",
  "proxy": "rho_final",
  "fallback_from_25d": {
    "state": "measured_and_failed",
    "error": "stack_to_voxel: transfer moved in-part dopant by 37.37 %, over the 2 % gate (Phase C staircase threshold). Refusing the transferred map.",
    "artifact": ".../grade/heatr/dopant_volume.npz"
  },
  "grid_n": 64,
  "null_correction": false
}
```

Both densify arms record `"engine": "heatr3d_native"` and the corrected arm records
`"correction_engine": "heatr3d_native_inversion"` (`grade/heatr3d/*/results.json`).

The selection cascade that produced this is
`studio3d/correction.py::build_correction` (repo MAIN tree; this is the tree
`grade_server.py:26-27` puts on the path, and the Studio worktree
`.claude/worktrees/strange-leakey-01a387` has no `studio3d/correction.py`):

1. Fresh direct solve, `grade_dir/heatr3d/solve/studio_solve_map.npz` (correction.py:119).
   NOT PRESENT for this job; the `solve/` directory does not exist. The Studio's
   on-demand direct-solve endpoint (`grade_server.py:1043` invoking
   `solve3d.studio_solve`) is a separate user-triggered route and was never run on the
   Tamper. The `/grade/auto` pipeline does not schedule it.
2. Registry lookup, `studio3d/registry.py::find_solved_map` (registry.py:33). The
   registry (`studio3d/solved_registry.json`) contains exactly ONE entry, the Phase C
   anchor cylinder at n=64, matched by EXACT part-mask sha256. A Tamper can never match.
3. 2.5-D per-slice map transfer (correction.py:83-100). The map exists and is real
   (see section 3f) but `stack_to_voxel` measured a "37.37" percent in-part dopant move,
   over the "2" percent gate, and refused it. This gate WORKED as designed; refusing
   was correct.
4. `_native_inversion` (correction.py:36-73): `heatr3d.make_fgm(res, magnitude=1.0,
   bpp=4, proxy=rho_final)` on the BEFORE arm's own fields. This is what ran.

## 2. The failure mechanism, ranked by evidence strength

### (1) STRONGEST: the inversion heuristic emitted a near-binary dopant-removal map

`heatr3d.py:1292-1308` (`make_fgm`): `norm = clip((P - p2) / (p98 - p2), 0, 1)`;
`sat = 0.5 + 1.0 * ((1 - norm) - 0.5) = 1 - norm` at `magnitude=1.0`.

The proxy is the BEFORE arm's `rho_final`, which is nearly saturated: in-part min
"0.5566", max "1.0000", mean "0.9800", std "0.0596", and only "6.7" percent of voxels
below "0.9" (measured from `grade/heatr3d/uncorrected/fields.npz`). Percentile
normalization against a saturated field maps the dense bulk to norm ~ 1, hence sat ~ 0.

Measured from the emitted `grade/heatr3d/correction_sat.npz`:

- in-part sat: min "0.0000", max "1.0000", mean "0.0707", std "0.1982"
- "7972" of "10076" in-part voxels ("79.1" percent) are exactly "0.0"
- correlation of sat with rho_final in-part: "-0.977" (pure anti-correlation, i.e. the
  proportional-inverse rule did exactly what it says, on the wrong baseline)

In heatr3d, sat multiplicatively scales the dopant coupling (gamma blends `part * sat`,
correction.py:63-64; runner passes it into `H.run`/`_march`, studio3d/runner.py:179,203).
sat = 0 means those voxels barely absorb RF. The "corrected" part is a mostly undoped
body with a thin fully doped under-dense skin. Consequences, all in
`grade/heatr3d/corrected/results.json` vs `uncorrected/results.json`:

| metric | uncorrected (uniform) | corrected (inversion) |
|---|---|---|
| sigma_T (C) | "28.545" | "53.179" |
| t_phi90 (s) | "789.0" | "1348.0" |
| sim_time (s) | "1075.35" (reached rho stop) | "1500.0" (hit max_time cap) |
| T_max (C), ceiling "250" | "288.0" | "448.4" |
| rho_final_mean | "0.98" | "0.9495" |
| rho_final_std | "0.0596" | "0.1046" |
| warp_std_pct | "6.371" | "11.041" |

This is also the known over-critical-sigma failure branch (project memory: at
sigma0 above sigma*, the HEATR proportional-inverse law is anti-correlated with ground
truth). The heuristic has the wrong functional form for this regime, and at
`magnitude=1.0` it is applied at full strength (the 2.5-D route's own accepted magnitude
for this part was "0.7").

### (2) STRONG: rho_final is the wrong proxy for a run that stops at a density target

The BEFORE arm marches until `stop_mean_rho` = "0.98". By construction its final density
is nearly flat. Inverting a field the run already flattened concentrates the entire
16-level quantized range onto the residual "6.7" percent of skin voxels and deletes the
dopant everywhere else. A T_phi90-proxied inversion would have been wrong more gently;
a solved map would have been right.

### (3) SUPPORTING: no optimizer, no objective, no gates ever saw this part

Zero gradient evaluations were performed. There is no J trajectory, no uniform-baseline
comparison inside the correction builder, no hold-out, no acceptance gate. The only
quality signal attached to the map is the trust badge string itself. The two arms WERE
compared under the same conventions (same grid n=64, same drive, same
`stop_mean_rho` = "0.98" with the same "1500" s cap, each read at its own phi90), so the
comparison is fair (checklist item 3f): the graded arm honestly lost.

### Failure modes checked and EXCLUDED

- (a) Rail stall / factr stop after ~2 evals: not applicable, no L-BFGS-B ran at all.
- (b) Budget starvation: not applicable, budget was zero by design of the heuristic path.
- (c) Transfer/staircase loss: did NOT corrupt the printed map. The staircase gate is
  what correctly rejected the 2.5-D map ("37.37" percent move vs "2" percent gate); the
  native-inversion map is same-grid ("dopant_mass_move_rel": "0.0",
  "transfer_not_applicable"). The transfer discipline worked; the fallback it fell into
  is the problem.
- (d) Objective/read mismatch: no objective existed to mismatch. Both arms were scored
  at their own phi90 reach under identical stop rules.
- (e) Geometry: the Tamper is CLEAN. `grade/intake.json`: watertight ("0" open edges),
  no self-intersections, chamber fit ok (bbox "44.4" x "44.33" x "11.0" mm in a "60" mm
  chamber). "2436" triangles. No intake refusal fired and none was bypassed.
- (f) The 2.5-D verification itself PREDICTED a large benefit for this part
  (`grade/heatr/report.json`, cluster rep z="5.5": peak_benefit "29.49" percent,
  melt_benefit "52.17" percent at magnitude "0.7", accepted=true). The part is gradable;
  the Studio just could not deliver that map to the 3-D grid (chamber/grid mismatch:
  2.5-D reps solved at "65"/"85" mm chambers, grid_n "160", dx ~ "0.5" mm; the voxel
  arm is a "60" mm chamber at n="64", dx "0.9375" mm).

## 3. What the Phase E pyramid/cube stack does differently, point by point

Phase E (`solve3d/phase_e/run.py`, `solve3d/objective.py`, `solve3d/design_chain.py`)
vs what the Tamper got:

| convention | Phase E pyramid/cube | Tamper job |
|---|---|---|
| generator | FEM adjoint solve on a conformal mesh (solve3d.forward/adjoint) | closed-form heuristic `make_fgm` on a voxel grid |
| objective | asymmetric, overheat-weighted, `phi_floor` = "0.85" (objective.py:21-30,69) | none |
| stop / read | envelope argmin over the transient (run.py:27,190) | fixed rho stop with a "1500" s cap |
| descent | L-BFGS-B, `1/|g0|` first-step rescale as STANDING convention (run.py:225) | none |
| budget | "12" gradient evaluations (run.py:222) | "0" |
| regularization | 1.0 mm physical filter chain (design_chain, Phase C prereg) | 4-bpp quantize only, no spatial filter |
| acceptance | hold-out bands + sub-filter smoothing gates, solved_label | trust badge string only |
| baseline comparison | uniform arm at matched read inside the protocol | arms compared only after the fact |
| map character | modest modulation around uniform | "79" percent of part at zero dopant |

The Studio ALREADY HAS the correct engine wrapped and reachable: `solve3d/studio_solve.py`
("the Phase C solve, wrapped so the Studio can run it on an IMPORTED part", with the
frozen conventions and both acceptance gates, deviations D1-D5 recorded), and
`build_correction` ranks its artifact FIRST (correction.py:119-139). It simply was not
invoked for this job; the auto pipeline fell through the whole ladder to the last rung.

## 4. Recommended fix path (for the Studio lane; not implemented here)

1. Wire the correction slot to the direct solve by default: when no registry match and
   no fresh artifact exist, the `/grade/auto` pipeline should run
   `solve3d.studio_solve` (budget "40" forward-equivalents, cold start) on the imported
   part BEFORE the corrected densify arm, instead of falling through to the heuristic.
   The plumbing exists (`grade_server.py:1043`); it only needs scheduling plus a load
   policy honoring the one-heavy-solve convention.
2. Gate or retire the `heatr3d_native_inversion` rung. At minimum: (a) never run it with
   proxy `rho_final` off an arm that stopped on a density target; (b) cap its magnitude
   (the 2.5-D rulebook chose "0.7" for this very part); (c) require a predicted-benefit
   check against the uniform arm before the corrected arm is printed, and REFUSE (ship
   uniform, say so loudly) when the map is worse. A correction path that can only assert
   a trust badge, never a number, should not be allowed to emit a print map.
3. Alternatively make the refusal terminal: if the 2.5-D map fails the transfer gate and
   no solved map exists, report "no deliverable correction, run the direct solve" rather
   than substituting a known-legacy heuristic. The provenance system already records
   everything needed to say this honestly.
4. Close the 2.5-D transfer gap for gradable parts: the per-slice maps are solved at
   per-cluster chambers ("65"/"85" mm, dx ~ "0.5" mm) and refused at 3-D ("60" mm,
   dx "0.9375" mm). Emitting the 2.5-D maps on (or resampling support-aware to) the
   voxel arm's chamber/grid would let the "29" to "52" percent predicted benefit reach
   the printer.

## Verification performed

All numbers above were read directly from the job artifacts
(`state.json`, `intake.json`, `correction_provenance.json`, both `results.json`,
`report.json`, `verify.log`, `densify_corrected.log`) and from
`correction_sat.npz` / `uncorrected/fields.npz` via a read-only numpy inspection
(single-threaded, OMP/OPENBLAS=1). No solver was re-run. Code citations are from the
main tree, which is the tree the server actually imports: `grade_server.py:26-27`
resolves `binderjet/code/geo-prewarp` (the MAIN tree, not the worktree) onto sys.path
and into its subprocess commands. The worktree
`.claude/worktrees/strange-leakey-01a387` does not contain `studio3d/correction.py`
(the studio3d correction module lives in the merged main tree); its `heatr3d.py`
carries the same `make_fgm` proportional-inverse rule (worktree heatr3d.py:697).

## Addendum 2026-08-05: the stale melt-onset snapshot (second incident, same job)

Matt flagged the "fixed" Tamper's densified view: a two-lobed pancake instead
of a disc. Root cause chain, verified on the archived fields plus a
full-horizon probe (uncorrected arm, no density stop, 1500 s):

1. heatr3d's T_phi90 / phi_final / T_max_c are MELT-ONSET reads by that
   solver's documented convention (heatr3d.py:1248-1263, snapshot at t90).
   For densify runs the march continues long past t90.
2. The Studio consumed them as end-of-run truth. Consequences on this job:
   the viewer classified ~1000 flange-rim voxels as loose powder from the
   stale phi (their END-state rho_final, mean 0.83, proves they melted and
   were consolidating); and the ceiling gate read 239.1 C while the march
   log of the same drive shows ~317 C at the density stop and 360 C at the
   horizon. The ACCEPTED benefit verdict rested on that under-report.
3. The under-melt is also ANISOTROPIC by real solver physics (y = electrode
   axis, heatr3d.py:227): uniform dopant leaves the x-rim cold; the
   inversion boosted it but cut the y-rim 39 percent, swapping which rim
   lags. Regional melt swap measured across ALL archived pairs: 7-41
   percent of part voxels change melt state between arms.

Fixes (Studio-side only; heatr3d.py untouched, its convention is documented):
end-state solid classification from rho_final (FUSED_RHO 0.60 above the 0.55
bed initial, CONSOLIDATED_RHO 0.90, under-consolidated count loud) in
warped_mesh + viewer; gates record T_end_max_C from T_final and
T_ceiling_ok uses the true peak; the benefit gate compares end-state peaks
when both arms carry them and says "melt-onset read" out loud when not.
Tests: studio3d/tests/test_stale_snapshot_fixes.py (red-first).

Open finding: with true-peak reads the Tamper likely FAILS the 250 C ceiling
in both arms at the mean-rho-0.98 stop (~300+ C). Full consolidation costs
360 C at the horizon. Under this uniform drive the part is ceiling-limited;
honest handling is a red ceiling gate until lower-power/longer schedules or
the direct solve handle it.
