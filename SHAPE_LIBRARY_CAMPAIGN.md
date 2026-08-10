# Shape-Library Generalization Campaign

**Goal:** Generalize the closed thermal-ceiling loop (B1-B4: ceiling-coupled dopant
shaping + backed-off drive, cross-engine `is_sendable`) across the full 14-shape
`shape_library_3d/stl/` library, to turn "closes on 3 shapes" into a defensible
"generalizes across the shape family" dissertation claim - AND to produce a real
matched-grid cross-engine OFFSET study (replacing the walked-back "stable constant").

**Approach (Matt 2026-08-10): DE-RISK FIRST on cone + sphere + cylinder** (one sharp,
one smooth, one thin-wall), prove the harness handles all three, THEN commit the
remaining 9 as an unattended multi-day run.

## The per-shape pipeline (each shape)
1. **Fidelity pre-gate** (the pyramid lesson): DG0->voxel transfer mass-move must be
   under 2%. Pick the faithful grid, or honestly flag the shape `transfer-limited`
   and do NOT ship a grid-confounded verify. Non-monotonic in n (pyramid: n80 ok,
   n64+n96 fail) - so probe grids, don't assume.
2. **Drive probe** (fine hold-out): backed-off drive whose UNIFORM true peak is under
   the effective ceiling 235 C (= 250 - 15 headroom). Per-part; solved not assumed.
3. **FD-GATE on the geometry** (IRON LAW): combined AL gradient + density co-state,
   frozen 1e-6, mutation must bite, in a densifying regime. A correct co-state on
   one mesh is NOT proof on another. If it fails, STOP that shape.
4. **AL solve** at the chosen drive (run_solve_al_b4 --shape). Sharp shapes: lower
   DELTA_EMA (pyramid oscillated at 0.5, converged=False) - watch the outer trajectory.
5. **Cross-engine verify** (Studio lane heatr3d is_sendable): true peak <= 250 both
   engines. Record the offset (heatr3d - dolfinx) at the matched grid.

## Compute discipline
One heavy AL solve at a time (multi-hour on conforming/voxel meshes), sequential,
`caffeinate -i`, checkpointed/resumable. The HEATR session confirmed no slot
contention. Engineer builds + FD-gates + probes (light); I launch heavy solves.

## Status table (14 shapes)

| shape | class | fidelity gate | drive | FD-gate | dolfinx peak | heatr3d | is_sendable | offset | status |
|---|---|---|---|---|---|---|---|---|---|
| square (anchor) | extrusion | - | 0.34x | 3e-9 | 235.03 | 246.3 | YES | +11.3 | DONE |
| cube | smooth convex | n64 ok | 0.57x | 3.58e-9 | 235.08 | 246.9 | YES | +11.8 | DONE |
| pyramid | sharp | n80 only (n64/96 alias) | 0.585x | 2.44e-9 | 235.43 | 244.2 | YES | +8.8 | DONE (converged=False) |
| **cone** | sharp | n96 (n64/80 alias apex) | probing | 5.95e-9 GREEN | - | - | - | - | READY (drive pending) |
| **sphere** | smooth convex | n64 ok | probing | 1.45e-9 GREEN | - | - | - | - | READY (drive pending) |
| **cylinder** | thin/elongated | n64 ok | probing | 1.26e-9 GREEN | - | - | - | - | READY (drive pending) |
| trunc_octahedron | smooth convex | - | - | - | - | - | - | - | queued |
| icosphere_coarse | smooth convex | - | - | - | - | - | - | - | queued |
| uv_sphere | smooth convex | - | - | - | - | - | - | - | queued |
| toroid | concave | - | - | - | - | - | - | - | queued |
| l_extrusion | reentrant | - | - | - | - | - | - | - | queued |
| flat_plane | thin slab | - | - | - | - | - | - | - | queued |
| open_cylinder | thin wall | - | - | - | - | - | - | - | queued |
| pipe | thin wall | - | - | - | - | - | - | - | queued |
| lattice | thin struts | - | - | - | - | - | - | - | queued |

## Decisions / notes
- 2026-08-10: de-risk on cone+sphere+cylinder first (Matt). Full 9 remaining after harness proves out.
- 2026-08-10: HARNESS BUILT + de-risked on all three. `solve3d/shape_campaign.py`
  (reusable for all 14): fidelity pre-gate = a two-env bridge (spike exports the
  SOLVE-mesh DG0 centroids; .venv312 voxelizes the STL with the SAME heatr3d
  voxelizer the cross-engine verify uses and measures the transfer mass-move).
  Conforming-mesh support extended to cone/sphere/cylinder in
  `solve3d/phase_e/geometry.py` (OCC solids sized to the equal-volume invariant
  4188.79 mm^3, so the pre-registered 1e-9 volume check still passes; all three
  are z-axis bodies of revolution, cone apex-up).
- FIDELITY PROBE, honest limit: the transfer mass-move is exactly 0 for a uniform
  field, so the pre-gate probes an O(1) basis (three axis ramps + radial, [0,1]
  normalised) and gates on the WORST axis ramp; radial is reported, not gated
  (pathologically conservative on convex parts). Validated against the PINNED
  pyramid: the worst-axis argmin correctly SELECTS the faithful grid (n80). It
  does NOT reproduce the solved map's exact per-grid value (that needs the solved
  map) -- the definitive mass-move stays the post-solve cross-engine number the
  pyramid verify measured. The chosen grid is the launch recommendation; if the
  post-solve map exceeds 2% there, step to the next finer grid.
- FIDELITY RESULTS (solve-mesh resolution, ~9-9.5k design cells):
  - cone n96 0.92% (n64 3.14% / n80 3.59% z-alias at the apex -> needs the FINE
    grid, opposite of the pyramid which needed n80). NOT transfer-limited.
  - sphere n64 1.07% (smooth; coarsest grid suffices; consistent with cube n64).
  - cylinder n64 1.25% (thin/flat caps; n80 z-ramp sits at 2.00%, so it is the
    likeliest to need a post-solve bump to n96).
- FD-GATE (IRON LAW) GREEN for all three at the 2.0x densifying gate drive: the
  combined AL gradient, the density co-state, and the march fidelity all pass at
  1e-6 with the drop-lambda_rho mutation biting (cone 3.41% / sphere 1.06% /
  cylinder 2.13% mutation) -- the density co-state is load-bearing on each mesh.
- Offset is NOT a stable constant (pyramid +8.8 vs square/cube +11-12, confounded by grid/aliasing/convergence). The campaign's matched-grid offsets replace the walked-back claim. See [[thermal-ceiling-loop-closed]].
- Expected transfer-limited candidates (fidelity gate may flag): cone (apex), lattice (struts), open_cylinder/pipe (walls). Honest flag > forced grid-confounded verify.
