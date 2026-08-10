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
| **cone** | sharp | - | - | - | - | - | - | - | DE-RISK |
| **sphere** | smooth convex | - | - | - | - | - | - | - | DE-RISK |
| **cylinder** | thin/elongated | - | - | - | - | - | - | - | DE-RISK |
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
- Offset is NOT a stable constant (pyramid +8.8 vs square/cube +11-12, confounded by grid/aliasing/convergence). The campaign's matched-grid offsets replace the walked-back claim. See [[thermal-ceiling-loop-closed]].
- Expected transfer-limited candidates (fidelity gate may flag): cone (apex), lattice (struts), open_cylinder/pipe (walls). Honest flag > forced grid-confounded verify.
