# shape_library_3d

The 3-D analog of the 2-D/2.5-D HEATR shape library: a curated set of watertight
STL primitives, each chosen to stress one specific RF-heating or numerical
behavior of the RFAM solvers **heatr3d** (voxel FDM) and **solve3d** (dolfinx
FEM). This is the geometry substrate for Phase E of the direct-solve 3-D port
(`docs/superpowers/specs/2026-07-31-solve-port-3d-design.md`) and a stable input
library for the eventual **Grade and Print** tool.

Spec: `docs/superpowers/specs/2026-08-01-shape-library-3d-design.md`.
Plan: `docs/superpowers/plans/2026-08-01-shape-library-3d.md`.

## Sizing convention: equal volume to V*

Every Tier-1/2 primitive is uniform-scaled so its **exact trimesh solid volume**
equals **V\* = 4188.79 mm³ — the 20 mm-diameter sphere** — the 3-D analog of the
2-D equal-area reference (A\* = π·10² = 314.16 mm², the 20 mm circle). Scaling is
done on the mesh volume, not a voxel raster, so it is grid- and mesh-independent:
both consumers resample from the one canonical STL. Voxelization staircase error
is *reported* (`voxelize.voxel_volume_report`), never corrected away.

All parts fit the heatr3d 60 mm cubic chamber; the tightest powder gap is the
toroid (15.8 mm/side).

## The library (generated; see `meta/library_manifest.json`)

| # | name | tier | RF characteristic stressed | genus | faces | gap/side |
|---|------|------|----------------------------|-------|-------|----------|
| 1 | cube | 1 | flat-faced orthogonal baseline control | 0 | 12 | 21.9 |
| 2 | cylinder | 1 | curved wall meeting flat end caps (90° junctions) | 0 | 384 | 21.3 |
| 3 | sphere | 1 | smoothest control; the 2-D circle's 3-D analog | 0 | 5120 | 20.0 |
| 4 | toroid | 1 | genus-1 through-hole field shadowing, no flats | 1 | 9216 | 15.8 |
| 5 | cone | 1 | smooth base → apex singularity (point concentration) | 0 | 192 | 17.4 |
| 6 | pyramid | 1 | apex **plus** sharp edges + flat facets (vs cone) | 0 | 6 | 18.4 |
| 7 | pipe | 1 | wall-thickness vs interior field (thin-wall heating) | 1 | 1024 | 18.0 |
| 8 | lattice | 1 | strut-junction hot spots, internal occlusion | 28 | 618 | 16.5 |
| 9 | l_extrusion | 1 | reentrant corner; 2-D L-outlier continuity | 0 | 24 | 21.1 |
| 10 | trunc_octahedron | 1 | oblique planar facets at mixed angles | 0 | 44 | 19.8 |
| 11 | icosphere_coarse | 2 | same sphere, coarser uniform mesh (resolution axis) | 0 | 1280 | 20.0 |
| 12 | uv_sphere | 2 | same sphere, pole-clustered facets (distribution axis) | 0 | 1152 | 20.0 |
| 13 | open_cylinder | 3 | **NOT A PART** — naked-edge shell (reject fixture) | — | 96 | — |
| 14 | flat_plane | 3 | **NOT A PART** — zero-volume sheet (reject fixture) | — | 32 | — |

Shapes 3 / 11 / 12 are the same nominal sphere in three meshes: **3→11** isolates
uniform-mesh **resolution**; **11→12** isolates facet **distribution** at matched
face count. Any physics difference across the trio is pure mesh sensitivity.

## Ingestion contracts

- **heatr3d (voxel):** `voxelize.stl_to_mask(mesh, grid)` → boolean
  `(n,n,n)` part mask usable directly as `heatr3d.run(grid, part, p)`. It
  validates first, so a non-part can never produce a silently-wrong mask.
- **solve3d (mesh):** consume the watertight `stl/<name>.stl` directly (its
  conforming-mesh path). The mesh-independent V\* is exactly what solve3d's chi
  construction requires.
- **Tier-3 rejection:** `validate.validate_part_mesh(mesh)` raises typed
  `NonWatertightMeshError` / `ZeroVolumeError` (both subclass
  `InvalidPartGeometryError`). Shapes 13/14 exist to prove this fires.

## Public API (Grade-and-Print handoff)

```python
from shape_library_3d import SHAPES, iter_parts, load_part_stl
for name, spec in iter_parts():          # Tier-1/2 only (loadable parts)
    mesh = load_part_stl(name)           # normalized watertight trimesh
    print(name, spec.rf_characteristic, spec.tier)
```

Stable on-disk artifacts a tool can enumerate without importing internals:
`stl/*.stl` and `meta/library_manifest.json` (name, tier, RF/numerical
characteristic, volume, genus, facet stats, bbox, realized gap, sha256).

## Regenerate

Environment: `../geo-prewarp/.venv312` (trimesh 4.12, manifold3d, rtree — see
`requirements.txt`). From the worktree root:

```bash
../geo-prewarp/.venv312/bin/python -m shape_library_3d.build_library
../geo-prewarp/.venv312/bin/python -m pytest shape_library_3d/tests/ -q
```

## Caveats

- **Thin features under-resolve on coarse voxel grids.** The lattice loses ~49%
  of its volume voxelized at n=64 (struts ≈ 2 cells thick); the report surfaces
  this. Use a finer grid for thin-strut/thin-wall parts, or read
  `voxel_vs_Vstar_frac` before trusting a coarse run.
- Grid ceilings (heatr3d): n ≤ 96 full physics, n ≤ 128 EQS-only
  (`HEATR_STANDARD_PARAMETERS.md`).
- The full 14-shape physics campaign is a **separate** plan; this library only
  ships a 3-shape densification smoke run (`smoke_campaign.py`).
