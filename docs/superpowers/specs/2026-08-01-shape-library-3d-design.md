# 3-D Primitive Geometry Test Library for heatr3d / solve3d

Date: 2026-08-01
Status: DRAFT — awaiting Matt's approval of the curated list, sizing, and
conventions BEFORE any build (house rule).
Owner: Matt McCoy
Worktree: `../geo-prewarp-shapelib3d`, branch `feat/shape-library-3d`
(off `feat/pernode-twosided-tuning`).

## 1. Goal

A curated library of watertight STL primitives, each chosen to stress ONE
specific RF-heating or numerical behavior of heatr3d, delivered as
`shape_library_3d/` at repo root. This is the 3-D analog of the 2-D/2.5-D
HEATR shape library (`shapes.py::make_shape`, the 18/19-shape standardized
campaign) and the **substrate for Phase E** of the solve-port spec
(`docs/superpowers/specs/2026-07-31-solve-port-3d-design.md`). The library
must be consumable by BOTH engines:
- **heatr3d** (voxel FDM) via voxelization onto its `Grid` boolean mask
  (`heatr3d.run(grid, part, p)`),
- **solve3d** (dolfinx FEM) via the STL/mesh itself (its future conforming-mesh
  ingestion path).

Deliverables: generator code, per-shape metadata (name, tier, RF characteristic
stressed, volume, genus, facet stats, bbox, realized chamber gap), TDD tests,
README, and a densification-visual smoke campaign on 2–3 Tier-1 shapes.

This spec covers the LIBRARY only. The full-library physics campaign is a
separate plan (explicitly out of scope here).

**Downstream consumer (Matt, 2026-08-01): the "Grade and Print" tool** — the
intended ultimate RFAM 3-D print tool — must be able to enumerate and load this
library programmatically. Design consequence: a clean, importable public API
(`shape_library_3d.iter_parts()`, `load_part_stl(name)`, the `SHAPES` registry
with full metadata) plus stable on-disk artifacts (`stl/*.stl` +
`library_manifest.json`) that a tool can discover without importing internals.
No Grade-and-Print code is written here; the API contract is the handoff, same
posture as the solve3d STL handoff.

## 2. Sizing / normalization convention (the one Matt approved)

**Equal-VOLUME normalization**, the 3-D analog of the 2-D equal-area
convention (`A* = π·10² = 314.159 mm²`, the 20 mm circle; `dose_control_lib.A_STAR_MM2`).

- **Reference volume `V* = (4/3)·π·10³ = 4188.7902 mm³` — the 20 mm-diameter
  sphere** (Matt, 2026-08-01). The reference IS the round primitive, exactly
  mirroring the 2-D convention where the reference is the circle.
- Normalization operates on the shape's **actual watertight `trimesh` volume**,
  by a single uniform scale factor, until `|vol − V*| / V* < 1e-4`. This is
  **grid- and mesh-independent** — it is NOT a voxel-raster area/volume on any
  solve grid. This is deliberate and required by the solve-port spec ("chi must
  be MESH-INDEPENDENT … never a binary raster on the solve grid"): both
  consumers (voxel heatr3d, FEM solve3d) resample from the same canonical STL,
  so the canonical size must not be tied to either one's resolution.
- Consequence, measured (not assumed): trimesh's faceted "smooth" solids
  (sphere, cylinder, cone, torus) undershoot the ideal analytic volume because
  faceting cuts corners (icosphere L4 measures 4179.7 mm³ before scaling). The
  uniform-scale-to-mesh-volume step corrects this for EVERY shape, so all
  shapes — flat-faceted and curved alike — carry the same true solid volume V*.
- **Voxelization staircase error is reported, never corrected away.** When a
  library STL is voxelized onto a heatr3d `Grid`, the voxel-count volume will
  differ from V* by a staircase error that shrinks with grid n; the voxelizer
  reports `voxel_volume_mm3` and `voxel_vs_Vstar_frac` at the chosen n. The
  canonical STL volume stays V*.

### Realized sizing table (closed-form, all hit V* exactly; grounded 2026-08-01)

Chamber is the heatr3d cubic domain `L = 60 mm` (`heatr3d.Grid.L = 0.060`).
"gap/side" = (60 − max bbox extent)/2, the minimum powder margin. All fit with
≥ 15.8 mm/side. Aspect ratios are fixed, defensible defaults (equant where a
free parameter exists); Matt can override any in review.

| # | shape | fixed aspect | key dims (mm) | bbox (mm) | genus | gap/side |
|---|---|---|---|---|---|---|
| 1 | cube | — | side 16.120 | 16.1³ | 0 | 21.9 |
| 2 | cylinder | h = d | d 17.472, h 17.472 | 17.5³ | 0 | 21.3 |
| 3 | sphere | — | d 20.000 | 20.0³ | 0 | 20.0 |
| 4 | toroid | R = 2r | r 4.734, R 9.468 | 28.4×28.4×9.5 | 1 | 15.8 |
| 5 | cone | h = d | d 25.198, h 25.198 | 25.2³ | 0 | 17.4 |
| 6 | pyramid (sq base) | h = base | base 23.249, h 23.249 | 23.2³ | 0 | 18.4 |
| 7 | pipe (open-ended hollow cyl) | od 24, H 20 fixed; t solved | wall t 3.206, id 17.588 | 24×24×20 | 1 | 18.0 |
| 8 | thick-strut cubic lattice | 2 cells/axis, strut = 0.22·pitch | bbox solved to V* | ~mid-20s³ | ≥1 (recorded) | recorded |
| 9 | L-extrusion (reentrant) | H = c, arm = c/2 | c 17.742, arm 8.871 | 17.7³ | 0 | 21.1 |
| 10 | truncated octahedron | Archimedean | edge ≈ 7.18 | 20.3³ | 0 | 19.8 |

Tier-2 (same nominal sphere as #3, three meshes; all uniform-scaled to V*):

| # | shape | tessellation | note |
|---|---|---|---|
| 3 | sphere (control) | icosphere, subdiv 4 (5120 uniform faces) | smoothest control |
| 11 | icosphere (coarse uniform) | icosphere, subdiv 3 (1280 uniform faces) | 3→11 isolates RESOLUTION |
| 12 | UV sphere (pole-singular) | uv-sphere, face count matched to #11 ±10% | 11→12 isolates facet DISTRIBUTION at matched count |

Two clean axes: **3→11 = uniform resolution** (fine→coarse, distribution held
uniform); **11→12 = facet distribution** (uniform→pole-clustered, count held).
Any physics difference across 3/11/12 is pure mesh sensitivity.

Tier-3 (NOT parts — rejection tests, never fed to physics):

| # | shape | defect | expected rejection |
|---|---|---|---|
| 13 | zero-thickness open cylinder | side wall only, no caps, no wall thickness | non-watertight (naked edges) |
| 14 | flat single-face plane grid | 2-D triangulated sheet at z=0 | zero enclosed volume |

## 3. Per-shape metadata schema

One JSON per shape in `shape_library_3d/meta/<name>.json`, plus a combined
`library_manifest.json`. Fields (all MEASURED from the final STL, never
hand-transcribed — data-contract rule):

```
name, tier (1|2|3), role ("control"|"stressor"|"reject"),
rf_characteristic  (one-line: the behavior stressed),
numerical_characteristic (one-line: the mesh/solver behavior stressed),
target_volume_mm3 (V* for tiers 1-2; null for tier 3),
actual_volume_mm3, volume_err_frac,
is_watertight, is_winding_consistent,
euler_number, genus,
n_vertices, n_faces,
facet_area_mm2 {min, max, mean, std},
bbox_mm [dx, dy, dz], realized_chamber_gap_mm,
scale_factor, stl_path, sha256
```

RF-characteristic strings (the curation rationale, one per shape) are the
column that makes this a *curated* library rather than a shape dump: e.g.
cube="flat-faced orthogonal baseline control", cone="smooth base → apex
singularity, field concentration at a point", pyramid="apex singularity PLUS
sharp edges + flat facets — separates apex from edge effects vs the cone",
toroid="genus-1 through-hole: field shadowing inside the hole, no flat faces",
pipe="wall-thickness vs interior-field, the thin-wall heating question",
lattice="intersecting trusses, internal occlusion, strut-junction hot spots",
L="reentrant corner, continuity with the 2-D outlier", trunc-oct="oblique
planar facets at mixed angles, between cube and sphere".

## 4. Ingestion contracts

### 4a. heatr3d (voxel) — `shape_library_3d/voxelize.py`
`stl_to_mask(mesh_or_path, grid) -> np.ndarray[bool]` of shape
`(grid.n, grid.n, grid.n)`, centered in the 60 mm domain, matching the
`heatr3d.make_geometry` convention (X,Y,Z meshgrid, y = field axis). Uses
`trimesh.voxelized`/`contains` on cell centers. Returns a mask directly usable
as `heatr3d.run(grid, part=mask, p=...)`. Reports voxel volume vs V*.
**Validation gate first:** rejects tier-3 meshes before voxelizing.

### 4b. solve3d (mesh) — STL is the deliverable
solve3d's conforming-mesh path is future work (Phase A ingests analytic
predicates for circle/square only). This library ships the watertight STL that
that path will mesh; no solve3d code is written here. The STL's watertightness
and mesh-independent V* are exactly the properties solve3d's chi construction
requires.

### 4c. Tier-3 rejection contract — `shape_library_3d/validate.py`
`validate_part_mesh(mesh) -> None` raises with an informative message:
- `NonWatertightMeshError` if `not mesh.is_watertight` (names naked-edge count),
- `ZeroVolumeError` if `mesh.volume <= vol_eps` (names the measured volume).
Both subclass `InvalidPartGeometryError`. The heatr3d voxel bridge and any
"load as part" entry call this first, so shapes 13/14 fail loudly and cannot
reach physics. This is the RED-first test target.

## 5. Directory layout

```
shape_library_3d/
  __init__.py          # public API: SHAPES registry, load_part_stl, iter_parts
  constants.py         # V_STAR_MM3, CHAMBER_L_MM=60, tolerances
  registry.py          # SHAPES: ordered dict name -> ShapeSpec (tier, role, characteristics, generator ref, aspect params)
  generators.py        # trimesh generator per shape (split to generators_tier1/2/3 if >400 lines)
  normalize.py         # scale_to_volume(mesh, V*) uniform-scale bisection/closed-form
  validate.py          # validate_part_mesh + exception classes (tier-3 contract)
  voxelize.py          # stl_to_mask(mesh, grid) heatr3d bridge
  build_library.py     # CLI: generate all -> stl/*.stl + meta/*.json + manifest
  smoke_campaign.py    # CLI: voxelize 2-3 tier-1 shapes, run heatr3d, render viz
  stl/                 # generated watertight STLs (committed)
  meta/                # per-shape json + library_manifest.json (committed)
  README.md
  tests/
    conftest.py        # sys.path shim (mirror solve3d/tests/conftest.py)
    test_normalize.py       # equal-volume convergence to V* (RED first)
    test_validate.py        # tier-3 rejection raises (RED first)
    test_generators.py      # each tier-1/2 STL: watertight, winding-consistent, vol==V*, expected genus/euler
    test_voxelize.py        # stl_to_mask volume-fraction sane vs analytic; rejects tier-3
    test_registry.py        # every registry entry has a generator + required metadata keys
```

Follows the coding-style rule (200–400 line files, dict registry idiom from
`dose_control_2d.py:100`, `__all__`, type hints, logger not print). Env:
`.venv312/bin/python` (trimesh 4.12.2).

## 6. TDD plan (red → green per unit; house rule)

Every generator behavior is unit-testable with trimesh. Order:
1. **normalize** — RED: `scale_to_volume` on a unit cube must reach V*; watch
   fail (function absent) → GREEN.
2. **validate** — RED: `validate_part_mesh` on a naked-edge open cylinder and
   on a flat plane must raise the two typed errors; watch fail → GREEN.
3. **generators** — per shape, RED: assert watertight + `vol==pytest.approx(V*,
   rel=1e-4)` + expected euler/genus; watch fail → GREEN by wiring the trimesh
   builder. Genus is ASSERTED where analytic (0 for solids; 1 for toroid, pipe)
   and RECORDED where emergent (lattice).
4. **voxelize** — RED: voxelized cube fill fraction ≈ (side/L)³ within staircase
   tolerance; tier-3 mesh raises before voxelizing.
5. **registry** — RED: every SHAPES entry resolves to a generator and yields all
   required metadata keys.

Every STL is verified watertight + volume-correct before it enters `stl/`
(`build_library.py` re-asserts and refuses to write a failing mesh).

## 7. Smoke campaign (end-to-end proof, then view figures personally)

`smoke_campaign.py` runs heatr3d with the STANDARD parameter set on **cube
(#1, flat control), sphere (#3, smooth control), cone (#5, apex stressor)** —
a clean control/control/stressor spread — within grid ceilings (n = 64,
full-physics safe; ≤ 96 ceiling). Pipeline per shape:
STL → `stl_to_mask(grid n=64)` → `heatr3d.run` with standard `Params`,
exposure-time optimizer to `phi_bar = 0.90`, 250 °C ceiling margin checked,
`|energy residual|/dose` gate printed → densification visuals via the existing
standard (`heatr3d_job._render_summary_plots`: melt_progression, ortho_slices,
melt_vs_cad, density_hist, radial_density, temperature_hist).

I will **Read the rendered PNGs myself** before delivering (memory:
view-figure-renders-personally; F7-sliver lesson). This proves the STL→mask→
heatr3d→viz path end to end. The FULL 14-shape campaign is NOT run here — it
gets its own plan.

## 8. Out of scope

- The full-library physics campaign (separate plan).
- solve3d conforming-mesh ingestion code (future Phase; STL is the handoff).
- Any dissertation edits.
- Dwell/rotation/FGM solving (this is geometry substrate only).

## 9. Assumptions (stated, Matt may correct in review)

- Chamber = the 60 mm heatr3d cubic domain; parts centered; gap reported.
- Fixed aspect ratios chosen equant where free (cylinder h=d, cone h=d,
  pyramid h=base, torus R=2r, L H=c); pipe fixes od=24/H=20 and solves wall t.
- Lattice topology = 2 cells/axis cubic frame, strut square section 0.22·pitch,
  overall size solved to V*; exact genus recorded from the mesh, not guessed.
- Sphere trio meshes as in §2; uv-sphere count matched to icosphere-L3 ±10%.
- Smoke shapes = cube/sphere/cone at n=64.

## 10. Commit discipline (house rule)

Commit ONLY files under `shape_library_3d/` and this spec, by explicit path —
never `git add -A`. `git pull --rebase` before any push. Other sessions own
`feat/pernode-twosided-tuning`.
