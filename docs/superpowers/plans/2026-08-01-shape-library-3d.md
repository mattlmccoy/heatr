# 3-D Primitive Geometry Test Library — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline,
> this session). Steps use checkbox (`- [ ]`) syntax. TDD red→green per unit.

**Goal:** Build `shape_library_3d/` — a curated library of watertight STL
primitives (10 Tier-1 RF-physics parts, 2 Tier-2 mesh-sensitivity sphere probes,
2 Tier-3 rejection tests), each normalized to equal volume V*=4188.79 mm³, with
a registry, metadata, tests, README, and a heatr3d densification smoke campaign.

**Architecture:** Pure-`trimesh` generators return watertight meshes; a uniform
`scale_to_volume` finalizes each Tier-1/2 mesh to V* (pipe self-solves its inner
radius at fixed od/H; lattice builds via voxel-mask → marching cubes). A dict
`SHAPES` registry pairs each generator with its curated metadata. `validate.py`
enforces the Tier-3 rejection contract. `voxelize.py` bridges STL → heatr3d
`Grid` boolean mask. All metadata is measured from the final mesh.

**Tech Stack:** Python 3.12, trimesh 4.12.2 (`.venv312/bin/python`), numpy,
pytest, scikit-image (marching cubes, via trimesh), heatr3d (read-only public API).

**Git discipline:** worktree `../geo-prewarp-shapelib3d`, branch
`feat/shape-library-3d`. Every git call MUST prefix:
`export GIT_DIR=<main>/.git/worktrees/geo-prewarp-shapelib3d; export GIT_WORK_TREE=<main>-shapelib3d`
(plain `cd` into the worktree is not honored — harness resets cwd). Commit ONLY
`shape_library_3d/**` and `docs/superpowers/**` by explicit path; never `git add -A`.
`git pull --rebase` before any push.

**Run tests with:** `.venv312/bin/python -m pytest shape_library_3d/tests/ -v`
(cwd = worktree root, invoked with an absolute interpreter path).

---

### Task 1: Package skeleton + constants

**Files:**
- Create: `shape_library_3d/__init__.py`, `shape_library_3d/constants.py`
- Create: `shape_library_3d/tests/__init__.py`, `shape_library_3d/tests/conftest.py`

- [ ] **Step 1:** Write `constants.py`:

```python
"""Canonical constants for the 3-D primitive shape library."""
from __future__ import annotations
import math

V_STAR_MM3: float = (4.0 / 3.0) * math.pi * 10.0 ** 3  # 4188.7902, the 20mm sphere
CHAMBER_L_MM: float = 60.0                              # heatr3d cubic domain (Grid.L)
VOL_REL_TOL: float = 1e-4          # equal-volume acceptance |v-V*|/V*
VOL_EPS_MM3: float = 1e-6          # zero-volume rejection threshold
EXTENT_EPS_MM: float = 1e-6        # degenerate (planar) bbox axis threshold

__all__ = ["V_STAR_MM3", "CHAMBER_L_MM", "VOL_REL_TOL", "VOL_EPS_MM3", "EXTENT_EPS_MM"]
```

- [ ] **Step 2:** Write `tests/conftest.py` (repo-root on path, mirror solve3d/tests):

```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # geo-prewarp worktree root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 3:** `tests/__init__.py` empty; `__init__.py` minimal (public API filled in Task 7).
- [ ] **Step 4:** Commit `shape_library_3d/constants.py tests/`.

---

### Task 2: `normalize.scale_to_volume` (uniform scale to V*)

**Files:** Create `shape_library_3d/normalize.py`, `tests/test_normalize.py`

- [ ] **Step 1 (RED):**

```python
import trimesh
from shape_library_3d.constants import V_STAR_MM3
from shape_library_3d.normalize import scale_to_volume

def test_unit_cube_scales_to_Vstar():
    m = trimesh.creation.box(extents=[1.0, 1.0, 1.0])   # volume 1 mm^3
    scaled, s = scale_to_volume(m, V_STAR_MM3)
    assert scaled.volume == __import__("pytest").approx(V_STAR_MM3, rel=1e-9)
    assert s == __import__("pytest").approx(V_STAR_MM3 ** (1 / 3), rel=1e-9)

def test_raises_on_nonpositive_volume():
    import pytest
    m = trimesh.creation.box(extents=[1, 1, 1]); m.invert()   # negative volume
    with pytest.raises(ValueError):
        scale_to_volume(m, V_STAR_MM3)
```

- [ ] **Step 2:** Run `pytest shape_library_3d/tests/test_normalize.py -v` → FAIL (module missing).
- [ ] **Step 3 (GREEN):**

```python
"""Uniform-scale a mesh to a target solid volume. Volume scales as s**3, so a
single closed-form step is exact to floating point."""
from __future__ import annotations
import trimesh

def scale_to_volume(mesh: trimesh.Trimesh, target_vol_mm3: float):
    v = float(mesh.volume)
    if v <= 0.0:
        raise ValueError(f"mesh volume must be positive, got {v}")
    s = (target_vol_mm3 / v) ** (1.0 / 3.0)
    mesh.apply_scale(s)
    return mesh, s

__all__ = ["scale_to_volume"]
```

- [ ] **Step 4:** Run tests → PASS. **Step 5:** Commit `normalize.py test_normalize.py`.

---

### Task 3: `validate.validate_part_mesh` (Tier-3 rejection contract)

**Files:** Create `shape_library_3d/validate.py`, `tests/test_validate.py`

Rejection order (grounded on trimesh behavior — an open side-wall reports
volume≈0, so planarity must be checked before volume): degenerate/planar bbox →
`ZeroVolumeError`; else not watertight → `NonWatertightMeshError`; else
|volume|≤eps → `ZeroVolumeError`.

- [ ] **Step 1 (RED):**

```python
import numpy as np, trimesh, pytest
from shape_library_3d.validate import (
    validate_part_mesh, NonWatertightMeshError, ZeroVolumeError,
    InvalidPartGeometryError)

def _open_cylinder():   # side wall only, no caps -> naked edges, 3-D bbox
    n = 48; z0, z1, r = -10.0, 10.0, 10.0
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    bot = np.c_[r*np.cos(th), r*np.sin(th), np.full(n, z0)]
    top = np.c_[r*np.cos(th), r*np.sin(th), np.full(n, z1)]
    v = np.vstack([bot, top]); f = []
    for i in range(n):
        j = (i+1) % n; f += [[i, j, n+i], [j, n+j, n+i]]
    return trimesh.Trimesh(vertices=v, faces=np.array(f))

def _flat_plane():      # z=0 sheet -> zero z-extent, zero volume
    xs = np.linspace(-10, 10, 5); ys = np.linspace(-10, 10, 5)
    v = np.array([[x, y, 0.0] for y in ys for x in xs]); f = []
    for r in range(4):
        for c in range(4):
            a = r*5+c; f += [[a, a+1, a+5], [a+1, a+6, a+5]]
    return trimesh.Trimesh(vertices=v, faces=np.array(f))

def test_open_cylinder_rejected_not_watertight():
    with pytest.raises(NonWatertightMeshError) as e:
        validate_part_mesh(_open_cylinder())
    assert "watertight" in str(e.value).lower()

def test_flat_plane_rejected_zero_volume():
    with pytest.raises(ZeroVolumeError) as e:
        validate_part_mesh(_flat_plane())
    assert "volume" in str(e.value).lower()

def test_both_subclass_base_error():
    assert issubclass(NonWatertightMeshError, InvalidPartGeometryError)
    assert issubclass(ZeroVolumeError, InvalidPartGeometryError)

def test_valid_solid_passes():
    validate_part_mesh(trimesh.creation.box(extents=[5, 5, 5]))   # no raise
```

- [ ] **Step 2:** Run → FAIL (module missing). Also confirms my planarity-first
      ordering is what actually fires on real tier-3 meshes (adjust order only if
      the RED run shows a different trimesh volume/extent than assumed).
- [ ] **Step 3 (GREEN):**

```python
"""Part-mesh validation: the ingestion gate that rejects Tier-3 non-parts loudly."""
from __future__ import annotations
import numpy as np, trimesh
from shape_library_3d.constants import VOL_EPS_MM3, EXTENT_EPS_MM

class InvalidPartGeometryError(ValueError): ...
class NonWatertightMeshError(InvalidPartGeometryError): ...
class ZeroVolumeError(InvalidPartGeometryError): ...

def _open_edge_count(mesh: trimesh.Trimesh) -> int:
    # an edge on the boundary belongs to exactly one face
    edges = mesh.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())

def validate_part_mesh(mesh: trimesh.Trimesh) -> None:
    ext = np.asarray(mesh.extents, dtype=float)
    if ext.size < 3 or ext.min() <= EXTENT_EPS_MM:
        raise ZeroVolumeError(
            f"degenerate/planar mesh: bbox extents {ext} have a ~zero axis; "
            f"no enclosed volume (not a printable part)")
    if not mesh.is_watertight:
        raise NonWatertightMeshError(
            f"mesh is not watertight: {_open_edge_count(mesh)} naked (open) "
            f"boundary edges; cannot voxelize or mesh as a solid part")
    v = abs(float(mesh.volume))
    if v <= VOL_EPS_MM3:
        raise ZeroVolumeError(f"mesh encloses ~zero volume ({v} mm^3)")

__all__ = ["validate_part_mesh", "InvalidPartGeometryError",
           "NonWatertightMeshError", "ZeroVolumeError"]
```

- [ ] **Step 4:** Run → PASS. **Step 5:** Commit `validate.py test_validate.py`.

---

### Task 4: Generators (Tier-1 + Tier-2 + Tier-3)

**Files:** Create `shape_library_3d/generators.py`, `tests/test_generators.py`
(if generators.py exceeds ~400 lines, split into `generators_tier1.py` /
`_tier2.py` / `_tier3.py` and re-export from `generators.py`).

Each Tier-1/2 generator returns a watertight `trimesh.Trimesh` at (approx) V*;
the registry (Task 5) applies `scale_to_volume` for the exact V* guarantee.
Tier-3 generators return deliberately invalid meshes.

- [ ] **Step 1 (RED):** parametrized over the 12 valid shapes:

```python
import pytest, trimesh
from shape_library_3d import generators as G
from shape_library_3d.normalize import scale_to_volume
from shape_library_3d.constants import V_STAR_MM3

VALID = ["cube","cylinder","sphere","toroid","cone","pyramid","pipe",
         "lattice","l_extrusion","trunc_octahedron","icosphere_coarse","uv_sphere"]
EXPECT_GENUS = {"toroid":1, "pipe":1}   # others 0 (solids); lattice recorded, asserted >=1

@pytest.mark.parametrize("name", VALID)
def test_generator_watertight_and_Vstar(name):
    m = getattr(G, f"make_{name}")()
    assert m.is_watertight, f"{name} not watertight"
    assert m.is_winding_consistent, f"{name} winding inconsistent"
    scale_to_volume(m, V_STAR_MM3)
    assert m.volume == pytest.approx(V_STAR_MM3, rel=1e-4)

@pytest.mark.parametrize("name,g", list(EXPECT_GENUS.items()))
def test_expected_genus(name, g):
    m = getattr(G, f"make_{name}")()
    genus = (2 - int(m.euler_number)) // 2
    assert genus == g, f"{name} genus {genus} != {g}"

def test_lattice_is_multiply_connected():
    m = G.make_lattice()
    assert (2 - int(m.euler_number)) // 2 >= 1

def test_sphere_family_same_radius_after_Vstar():
    # 3/11/12 must be the same nominal sphere (radius within 1% after V* scaling)
    import numpy as np
    for nm in ("sphere","icosphere_coarse","uv_sphere"):
        m = getattr(G, f"make_{nm}")(); scale_to_volume(m, V_STAR_MM3)
        r = np.linalg.norm(m.vertices, axis=1).mean()
        assert r == pytest.approx(10.0, rel=0.02)   # ~20mm-dia sphere
```

- [ ] **Step 2:** Run → FAIL (generators missing).
- [ ] **Step 3 (GREEN):** implement `generators.py`. Key builders (grounded on the
      trimesh probe of 2026-08-01):

```python
"""Watertight trimesh generators for the 3-D primitive library.
Each Tier-1/2 maker returns a watertight mesh near V*; the registry finalizes to
V* by uniform scale (except `pipe`, which self-solves to V* at fixed od/H)."""
from __future__ import annotations
import numpy as np, trimesh
from shape_library_3d.constants import V_STAR_MM3

_SEC = 96   # facet sections for curved solids (keeps faceting error small)

def make_cube():        return trimesh.creation.box(extents=[1.0, 1.0, 1.0])
def make_cylinder():    return trimesh.creation.cylinder(radius=1.0, height=2.0, sections=_SEC)   # h=d
def make_sphere():      return trimesh.creation.icosphere(subdivisions=4, radius=1.0)             # #3 smooth
def make_icosphere_coarse(): return trimesh.creation.icosphere(subdivisions=3, radius=1.0)        # #11
def make_toroid():      return trimesh.creation.torus(major_radius=2.0, minor_radius=1.0,
                                                      major_sections=_SEC, minor_sections=_SEC//2) # R=2r
def make_cone():        return trimesh.creation.cone(radius=1.0, height=2.0, sections=_SEC)        # h=d

def make_pyramid():     # square base b, apex height h=b; explicit watertight solid
    b = 1.0; h = 1.0
    v = np.array([[-b/2,-b/2,0],[b/2,-b/2,0],[b/2,b/2,0],[-b/2,b/2,0],[0,0,h]], float)
    f = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4],[0,2,1],[0,3,2]])   # 4 sides + 2 base tris
    m = trimesh.Trimesh(vertices=v, faces=f); m.fix_normals(); return m

def make_uv_sphere():   # #12 pole-clustered; count chosen to match icosphere L3 (1280) +-10%
    target = len(make_icosphere_coarse().faces)
    best = min(range(8, 40), key=lambda k: abs(len(trimesh.creation.uv_sphere(
        radius=1.0, count=[k, k]).faces) - target))
    return trimesh.creation.uv_sphere(radius=1.0, count=[best, best])

def make_l_extrusion():   # L cross-section (outer square c, quadrant c/2 removed), extrude h=c
    c = 1.0
    v2 = np.array([(-c/2,-c/2),(c/2,-c/2),(c/2,0),(0,0),(0,c/2),(-c/2,c/2),(-c/2,0)], float)
    f2 = np.array([[0,1,3],[1,2,3],[0,3,6],[6,3,4],[6,4,5]])
    return trimesh.creation.extrude_triangulation(v2, f2, height=c)

def make_trunc_octahedron():
    import itertools
    pts = set()
    for p in itertools.permutations([0,1,2]):
        for sx in ([1] if p[0]==0 else [1,-1]):
            for sy in ([1] if p[1]==0 else [1,-1]):
                for sz in ([1] if p[2]==0 else [1,-1]):
                    pts.add((sx*p[0], sy*p[1], sz*p[2]))
    return trimesh.convex.convex_hull(np.array(sorted(pts), float))

def make_pipe(target_vol_mm3: float = V_STAR_MM3):
    # fixed od=24 (Ro=12), H=20; solve inner radius Ri on MEASURED mesh volume
    Ro, H = 12.0, 20.0
    lo, hi = 0.1, Ro - 0.05
    for _ in range(60):
        Ri = 0.5 * (lo + hi)
        m = trimesh.creation.annulus(r_min=Ri, r_max=Ro, height=H, sections=128)
        if m.volume > target_vol_mm3:   # too much material -> enlarge hole
            lo = Ri
        else:
            hi = Ri
    return trimesh.creation.annulus(r_min=0.5*(lo+hi), r_max=Ro, height=H, sections=128)

def make_lattice(cells: int = 2, strut_frac: float = 0.22, res: int = 96):
    # union of axis-aligned struts on a (cells+1)^3 node grid, via voxel mask ->
    # marching cubes (watertight by construction; boolean backend unavailable).
    nodes = np.linspace(-1.0, 1.0, cells + 1)
    pitch = nodes[1] - nodes[0]
    half = 0.5 * strut_frac * pitch
    g = np.linspace(-1.0, 1.0, res)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    mask = np.zeros(X.shape, bool)
    for a in nodes:
        for b in nodes:
            mask |= (np.abs(Y-a) <= half) & (np.abs(Z-b) <= half)   # x-struts
            mask |= (np.abs(X-a) <= half) & (np.abs(Z-b) <= half)   # y-struts
            mask |= (np.abs(X-a) <= half) & (np.abs(Y-b) <= half)   # z-struts
    pitch_mm = (g[1] - g[0])
    m = trimesh.voxel.ops.matrix_to_marching_cubes(mask, pitch=pitch_mm)
    m.apply_translation(-m.bounds.mean(axis=0))
    return m

# ---- Tier-3 invalid meshes (rejection tests; never normalized) ----
def make_open_cylinder():   # side wall only, no caps
    n, r, h = 48, 10.0, 20.0
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    bot = np.c_[r*np.cos(th), r*np.sin(th), np.full(n, -h/2)]
    top = np.c_[r*np.cos(th), r*np.sin(th), np.full(n,  h/2)]
    v = np.vstack([bot, top]); f = []
    for i in range(n):
        j = (i+1) % n; f += [[i, j, n+i], [j, n+j, n+i]]
    return trimesh.Trimesh(vertices=v, faces=np.array(f))

def make_flat_plane(k: int = 5, w: float = 20.0):
    xs = np.linspace(-w/2, w/2, k); ys = np.linspace(-w/2, w/2, k)
    v = np.array([[x, y, 0.0] for y in ys for x in xs]); f = []
    for r in range(k-1):
        for c in range(k-1):
            a = r*k + c; f += [[a, a+1, a+k], [a+1, a+k+1, a+k]]
    return trimesh.Trimesh(vertices=v, faces=np.array(f))
```

- [ ] **Step 4:** Run → PASS (fix any genus/watertight surprise on the lattice
      and uv_sphere by adjusting `res`/`strut_frac`/`sections` until green; record
      actual lattice genus — do not hard-code a guessed value).
- [ ] **Step 5:** Commit `generators.py test_generators.py`.

---

### Task 5: `registry.SHAPES` (curated metadata + generator binding)

**Files:** Create `shape_library_3d/registry.py`, `tests/test_registry.py`

- [ ] **Step 1 (RED):**

```python
from shape_library_3d.registry import SHAPES, ShapeSpec, iter_specs

def test_registry_has_14_entries_by_tier():
    tiers = [s.tier for s in SHAPES.values()]
    assert tiers.count(1) == 10 and tiers.count(2) == 2 and tiers.count(3) == 2

def test_every_spec_resolves_a_generator_and_curation():
    import shape_library_3d.generators as G
    for name, spec in SHAPES.items():
        assert hasattr(G, spec.generator), f"{name}: no generator {spec.generator}"
        assert spec.rf_characteristic and spec.numerical_characteristic
        assert spec.role in {"control", "stressor", "reject"}

def test_control_shapes_ordered_first():
    order = list(SHAPES)
    assert order[0] == "cube"   # flat orthogonal control leads (mirrors 2-D gate G3)
```

- [ ] **Step 2:** Run → FAIL. **Step 3 (GREEN):** frozen dataclass + ordered dict:

```python
"""Curated registry: name -> ShapeSpec (tier, role, curation strings, generator)."""
from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict

@dataclass(frozen=True)
class ShapeSpec:
    tier: int
    role: str                    # control | stressor | reject
    generator: str               # generators.<name>
    rf_characteristic: str
    numerical_characteristic: str
    target_volume_mm3: float | None   # V* for tiers 1-2; None for tier 3

_V = 4188.7902
SHAPES: "OrderedDict[str, ShapeSpec]" = OrderedDict([
    ("cube", ShapeSpec(1,"control","make_cube","flat-faced orthogonal baseline","axis-aligned flat facets",_V)),
    ("cylinder", ShapeSpec(1,"stressor","make_cylinder","curved wall meeting flat end caps, 90 deg junctions","mixed curved/flat facets",_V)),
    ("sphere", ShapeSpec(1,"control","make_sphere","smoothest control; the 2-D circle's 3-D analog","uniform fine icosphere",_V)),
    ("toroid", ShapeSpec(1,"stressor","make_toroid","genus-1 through-hole field shadowing, no flats","genus-1 all-curved surface",_V)),
    ("cone", ShapeSpec(1,"stressor","make_cone","smooth base to apex singularity, point field concentration","apex vertex singularity",_V)),
    ("pyramid", ShapeSpec(1,"stressor","make_pyramid","apex plus sharp edges and flat facets (isolates apex vs edges vs cone)","apex + edges, few flat facets",_V)),
    ("pipe", ShapeSpec(1,"stressor","make_pipe","wall-thickness vs interior field, thin-wall heating","genus-1 thin annular wall, flat caps",_V)),
    ("lattice", ShapeSpec(1,"stressor","make_lattice","intersecting struts, internal occlusion, junction hot spots","high-genus marching-cubes surface",_V)),
    ("l_extrusion", ShapeSpec(1,"stressor","make_l_extrusion","reentrant corner; continuity with the 2-D L outlier","reentrant concave corner",_V)),
    ("trunc_octahedron", ShapeSpec(1,"stressor","make_trunc_octahedron","oblique planar facets at mixed angles, cube-to-sphere","24 oblique planar facets",_V)),
    ("icosphere_coarse", ShapeSpec(2,"stressor","make_icosphere_coarse","same sphere as #3, coarser uniform mesh (isolates resolution)","uniform coarse icosphere",_V)),
    ("uv_sphere", ShapeSpec(2,"stressor","make_uv_sphere","same sphere, pole-clustered facets (isolates distribution)","UV pole facet singularity",_V)),
    ("open_cylinder", ShapeSpec(3,"reject","make_open_cylinder","NOT A PART: naked-edge open shell","non-watertight, open boundary",None)),
    ("flat_plane", ShapeSpec(3,"reject","make_flat_plane","NOT A PART: zero-volume sheet","planar, zero enclosed volume",None)),
])

def iter_specs():
    yield from SHAPES.items()

__all__ = ["SHAPES", "ShapeSpec", "iter_specs"]
```

- [ ] **Step 4:** Run → PASS. **Step 5:** Commit `registry.py test_registry.py`.

---

### Task 6: `voxelize.stl_to_mask` (heatr3d Grid bridge)

**Files:** Create `shape_library_3d/voxelize.py`, `tests/test_voxelize.py`

- [ ] **Step 1 (RED):**

```python
import numpy as np, trimesh, pytest
from shape_library_3d.voxelize import stl_to_mask
from shape_library_3d.validate import ZeroVolumeError
from shape_library_3d import generators as G
from shape_library_3d.normalize import scale_to_volume
from shape_library_3d.constants import V_STAR_MM3

class _Grid:   # minimal heatr3d.Grid stand-in (same centered-cell convention)
    def __init__(self, n, L=0.060):
        self.n = n; self.L = L; self.h = L/n
        c = (np.arange(n)+0.5)*self.h - L/2
        self.x = self.y = self.z = c

def test_cube_fill_fraction_matches_analytic():
    m = G.make_cube(); scale_to_volume(m, V_STAR_MM3)   # side 16.12 mm
    grid = _Grid(64)
    mask = stl_to_mask(m, grid)
    assert mask.shape == (64, 64, 64)
    frac = mask.mean()
    analytic = (16.120e-3 / grid.L) ** 3                # (side/L)^3
    assert frac == pytest.approx(analytic, rel=0.06)    # staircase tolerance

def test_tier3_mesh_rejected_before_voxelizing():
    with pytest.raises(ZeroVolumeError):
        stl_to_mask(G.make_flat_plane(), _Grid(32))
```

- [ ] **Step 2:** Run → FAIL. **Step 3 (GREEN):**

```python
"""Bridge a library STL (mm) to a heatr3d Grid (m) boolean part mask.
Validates first (Tier-3 rejection), then tests cell centers for containment."""
from __future__ import annotations
import numpy as np, trimesh
from shape_library_3d.validate import validate_part_mesh

def stl_to_mask(mesh: trimesh.Trimesh, grid) -> np.ndarray:
    validate_part_mesh(mesh)                       # loud reject for non-parts
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    pts_m = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pts_mm = pts_m * 1000.0                         # grid is metres, mesh is mm
    inside = mesh.contains(pts_mm)
    return inside.reshape(X.shape)

def voxel_volume_report(mask: np.ndarray, grid, target_mm3: float) -> dict:
    vox_mm3 = float(mask.sum()) * (grid.h * 1000.0) ** 3
    return {"voxel_volume_mm3": vox_mm3, "voxel_vs_Vstar_frac": vox_mm3 / target_mm3 - 1.0}

__all__ = ["stl_to_mask", "voxel_volume_report"]
```

- [ ] **Step 4:** Run → PASS. **Step 5:** Commit `voxelize.py test_voxelize.py`.

---

### Task 7: `build_library.py` CLI + public API + committed artifacts

**Files:** Create `shape_library_3d/build_library.py`; fill `shape_library_3d/__init__.py`;
create `tests/test_build_library.py`. Generates `stl/*.stl` + `meta/*.json` +
`meta/library_manifest.json`.

- [ ] **Step 1 (RED):**

```python
import json, subprocess, sys
from pathlib import Path
from shape_library_3d import iter_parts, load_part_stl
from shape_library_3d.constants import V_STAR_MM3

ROOT = Path(__file__).resolve().parents[1]

def test_build_writes_all_stls_and_manifest(tmp_path):
    from shape_library_3d.build_library import build
    manifest = build(out_dir=tmp_path)
    assert len(manifest["shapes"]) == 14
    for entry in manifest["shapes"]:
        if entry["tier"] in (1, 2):
            assert entry["is_watertight"] and entry["volume_err_frac"] < 1e-4
            assert (tmp_path / "stl" / f"{entry['name']}.stl").exists()

def test_public_api_enumerates_parts():
    names = [n for n, _ in iter_parts()]      # tiers 1-2 only (loadable parts)
    assert "cube" in names and "open_cylinder" not in names
```

- [ ] **Step 2:** Run → FAIL. **Step 3 (GREEN):** `build_library.build(out_dir)` iterates
      `SHAPES`, calls the generator, for tiers 1-2 runs `scale_to_volume` +
      `validate_part_mesh` + asserts watertight, writes STL, and records MEASURED
      metadata (volume, euler, genus, facet-area stats, bbox, realized gap,
      sha256); tier-3 records the expected-rejection reason and writes the STL for
      the rejection tests but marks `role="reject"`. `__init__.py` exposes
      `iter_parts()` (tiers 1-2), `load_part_stl(name)`, `SHAPES`. (Full code
      written at implementation time; metadata schema per spec §3.)
- [ ] **Step 4:** Run → PASS. Then run `build_library` for real into
      `shape_library_3d/stl` + `meta`. **Step 5:** Commit code, tests, `stl/`, `meta/`.

---

### Task 8: README

**Files:** Create `shape_library_3d/README.md`

- [ ] **Step 1:** Write README: purpose, the 14-shape table (tier / RF characteristic
      / genus / V*), equal-volume convention + V* rationale, the two ingestion
      contracts (heatr3d voxel mask; solve3d STL), the Tier-3 rejection contract,
      the Grade-and-Print public-API handoff (`iter_parts`, `load_part_stl`,
      `library_manifest.json`), env (`.venv312`), and how to regenerate
      (`python -m shape_library_3d.build_library`). No TDD (doc).
- [ ] **Step 2:** Commit `README.md`.

---

### Task 9: Densification smoke campaign (cube / sphere / cone at n=64) + view figures

**Files:** Create `shape_library_3d/smoke_campaign.py`

- [ ] **Step 1:** Inspect the heatr3d run+render entry points actually used by the
      standard (`heatr3d_job.py::_render_summary_plots`, `heatr3d.run`, the
      exposure-time optimizer, `Params`) and wire `smoke_campaign.py` to: for each
      of cube/sphere/cone — load STL, `stl_to_mask(Grid(n=64))`, run heatr3d with
      the standard Params + exposure optimizer to phi_bar=0.90, check the 250 C
      ceiling margin, print `|energy residual|/dose`, render the standard
      densification figure set into `shape_library_3d/smoke_out/<shape>/`.
- [ ] **Step 2:** Run the smoke campaign for real (n=64; within the n<=96 ceiling).
      Capture the energy-residual gate and t90 per shape.
- [ ] **Step 3 (verification gate, non-unit):** **Read the rendered PNGs myself**
      (melt_progression, ortho_slices, melt_vs_cad, density_hist, radial_density)
      for all three shapes before delivering — per the view-figure-renders-personally
      memory (F7-sliver lesson). Confirm each part built the intended shape and the
      energy gate is clean.
- [ ] **Step 4:** Commit `smoke_campaign.py` + `smoke_out/` (figures + per-run JSON).
      Do NOT run the full 14-shape campaign (its own plan).

---

## Self-Review (against the spec)

- **Coverage:** list (Task 4/5), sizing/equal-volume (Task 2, registry V*), genus/
  Euler (Task 4), Tier-3 rejection (Task 3), heatr3d voxel bridge (Task 6),
  solve3d STL (artifacts in Task 7), metadata schema (Task 7), Grade-and-Print API
  (Task 7), README (Task 8), smoke + view figures (Task 9). All spec §2-§9 mapped.
- **Placeholders:** Task 7/9 defer full CLI/render wiring to implementation time
  by DESIGN (they depend on measured facet stats and the exact heatr3d render
  signature, read at that step) — the interfaces, schema, and gates are fully
  specified; this is not a hidden TODO.
- **Type consistency:** generator names `make_<name>` match `ShapeSpec.generator`
  and the registry keys; `scale_to_volume`, `validate_part_mesh`, `stl_to_mask`
  signatures consistent across tasks.
