"""Curated registry: name -> ShapeSpec (tier, role, curation strings, generator).

The `rf_characteristic` / `numerical_characteristic` strings are what make this a
*curated* library rather than a shape dump: each states the single RF-heating or
numerical behavior that shape is chosen to stress. Order is deliberate -- the
flat orthogonal control (`cube`) leads, mirroring the 2-D campaign's control-first
gate.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from shape_library_3d.constants import V_STAR_MM3


@dataclass(frozen=True)
class ShapeSpec:
    tier: int                          # 1 RF-physics, 2 mesh-sensitivity, 3 rejection
    role: str                          # control | stressor | reject
    generator: str                     # generators.<name>
    rf_characteristic: str             # the RF-heating behavior stressed
    numerical_characteristic: str      # the mesh/solver behavior stressed
    target_volume_mm3: Optional[float]  # V* for tiers 1-2; None for tier 3


_V = V_STAR_MM3

SHAPES: "OrderedDict[str, ShapeSpec]" = OrderedDict([
    ("cube", ShapeSpec(
        1, "control", "make_cube",
        "flat-faced orthogonal baseline control",
        "axis-aligned flat facets", _V)),
    ("cylinder", ShapeSpec(
        1, "stressor", "make_cylinder",
        "curved wall meeting flat end caps (90-degree curved-to-flat junctions)",
        "mixed curved/flat facets", _V)),
    ("sphere", ShapeSpec(
        1, "control", "make_sphere",
        "smoothest control; the 2-D circle's true 3-D analog",
        "uniform fine icosphere", _V)),
    ("toroid", ShapeSpec(
        1, "stressor", "make_toroid",
        "genus-1 through-hole: field shadowing/shielding inside the hole, no flat faces",
        "genus-1 all-curved surface", _V)),
    ("cone", ShapeSpec(
        1, "stressor", "make_cone",
        "smooth base converging to an apex singularity (field concentration at a point)",
        "apex vertex singularity", _V)),
    ("pyramid", ShapeSpec(
        1, "stressor", "make_pyramid",
        "apex singularity PLUS sharp edges and flat facets (separates apex from edge effects vs the cone)",
        "apex vertex + sharp edges, few flat facets", _V)),
    ("pipe", ShapeSpec(
        1, "stressor", "make_pipe",
        "open-ended hollow wall: wall-thickness vs interior-field, the thin-wall heating question",
        "genus-1 thin annular wall with flat caps", _V)),
    ("lattice", ShapeSpec(
        1, "stressor", "make_lattice",
        "intersecting trusses with internal occlusion; strut junctions are known hot spots",
        "high-genus fused strut network", _V)),
    ("l_extrusion", ShapeSpec(
        1, "stressor", "make_l_extrusion",
        "reentrant corner in 3-D; continuity with the 2-D L-shape outlier",
        "reentrant concave corner", _V)),
    ("trunc_octahedron", ShapeSpec(
        1, "stressor", "make_trunc_octahedron",
        "many oblique planar facets at mixed angles, between cube and sphere",
        "24 oblique planar facets", _V)),
    ("icosphere_coarse", ShapeSpec(
        2, "stressor", "make_icosphere_coarse",
        "same sphere as the control, coarser uniform mesh (isolates resolution sensitivity)",
        "uniform coarse icosphere", _V)),
    ("uv_sphere", ShapeSpec(
        2, "stressor", "make_uv_sphere",
        "same sphere, facet clusters piled at the poles (isolates facet-distribution sensitivity)",
        "UV pole facet singularity", _V)),
    ("open_cylinder", ShapeSpec(
        3, "reject", "make_open_cylinder",
        "NOT A PART: naked-edge open shell (ingestion must reject as non-watertight)",
        "non-watertight, open boundary edges", None)),
    ("flat_plane", ShapeSpec(
        3, "reject", "make_flat_plane",
        "NOT A PART: zero-volume 2-D sheet (ingestion must reject as zero volume)",
        "planar, zero enclosed volume", None)),
])


def iter_specs():
    """Yield (name, ShapeSpec) for every registered shape, in registry order."""
    yield from SHAPES.items()


__all__ = ["SHAPES", "ShapeSpec", "iter_specs"]
