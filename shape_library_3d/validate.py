"""Part-mesh validation: the ingestion gate that rejects Tier-3 non-parts loudly.

heatr3d/solve3d must never silently accept a non-watertight shell or a
zero-volume sheet as a printable part. `validate_part_mesh` is called first by
the voxel bridge (and any "load as part" path) so shapes 13/14 fail with an
informative, typed error instead of producing a corrupt mask.

Rejection order is grounded on measured trimesh behavior (2026-08-01): an open
side-wall reports a *nonzero* signed volume (6265 mm^3 for the naked cylinder),
so a volume-first check would mislabel it. Planarity is therefore checked first
(catches the flat sheet), then watertightness (catches the open shell), then a
final zero-volume guard.
"""
from __future__ import annotations

import numpy as np
import trimesh

from shape_library_3d.constants import EXTENT_EPS_MM, VOL_EPS_MM3


class InvalidPartGeometryError(ValueError):
    """Base for a mesh that cannot serve as a printable solid part."""


class NonWatertightMeshError(InvalidPartGeometryError):
    """Mesh has open boundary (naked) edges; not a closed solid."""


class ZeroVolumeError(InvalidPartGeometryError):
    """Mesh is planar/degenerate and encloses no volume."""


def _open_edge_count(mesh: trimesh.Trimesh) -> int:
    """Number of edges belonging to exactly one face (open boundary edges)."""
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    return int((counts == 1).sum())


def validate_part_mesh(mesh: trimesh.Trimesh) -> None:
    """Raise if ``mesh`` is not a valid watertight solid part.

    Raises:
        ZeroVolumeError: degenerate/planar mesh (a bbox axis is ~zero) or a
            watertight mesh enclosing ~zero volume.
        NonWatertightMeshError: mesh has open boundary edges.
    """
    ext = np.asarray(mesh.extents, dtype=float)
    if ext.size < 3 or float(ext.min()) <= EXTENT_EPS_MM:
        raise ZeroVolumeError(
            f"degenerate/planar mesh: bbox extents {np.round(ext, 4)} have a "
            f"~zero axis; the mesh encloses no volume and is not a printable part")
    if not mesh.is_watertight:
        raise NonWatertightMeshError(
            f"mesh is not watertight: {_open_edge_count(mesh)} naked (open) "
            f"boundary edges; it cannot be voxelized or meshed as a solid part")
    v = abs(float(mesh.volume))
    if v <= VOL_EPS_MM3:
        raise ZeroVolumeError(f"mesh encloses ~zero volume ({v} mm^3)")


__all__ = [
    "validate_part_mesh",
    "InvalidPartGeometryError",
    "NonWatertightMeshError",
    "ZeroVolumeError",
]
