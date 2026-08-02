"""Bridge a library STL (millimetres) to a heatr3d Grid (metres) boolean part mask.

heatr3d ingests geometry as a boolean voxel array on its structured `Grid`
(`heatr3d.run(grid, part, p)`), matching the `make_geometry` convention
(X,Y,Z meshgrid, indexing="ij", centered cells). This module tests each cell
center for containment in the mesh. Tier-3 non-parts are rejected first, so a
naked shell or a flat sheet can never produce a silently-wrong mask.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import trimesh

from shape_library_3d.validate import validate_part_mesh


def stl_to_mask(mesh: trimesh.Trimesh, grid) -> np.ndarray:
    """Return an ``(grid.n, grid.n, grid.n)`` boolean part mask, centered in the domain.

    ``grid`` is any object exposing ``n``, ``h`` and centered coordinate arrays
    ``x``, ``y``, ``z`` in metres (heatr3d.Grid or a compatible stand-in). The
    mesh is interpreted in millimetres and converted to the grid's metres.

    Raises the Tier-3 rejection errors from ``validate_part_mesh`` before any
    voxelization work.
    """
    validate_part_mesh(mesh)
    xx, yy, zz = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    pts_mm = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]) * 1000.0
    inside = mesh.contains(pts_mm)
    return inside.reshape(xx.shape)


def voxel_volume_report(mask: np.ndarray, grid, target_mm3: float) -> Dict[str, float]:
    """Report the voxelized volume and its staircase deviation from V*.

    The canonical STL volume is exactly V*; the voxelized volume differs by a
    grid-dependent staircase error that shrinks with grid.n. This is reported,
    never corrected away.
    """
    h_mm = grid.h * 1000.0
    vox_mm3 = float(mask.sum()) * h_mm ** 3
    return {
        "voxel_volume_mm3": vox_mm3,
        "voxel_vs_Vstar_frac": vox_mm3 / target_mm3 - 1.0,
    }


__all__ = ["stl_to_mask", "voxel_volume_report"]
