"""Uniform-scale a mesh to a target solid volume.

Solid volume scales as s**3 under a uniform scale, so a single closed-form step
hits the target to floating-point precision -- no iteration needed. This is the
equal-volume convention (mesh-independent, operating on the exact trimesh volume,
never a voxel raster).
"""
from __future__ import annotations

from typing import Tuple

import trimesh


def scale_to_volume(mesh: trimesh.Trimesh, target_vol_mm3: float) -> Tuple[trimesh.Trimesh, float]:
    """Uniform-scale ``mesh`` in place so its solid volume equals ``target_vol_mm3``.

    Returns the (mutated) mesh and the scale factor applied.

    Raises:
        ValueError: if the mesh has non-positive volume (open or inverted mesh);
            such meshes are not valid solids to normalize.
    """
    v = float(mesh.volume)
    if v <= 0.0:
        raise ValueError(
            f"cannot normalize a mesh with non-positive volume ({v} mm^3); "
            f"the mesh is not a valid closed solid")
    s = (target_vol_mm3 / v) ** (1.0 / 3.0)
    mesh.apply_scale(s)
    return mesh, s


__all__ = ["scale_to_volume"]
