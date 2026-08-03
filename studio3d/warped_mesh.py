"""The densified part as a 3-D surface mesh, for the Studio viewport.

Reuses heatr3d_job's shrink machinery verbatim (_shrink_factor_fields,
_warped_centers): per-voxel anisotropic shrink factors from the final
density via the solver's own shrink law, warped centers with z compacting
from the build plate. This module only turns those warped voxels into a
renderable surface: one box per surface voxel at its warped center, scaled
by its local shrink factors. Display geometry, not new physics; the
authoritative record stays warped_geometry.json.
"""
from __future__ import annotations

import numpy as np
import trimesh

import heatr3d_job as J


def build_warped_mesh(part: np.ndarray, rho_final: np.ndarray, p,
                      grid) -> trimesh.Trimesh:
    """Surface mesh of the densified (warped) part, coordinates in mm."""
    part = np.asarray(part, bool)
    lam_xy, lam_z = J._shrink_factor_fields(rho_final, part, p)
    wx, wy, wz = J._warped_centers(part, lam_xy, lam_z, grid.h, grid.L)
    surf = J._surface_voxels(part)
    ix, iy, iz = surf[:, 0], surf[:, 1], surf[:, 2]
    centers = np.stack([wx[ix, iy, iz], wy[ix, iy, iz], wz[ix, iy, iz]],
                       axis=1) * 1e3                       # mm
    sx = lam_xy[ix, iy, iz] * grid.h * 1e3
    sz = lam_z[ix, iy, iz] * grid.h * 1e3

    unit = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    v0 = np.asarray(unit.vertices)                          # (8, 3)
    f0 = np.asarray(unit.faces)                             # (12, 3)
    n = len(centers)
    scale = np.stack([sx, sx, sz], axis=1)                  # (n, 3)
    verts = (v0[None, :, :] * scale[:, None, :]
             + centers[:, None, :]).reshape(-1, 3)
    faces = (f0[None, :, :] + (np.arange(n) * len(v0))[:, None, None]
             ).reshape(-1, 3)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
