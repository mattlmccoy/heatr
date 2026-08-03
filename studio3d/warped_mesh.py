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


SINTER_PHI = 0.5      # the solver's own sinter threshold (H.sinter_metrics)


def _boxes_mesh(centers_mm: np.ndarray, sx_mm: np.ndarray,
                sz_mm: np.ndarray) -> trimesh.Trimesh:
    """One axis-aligned box per voxel; scales per voxel (sx, sx, sz)."""
    if len(centers_mm) == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)),
                               faces=np.zeros((0, 3), int), process=False)
    unit = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    v0 = np.asarray(unit.vertices)
    f0 = np.asarray(unit.faces)
    scale = np.stack([sx_mm, sx_mm, sz_mm], axis=1)
    verts = (v0[None, :, :] * scale[:, None, :]
             + centers_mm[:, None, :]).reshape(-1, 3)
    faces = (f0[None, :, :]
             + (np.arange(len(centers_mm)) * len(v0))[:, None, None]
             ).reshape(-1, 3)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def build_densified_meshes(part: np.ndarray, rho_final: np.ndarray,
                           phi_final: np.ndarray, p, grid):
    """Voxel-conserving densified form: (solid_mesh, powder_mesh, info).

    Only voxels past the solver's sinter threshold form the part; each
    shrinks by its local law and SETTLES per column (nothing floats over
    removed powder). Unsintered in-part voxels are LOOSE POWDER, returned
    as a separate mesh at their nominal positions. Same shrink law as
    heatr3d's shrinkage_analysis; classification per H.sinter_metrics.
    """
    part = np.asarray(part, bool)
    phi = np.asarray(phi_final, float)
    sintered = part & (phi >= SINTER_PHI)
    unsintered = part & ~sintered
    lam_xy, lam_z = J._shrink_factor_fields(rho_final, part, p)
    h, L = grid.h, grid.L

    idx = np.indices(part.shape).astype(float)
    xc = (idx[0] + 0.5) * h - L / 2.0
    yc = (idx[1] + 0.5) * h - L / 2.0
    zc = (idx[2] + 0.5) * h - L / 2.0
    cx = float(xc[part].mean())
    cy = float(yc[part].mean())
    wx = cx + (xc - cx) * lam_xy
    wy = cy + (yc - cy) * lam_xy
    # settle: per column, stack SINTERED voxels only, anchored at the
    # column's first part voxel's nominal bottom face
    hz = np.where(sintered, h * lam_z, 0.0)
    cfb = np.cumsum(hz, axis=2) - 0.5 * hz
    first = np.argmax(part, axis=2)
    anchor = first * h - L / 2.0
    wz = anchor[:, :, None] + cfb

    si = np.argwhere(sintered)
    sx, sy, sz = si[:, 0], si[:, 1], si[:, 2]
    solid = _boxes_mesh(
        np.stack([wx[sx, sy, sz], wy[sx, sy, sz], wz[sx, sy, sz]],
                 axis=1) * 1e3,
        lam_xy[sx, sy, sz] * h * 1e3, lam_z[sx, sy, sz] * h * 1e3)

    ui = np.argwhere(unsintered)
    ux, uy, uz = ui[:, 0], ui[:, 1], ui[:, 2]
    powder = _boxes_mesh(
        np.stack([xc[ux, uy, uz], yc[ux, uy, uz], zc[ux, uy, uz]],
                 axis=1) * 1e3,
        np.full(len(ui), h * 1e3), np.full(len(ui), h * 1e3))

    # rest the assembly on the build plate: the viewer's frame is plate
    # z = 0, matching every other mesh it shows
    if len(solid.vertices):
        shift = -float(solid.bounds[0][2])
        solid.apply_translation((0.0, 0.0, shift))
        if len(powder.vertices):
            powder.apply_translation((0.0, 0.0, shift))

    info = {"n_sintered": int(sintered.sum()),
            "n_unsintered": int(unsintered.sum()),
            "sinter_phi_threshold": SINTER_PHI,
            "solid_top_mm": (float(solid.bounds[1][2])
                             if len(solid.vertices) else None),
            "statement": ("sintered material only, settled per column by "
                          "the solver's shrink law; unsintered in-part "
                          "voxels are loose powder, shown separately, "
                          "never as solid part")}
    return solid, powder, info


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
