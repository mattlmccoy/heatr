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

# End-state classification for DENSIFY runs (Tamper incident 2026-08-05):
# heatr3d's phi_final is a MELT-ONSET snapshot (heatr3d.py:1275, taken at t90),
# so for densify runs it describes a mid-run state - on the real Tamper job it
# rendered ~1000 rim voxels as loose powder whose end-state rho_final (0.83
# mean) proves they melted and were consolidating. rho_final IS the end state
# (rho_rel at the last step), so solid-vs-powder comes from it alone:
#   fused        rho >= FUSED_RHO (above the 0.55 powder-bed initial; any
#                consolidation implies the voxel melted)
#   consolidated rho >= CONSOLIDATED_RHO (essentially finished densifying)
# Fused-but-under-consolidated voxels are SOLID (porous material attached to
# the part, physically nothing like loose powder) and their count is reported
# loudly rather than hidden.
FUSED_RHO = 0.60
CONSOLIDATED_RHO = 0.90


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
    rho = np.asarray(rho_final, float)
    # END-STATE classification (see FUSED_RHO above): phi_final is a stale
    # melt-onset snapshot for densify runs and must not decide geometry.
    sintered = part & (rho >= FUSED_RHO)
    unsintered = part & ~sintered
    under_consolidated = part & (rho >= FUSED_RHO) & (rho < CONSOLIDATED_RHO)
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
    # BED-SUSPENDED z (Matt 2026-08-03): the part is never anchored to a
    # substrate; it sinters suspended in the nylon powder bed, so
    # contraction is about the sintered material's own z-centroid. Per
    # column: settle sintered voxels into a contiguous stack (internal
    # offsets from the local shrink law), then place the stack so its
    # column center maps to the column's nominal sintered center shrunk
    # toward the global sintered z-centroid. Same shrink-law magnitudes;
    # only the anchoring gauge changes.
    hz = np.where(sintered, h * lam_z, 0.0)
    cfb = np.cumsum(hz, axis=2) - 0.5 * hz          # offsets within stack
    col_h = hz.sum(axis=2)                          # compacted stack height
    ns = sintered.sum(axis=2)
    with np.errstate(invalid="ignore"):
        col_znom = np.where(ns > 0,
                            (zc * sintered).sum(axis=2) / np.maximum(ns, 1),
                            0.0)                    # nominal sintered center
        col_lam = np.where(ns > 0,
                           (lam_z * sintered).sum(axis=2)
                           / np.maximum(ns, 1), 1.0)
    cz = float(zc[sintered].mean()) if sintered.any() else 0.0
    col_center = cz + (col_znom - cz) * col_lam     # suspended contraction
    stack_bottom = col_center - col_h / 2.0
    wz = stack_bottom[:, :, None] + cfb

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

    # display frame: the NOMINAL part bottom face maps to z = 0 (the same
    # frame the imported view uses), so the suspended lift of the bottom
    # is visible against the imported reference; no plate anchoring
    if part.any():
        zs = np.where(part.any(axis=(0, 1)))[0]
        nominal_bottom = zs[0] * h - L / 2.0
        shift = -nominal_bottom * 1e3
        for mesh_ in (solid, powder):
            if len(mesh_.vertices):
                mesh_.apply_translation((0.0, 0.0, shift))

    info = {"n_solid": int(sintered.sum()),
            "n_powder": int(unsintered.sum()),
            "n_under_consolidated": int(under_consolidated.sum()),
            "min_rho_solid": (float(rho[sintered].min())
                              if sintered.any() else None),
            "classification": "end-state rho_final",
            "fused_rho_threshold": FUSED_RHO,
            "consolidated_rho_threshold": CONSOLIDATED_RHO,
            # legacy keys, kept so existing consumers keep working
            "n_sintered": int(sintered.sum()),
            "n_unsintered": int(unsintered.sum()),
            "solid_top_mm": (float(solid.bounds[1][2])
                             if len(solid.vertices) else None),
            "statement": ("fused material only (end-state rho_final >= "
                          f"{FUSED_RHO}, NOT the stale melt-onset phi "
                          "snapshot), bed-suspended (no plate anchor): "
                          "contraction about the fused mass's own center per "
                          "the solver's shrink law, columns settled; never-"
                          "fused in-part voxels are loose powder, shown "
                          "separately; fused-but-under-consolidated voxels "
                          "(rho < "
                          f"{CONSOLIDATED_RHO}) are counted, not hidden")}
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
