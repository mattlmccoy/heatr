"""Densified-form v2 (Matt 2026-08-03): voxel-conserving, sinter-aware,
BED-SUSPENDED.

Rules:
- only voxels past the solver's sinter threshold form the part mesh;
- each sintered voxel shrinks by its local law and settles within its
  column (no voxel floats over removed powder);
- the part is SUSPENDED IN THE POWDER BED, never anchored to a plate:
  contraction is about the sintered material's own z-centroid, so the
  bottom rises and the top drops - no artificial flat bottom;
- unsintered voxels are returned separately as loose powder;
- the display frame maps the NOMINAL part bottom to z = 0, so the
  suspended lift is visible against the imported reference.
"""
from __future__ import annotations

import numpy as np

import heatr3d as H
from studio3d.warped_mesh import build_densified_meshes


def _column_part(n: int = 16) -> np.ndarray:
    part = np.zeros((n, n, n), bool)
    part[7:9, 7:9, 4:12] = True          # a 2x2 column, 8 voxels tall
    return part


def test_suspended_contraction_preserves_the_center_not_the_bottom():
    grid = H.Grid(n=16)
    p = H.Params()
    part = _column_part()
    phi = np.where(part, 1.0, 0.0)
    rho = np.where(part, 1.0, 0.0)
    solid, powder, info = build_densified_meshes(part, rho, phi, p, grid)
    h_mm = grid.h * 1e3
    # frame: nominal part bottom face at z = 0; nominal center at 4 voxels
    nominal_center = 4 * h_mm
    center = (solid.bounds[0][2] + solid.bounds[1][2]) / 2
    assert abs(center - nominal_center) < 0.6 * h_mm
    # bed-suspended: the bottom LIFTS off the nominal bottom face
    assert solid.bounds[0][2] > 0.5
    lam_z = H.shrinkage_factors(np.array([p.rho_rel]))[1][0]
    ext = solid.bounds[1][2] - solid.bounds[0][2]
    assert abs(ext - 8 * h_mm * lam_z) < 0.6 * h_mm
    assert info["n_unsintered"] == 0
    assert len(powder.vertices) == 0


def test_unsintered_voxels_are_powder_not_part():
    grid = H.Grid(n=16)
    p = H.Params()
    part = _column_part()
    phi = np.where(part, 1.0, 0.0)
    phi[:, :, 8:] = np.where(part[:, :, 8:], 0.2, 0.0)   # top half unsintered
    rho = np.where(phi >= 0.5, 1.0, p.rho_rel) * part
    solid, powder, info = build_densified_meshes(part, rho, phi, p, grid)
    h_mm = grid.h * 1e3
    assert info["n_sintered"] == 2 * 2 * 4
    assert info["n_unsintered"] == 2 * 2 * 4
    # the solid contracts about the SINTERED material's own centroid
    # (bottom 4 voxels, nominal centers 0.5h..3.5h -> mean 2h)
    center = (solid.bounds[0][2] + solid.bounds[1][2]) / 2
    assert abs(center - 2 * h_mm) < 0.6 * h_mm
    lam_z = H.shrinkage_factors(np.array([p.rho_rel]))[1][0]
    ext = solid.bounds[1][2] - solid.bounds[0][2]
    assert abs(ext - 4 * h_mm * lam_z) < 0.6 * h_mm
    assert len(powder.vertices) > 0


def test_settling_leaves_no_floating_voxels():
    grid = H.Grid(n=16)
    p = H.Params()
    part = _column_part()
    phi = np.where(part, 1.0, 0.0)
    phi[:, :, 7:9] = np.where(part[:, :, 7:9], 0.2, 0.0)  # a mid-gap of powder
    rho = np.where(phi >= 0.5, 1.0, p.rho_rel) * part
    solid, powder, info = build_densified_meshes(part, rho, phi, p, grid)
    # 6 sintered voxels settle into one contiguous stack
    lam_z = H.shrinkage_factors(np.array([p.rho_rel]))[1][0]
    ext = solid.bounds[1][2] - solid.bounds[0][2]
    assert abs(ext - 6 * grid.h * 1e3 * lam_z) < grid.h * 1e3 * 0.6
