"""Densified-form v2 (Matt 2026-08-03): voxel-conserving, sinter-aware.

The old view rendered UNSINTERED voxels as solid part, so an undensified
region looked like a tall spike instead of what it is: loose powder. Rules:
- only voxels past the solver's sinter threshold form the part mesh;
- each sintered voxel shrinks by its local law and SETTLES (no voxel
  floats above removed powder);
- unsintered voxels are returned separately as loose powder.
"""
from __future__ import annotations

import numpy as np

import heatr3d as H
from studio3d.warped_mesh import build_densified_meshes


def _column_part(n: int = 16) -> np.ndarray:
    part = np.zeros((n, n, n), bool)
    part[7:9, 7:9, 4:12] = True          # a 2x2 column, 8 voxels tall
    return part


def test_unsintered_voxels_are_powder_not_part():
    grid = H.Grid(n=16)
    p = H.Params()
    part = _column_part()
    phi = np.where(part, 1.0, 0.0)
    phi[:, :, 8:] = np.where(part[:, :, 8:], 0.2, 0.0)   # top half unsintered
    rho = np.where(phi >= 0.5, 1.0, p.rho_rel) * part
    solid, powder, info = build_densified_meshes(part, rho, phi, p, grid)
    assert info["n_sintered"] == 2 * 2 * 4
    assert info["n_unsintered"] == 2 * 2 * 4
    # the solid rests on the plate (z = 0) and must NOT reach the
    # unsintered region's nominal height: dense voxels compact, powder is
    # excluded, nothing rides on top
    assert abs(solid.bounds[0][2]) < 1e-6
    lam_z_dense = H.shrinkage_factors(np.array([p.rho_rel]))[1][0]
    assert solid.bounds[1][2] < 8 * grid.h * 1e3   # far below nominal stack
    assert abs((solid.bounds[1][2] - solid.bounds[0][2])
               - 4 * grid.h * 1e3 * lam_z_dense) < grid.h * 1e3 * 0.6
    assert len(powder.vertices) > 0


def test_settling_leaves_no_floating_voxels():
    """A sintered voxel above an unsintered gap settles down onto the
    sintered stack below the gap."""
    grid = H.Grid(n=16)
    p = H.Params()
    part = _column_part()
    phi = np.where(part, 1.0, 0.0)
    phi[:, :, 7:9] = np.where(part[:, :, 7:9], 0.2, 0.0)  # a mid-gap of powder
    rho = np.where(phi >= 0.5, 1.0, p.rho_rel) * part
    solid, powder, info = build_densified_meshes(part, rho, phi, p, grid)
    # solid extent in z ~ contiguous stack of the 6 sintered voxels' shrunk
    # heights: no gap-sized hole, top well below the nominal stack top
    ext_z = solid.bounds[1][2] - solid.bounds[0][2]
    lam_z_full = H.shrinkage_factors(np.array([p.rho_rel]))[1][0]
    expected = 6 * grid.h * 1e3 * lam_z_full
    assert abs(ext_z - expected) < grid.h * 1e3 * 0.6


def test_fully_sintered_part_keeps_one_solid_body():
    grid = H.Grid(n=16)
    p = H.Params()
    part = _column_part()
    phi = np.where(part, 1.0, 0.0)
    rho = np.where(part, 1.0, 0.0)
    solid, powder, info = build_densified_meshes(part, rho, phi, p, grid)
    assert info["n_unsintered"] == 0
    assert len(powder.vertices) == 0
