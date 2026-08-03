"""Densified-form 3-D mesh (spec 7d follow-on, Matt 2026-08-03): the warped
part surface as a loadable mesh, built from heatr3d_job's own shrink fields
(no new physics)."""
from __future__ import annotations

import numpy as np
import trimesh

import heatr3d as H
from studio3d.warped_mesh import build_warped_mesh


def _box_part(n: int = 16, w: int = 6) -> np.ndarray:
    part = np.zeros((n, n, n), bool)
    lo = (n - w) // 2
    part[lo:lo + w, lo:lo + w, lo:lo + w] = True
    return part


def test_no_densification_keeps_nominal_extent():
    grid = H.Grid(n=16)
    p = H.Params()
    part = _box_part()
    rho = np.where(part, p.rho_rel, 0.0)      # still at powder density
    mesh = build_warped_mesh(part, rho, p, grid)
    assert isinstance(mesh, trimesh.Trimesh)
    ext_mm = mesh.extents * 1e3 if mesh.units == "m" else mesh.extents
    nominal_mm = 6 * grid.h * 1e3
    assert np.allclose(ext_mm, nominal_mm, rtol=0.05)


def test_full_densification_shrinks_the_part():
    grid = H.Grid(n=16)
    p = H.Params()
    part = _box_part()
    rho_full = np.where(part, 1.0, 0.0)
    m_full = build_warped_mesh(part, rho_full, p, grid)
    m_none = build_warped_mesh(part, np.where(part, p.rho_rel, 0.0), p, grid)
    # z compacts hardest (anisotropic shrink law); every extent shrinks or holds
    assert m_full.extents[2] < m_none.extents[2] * 0.9
    assert (m_full.extents <= m_none.extents * 1.001).all()
