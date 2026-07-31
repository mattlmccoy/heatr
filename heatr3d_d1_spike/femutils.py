"""Small dolfinx utilities shared by the D1 Task 2 / Task 3 scripts:
DG0 projection of an elementwise quantity, and point evaluation on the
heatr3d voxel-centre grid (so both engines are read at the SAME points).
"""
from __future__ import annotations

import resource
import sys

import numpy as np
import ufl
from dolfinx import geometry

import eqs_common as ec           # applies jit_fix before dolfinx


def peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(ru) / 1024.0 ** 3 if sys.platform == "darwin" else float(ru) / 1024.0 ** 2


def dg0_project(expr, Q):
    """L2 projection onto DG0 = exact cell average. For P1 fields any function
    of grad(V) is already cell-wise constant, so this is exact, not a smoothing."""
    p_, w_ = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    a = ufl.inner(p_, w_) * ufl.dx
    L = ufl.inner(expr, w_) * ufl.dx
    return ec._solve_linear(a, L, [], {"ksp_type": "preonly", "pc_type": "jacobi"})


def emag_dg0(Vr, Vi, Q):
    """|E| = sqrt(|grad Vr|^2 + |grad Vi|^2) as a DG0 function (heatr3d
    convention 3: E = -grad V, sign irrelevant to the magnitude)."""
    e2 = (ufl.inner(ufl.grad(Vr), ufl.grad(Vr))
          + ufl.inner(ufl.grad(Vi), ufl.grad(Vi)))
    f = dg0_project(e2, Q)
    arr = np.sqrt(np.clip(np.real(f.x.array), 0.0, None))
    f.x.array[:] = arr.astype(f.x.array.dtype)
    return f


def cell_volumes(msh, Q):
    """Volume of every DG0 cell: assemble inner(1, w)*dx, whose entry for cell
    k is exactly |K| because the DG0 test function is 1 on K and 0 elsewhere."""
    import dolfinx
    from dolfinx import fem
    w = ufl.TestFunction(Q)
    one = fem.Constant(msh, dolfinx.default_scalar_type(1.0))
    b = fem.assemble_vector(fem.form(ufl.inner(one, w) * ufl.dx))
    return np.real(b.array).copy()


def cell_midpoints(msh):
    import dolfinx
    tdim = msh.topology.dim
    n = (msh.topology.index_map(tdim).size_local
         + msh.topology.index_map(tdim).num_ghosts)
    cells = np.arange(n, dtype=np.int32)
    return dolfinx.mesh.compute_midpoints(msh, tdim, cells)


def eval_points(fn, msh, pts: np.ndarray):
    """Evaluate `fn` at pts (N,3). Returns (values[N], n_missed).

    Points that collide with no cell (should never happen for interior part
    points) are returned as NaN and counted -- never silently dropped."""
    pts = np.ascontiguousarray(np.asarray(pts, dtype=np.float64))
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = np.zeros(pts.shape[0], dtype=np.int32)
    missed = np.zeros(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) == 0:
            missed[i] = True
        else:
            cells[i] = links[0]
    vals = np.real(fn.eval(pts, cells)).reshape(-1)
    vals[missed] = np.nan
    return vals, int(missed.sum())


def voxel_plane_points(x: np.ndarray, y: np.ndarray, z_mid: float,
                       mask: np.ndarray):
    """The heatr3d mid-plane voxel centres that lie inside the part, as an
    (N,3) point array plus the flat index used to unpack results."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    sel = np.asarray(mask, dtype=bool)
    pts = np.column_stack([X[sel], Y[sel], np.full(int(sel.sum()), float(z_mid))])
    return pts, sel
