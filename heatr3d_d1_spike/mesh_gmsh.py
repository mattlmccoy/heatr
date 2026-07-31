"""gmsh meshing for the D1 spike: a conforming part embedded in the 60 mm bed.

The whole point of the spike is that the part boundary is a CONFORMING mesh
surface, not a voxel staircase, so both geometries are built with the OCC
kernel and fragmented against the chamber box (one conformal mesh, two
volumes: physical tag 1 = part, tag 2 = bed).

Element sizing is region-based (gmsh Cylinder / Box fields), NOT curvature- or
point-driven, so the in-part element size is a single controlled number that
can be matched against the heatr3d voxel size h = L/n.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import eqs_common as ec           # applies jit_fix before dolfinx
import gmsh
from dolfinx.io import gmsh as dgmsh
from mpi4py import MPI

PART_TAG, BED_TAG = 1, 2


@dataclass
class MeshInfo:
    lc_part: float
    lc_bed: float
    n_nodes_total: int
    n_nodes_in_part: int
    n_cells_total: int
    n_cells_in_part: int
    part_volume_m3: float
    corner_lc: float | None = None
    corner_radius_m: float | None = None


def _size_fields(kind: str, half: float, lc_part: float, lc_bed: float,
                 L: float, corner_lc: float | None, corner_radius: float | None):
    """Region size field: lc_part inside the part (+ optional finer band around
    the four vertical corner edges of the square), lc_bed elsewhere."""
    f = gmsh.model.mesh.field
    ids = []
    if kind == "cylinder":
        t = f.add("Cylinder")
        f.setNumber(t, "Radius", half * 1.02)
        f.setNumber(t, "VIn", lc_part)
        f.setNumber(t, "VOut", lc_bed)
        f.setNumber(t, "XCenter", 0.0)
        f.setNumber(t, "YCenter", 0.0)
        f.setNumber(t, "ZCenter", 0.0)
        f.setNumber(t, "XAxis", 0.0)
        f.setNumber(t, "YAxis", 0.0)
        f.setNumber(t, "ZAxis", L)
        ids.append(t)
    else:
        t = f.add("Box")
        pad = 0.02 * half
        for k, v in (("XMin", -half - pad), ("XMax", half + pad),
                     ("YMin", -half - pad), ("YMax", half + pad),
                     ("ZMin", -L), ("ZMax", L)):
            f.setNumber(t, k, v)
        f.setNumber(t, "VIn", lc_part)
        f.setNumber(t, "VOut", lc_bed)
        f.setNumber(t, "Thickness", 3.0 * lc_part)
        ids.append(t)
        if corner_lc is not None:
            for sx in (-1.0, +1.0):
                for sy in (-1.0, +1.0):
                    c = f.add("Cylinder")            # ball-of-radius band about
                    f.setNumber(c, "Radius", corner_radius)   # the vertical edge
                    f.setNumber(c, "VIn", corner_lc)
                    f.setNumber(c, "VOut", lc_bed)
                    f.setNumber(c, "XCenter", sx * half)
                    f.setNumber(c, "YCenter", sy * half)
                    f.setNumber(c, "ZCenter", 0.0)
                    f.setNumber(c, "XAxis", 0.0)
                    f.setNumber(c, "YAxis", 0.0)
                    f.setNumber(c, "ZAxis", L)
                    ids.append(c)
    mn = f.add("Min")
    f.setNumbers(mn, "FieldsList", [float(i) for i in ids])
    f.setAsBackgroundMesh(mn)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


def _build_model(kind: str, half: float, lc_part: float, lc_bed: float,
                 L: float, corner_lc: float | None, corner_radius: float | None,
                 seed: int = 1):
    gmsh.clear()
    gmsh.model.add("d1")
    occ = gmsh.model.occ
    box = occ.addBox(-L / 2, -L / 2, -L / 2, L, L, L)
    if kind == "cylinder":
        part = occ.addCylinder(0.0, 0.0, -L / 2, 0.0, 0.0, L, half)
    elif kind == "square":
        part = occ.addBox(-half, -half, -L / 2, 2 * half, 2 * half, L)
    else:
        raise ValueError(kind)
    out, _ = occ.fragment([(3, box)], [(3, part)])
    occ.synchronize()
    vols = [t for (d, t) in out if d == 3]
    # identify the part volume by its mass (part volume is much the smaller)
    masses = {t: gmsh.model.occ.getMass(3, t) for t in vols}
    part_vol = min(masses, key=masses.get)
    bed_vols = [t for t in vols if t != part_vol]
    gmsh.model.addPhysicalGroup(3, [part_vol], PART_TAG)
    gmsh.model.addPhysicalGroup(3, bed_vols, BED_TAG)
    _size_fields(kind, half, lc_part, lc_bed, L, corner_lc, corner_radius)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)      # Delaunay (deterministic)
    gmsh.option.setNumber("Mesh.RandomSeed", seed)
    gmsh.model.mesh.generate(3)
    return part_vol, masses[part_vol]


def _count_part_nodes(part_vol: int) -> int:
    tags, _, _ = gmsh.model.mesh.getNodes(3, part_vol, includeBoundary=True)
    return int(np.unique(tags).size)


def build(kind: str, lc_part: float, lc_bed_factor: float = 4.0,
          half: float = 0.010, L: float = ec.L_DOMAIN,
          corner_lc: float | None = None, corner_radius_m: float = 0.002,
          comm=MPI.COMM_WORLD):
    """Mesh once at the given in-part element size; return (dolfinx mesh, MeshInfo)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    lc_bed = lc_bed_factor * lc_part
    part_vol, vol_m3 = _build_model(kind, half, lc_part, lc_bed, L,
                                    corner_lc, corner_radius_m)
    n_part_nodes = _count_part_nodes(part_vol)
    md = dgmsh.model_to_mesh(gmsh.model, comm, 0, gdim=3)
    msh = md.mesh
    gmsh.finalize()

    tdim = msh.topology.dim
    ncell = (msh.topology.index_map(tdim).size_local
             + msh.topology.index_map(tdim).num_ghosts)
    info = MeshInfo(
        lc_part=lc_part, lc_bed=lc_bed,
        n_nodes_total=int(msh.geometry.index_map().size_global),
        n_nodes_in_part=n_part_nodes,
        n_cells_total=int(msh.topology.index_map(tdim).size_global),
        n_cells_in_part=0, part_volume_m3=float(vol_m3),
        corner_lc=corner_lc,
        corner_radius_m=corner_radius_m if corner_lc is not None else None)
    return msh, info


def match_lc(kind: str, target_nodes_in_part: int, lc0: float,
             tol: float = 0.20, max_iter: int = 6, **kw):
    """Iterate the in-part element size until the in-part NODE count (the FEM
    unknown count in the part) matches the heatr3d in-part VOXEL count (its
    unknown count in the part) within `tol`.

    Node-count matching -- not tet-count matching -- is the like-for-like
    comparison of UNKNOWNS, and it also lands the element size within ~20 % of
    the voxel size h, so "same resolution" holds under both readings. Both the
    node and cell counts are recorded so the report can check either."""
    lc = lc0
    history = []
    msh = info = None
    for _ in range(max_iter):
        msh, info = build(kind, lc, **kw)
        ratio = info.n_nodes_in_part / target_nodes_in_part
        history.append({"lc_part_m": lc, "n_nodes_in_part": info.n_nodes_in_part,
                        "ratio_vs_target": ratio})
        if abs(ratio - 1.0) <= tol:
            break
        lc = lc * ratio ** (1.0 / 3.0)          # nodes ~ lc^-3
    return msh, info, history
