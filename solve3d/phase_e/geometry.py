"""Phase E geometry: the library pyramid and cube as ANALYTIC OCC solids.

RUNS IN THE SPIKE ENV (gmsh + dolfinx).

WHY NOT IMPORT THE STL. The library ships watertight STLs, but an STL is a
TESSELLATION: importing it would carry the library's triangle budget into the
solve mesh and make the part boundary a faceted approximation whose facets are
not the ones the solver would choose. Both shapes are exactly constructible in
OCC, so they are constructed and then VERIFIED against the STL's recorded
volume at 1e-9 relative (the pre-registered check). That check is what proves
the analytic solid IS the library shape.

Dimensions, from shape_library_3d/meta/*.json (both equal-volume to the 20 mm
sphere, 4188.790204786391 mm^3):
    cube     side 16.119919540164695 mm
    pyramid  square base side = height = 23.248947030192525 mm, APEX UP

The bounding box of each is centred on the chamber origin, matching how
heatr3d.make_geometry and the Phase A anchors centre their parts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solve3d import forward as fwd

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "shape_library_3d" / "meta"

CUBE_A_M = 16.119919540164695e-3
PYR_B_M = 23.248947030192525e-3          # square base side
PYR_H_M = 23.248947030192525e-3          # height, apex up
L_DOMAIN = fwd.L_DOMAIN
SHAPES = ("pyramid", "cube")


def library_volume_m3(shape: str) -> float:
    return float(json.loads((META / f"{shape}.json").read_text())
                 ["actual_volume_mm3"]) * 1e-9


# --------------------------------------------------------------------------- #
# Predicates (feed materials, chi and the nominal masks)
# --------------------------------------------------------------------------- #
def in_part_predicate(shape: str):
    """`mp` is (3, N) points in metres."""
    if shape == "cube":
        h = CUBE_A_M / 2.0
        return lambda mp: ((np.abs(mp[0]) <= h) & (np.abs(mp[1]) <= h)
                           & (np.abs(mp[2]) <= h))
    if shape == "pyramid":
        b2, h = PYR_B_M / 2.0, PYR_H_M

        def pred(mp):
            z = mp[2]
            # half-width of the square cross-section at height z; full at the
            # base (z = -h/2), zero at the apex (z = +h/2)
            s = b2 * (h / 2.0 - z) / h
            return ((np.abs(mp[2]) <= h / 2.0) & (np.abs(mp[0]) <= s)
                    & (np.abs(mp[1]) <= s))
        return pred
    raise ValueError(f"unknown Phase E shape {shape!r}")


def nominal_mask_2d(shape: str, z: float = 0.0) -> np.ndarray:
    """The ANALYTIC nominal cross-section at height z on the shared evaluation
    grid, for the shape metrics."""
    from solve3d import gates as sg
    x, y, _, _ = sg.eval_grid_axes()
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.full(X.size, float(z))
    mp = np.vstack([X.ravel(), Y.ravel(), Z])
    return in_part_predicate(shape)(mp).reshape(X.shape)


# --------------------------------------------------------------------------- #
# OCC construction
# --------------------------------------------------------------------------- #
def _add_solid(occ, shape: str) -> int:
    if shape == "cube":
        a = CUBE_A_M
        return occ.addBox(-a / 2, -a / 2, -a / 2, a, a, a)
    b2, h = PYR_B_M / 2.0, PYR_H_M / 2.0
    p = [occ.addPoint(-b2, -b2, -h), occ.addPoint(b2, -b2, -h),
         occ.addPoint(b2, b2, -h), occ.addPoint(-b2, b2, -h),
         occ.addPoint(0.0, 0.0, h)]
    base = [occ.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    up = [occ.addLine(p[i], p[4]) for i in range(4)]
    faces = [occ.addPlaneSurface([occ.addCurveLoop(base)])]
    for i in range(4):
        loop = occ.addCurveLoop([base[i], up[(i + 1) % 4], -up[i]])
        faces.append(occ.addPlaneSurface([loop]))
    return occ.addVolume([occ.addSurfaceLoop(faces)])


def occ_volume_m3(shape: str) -> float:
    """Volume of the constructed solid, straight from OCC."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("vol")
        tag = _add_solid(gmsh.model.occ, shape)
        gmsh.model.occ.synchronize()
        return float(gmsh.model.occ.getMass(3, tag))
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------- #
# Meshing (modelled on heatr3d_d1_spike/mesh_gmsh.py, which is read-only)
# --------------------------------------------------------------------------- #
class MeshInfo:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"MeshInfo({self.__dict__})"


def build_mesh(shape: str, lc_part: float, lc_bed_factor: float = 4.0,
               L: float = L_DOMAIN, seed: int = 1):
    """Conforming tetrahedral mesh of the part embedded in the chamber.

    The part and the chamber box are FRAGMENTED so the part boundary is a
    conforming mesh surface -- which is the whole reason Phase E uses solve3d
    rather than the voxel engine (registration: instrument.why_not_heatr3d).
    """
    import gmsh
    from dolfinx.io import gmsh as dgmsh
    from mpi4py import MPI

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("phase_e")
        occ = gmsh.model.occ
        box = occ.addBox(-L / 2, -L / 2, -L / 2, L, L, L)
        part = _add_solid(occ, shape)
        out, _ = occ.fragment([(3, box)], [(3, part)])
        occ.synchronize()
        vols = [t for (d, t) in out if d == 3]
        masses = {t: occ.getMass(3, t) for t in vols}
        part_vol = min(masses, key=masses.get)
        bed = [t for t in vols if t != part_vol]
        gmsh.model.addPhysicalGroup(3, [part_vol], 1)
        gmsh.model.addPhysicalGroup(3, bed, 2)

        half = max(PYR_B_M, CUBE_A_M) / 2.0
        f = gmsh.model.mesh.field
        t = f.add("Box")
        pad = 0.02 * half
        for k, v in (("XMin", -half - pad), ("XMax", half + pad),
                     ("YMin", -half - pad), ("YMax", half + pad),
                     ("ZMin", -half - pad), ("ZMax", half + pad)):
            f.setNumber(t, k, v)
        f.setNumber(t, "VIn", lc_part)
        f.setNumber(t, "VOut", lc_bed_factor * lc_part)
        f.setNumber(t, "Thickness", 3.0 * lc_part)
        f.setAsBackgroundMesh(t)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", seed)
        gmsh.model.mesh.generate(3)

        tags, _, _ = gmsh.model.mesh.getNodes(3, part_vol, includeBoundary=True)
        n_part_nodes = int(np.unique(tags).size)
        vol_m3 = float(masses[part_vol])
        md = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
        msh = md.mesh
    finally:
        gmsh.finalize()

    tdim = msh.topology.dim
    info = MeshInfo(
        shape=shape, lc_part=float(lc_part),
        lc_bed=float(lc_bed_factor * lc_part),
        n_nodes_total=int(msh.geometry.index_map().size_global),
        n_nodes_in_part=n_part_nodes,
        n_cells_total=int(msh.topology.index_map(tdim).size_global),
        part_volume_m3=vol_m3,
        part_volume_rel_err_vs_library=vol_m3 / library_volume_m3(shape) - 1.0)
    return msh, info


def match_lc(shape: str, target_nodes_in_part: int, lc0: float,
             tol: float = 0.20, max_iter: int = 6):
    """Iterate the in-part element size until the in-part NODE count matches
    the target within `tol` -- the Phase A unknown-matching rule."""
    lc = float(lc0)
    hist = []
    msh = info = None
    for _ in range(max_iter):
        msh, info = build_mesh(shape, lc)
        ratio = info.n_nodes_in_part / float(target_nodes_in_part)
        hist.append({"lc_part_m": lc, "n_nodes_in_part": info.n_nodes_in_part,
                     "ratio_vs_target": ratio})
        if abs(ratio - 1.0) <= tol:
            break
        lc = lc * ratio ** (1.0 / 3.0)
    return msh, info, hist
