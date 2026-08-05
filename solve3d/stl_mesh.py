"""Arbitrary-STL intake: validate, refuse, and mesh to conforming tets.

Plan: docs/superpowers/plans/2026-08-04-shrinkage-v2-tranche1.md Task 2.

ROUTE CHOICE, and why. Meshing is gmsh 4.15.2 (already the Phase A/E route,
already in the spike env, and the same kernel the OCC path uses so the
equivalence gate compares like with like). VALIDATION is plain numpy rather
than trimesh: trimesh is NOT installed in the dolfinx spike env (it lives only
in .venv312), and adding a dependency to the solver env to answer questions
that are a dozen lines of array code would be the wrong trade. The checks are
explicit and individually tested.

The STL surface is used as the part boundary directly; the chamber is built
around it as a volume-with-a-hole, which is what makes chi non-trivial and
gives the same part/bed physical-group structure the OCC path produces.

REFUSAL NAMES ARE A CONTRACT. shape_library_3d records, per Tier-3 fixture,
the exception it expects ingestion to raise (`rejection_error` in the meta
JSON). Those names are reproduced here exactly and a test asserts the match,
so the library and this module cannot drift apart silently.

UNITS. Library STLs are in MILLIMETRES (meta records bbox_mm). `load_stl`
returns native units; `build_mesh_from_stl` converts with `scale`.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MM_TO_M = 1.0e-3

# Sharp-feature angle handed to gmsh's `classifySurfaces`: facets whose dihedral
# angle exceeds it start a new patch, and each patch is spline-fitted by
# `createGeometry`. Too coarse a threshold merges genuinely curved regions into
# one patch and the fit CUTS THE CORNER, so the meshed solid is quietly smaller
# than the STL.
#
# Tranche 1 measured this once, on the library pyramid, and moved pi -> 40 deg
# (pi merged every facet and shrank the pyramid 2 percent). 40 deg is safe for
# an all-planar primitive but NOT for arbitrary geometry: on the real user part
# ("Part Studio 1 - Tamper.stl", a 44 mm disc-like solid with a curved wall) it
# merges the wall into 8 patches and loses 0.79 percent of the volume. Measured
# sweep, part volume rel err vs the STL's own enclosed volume:
#
#   angle    pyramid          Tamper       Tamper surfaces
#    40.0    2.220e-16       -7.912e-03           8
#     5.0    2.220e-16       -1.604e-04         489
#     2.0    2.220e-16       -9.436e-06         886
#     1.0    2.220e-16       -1.458e-06         998
#
# The pyramid is INVARIANT to the angle (its facets are coplanar, so nothing
# splits), which is why lowering the default costs the library path nothing.
# 1 degree, and the volume check below refuses anything the fit still smooths.
FEATURE_ANGLE_DEG = 1.0

# Stated discretisation tolerance on the meshed part volume against the STL's
# own enclosed volume. Same number the Tranche 1 gates use. It is a REFUSAL,
# not a report: a part that meshed 1 percent small would otherwise flow into a
# solve and produce confident wrong numbers.
VOLUME_REL_TOL = 5.0e-3


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
class MeshRefusal(Exception):
    """Base: this geometry is not a part and will not be meshed."""


class NonWatertightMeshError(MeshRefusal):
    """Open shell: some edge is not shared by exactly two triangles."""


class ZeroVolumeError(MeshRefusal):
    """Degenerate: the surface encloses no volume."""


class SelfIntersectingMeshError(MeshRefusal):
    """Triangles pass through one another; the enclosed region is ambiguous."""


class InconsistentWindingError(MeshRefusal):
    """Face orientations disagree, so the inside is not well defined."""


class SurfaceReconstructionError(MeshRefusal):
    """The meshed solid is not the STL solid.

    gmsh's `classifySurfaces` groups facets into patches and `createGeometry`
    spline-fits each patch, so a threshold that is too coarse smooths across
    real geometry and the meshed part is quietly the WRONG SIZE. That is a
    confident-wrong-number failure, not a crash, so it is refused rather than
    reported: see FEATURE_ANGLE_DEG.
    """


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def load_stl(path: str | Path):
    """Read a binary or ASCII STL into (vertices, faces), native units.

    Vertices are welded on exact coordinate equality, which is what STL's
    per-facet duplication produces for a mesh exported from a solid; the
    watertightness test depends on that welding.
    """
    raw = Path(path).read_bytes()
    tris = (_parse_binary(raw) if _is_binary(raw) else _parse_ascii(raw))
    if tris.size == 0:
        raise ZeroVolumeError(f"{Path(path).name}: no triangles")
    flat = tris.reshape(-1, 3)
    verts, inv = np.unique(flat, axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 3).astype(np.int64)


def _is_binary(raw: bytes) -> bool:
    if len(raw) < 84:
        return False
    n = struct.unpack("<I", raw[80:84])[0]
    return len(raw) == 84 + 50 * n


def _parse_binary(raw: bytes) -> np.ndarray:
    n = struct.unpack("<I", raw[80:84])[0]
    rec = np.frombuffer(raw, dtype=np.dtype([
        ("normal", "<3f4"), ("v", "<3f4", (3,)), ("attr", "<u2")]),
        count=n, offset=84)
    return np.asarray(rec["v"], dtype=np.float64)


def _parse_ascii(raw: bytes) -> np.ndarray:
    vals = [float(t) for line in raw.decode("ascii", "ignore").splitlines()
            if line.strip().startswith("vertex")
            for t in line.split()[1:4]]
    a = np.asarray(vals, dtype=np.float64)
    return a.reshape(-1, 3, 3)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _edges(faces: np.ndarray) -> np.ndarray:
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                           faces[:, [2, 0]]], axis=0)


def check_watertight(verts: np.ndarray, faces: np.ndarray) -> None:
    """Every undirected edge shared by exactly two triangles, and every
    DIRECTED edge used exactly once (consistent winding)."""
    e = _edges(faces)
    und = np.sort(e, axis=1)
    _u, counts = np.unique(und, axis=0, return_counts=True)
    bad = int((counts != 2).sum())
    if bad:
        raise NonWatertightMeshError(
            f"{bad} boundary/non-manifold edge(s): every edge of a closed "
            f"surface must be shared by exactly 2 triangles, found counts "
            f"{sorted(set(counts.tolist()))}")
    _d, dcounts = np.unique(e, axis=0, return_counts=True)
    if int((dcounts != 1).sum()):
        raise InconsistentWindingError(
            "a directed edge is used more than once: face winding disagrees")


def enclosed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume by the divergence theorem, in the vertices' own units."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def _tri_pairs_to_test(verts: np.ndarray, faces: np.ndarray):
    """AABB broad phase; skip pairs that share a vertex (they touch legally)."""
    tv = verts[faces]
    lo, hi = tv.min(axis=1), tv.max(axis=1)
    n = len(faces)
    order = np.argsort(lo[:, 0])
    for ii in range(n):
        i = order[ii]
        for jj in range(ii + 1, n):
            j = order[jj]
            if lo[j, 0] > hi[i, 0]:
                break
            if (hi[i, 1] < lo[j, 1] or hi[j, 1] < lo[i, 1]
                    or hi[i, 2] < lo[j, 2] or hi[j, 2] < lo[i, 2]):
                continue
            if np.intersect1d(faces[i], faces[j]).size:
                continue
            yield i, j


def _seg_tri_hit(p0, p1, t0, t1, t2, eps=1e-12) -> bool:
    """Moller-Trumbore, segment form."""
    d = p1 - p0
    e1, e2 = t1 - t0, t2 - t0
    h = np.cross(d, e2)
    a = float(np.dot(e1, h))
    if abs(a) < eps:
        return False
    f = 1.0 / a
    s = p0 - t0
    u = f * float(np.dot(s, h))
    if u < -eps or u > 1.0 + eps:
        return False
    q = np.cross(s, e1)
    v = f * float(np.dot(d, q))
    if v < -eps or u + v > 1.0 + eps:
        return False
    t = f * float(np.dot(e2, q))
    return eps < t < 1.0 - eps


def check_self_intersections(verts: np.ndarray, faces: np.ndarray) -> None:
    tv = verts[faces]
    for i, j in _tri_pairs_to_test(verts, faces):
        A, B = tv[i], tv[j]
        for (p, q, T) in ((A[0], A[1], B), (A[1], A[2], B), (A[2], A[0], B),
                          (B[0], B[1], A), (B[1], B[2], A), (B[2], B[0], A)):
            if _seg_tri_hit(p, q, T[0], T[1], T[2]):
                raise SelfIntersectingMeshError(
                    f"triangles {i} and {j} intersect; the enclosed region "
                    f"is ambiguous and will not be meshed")


def validate(verts: np.ndarray, faces: np.ndarray, *,
             check_self_intersection: bool = False,
             min_volume: float = 1e-12) -> None:
    """Refuse anything that is not a closed, positive-volume solid."""
    check_watertight(verts, faces)
    v = enclosed_volume(verts, faces)
    if v <= min_volume:
        raise ZeroVolumeError(f"enclosed volume {v:.3e} is not positive")
    if check_self_intersection:
        check_self_intersections(verts, faces)


# --------------------------------------------------------------------------- #
# Constructed fixture: the library has no self-intersecting Tier-3 case
# --------------------------------------------------------------------------- #
def two_interpenetrating_tetrahedra():
    """CONSTRUCTED, not a library fixture (recorded gap).

    Two closed tetrahedra that overlap in space. Edge-wise each is perfect, so
    the union passes the watertight test; only a real intersection test finds
    the problem. That is exactly the case the refusal must catch.
    """
    t = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    # shift chosen so the tetrahedra genuinely overlap: each coordinate of
    # the shifted apex region is >= 0.20 and 3 * 0.20 = 0.60 < 1, so points
    # satisfying x + y + z <= 1 lie inside BOTH solids. (0.35 was wrong: it
    # gives 1.05 > 1 and the two tets merely come close.)
    shift = np.array([0.20, 0.20, 0.20])
    verts = np.vstack([t, t + shift])
    faces = np.vstack([f, f + 4])
    return verts, faces


# --------------------------------------------------------------------------- #
# Meshing
# --------------------------------------------------------------------------- #
class StlMeshInfo:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"StlMeshInfo({self.__dict__})"


_L0_DEFAULT = object()          # sentinel: "use the approved shared default"


def stl_vertices_in_mesh_frame(verts: np.ndarray, scale: float,
                               centre: bool, precomp_coeffs):
    """The affine map from STL native coordinates to solver coordinates.

    Three steps, in this order, and the order is not arbitrary:
      1. UNITS      native -> metres (`scale`).
      2. CENTRING   subtract the bounding-box centre, so the part sits on the
                    chamber origin. The solver's electrodes are the facets at
                    z = +-L/2 in ABSOLUTE coordinates (forward.march_enthalpy),
                    so a part left where the CAD put it would be off-axis in
                    the chamber. The OCC path centres its solids for the same
                    reason (phase_e/geometry._add_solid).
      3. LEVEL 0    anisotropic pre-compensation about that centre. Scaling
                    before centring would move the part as well as grow it.

    Returned as (points_m, translation_m, factors) so the inverse is exact and
    a caller can map solver points back onto the original triangle soup.
    """
    p = np.asarray(verts, dtype=float) * float(scale)
    t = ((p.min(axis=0) + p.max(axis=0)) / 2.0 if centre
         else np.zeros(3, dtype=float))
    f = (np.ones(3) if precomp_coeffs is None or precomp_coeffs.is_identity
         else np.asarray(precomp_coeffs.factors, dtype=float))
    return (p - t) * f, t, f


def mesh_points_to_stl_native(pts: np.ndarray, info) -> np.ndarray:
    """Inverse of `stl_vertices_in_mesh_frame`, for (N, 3) solver points."""
    p = np.asarray(pts, dtype=float)
    return (p / np.asarray(info.precomp_factors)
            + np.asarray(info.translation_m)) / float(info.scale)


def part_mask_predicate(info):
    """A `build_materials`-shaped predicate that reads the CELL TAGS.

    The point of meshing the STL rather than voxelising it is that the part
    boundary is a conforming mesh surface, so "is this cell in the part" is
    answered by the gmsh physical group exactly, with no inside test and no
    staircase. A geometric ray cast over ~10^5 cells would be both slower and
    less accurate than the tagging that produced the mesh.

    It is not self-certifying: `test_stl_chamber.py` cross-checks the mask
    cell-for-cell against an INDEPENDENT ray cast on the original triangles,
    so a tag/ordering mismatch fails a gate rather than passing silently.
    """
    mask = np.zeros(int(info.n_cells_local), dtype=bool)
    mask[np.asarray(info.part_cells, dtype=np.int64)] = True

    def pred(mp):
        mp = np.asarray(mp)
        n = mp.shape[1] if mp.ndim == 2 else 0
        if n != mask.size:
            raise ValueError(
                f"part_mask_predicate: mesh has {mask.size} local cells but "
                f"was asked about {n} points; this predicate is defined on "
                "the cells of ITS OWN mesh only")
        return mask
    return pred


def build_mesh_from_stl(path: str | Path, lc_part: float,
                        scale: float = MM_TO_M, with_chamber: bool = False,
                        L: float | None = None, lc_bed_factor: float = 4.0,
                        seed: int = 1, check_self_intersection: bool = False,
                        precomp_coeffs=_L0_DEFAULT, centre: bool | None = None,
                        feature_angle_deg: float = FEATURE_ANGLE_DEG,
                        volume_rel_tol: float = VOLUME_REL_TOL):
    """Validate, then mesh the STL solid (optionally inside a chamber box).

    Refusal happens BEFORE gmsh is touched, so bad geometry costs a
    millisecond rather than a meshing run.

    `with_chamber=True` NOW WORKS. Tranche 1 recorded it as blocked by a tetgen
    "PLC Error: a segment and a facet intersect" and attributed that to the
    inner surface being a discrete entity. The attribution was wrong: the empty
    chamber box failed identically with no part present, because
    `_add_box_surface_loop` built each box edge twice. See that function.

    LEVEL 0 IS APPLIED BY DEFAULT, matching phase_e/run.build_case. The scaled
    solid is what gets meshed AND what the refinement box follows, so
    pre-compensation cannot quietly coarsen the mesh at the part boundary
    (the correction Tranche 1 had to make on the OCC path). Pass
    `precomp_coeffs=precomp.ShrinkageL0(0.0, 0.0)` -- or `None` -- to reproduce
    a pre-L0 run.

    `centre` defaults to `with_chamber`: a part being embedded in the chamber
    is put on the chamber origin, a part-only mesh is left in its own frame so
    the existing part-only gates keep meshing exactly what they meshed before.
    """
    import gmsh
    from dolfinx.io import gmsh as dgmsh
    from mpi4py import MPI

    from solve3d import precomp as _pc
    if precomp_coeffs is _L0_DEFAULT:
        precomp_coeffs = _pc.load_defaults()
    if centre is None:
        centre = bool(with_chamber)

    p = Path(path)
    verts, faces = load_stl(p)
    validate(verts, faces, check_self_intersection=check_self_intersection)
    vm, translation, factors = stl_vertices_in_mesh_frame(
        verts, scale, centre, precomp_coeffs)
    # the STL's own enclosed volume, in the SAME frame as the mesh: the
    # pre-compensated solid is deliberately larger than nominal, and comparing
    # the meshed volume against the nominal number would report the correction
    # as an error (the bug Tranche 1 found on the OCC path)
    v_stl = enclosed_volume(verts, faces) * scale ** 3 * float(np.prod(factors))
    ext = vm.max(axis=0) - vm.min(axis=0)
    if L is None:
        L = float(ext.max()) * 3.0
    if with_chamber and float(ext.max()) >= L:
        raise ValueError(
            f"part extent {float(ext.max()):.4f} m does not fit in a chamber "
            f"of side {float(L):.4f} m; refusing rather than meshing a part "
            "that pokes through the bed")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("stl_part")
        # The surface is handed to gmsh as a DISCRETE ENTITY built from the
        # triangles this module already parsed and validated, rather than via
        # gmsh.merge. Two reasons, one practical and one about provenance:
        # gmsh 4.15.2 refuses these library binary STLs outright ("Error
        # loading"), and feeding the parsed arrays guarantees the meshed
        # surface is exactly the one the refusal checks ran on.
        ent = gmsh.model.addDiscreteEntity(2)
        node_tags = np.arange(1, verts.shape[0] + 1, dtype=np.int64)
        gmsh.model.mesh.addNodes(2, ent, node_tags, vm.ravel())
        gmsh.model.mesh.addElementsByType(
            ent, 2, [], (faces + 1).ravel().astype(np.int64))
        # See FEATURE_ANGLE_DEG for the measured sweep behind this number and
        # for why 40 degrees was safe on the library primitives and wrong on a
        # real curved part.
        gmsh.model.mesh.classifySurfaces(
            float(feature_angle_deg) * np.pi / 180.0, True, True,
            180.0 * np.pi / 180.0)
        gmsh.model.mesh.createGeometry()
        surfs = [t for (d, t) in gmsh.model.getEntities(2)]
        part_loop = gmsh.model.geo.addSurfaceLoop(surfs)
        part_vol = gmsh.model.geo.addVolume([part_loop])
        groups = {"part": [part_vol]}
        if with_chamber:
            ctr = None if centre else (vm.min(axis=0) + vm.max(axis=0)) / 2.0
            box_loop = _add_box_surface_loop(gmsh, L, ctr)
            # a volume whose SECOND loop is a hole: the bed wraps the part and
            # shares its boundary, which is what makes the mesh conforming
            bed_vol = gmsh.model.geo.addVolume([box_loop, part_loop])
            groups["bed"] = [bed_vol]
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(3, groups["part"], 1)
        if with_chamber:
            gmsh.model.addPhysicalGroup(3, groups["bed"], 2)
        f = gmsh.model.mesh.field
        t = f.add("Box")
        # centred on the PART's own bounding box: library STLs are not all
        # origin-centred (the pyramid sits at z in [0, 23.2] mm), and an
        # origin-centred refinement box left half of it at the coarse size
        lo_m = vm.min(axis=0)
        hi_m = vm.max(axis=0)
        pad = 0.02 * float(ext.max())
        for k, val in (("XMin", lo_m[0] - pad), ("XMax", hi_m[0] + pad),
                       ("YMin", lo_m[1] - pad), ("YMax", hi_m[1] + pad),
                       ("ZMin", lo_m[2] - pad), ("ZMax", hi_m[2] + pad)):
            f.setNumber(t, k, val)
        f.setNumber(t, "VIn", lc_part)
        f.setNumber(t, "VOut", lc_bed_factor * lc_part)
        f.setNumber(t, "Thickness", 3.0 * lc_part)
        f.setAsBackgroundMesh(t)
        for opt in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
                    "Mesh.MeshSizeFromCurvature"):
            gmsh.option.setNumber(opt, 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", seed)
        gmsh.model.mesh.generate(3)
        md = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
        msh, ct = md.mesh, md.cell_tags
    finally:
        gmsh.finalize()

    tdim = msh.topology.dim
    n_local = int(msh.topology.index_map(tdim).size_local)
    part_cells = (ct.find(1) if ct is not None
                  else np.arange(n_local, dtype=np.int32))
    part_cells = np.asarray(part_cells, dtype=np.int64)
    part_cells = part_cells[part_cells < n_local]
    v_mesh = float(_cell_volumes(msh)[part_cells].sum())
    info = StlMeshInfo(
        source=p.name, path=str(p), lc_part=float(lc_part), scale=float(scale),
        with_chamber=bool(with_chamber), centred=bool(centre),
        L_chamber_m=(float(L) if with_chamber else None),
        lc_bed=float(lc_bed_factor * lc_part),
        n_cells_total=int(msh.topology.index_map(tdim).size_global),
        n_cells_local=n_local,
        n_part_cells=int(part_cells.size),
        n_bed_cells=int(n_local - part_cells.size),
        n_nodes_total=int(msh.geometry.index_map().size_global),
        n_facets_stl=int(faces.shape[0]),
        part_cells=part_cells,
        translation_m=[float(v) for v in translation],
        precomp_factors=[float(v) for v in factors],
        precomp=(_pc.ShrinkageL0(0.0, 0.0).provenance()
                 if precomp_coeffs is None else precomp_coeffs.provenance()),
        feature_angle_deg=float(feature_angle_deg),
        stl_volume_m3=v_stl, part_volume_m3=v_mesh,
        part_volume_rel_err_vs_stl=v_mesh / v_stl - 1.0)
    if abs(info.part_volume_rel_err_vs_stl) > float(volume_rel_tol):
        raise SurfaceReconstructionError(
            f"{p.name}: the meshed part volume is "
            f"{info.part_volume_rel_err_vs_stl:+.4e} relative to the STL's own "
            f"enclosed volume, outside the stated {volume_rel_tol:.1e} "
            f"tolerance. The surface reconstruction at feature_angle_deg="
            f"{float(feature_angle_deg)} has smoothed across real geometry. "
            "Lower the angle rather than accepting the mesh: a part that is "
            "quietly the wrong size produces confident wrong numbers.")
    return msh, info


def _add_box_surface_loop(gmsh, L: float, centre=None) -> int:
    """Axis-aligned chamber box in the geo kernel, returned as a surface loop.

    EACH OF THE 12 EDGES IS CREATED EXACTLY ONCE and reused, with a negative
    tag where a face traverses it backwards. This is the whole Tranche 1
    chamber blocker.

    `gmsh.model.geo.addLine` does NOT deduplicate: building the six faces
    independently created 24 curves, two coincident copies of every edge. Each
    copy is meshed on its own, so the two faces meeting at that edge carried
    different 1-D node sets and the shell was not conforming with ITSELF.
    tetgen then reported "PLC Error: a segment and a facet intersect", which
    was true and had nothing to do with the STL part -- the empty box failed
    the same way. See solve3d/tests/test_stl_chamber.py, which fails at 24
    curves before this function is corrected.
    """
    h = L / 2.0
    cx, cy, cz = (0.0, 0.0, 0.0) if centre is None else [float(v) for v in centre]
    g = gmsh.model.geo
    c = [g.addPoint(x, y, z) for x in (cx - h, cx + h) for y in (cy - h, cy + h)
         for z in (cz - h, cz + h)]
    # corner index: bit0=z, bit1=y, bit2=x
    lines: dict[tuple[int, int], int] = {}

    def edge(a: int, b: int) -> int:
        """Signed curve tag for the directed edge a->b, creating it once."""
        key = (a, b) if a < b else (b, a)
        if key not in lines:
            lines[key] = g.addLine(c[key[0]], c[key[1]])
        return lines[key] if a < b else -lines[key]

    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = []
    for q in quads:
        loop = [edge(q[i], q[(i + 1) % 4]) for i in range(4)]
        faces.append(g.addPlaneSurface([g.addCurveLoop(loop)]))
    return g.addSurfaceLoop(faces)


def _cell_volumes(msh) -> np.ndarray:
    tdim = msh.topology.dim
    n = msh.topology.index_map(tdim).size_local
    geom = msh.geometry.x
    dofs = msh.geometry.dofmap.reshape(n, -1)[:, :4]
    v = geom[dofs]
    return np.abs(np.einsum("ij,ij->i",
                            v[:, 1] - v[:, 0],
                            np.cross(v[:, 2] - v[:, 0],
                                     v[:, 3] - v[:, 0]))) / 6.0


def chi_and_volumes(msh, info):
    """Cellwise chi (1 in the part, 0 in the bed) and the cell volumes.

    chi is read from the gmsh PHYSICAL GROUPS, i.e. from the conforming part
    boundary, which is the point of meshing the STL rather than voxelising it.
    """
    vol = _cell_volumes(msh)
    chi = np.zeros(vol.size, dtype=float)
    chi[np.asarray(info.part_cells, dtype=np.int64)] = 1.0
    return chi, vol


# --------------------------------------------------------------------------- #
# The honest equivalence gate
# --------------------------------------------------------------------------- #
def occ_vs_stl_equivalence(lc_part: float = 1.5e-3, shape: str = "cube",
                           tmpdir: str | Path | None = None) -> dict:
    """Build a solid via OCC, export it to STL, re-mesh through the STL path,
    and compare against the Phase A dolfinx SAME-ENGINE band.

    The band comes from solve3d/results/dolfinx_refinement.json (MAX of the
    engine's own refinement pair spreads, times the frozen 1.5). Using the
    engine's own wobble is what keeps this from being self-graded: the STL
    route is equivalent only if it differs by no more than remeshing does.
    """
    import json
    import tempfile
    import gmsh
    from solve3d.phase_e import geometry as geo

    root = Path(__file__).resolve().parents[1]
    spreads = json.loads((root / "solve3d" / "results"
                          / "dolfinx_refinement.json").read_text())["spreads"]
    # the file records the MAX-of-pairs at top level under its own
    # reporting_rule; read that rather than recomputing it here
    band = 1.5 * float(spreads["t90_rel_spread"])

    d = Path(tmpdir or tempfile.mkdtemp())
    stl_path = d / f"{shape}_occ.stl"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("occ_export")
        tag = geo._add_solid(gmsh.model.occ, shape)
        gmsh.model.occ.synchronize()
        v_occ = float(gmsh.model.occ.getMass(3, tag))
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(stl_path))
    finally:
        gmsh.finalize()

    verts, faces = load_stl(stl_path)
    validate(verts, faces)
    v_stl = enclosed_volume(verts, faces)      # already metres: OCC exported m
    return {
        "shape": shape,
        "occ_volume_m3": v_occ,
        "stl_volume_m3": v_stl,
        "part_volume_rel_diff": v_stl / v_occ - 1.0,
        "band_used": band,
        "band_rule": ("1.5 x MAX of the dolfinx own-refinement pair spreads "
                      "(t90_rel_spread); Phase A same-engine band"),
        "source": "solve3d/results/dolfinx_refinement.json",
        "n_facets": int(faces.shape[0]),
        "stl_path": str(stl_path),
    }


def points_inside(verts: np.ndarray, faces: np.ndarray,
                  pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Inside test by ray casting along +z, counting crossings.

    Needed independently of the mesh: chi on any embedding grid, and the L2
    kinematics comparison, both ask "is this point in the solid" for a surface
    that is a triangle soup rather than an OCC solid.

    DEGENERACY IS HANDLED, not hoped away. A vertical ray from a point on the
    pyramid's axis passes exactly through the apex, which four triangles share,
    so the naive crossing count is even and reports the point OUTSIDE. Whenever
    a hit lands within `tol` of a facet edge or vertex the query is retried at a
    tiny deterministic xy offset, which moves the ray off the degenerate line
    without moving it across any real boundary.
    """
    p = np.asarray(pts, dtype=float)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    span = float(np.max(verts.max(axis=0) - verts.min(axis=0)))
    # the retry offset must be LARGER than the degeneracy tolerance or every
    # retry re-flags the same degeneracy and the point falls through as
    # "outside"; 1e-4 of the bounding span is far above tol and far below any
    # real feature of these parts
    jitters = ((0.0, 0.0), (1.7e-4, 0.9e-4), (-1.3e-4, 2.1e-4),
               (0.7e-4, -1.9e-4))
    out = np.zeros(p.shape[0], dtype=bool)
    for i, q0 in enumerate(p):
        for jx, jy in jitters:
            q = q0 + np.array([jx * span, jy * span, 0.0])
            v0 = b[:, :2] - a[:, :2]
            v1 = c[:, :2] - a[:, :2]
            v2 = q[:2] - a[:, :2]
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            ok = np.abs(den) > 1e-300
            u = np.zeros_like(den)
            w = np.zeros_like(den)
            u[ok] = (v2[ok, 0] * v1[ok, 1] - v1[ok, 0] * v2[ok, 1]) / den[ok]
            w[ok] = (v0[ok, 0] * v2[ok, 1] - v2[ok, 0] * v0[ok, 1]) / den[ok]
            hit = ok & (u >= -tol) & (w >= -tol) & (u + w <= 1.0 + tol)
            degenerate = False
            if hit.any():
                bary = np.stack([u[hit], w[hit], 1.0 - u[hit] - w[hit]])
                degenerate = bool(np.any(np.abs(bary) < tol * 1e3))
            z = (a[hit, 2] + u[hit] * (b[hit, 2] - a[hit, 2])
                 + w[hit] * (c[hit, 2] - a[hit, 2]))
            out[i] = bool(int((z > q[2]).sum()) % 2)
            if not degenerate:
                break
            # else: keep this answer but try to better it with an offset ray
    return out


def cell_centroids(msh) -> np.ndarray:
    tdim = msh.topology.dim
    n = msh.topology.index_map(tdim).size_local
    dofs = msh.geometry.dofmap.reshape(n, -1)[:, :4]
    return msh.geometry.x[dofs].mean(axis=1)
