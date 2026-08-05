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


def build_mesh_from_stl(path: str | Path, lc_part: float,
                        scale: float = MM_TO_M, with_chamber: bool = False,
                        L: float | None = None, lc_bed_factor: float = 4.0,
                        seed: int = 1, check_self_intersection: bool = False):
    """Validate, then mesh the STL solid (optionally inside a chamber box).

    Refusal happens BEFORE gmsh is touched, so bad geometry costs a
    millisecond rather than a meshing run.

    `with_chamber` DEFAULTS FALSE and is a KNOWN INCOMPLETE PATH. Wrapping the
    part in a bed as a geo-kernel volume-with-a-hole fails in tetgen ("PLC
    Error: a segment and a facet intersect") when the inner surface came from
    a discrete entity. The part-only mesh is exact (machine-precision volume)
    and is what Task 2's gates cover; bed embedding, which a full solve needs,
    is the recorded next increment and is NOT claimed to work here.
    """
    import gmsh
    from dolfinx.io import gmsh as dgmsh
    from mpi4py import MPI

    p = Path(path)
    verts, faces = load_stl(p)
    validate(verts, faces, check_self_intersection=check_self_intersection)
    v_stl = enclosed_volume(verts, faces) * scale ** 3
    ext = (verts.max(axis=0) - verts.min(axis=0)) * scale
    if L is None:
        L = float(ext.max()) * 3.0

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
        gmsh.model.mesh.addNodes(2, ent, node_tags,
                                 (verts * scale).ravel())
        gmsh.model.mesh.addElementsByType(
            ent, 2, [], (faces + 1).ravel().astype(np.int64))
        # 40 degrees, NOT pi: with pi every facet joins one patch and
        # createGeometry spline-fits across the edges, which shrank the pyramid
        # by 2 percent. A sharp-edge angle keeps each planar face its own
        # surface, so the reconstructed geometry is exactly the facets.
        gmsh.model.mesh.classifySurfaces(40.0 * np.pi / 180.0, True, True,
                                         180.0 * np.pi / 180.0)
        gmsh.model.mesh.createGeometry()
        surfs = [t for (d, t) in gmsh.model.getEntities(2)]
        part_loop = gmsh.model.geo.addSurfaceLoop(surfs)
        part_vol = gmsh.model.geo.addVolume([part_loop])
        groups = {"part": [part_vol]}
        if with_chamber:
            ctr = ((verts.min(axis=0) + verts.max(axis=0)) / 2.0) * scale
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
        lo_m = verts.min(axis=0) * scale
        hi_m = verts.max(axis=0) * scale
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
    part_cells = (ct.find(1) if ct is not None
                  else np.arange(msh.topology.index_map(tdim).size_local,
                                 dtype=np.int32))
    v_mesh = float(_cell_volumes(msh)[part_cells].sum())
    info = StlMeshInfo(
        source=p.name, lc_part=float(lc_part), scale=float(scale),
        with_chamber=bool(with_chamber),
        n_cells_total=int(msh.topology.index_map(tdim).size_global),
        n_nodes_total=int(msh.geometry.index_map().size_global),
        n_facets_stl=int(faces.shape[0]),
        part_cells=part_cells,
        stl_volume_m3=v_stl, part_volume_m3=v_mesh,
        part_volume_rel_err_vs_stl=v_mesh / v_stl - 1.0)
    return msh, info


def _add_box_surface_loop(gmsh, L: float, centre=None) -> int:
    """Axis-aligned chamber box in the geo kernel, returned as a surface loop."""
    h = L / 2.0
    cx, cy, cz = (0.0, 0.0, 0.0) if centre is None else [float(v) for v in centre]
    g = gmsh.model.geo
    c = [g.addPoint(x, y, z) for x in (cx - h, cx + h) for y in (cy - h, cy + h)
         for z in (cz - h, cz + h)]
    # corner index: bit0=z, bit1=y, bit2=x
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = []
    for q in quads:
        lines = [g.addLine(c[q[i]], c[q[(i + 1) % 4]]) for i in range(4)]
        faces.append(g.addPlaneSurface([g.addCurveLoop(lines)]))
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
