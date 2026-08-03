"""Grade and Print intake gate (spec 2026-08-02-grade-and-print-design.md
section 4): refuse non-watertight and self-intersecting meshes loudly.

Verdict shape:
    {"accepted": bool, "error": str | None,
     "checks": {"watertight": {"ok", "n_open_edges"},
                "self_intersection": {"ok", "n_intersecting_pairs"}}}

Through-hole refusal is enforced at analyze time on the slice loops (the
existing has_holes detection); a mesh-level intake cannot see slice holes
without duplicating the slicer.

Self-intersection test: rtree broad phase on triangle bounding boxes, then a
segment-vs-triangle narrow phase (each edge of one triangle against the other
triangle, both directions), excluding face pairs that share a vertex. Exactly
coplanar overlapping triangles are not detected by the edge test; real STL
defects of that kind co-occur with edge crossings.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import trimesh
import trimesh.grouping

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _open_edge_count(mesh: trimesh.Trimesh) -> int:
    """Edges referenced by exactly one face."""
    groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    return int(len(groups))


def _segments_cross_triangles(orig: np.ndarray, vec: np.ndarray,
                              tri: np.ndarray) -> np.ndarray:
    """Vectorized Moller-Trumbore for segments vs paired triangles.

    orig, vec: (m, 3) segment origins and full-length direction vectors.
    tri: (m, 3, 3) one triangle per segment.
    Returns a boolean mask: segment strictly crosses its triangle interior.
    """
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    pvec = np.cross(vec, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > _EPS
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = orig - tri[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = np.einsum("ij,ij->i", vec, qvec) * inv
    t = np.einsum("ij,ij->i", e2, qvec) * inv
    strict = 1e-9
    return (ok & (u > strict) & (v > strict) & (u + v < 1.0 - strict)
            & (t > strict) & (t < 1.0 - strict))


def _pairs_intersect(tris_a: np.ndarray, tris_b: np.ndarray) -> np.ndarray:
    """For each triangle pair, do any of the 6 edges cross the other one."""
    hit = np.zeros(len(tris_a), dtype=bool)
    for a, b in ((tris_a, tris_b), (tris_b, tris_a)):
        for i0, i1 in ((0, 1), (1, 2), (2, 0)):
            orig = a[:, i0]
            vec = a[:, i1] - a[:, i0]
            hit |= _segments_cross_triangles(orig, vec, b)
    return hit


def count_self_intersections(mesh: trimesh.Trimesh,
                             max_pairs: int = 2_000_000) -> int:
    """Number of face pairs that geometrically intersect.

    Face pairs sharing a merged vertex are excluded (mesh adjacency, not a
    defect). A mesh generating more candidate pairs than max_pairs is
    refused by ValueError, never silently truncated.
    """
    from rtree import index

    tris = mesh.triangles                       # (n, 3, 3)
    lo = tris.min(axis=1)
    hi = tris.max(axis=1)
    prop = index.Property()
    prop.dimension = 3
    boxes = ((int(i), tuple(lo[i]) + tuple(hi[i]), None)
             for i in range(len(tris)))
    idx = index.Index(boxes, properties=prop)

    faces = mesh.faces
    cand_a: list[int] = []
    cand_b: list[int] = []
    for i in range(len(tris)):
        for j in idx.intersection(tuple(lo[i]) + tuple(hi[i])):
            if j <= i:
                continue
            if len(set(faces[i]) & set(faces[j])):
                continue
            cand_a.append(i)
            cand_b.append(int(j))
    if not cand_a:
        return 0
    if len(cand_a) > max_pairs:
        raise ValueError(
            f"self-intersection narrow phase would test {len(cand_a)} "
            f"candidate pairs (> {max_pairs}); mesh too large for intake")
    hits = _pairs_intersect(tris[np.asarray(cand_a)], tris[np.asarray(cand_b)])
    return int(hits.sum())


CHAMBER_MM = 60.0


def intake_verdict(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """Run the refusal rules; return the JSON-serializable verdict record."""
    # chamber fit first: an import-time fact, refused at import time
    size_mm = (mesh.bounds[1] - mesh.bounds[0]).tolist()
    fit_ok = bool(all(s < CHAMBER_MM for s in size_mm))
    fit_check = {"ok": fit_ok,
                 "bbox_mm": [round(float(s), 2) for s in size_mm],
                 "chamber_mm": CHAMBER_MM}
    if not fit_ok:
        return {
            "accepted": False,
            "error": (f"REFUSED: part bbox "
                      f"{[round(float(s), 1) for s in size_mm]} mm does not "
                      f"fit the 60 mm RFAM chamber. Grading and the 3-D "
                      "simulation are chamber-limited; scale the part down "
                      "or import a chamber-sized piece of it."),
            "checks": {"chamber_fit": fit_check},
        }
    watertight_ok = bool(mesh.is_watertight)
    n_open = 0 if watertight_ok else _open_edge_count(mesh)
    checks: Dict[str, Any] = {
        "chamber_fit": fit_check,
        "watertight": {"ok": watertight_ok, "n_open_edges": n_open},
    }
    if not watertight_ok:
        # a leaking shell makes downstream volume modeling meaningless
        checks["self_intersection"] = {"ok": None,
                                       "n_intersecting_pairs": None,
                                       "skipped": "mesh not watertight"}
        return {
            "accepted": False,
            "error": (f"REFUSED: mesh is not watertight ({n_open} open "
                      "edges). Repair the mesh; grading a leaking volume "
                      "would model geometry that does not exist."),
            "checks": checks,
        }
    n_pairs = count_self_intersections(mesh)
    si_ok = n_pairs == 0
    checks["self_intersection"] = {"ok": si_ok,
                                   "n_intersecting_pairs": n_pairs}
    if not si_ok:
        return {
            "accepted": False,
            "error": (f"REFUSED: mesh self-intersects ({n_pairs} "
                      "intersecting face pairs). Repair the mesh before "
                      "grading."),
            "checks": checks,
        }
    return {"accepted": True, "error": None, "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description="Grade intake refusal gate")
    ap.add_argument("mesh")
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()
    try:
        mesh = trimesh.load_mesh(args.mesh)
        verdict = intake_verdict(mesh)
    except Exception as e:  # loud refusal on unreadable input too
        logger.error("intake failed for %s: %s", args.mesh, e)
        verdict = {"accepted": False,
                   "error": f"REFUSED: could not evaluate mesh ({e})",
                   "checks": {}}
    Path(args.json_out).write_text(json.dumps(verdict, indent=1))
    print(json.dumps({"accepted": verdict["accepted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
