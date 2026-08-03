"""Pure-numpy geometry front end for the Studio direct-solve service (spec 7e).

No dolfinx, no scipy, no heatr3d: this module is importable from BOTH
environments and is where every decision that can be checked WITHOUT running a
solve lives, so those decisions are unit-tested rather than argued.

Three jobs, in the order `studio_solve` performs them:

1. EXTRUSION DETECTION. Spec 7e forbids shape heuristics: a part is extruded
   iff every occupied z-slice is voxel-for-voxel identical. Anything else is
   REFUSED with a message naming the missing piece (Phase E STL tet meshing).

2. THE OUTLINE. The conforming mesh is built from the mid-slice outline, and
   that outline is a SUB-CELL marching-squares contour of the 2-D mask, not the
   voxel staircase. On a binary field the 0.5 level-set crossing of a dual-grid
   edge is exactly its midpoint, so every contour vertex sits half a cell
   outside the outermost occupied cell centre and the staircase corners are cut.
   Segments are oriented inside-on-the-LEFT, so an outer ring comes out
   counter-clockwise (positive signed area) and a hole clockwise -- the ring
   classification is a by-product of the orientation rule, not a second guess.

   The saddle cases (diagonally opposite corners inside) are resolved by pairing
   each "exit" crossing with the NEXT "entry" crossing in counter-clockwise
   order, which keeps the INSIDE connected through the cell centre. Stated here
   because a saddle rule is a real modelling choice, and the opposite choice
   would split a diagonal neck into two bodies.

3. THE ARTIFACT. `write_map_npz` is the Studio's consumer contract
   (studio3d/correction.py reads centroids / s_map / volumes), so its key set is
   pinned by a test rather than by convention.

Grid convention throughout: the heatr3d Grid, chamber L = 0.060 m, cell centres
c_k = (k + 0.5) h - L/2, part[i, j, k] indexed (x, y, z).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

L_DOMAIN = 0.060

REFUSAL_NOT_EXTRUDED = (
    "REFUSED: this part is not an extrusion. The direct-solve service builds "
    "its conforming mesh by extruding one cross-section, so it only accepts "
    "parts whose occupied z-slices are voxel-for-voxel identical "
    "({n_bad} of {n_tot} occupied slices differ from the first, first "
    "mismatch at z index {k_bad}). Non-extruded parts need STL tet meshing, "
    "which the D1 spike left explicitly unsettled and which lands with Phase E "
    "proper (spec section 7e, ALL-GEOMETRY requirement). No shape heuristic is "
    "applied and no approximate extrusion is substituted.")


class OutlineError(ValueError):
    """The mid-slice outline is not a single body with optional holes."""


class NotExtrudedError(ValueError):
    """The part failed the extrusion detection (see REFUSAL_NOT_EXTRUDED)."""


# --------------------------------------------------------------------------- #
# 1. Extrusion detection
# --------------------------------------------------------------------------- #
def detect_extrusion(part: np.ndarray) -> dict:
    """Is `part` (n, n, n) an extrusion along z? DETECTED, not assumed."""
    part = np.asarray(part, dtype=bool)
    occ = np.flatnonzero(part.any(axis=(0, 1)))
    if occ.size == 0:
        return {"is_extruded": False, "n_occupied_slices": 0,
                "z_index_lo": None, "z_index_hi": None,
                "n_mismatched_slices": 0, "first_mismatch_z": None,
                "refusal": "REFUSED: the part is empty (no occupied voxel)."}
    ref = part[:, :, occ[0]]
    bad = [int(k) for k in occ if not np.array_equal(part[:, :, k], ref)]
    contiguous = bool(occ[-1] - occ[0] + 1 == occ.size)
    rec = {"n_occupied_slices": int(occ.size),
           "z_index_lo": int(occ[0]), "z_index_hi": int(occ[-1]),
           "z_indices_contiguous": contiguous,
           "n_mismatched_slices": len(bad),
           "first_mismatch_z": (bad[0] if bad else None)}
    ok = (not bad) and contiguous
    rec["is_extruded"] = bool(ok)
    rec["refusal"] = None if ok else REFUSAL_NOT_EXTRUDED.format(
        n_bad=max(len(bad), 1), n_tot=int(occ.size),
        k_bad=(bad[0] if bad else "n/a (occupied slices are not contiguous)"))
    return rec


def z_extent(part: np.ndarray, h: float, L: float = L_DOMAIN) -> tuple:
    """Physical [z_lo, z_hi] of the occupied slab (voxel faces, not centres)."""
    occ = np.flatnonzero(np.asarray(part, bool).any(axis=(0, 1)))
    zc = (np.arange(part.shape[2]) + 0.5) * h - L / 2.0
    return float(zc[occ[0]] - h / 2.0), float(zc[occ[-1]] + h / 2.0)


def mid_slice(part: np.ndarray) -> np.ndarray:
    occ = np.flatnonzero(np.asarray(part, bool).any(axis=(0, 1)))
    return np.asarray(part, bool)[:, :, occ[occ.size // 2]]


# --------------------------------------------------------------------------- #
# 2. Marching-squares outline
# --------------------------------------------------------------------------- #
# Dual cell (i, j) has corners a=(i,j), b=(i+1,j), c=(i+1,j+1), d=(i,j+1) and
# edges, traversed counter-clockwise: 0 = a->b, 1 = b->c, 2 = c->d, 3 = d->a.
_EDGE_ENDS = ((0, 1), (1, 2), (2, 3), (3, 0))       # corner index pairs
# doubled-integer coordinates of each edge midpoint, relative to corner a
_EDGE_MID = ((1, 0), (2, 1), (1, 2), (0, 1))


def _segments(mask: np.ndarray) -> list:
    """(start_vertex, end_vertex) pairs in doubled integer index space."""
    m = np.asarray(mask, bool)
    segs = []
    ni, nj = m.shape
    corner = lambda ii, jj: (bool(m[ii, jj]) if 0 <= ii < ni and 0 <= jj < nj
                             else False)
    for i in range(-1, ni):
        for j in range(-1, nj):
            v = (corner(i, j), corner(i + 1, j),
                 corner(i + 1, j + 1), corner(i, j + 1))
            if all(v) or not any(v):
                continue
            exits, entries = [], []
            for e, (p, q) in enumerate(_EDGE_ENDS):
                if v[p] and not v[q]:
                    exits.append(e)
                elif v[q] and not v[p]:
                    entries.append(e)
            for e in exits:
                # pair with the NEXT entry counter-clockwise (saddle rule:
                # keeps the INSIDE connected through the cell centre)
                nxt = min(((f - e) % 4, f) for f in entries)[1]
                p0 = (2 * i + _EDGE_MID[e][0], 2 * j + _EDGE_MID[e][1])
                p1 = (2 * i + _EDGE_MID[nxt][0], 2 * j + _EDGE_MID[nxt][1])
                segs.append((p0, p1))
    return segs


def _chain(segs: list) -> list:
    nxt = {}
    for a, b in segs:
        if a in nxt:
            raise OutlineError(
                "marching-squares chaining found two segments leaving the same "
                f"vertex {a}; the mask is not a manifold boundary")
        nxt[a] = b
    rings, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        ring, v = [], start
        while v not in seen:
            seen.add(v)
            ring.append(v)
            v = nxt[v]
        if v == start and len(ring) >= 3:
            rings.append(ring)
    return rings


def outline_rings(mask2d: np.ndarray, h: float, L: float = L_DOMAIN) -> list:
    """Sub-cell contour rings of a 2-D voxel mask, in PHYSICAL metres.

    Returns [outer_ring, hole, hole, ...]; the outer ring is counter-clockwise
    (positive signed area), holes clockwise. Raises OutlineError for a mask with
    more than one connected body -- the service refuses instead of guessing.
    """
    mask2d = np.asarray(mask2d, bool)
    if not mask2d.any():
        raise OutlineError("empty slice: no outline to extract")
    rings_idx = _chain(_segments(mask2d))
    # doubled index -> physical: index 0 is the centre of cell 0, and one
    # doubled step is h/2. The padded ring at index -1 is handled by the same
    # affine map, which is why the contour can sit half a cell outside the part.
    x0 = 0.5 * h - L / 2.0
    rings = [_simplify(np.column_stack([x0 + np.asarray([p[0] for p in r]) * h / 2.0,
                                        x0 + np.asarray([p[1] for p in r]) * h / 2.0]))
             for r in rings_idx]
    outer = [r for r in rings if ring_area(r) > 0.0]
    if len(outer) != 1:
        raise OutlineError(
            f"the slice has {len(outer)} outer rings (disconnected bodies). "
            "The direct-solve service meshes ONE extruded body; split the part "
            "or use the Phase E path.")
    holes = [r for r in rings if ring_area(r) < 0.0]
    return [outer[0]] + holes


def _simplify(pts: np.ndarray) -> np.ndarray:
    """Drop collinear vertices (exact, on an axis-aligned half-cell lattice)."""
    p = np.asarray(pts, float)
    keep = []
    n = p.shape[0]
    for i in range(n):
        a, b, c = p[i - 1], p[i], p[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > 1e-18:
            keep.append(i)
    return p[keep] if len(keep) >= 3 else p


def ring_area(ring: np.ndarray) -> float:
    """Signed shoelace area [m^2]: > 0 counter-clockwise (outer), < 0 hole."""
    p = np.asarray(ring, float)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def rings_area(rings: list) -> float:
    """Net enclosed area of an outer ring plus its holes [m^2]."""
    return float(sum(ring_area(r) for r in rings))


def points_in_rings(rings: list, px, py) -> np.ndarray:
    """Even-odd inside test across ALL rings (outer XOR holes).

    Reuses the shared fill contract's point-in-polygon (solve3d.fill, which is
    what `fgm_solve_campaign/adjoint2d/tests/fill_contract.py` checks) once per
    ring. Even-odd parity across rings is exactly the with-holes rule: a point
    in a bore crosses both rings, so it is outside.
    """
    from solve3d import fill
    px = np.asarray(px, float).ravel()
    inside = np.zeros(px.size, dtype=bool)
    for r in rings:
        inside ^= fill.points_in_polygon(np.asarray(r, float), px,
                                         np.asarray(py, float).ravel())
    return inside


# --------------------------------------------------------------------------- #
# 3. Parts and artifacts
# --------------------------------------------------------------------------- #
def make_tube_part(n: int = 32, r_outer_m: float = 0.012,
                   r_bore_m: float = 0.005, height_m: float = 0.020,
                   L: float = L_DOMAIN) -> np.ndarray:
    """A tube voxel part on the heatr3d Grid, by the runner's convention:
    containment tested at cell centres, part centred on the chamber centre."""
    h = L / n
    c = (np.arange(n) + 0.5) * h - L / 2.0
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.hypot(X, Y)
    return ((r <= r_outer_m) & (r >= r_bore_m)
            & (np.abs(Z) <= height_m / 2.0))


def write_map_npz(path, centroids: np.ndarray, s_map: np.ndarray,
                  volumes: np.ndarray, v_raw: np.ndarray) -> Path:
    """The delivered DG0 artifact -- the SAME keys as phase_c_map_*.npz, which
    is what studio3d/correction.py reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, centroids=np.asarray(centroids, float),
                        s_map=np.asarray(s_map, float),
                        volumes=np.asarray(volumes, float),
                        v_raw=np.asarray(v_raw, float))
    return path
