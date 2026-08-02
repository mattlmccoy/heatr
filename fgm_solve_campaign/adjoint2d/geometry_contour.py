"""Mask to polygon, exactly, plus a dependency-light inside test.

WHY EXACT AND NOT SMOOTHED. The solve stack is polygon-based from end to end:
`rfam_eqs_coupled.make_domain` rasterizes a polygon into the part mask and the
material fill fraction, and `adjoint2d.chi_area` supersamples the SAME polygon
into the target indicator. An imported geometry that arrives as pixels (a
slicer mask, a portable network graphics (PNG) mask, a rasterized
stereolithography (STL) layer) therefore needs exactly one conversion, and that
conversion must not invent or lose material. A smoothed or spline-fitted
outline would move the boundary by a fraction of a pixel everywhere, which is
the same class of error the area-fill target was introduced to remove.

THE CONSTRUCTION. The pixel region is a union of rectangles, so its boundary is
the set of pixel edges that are NOT shared by two inside pixels. Those edges,
emitted counter-clockwise around each inside pixel, stitch into closed loops
with no ambiguity except at diagonal pixel touches, which are resolved by the
straight-ahead-first rule below. Collinear runs are then merged, which turns a
long straight edge into two vertices while leaving every staircase step intact.
The result is the pixel region itself, to the last bit: `signed_area` of the
loops equals the pixel count times the cell area exactly.

HOLES ARE REFUSED, LOUDLY. `chi_area.area_fill_union` combines parts with
`np.maximum`, reproducing `rfam_eqs_coupled.py:1595`, and a maximum cannot
express a hole: an annulus would come back as a filled disc. Rather than return
a target indicator that silently disagrees with the imported geometry, a
clockwise (interior) loop raises `UnsupportedGeometry`. Multiply-connected
geometry needs a subtractive fill rule and that is not built here.

ORIENTATION CONVENTION. Outer loops are counter-clockwise, positive signed
area, in the (x, y) frame where x increases with the column index and y with
the row index, which is the frame of `rfam_eqs_coupled.make_domain`'s axes.
Downstream, winding does not matter at all: the area fill uses the even-odd
rule (see `tests/fill_contract.py`), so a reversed ring gives the same fill.
"""
from __future__ import annotations

import numpy as np

__all__ = ["UnsupportedGeometry", "mask_to_polygons", "signed_area",
           "point_in_polygon_evenodd", "merge_collinear"]


class UnsupportedGeometry(RuntimeError):
    """The imported geometry uses a feature this intake cannot represent."""


def signed_area(poly: np.ndarray) -> float:
    """Shoelace area; positive for counter-clockwise rings."""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def merge_collinear(poly: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """Drop vertices whose two incident edges are parallel."""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return p
    prev = p - np.roll(p, 1, axis=0)
    nxt = np.roll(p, -1, axis=0) - p
    cross = prev[:, 0] * nxt[:, 1] - prev[:, 1] * nxt[:, 0]
    keep = np.abs(cross) > atol
    return p[keep] if keep.sum() >= 3 else p


def _boundary_edges(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Counter-clockwise pixel edges not shared by two inside pixels.

    Corner (I, J) is the lower-left corner of pixel (row J, column I), so pixel
    (j, i) has corners (i, j), (i+1, j), (i+1, j+1), (i, j+1).
    """
    m = np.asarray(mask, dtype=bool)
    ny, nx = m.shape
    pad = np.zeros((ny + 2, nx + 2), dtype=bool)
    pad[1:-1, 1:-1] = m
    inner = pad[1:-1, 1:-1]
    south = inner & ~pad[:-2, 1:-1]
    north = inner & ~pad[2:, 1:-1]
    west = inner & ~pad[1:-1, :-2]
    east = inner & ~pad[1:-1, 2:]

    out: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add(a: tuple[int, int], b: tuple[int, int]) -> None:
        out.setdefault(a, []).append(b)

    for jj, ii in zip(*np.nonzero(south)):
        add((int(ii), int(jj)), (int(ii) + 1, int(jj)))
    for jj, ii in zip(*np.nonzero(east)):
        add((int(ii) + 1, int(jj)), (int(ii) + 1, int(jj) + 1))
    for jj, ii in zip(*np.nonzero(north)):
        add((int(ii) + 1, int(jj) + 1), (int(ii), int(jj) + 1))
    for jj, ii in zip(*np.nonzero(west)):
        add((int(ii), int(jj) + 1), (int(ii), int(jj)))
    return out


def _stitch(edges: dict[tuple[int, int], list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    """Walk the edge multigraph into closed loops, straight ahead first.

    At a diagonal pixel touch a corner carries two outgoing edges. Continuing
    STRAIGHT AHEAD when that is available keeps the two touching blocks as two
    separate loops, which is the reading that matches the even-odd fill of the
    pixel region. Turning instead would merge them through a zero-width neck.
    """
    remaining = {k: list(v) for k, v in edges.items()}
    loops: list[list[tuple[int, int]]] = []
    starts = sorted(remaining)
    for s in starts:
        while remaining.get(s):
            loop = [s]
            cur = s
            prev_dir = None
            while True:
                outs = remaining.get(cur)
                if not outs:
                    raise UnsupportedGeometry(
                        "the pixel boundary does not close; this is a tracer bug")
                pick = 0
                if prev_dir is not None and len(outs) > 1:
                    for k, nxt in enumerate(outs):
                        d = (nxt[0] - cur[0], nxt[1] - cur[1])
                        if d == prev_dir:
                            pick = k
                            break
                nxt = outs.pop(pick)
                if not outs:
                    remaining.pop(cur, None)
                prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
                if nxt == loop[0]:
                    break
                loop.append(nxt)
                cur = nxt
            loops.append(loop)
    return loops


def mask_to_polygons(mask: np.ndarray, dx: float, dy: float,
                     x0: float, y0: float,
                     merge: bool = True) -> list[np.ndarray]:
    """Exact staircase polygons of a binary mask, in physical coordinates.

    Args:
        mask: boolean array, `[row, col]`, row index along y.
        dx, dy: cell size in metres.
        x0, y0: physical coordinates of the CENTRE of pixel (0, 0).
        merge: drop collinear vertices (default true).

    Returns:
        One counter-clockwise polygon per connected component, largest first.

    Raises:
        UnsupportedGeometry: the mask is empty, or it has an interior hole.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        raise UnsupportedGeometry("the imported mask is empty")
    loops_ij = _stitch(_boundary_edges(m))
    polys: list[np.ndarray] = []
    for loop in loops_ij:
        arr = np.asarray(loop, dtype=float)
        xy = np.column_stack([x0 + (arr[:, 0] - 0.5) * float(dx),
                              y0 + (arr[:, 1] - 0.5) * float(dy)])
        if merge:
            xy = merge_collinear(xy)
        a = signed_area(xy)
        if a < 0.0:
            raise UnsupportedGeometry(
                "the imported mask has an interior hole (a clockwise boundary "
                "loop). The union-by-maximum fill rule of "
                "chi_area.area_fill_union cannot express a hole, so the target "
                "indicator would silently fill it. Multiply-connected geometry "
                "needs a subtractive fill rule, which is not built.")
        polys.append(xy)
    polys.sort(key=lambda p: -signed_area(p))
    return polys


def point_in_polygon_evenodd(poly: np.ndarray, px: np.ndarray,
                             py: np.ndarray) -> np.ndarray:
    """Even-odd (crossing-number) inside test, numpy only.

    Provided so a lane without matplotlib can evaluate the same fill. It is
    checked point for point against `matplotlib.path.Path.contains_points` on a
    concave star in `test_geometry_contour.py`, because the two must agree or
    the two lanes disagree about what "inside" means.
    """
    p = np.asarray(poly, dtype=float)
    x = np.asarray(px, dtype=float).ravel()
    y = np.asarray(py, dtype=float).ravel()
    x1, y1 = p[:, 0][None, :], p[:, 1][None, :]
    x2, y2 = np.roll(p[:, 0], -1)[None, :], np.roll(p[:, 1], -1)[None, :]
    yy = y[:, None]
    straddles = (y1 > yy) != (y2 > yy)
    denom = np.where(np.abs(y2 - y1) > 0.0, y2 - y1, 1.0)
    xint = (x2 - x1) * (yy - y1) / denom + x1
    crosses = straddles & (x[:, None] < xint)
    return (np.sum(crosses, axis=1) % 2) == 1
