"""Stage ANY solved part: (solve3d DG0 dopant map + the part's STL) -> a validated,
pre-flighted MetPrint job, in one command.

    ./.venv312/bin/python -m solve3d.stage_part \\
        --map solve3d/phase_e/results/map_pyramid_solve_filter_only.npz \\
        --stl shape_library_3d/stl/pyramid.stl \\
        --densify solve3d/results/densify_pyramid/fields.npz \\
        --hot-folder "<...>/rfam-web/Hot Folder" --job-name pyramid_graded_3d

Geometry comes from the STL (authoritative, crisp). The dopant comes from the
map's cell centroids, REGISTERED into the STL frame and PROVEN to be the same
solid in the same pose before anything is staged:
  * units    : map centroids are metres, the STL is millimetres
  * rotation : the declared solve-frame build axis becomes +z and the declared
               base end goes to low z -- PROPER rotations only, never a mirror
  * shift    : the map's volume-weighted centroid moves onto the STL centre of
               mass (both are the true centroid of the same solid, so the
               half-cell inset of cell centres does not bias it)
  * proof    : volumes agree within 3 %, extents agree within ~1.5 cells, and
               the per-axis third-moment skew agrees -- a part on its side or
               upside down is REFUSED, never silently staged. (A part symmetric
               top-to-bottom, e.g. a cube, cannot be checked for a flip; its
               declared --build-axis/--base are trusted and the report says so.)
Scale and rotation on the bed are deliberately not offered: the solve's
orientation relative to the RF field IS the print orientation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

VOLUME_TOL = 0.03          # map vs STL volume
SKEW_ACTIVE = 0.10         # |STL skew| above which an axis is asymmetric enough to check
SKEW_TOL = 0.35            # allowed per-axis skew disagreement, map vs STL
_ORIENT_GRID = 40          # deterministic STL interior sample grid per axis

_BUILD = {  # proper rotations taking the declared build axis to +z
    "z": np.eye(3),
    "y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),   # (x, -z, y)
    "x": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float),   # (-z, y, x)
}
_FLIP = np.diag([1.0, -1.0, -1.0])   # 180 deg about x: base end at max -> low z


class RegistrationError(ValueError):
    """The map and the STL are not provably the same solid in the same pose."""


def load_map(map_npz) -> dict:
    """Load a solve3d DG0 dopant map (cell s_map + centroids + volumes).

    That format is the 3-D backsolve's output; proxy-inversion maps are 2-D grids
    and fail the key check."""
    d = np.load(Path(map_npz))
    missing = [k for k in ("s_map", "centroids", "volumes") if k not in d.files]
    if missing:
        raise ValueError(f"{map_npz}: missing {missing}; not a solve3d DG0 dopant map")
    c = np.asarray(d["centroids"], float)
    s = np.asarray(d["s_map"], float)
    v = np.asarray(d["volumes"], float)
    if c.ndim != 2 or c.shape[1] != 3 or not (len(c) == len(s) == len(v)):
        raise ValueError(f"{map_npz}: inconsistent map arrays "
                         f"{c.shape}, {s.shape}, {v.shape}")
    if not np.all(v > 0):
        raise ValueError(f"{map_npz}: non-positive cell volumes")
    return {"centroids_m": c, "s_map": s, "volumes_m3": v}


def rotation(build_axis: str = "z", base: str = "min") -> np.ndarray:
    """Proper rotation taking the solve frame to the print frame (column form)."""
    if build_axis not in _BUILD:
        raise ValueError(f"build_axis must be one of x/y/z, got {build_axis!r}")
    if base not in ("min", "max"):
        raise ValueError(f"base must be 'min' or 'max', got {base!r}")
    R = _BUILD[build_axis]
    return _FLIP @ R if base == "max" else R


def _skew(x: np.ndarray, w: np.ndarray) -> float:
    mu = np.average(x, weights=w)
    sd = math.sqrt(float(np.average((x - mu) ** 2, weights=w)))
    return 0.0 if sd == 0 else float(np.average(((x - mu) / sd) ** 3, weights=w))


def _stl_interior_points(mesh: trimesh.Trimesh, n: int = _ORIENT_GRID) -> np.ndarray:
    lo, hi = mesh.bounds
    axes = [lo[i] + (np.arange(n) + 0.5) * (hi[i] - lo[i]) / n for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return P[mesh.contains(P)]


def register(map_d: dict, mesh: trimesh.Trimesh, build_axis: str = "z",
             base: str = "min") -> dict:
    """Place the map's cells in the STL frame and prove it, or raise."""
    if not mesh.is_watertight:
        raise RegistrationError("STL is not watertight: its volume and interior are undefined")
    R = rotation(build_axis, base)
    p = (map_d["centroids_m"] * 1e3) @ R.T
    w = map_d["volumes_m3"]
    shift = np.asarray(mesh.center_mass, float) - np.average(p, axis=0, weights=w)
    p = p + shift
    v_map, v_stl = float(w.sum()) * 1e9, float(mesh.volume)
    rel_v = abs(v_map - v_stl) / v_stl
    cell = (float(np.mean(w)) * 1e9) ** (1.0 / 3.0)
    lo, hi = mesh.bounds
    ext_ok = bool(np.all(p.min(0) >= lo - 1.5 * cell)
                  and np.all(p.max(0) <= hi + 1.5 * cell)
                  and np.all(np.ptp(p, axis=0) >= (hi - lo) - 4.0 * cell))
    q = _stl_interior_points(mesh)
    sk_map = np.array([_skew(p[:, i], w) for i in range(3)])
    sk_stl = np.array([_skew(q[:, i], np.ones(len(q))) for i in range(3)])
    active = np.abs(sk_stl) > SKEW_ACTIVE
    orient_ok = bool(np.all(np.abs(sk_map - sk_stl)[active] < SKEW_TOL)
                     and np.all(np.abs(sk_map)[~active] < SKEW_TOL))
    report = {"build_axis": build_axis, "base": base,
              "shift_mm": [round(float(x), 4) for x in shift],
              "volume_map_mm3": round(v_map, 2), "volume_stl_mm3": round(v_stl, 2),
              "volume_rel_err": round(rel_v, 5), "cell_mm": round(cell, 4),
              "extent_ok": ext_ok,
              "skew_map": [round(float(x), 3) for x in sk_map],
              "skew_stl": [round(float(x), 3) for x in sk_stl],
              "orientation_checked": bool(active.any()),
              "orientation_ok": orient_ok}
    problems = []
    if rel_v > VOLUME_TOL:
        problems.append(f"volume map {v_map:.1f} vs STL {v_stl:.1f} mm3 "
                        f"({100 * rel_v:.1f}% > {100 * VOLUME_TOL:.0f}%): not the "
                        "same solid, or not the same units")
    if not ext_ok:
        problems.append("registered map extents do not match the STL bounding box")
    if not orient_ok:
        problems.append(f"shape skew map {report['skew_map']} vs STL "
                        f"{report['skew_stl']}: the declared build axis/base puts the "
                        "part on its side or upside down")
    if problems:
        raise RegistrationError("; ".join(problems)
                                + f" [declared build_axis={build_axis}, base={base}]")
    return {"points_mm": p, "s_map": map_d["s_map"], "report": report}
