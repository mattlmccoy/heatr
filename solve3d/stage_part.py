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

import argparse
import json
import math
import os
import subprocess
import sys
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


def build_spec(map_npz, stl_path, *, voxel_mm: float = 0.25,
               chamber_mm: float | None = None, pad: float = 1.25,
               build_axis: str = "z", base: str = "min") -> dict:
    """Staging spec (SOLVE_cont (nz, ny, nx), base at k = 0) spanning EXACTLY the
    chamber canvas, centred where stage_lib._stl_layer_masks centres the part (the
    STL triangle-vertex mean), and the STL's full z-span."""
    if not voxel_mm > 0:
        raise ValueError(f"voxel_mm must be > 0, got {voxel_mm}")
    mesh = trimesh.load(str(stl_path), force="mesh")
    reg = register(load_map(map_npz), mesh, build_axis, base)
    tri = np.asarray(mesh.triangles).reshape(-1, 3)
    cx, cy = float(tri[:, 0].mean()), float(tri[:, 1].mean())
    lo, hi = mesh.bounds
    need = 2.0 * max(abs(lo[0] - cx), abs(hi[0] - cx), abs(lo[1] - cy), abs(hi[1] - cy))
    if chamber_mm is None:
        chamber_mm = math.ceil(pad * need * 10.0) / 10.0
    chamber_mm = float(chamber_mm)
    if chamber_mm < need:
        raise ValueError(f"chamber_mm={chamber_mm} cannot hold the part "
                         f"(needs >= {need:.2f} mm around its centre)")
    n_xy = max(8, int(round(chamber_mm / voxel_mm)))
    height = float(hi[2] - lo[2])
    nz = max(2, int(math.ceil(height / voxel_mm)))
    dxy, dz = chamber_mm / n_xy, height / nz
    xs = cx - chamber_mm / 2 + (np.arange(n_xy) + 0.5) * dxy
    ys = cy - chamber_mm / 2 + (np.arange(n_xy) + 0.5) * dxy
    zs = lo[2] + (np.arange(nz) + 0.5) * dz
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")               # (nz, ny, nx)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    in_box = np.flatnonzero(np.all((pts >= lo) & (pts <= hi), axis=1))
    inside = np.zeros(len(pts), bool)
    for s in range(0, len(in_box), 200_000):
        chunk = in_box[s:s + 200_000]
        inside[chunk] = mesh.contains(pts[chunk])
    from scipy.spatial import cKDTree
    _, near = cKDTree(reg["points_mm"]).query(pts[inside])
    sol = np.zeros(len(pts), np.float32)
    sol[inside] = np.clip(reg["s_map"][near], 0.0, 1.0)
    shape = (nz, n_xy, n_xy)
    return {"SOLVE_cont": sol.reshape(shape), "part_mask": inside.reshape(shape),
            "proxy_field": "solve", "domain_mm": chamber_mm, "z_mm": height,
            "voxel_mm": [dz, dxy, dxy], "registration": reg["report"],
            "source_map": str(map_npz), "stl": str(stl_path)}


def write_spec(spec: dict, out_npz) -> Path:
    """Save a build_spec() result as a stage_job-ready npz (no pickled objects)."""
    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, SOLVE_cont=spec["SOLVE_cont"], part_mask=spec["part_mask"],
                        proxy_field=spec["proxy_field"], domain_mm=spec["domain_mm"],
                        z_mm=spec["z_mm"], voxel_mm=np.asarray(spec["voxel_mm"], float),
                        registration=json.dumps(spec["registration"]),
                        source_map=spec["source_map"], stl=spec["stl"])
    return out


def _default_meteor_tools() -> Path:
    env = os.environ.get("RFAM_METEOR_TOOLS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "software" / "meteor" / "tools"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="solve3d DG0 map npz")
    ap.add_argument("--stl", required=True, help="the part STL (mm), as solved")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--densify", metavar="FIELDS_NPZ",
                   help="densify march fields.npz: field-based factor (preferred)")
    g.add_argument("--densify-factor", type=float,
                   help="manual green->dense Z factor (recorded as 'manual')")
    g.add_argument("--no-densification", action="store_true",
                   help="the STL IS the green shape (recorded as 'none')")
    ap.add_argument("--hot-folder", required=True)
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--layer-height", type=float, default=0.2)
    ap.add_argument("--voxel-mm", type=float, default=0.25)
    ap.add_argument("--chamber-mm", type=float, default=None)
    ap.add_argument("--build-axis", choices=["x", "y", "z"], default="z")
    ap.add_argument("--base", choices=["min", "max"], default="min")
    ap.add_argument("--work-dir", default=None,
                    help="spec/summary/report dir (default solve3d/results/stage_inputs/<job>)")
    ap.add_argument("--meteor-tools", default=None)
    args = ap.parse_args(argv)

    tools = Path(args.meteor_tools) if args.meteor_tools else _default_meteor_tools()
    for f in ("stage_job.py", "preflight.py"):
        if not (tools / f).is_file():
            print(f"error: {tools / f} not found (set --meteor-tools or "
                  "RFAM_METEOR_TOOLS)", file=sys.stderr)
            return 2
    work = (Path(args.work_dir) if args.work_dir else
            Path(__file__).resolve().parent / "results" / "stage_inputs" / args.job_name)
    work.mkdir(parents=True, exist_ok=True)
    try:
        spec = build_spec(args.map, args.stl, voxel_mm=args.voxel_mm,
                          chamber_mm=args.chamber_mm, build_axis=args.build_axis,
                          base=args.base)
    except ValueError as e:                     # includes RegistrationError
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    spec_npz = write_spec(spec, work / f"{args.job_name}_spec.npz")
    report_json = work / "stage_report.json"
    cmd = [sys.executable, str(tools / "stage_job.py"), str(spec_npz), "--3d",
           "--stl", str(args.stl), "--chamber-mm", repr(spec["domain_mm"]),
           "--layer-height", repr(args.layer_height),
           "--hot-folder", str(args.hot_folder), "--job-name", args.job_name,
           "--report-json", str(report_json)]
    if args.densify:
        from solve3d import densify_summary as ds
        cmd += ["--densify-summary",
                str(ds.write_summary(args.densify, work / "densify_summary.json"))]
    elif args.densify_factor is not None:
        cmd += ["--z-densification", repr(args.densify_factor)]
    r = subprocess.run(cmd, cwd=str(tools), capture_output=True, text=True)
    if r.returncode != 0 or not report_json.is_file():
        sys.stderr.write(r.stdout + r.stderr)
        print(f"STAGE FAILED (stage_job exit {r.returncode})", file=sys.stderr)
        return 1
    rep = json.loads(report_json.read_text())
    pf = subprocess.run([sys.executable, str(tools / "preflight.py"), rep["out_dir"],
                         "--json"], cwd=str(tools), capture_output=True, text=True)
    pre = (json.loads(pf.stdout)[0] if pf.stdout.strip()
           else {"ready": False, "errors": [pf.stderr.strip()], "warnings": []})
    out = {"job": args.job_name, "out_dir": rep["out_dir"],
           "staged_all_pass": rep.get("all_pass"),
           "print_layers": rep.get("print_layers"),
           "printed_z_mm": rep.get("printed_z_mm"),
           "z_mode": rep.get("provenance", {}).get("z", {}).get("mode"),
           "factor": rep.get("z_densification"),
           "registration": spec["registration"],
           "preflight_ready": pre.get("ready"),
           "preflight_errors": pre.get("errors"),
           "preflight_warnings": pre.get("warnings")}
    print(json.dumps(out, indent=2))
    return 0 if (rep.get("all_pass") and pre.get("ready")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
