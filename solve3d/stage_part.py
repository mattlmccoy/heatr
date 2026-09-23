"""Stage ANY solved part: (solve3d DG0 dopant map + the part's STL) -> a validated,
pre-flighted MetPrint job, in one command.

    ./.venv312/bin/python -m solve3d.stage_part \\
        --map solve3d/phase_e/results/map_pyramid_solve_filter_only.npz \\
        --stl shape_library_3d/stl/pyramid.stl \\
        --densify solve3d/results/densify_pyramid/fields.npz \\
        --hot-folder "<...>/rfam-web/Hot Folder" --job-name pyramid_graded_3d \\
        --layer-height 0.2

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
               the full scale-normalised 2nd + 3rd central moments agree -- a
               part on its side, upside down or rotated in plane is REFUSED,
               never silently staged. (A part with axis-rotation symmetry, e.g.
               a square pyramid (4) or a cube (24), cannot be checked beyond that
               symmetry; the report counts it and says so in pose_note.)
Scale and rotation on the bed are deliberately not offered: the solve's
orientation relative to the RF field IS the print orientation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

VOLUME_TOL = 0.03          # map vs STL volume
SEC_TOL = 0.05             # max abs diff of normalised 2nd central moments, map vs STL
THR_TOL = 0.25             # max abs diff of normalised 3rd central moments
# Densify march vs STL. The march is a COARSE voxelisation (h = 1.25 mm, 13-19
# voxels across), so it gets its own moment tolerances. Measured 2026-09-22:
#   real marches (solve3d/results/densify_*/fields.npz vs shape_library_3d/stl):
#     same part : cube 0.000/0.000, pyramid 0.042/0.035        (2nd/3rd order)
#     cross part: cube-march vs pyramid.stl 0.182/0.639,
#                 pyramid-march vs cube.stl 0.224/0.604
#   every library STL voxelised at 1.25 mm, 4 grid offsets (same-part worst):
#     pipe 0.064/0.059, cylinder 0.061/0.005, lattice 0.005/0.161
# Tolerances sit near the geometric middle of the worst same-part and the
# closest real cross-part errors. Normalised 2nd moments cannot separate
# isotropic solids (cube, spheres, truncated octahedron); for those the physical
# extent check carries the proof, and it separates cube from the spheres
# (16.1 vs ~20 mm) but NOT the near-spheres from each other (extents within 1.5 %).
MARCH_SEC_TOL = 0.10
MARCH_THR_TOL = 0.30
MARCH_EXTENT_VOX = 2.0     # extent tolerance = max(2 voxels, 3 % of the STL extent)
MARCH_EXTENT_REL = 0.03
_ORIENT_GRID = 40          # deterministic STL interior sample grid per axis
_SUBPROCESS_TIMEOUT_S = 1800   # stage_job / preflight wall-clock limit

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


def _proper_axis_rotations() -> list:
    """The 24 proper rotations that permute/negate the coordinate axes."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for r, c in enumerate(perm):
                M[r, c] = signs[r]
            if np.linalg.det(M) > 0.5:
                out.append(M)
    return out


def _moment_signature(P: np.ndarray, w: np.ndarray):
    """Scale-normalised central moments: 6 second-order and 10 third-order."""
    mu = np.average(P, axis=0, weights=w)
    X = P - mu
    L = math.sqrt(float(np.average((X ** 2).sum(axis=1), weights=w)) / 3.0)
    Y = X / L
    idx2 = [(i, j) for i in range(3) for j in range(i, 3)]
    idx3 = [(i, j, k) for i in range(3) for j in range(i, 3) for k in range(j, 3)]
    sec = np.array([np.average(Y[:, i] * Y[:, j], weights=w) for i, j in idx2])
    thr = np.array([np.average(Y[:, i] * Y[:, j] * Y[:, k], weights=w) for i, j, k in idx3])
    return sec, thr


def _signature_err(a, b) -> tuple:
    return (float(np.max(np.abs(a[0] - b[0]))), float(np.max(np.abs(a[1] - b[1]))))


def _pose_symmetry_count(q: np.ndarray) -> int:
    """How many axis rotations (identity included) leave the STL's moment
    signature unchanged about its own centroid: the pose is verifiable only up
    to that many indistinguishable headings."""
    X = q - q.mean(axis=0)
    ones = np.ones(len(X))
    ref = _moment_signature(X, ones)
    n = 0
    for Rp in _proper_axis_rotations():
        e2, e3 = _signature_err(_moment_signature(X @ Rp.T, ones), ref)
        n += int(e2 < SEC_TOL and e3 < THR_TOL)
    return n


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
    if not mesh.is_winding_consistent:
        raise RegistrationError("STL face winding is inconsistent: its signed volume and "
                                "centre of mass are undefined")
    if mesh.body_count != 1:
        raise RegistrationError(f"STL has {mesh.body_count} separate bodies: a stray body "
                                "shifts the centre of mass; stage one solid part per STL")
    R = rotation(build_axis, base)
    p = (map_d["centroids_m"] * 1e3) @ R.T
    w = map_d["volumes_m3"]
    shift = np.asarray(mesh.center_mass, float) - np.average(p, axis=0, weights=w)
    p = p + shift
    # trimesh volume is SIGNED: a globally inverted (inside-out) mesh is negative
    v_map, v_stl = float(w.sum()) * 1e9, abs(float(mesh.volume))
    rel_v = abs(v_map - v_stl) / v_stl
    cell = (float(np.mean(w)) * 1e9) ** (1.0 / 3.0)
    lo, hi = mesh.bounds
    ext_ok = bool(np.all(p.min(0) >= lo - 1.5 * cell)
                  and np.all(p.max(0) <= hi + 1.5 * cell)
                  and np.all(np.ptp(p, axis=0) >= (hi - lo) - 4.0 * cell))
    q = _stl_interior_points(mesh)
    err2, err3 = _signature_err(_moment_signature(p, w),
                                _moment_signature(q, np.ones(len(q))))
    pose_ok = bool(err2 < SEC_TOL and err3 < THR_TOL)
    n_sym = _pose_symmetry_count(q)
    pose_unique = n_sym == 1
    note = ("pose verified by shape moments" if pose_unique else
            f"part is symmetric under {n_sym} axis rotations: its pose is verified only "
            "up to that symmetry; the remaining heading relies on the solve frame and "
            "the STL sharing axes")
    report = {"build_axis": build_axis, "base": base,
              "shift_mm": [round(float(x), 4) for x in shift],
              "volume_map_mm3": round(v_map, 2), "volume_stl_mm3": round(v_stl, 2),
              "volume_rel_err": round(rel_v, 5), "cell_mm": round(cell, 4),
              "extent_ok": ext_ok,
              "moment_err_2nd": round(err2, 4), "moment_err_3rd": round(err3, 4),
              "pose_ok": pose_ok, "pose_symmetry_count": n_sym,
              "pose_unique": pose_unique, "pose_note": note}
    problems = []
    if rel_v > VOLUME_TOL:
        problems.append(f"volume map {v_map:.1f} vs STL {v_stl:.1f} mm3 "
                        f"({100 * rel_v:.1f}% > {100 * VOLUME_TOL:.0f}%): not the "
                        "same solid, or not the same units")
    if not ext_ok:
        problems.append("registered map extents do not match the STL bounding box")
    if not pose_ok:
        problems.append(f"shape moments map vs STL differ (2nd-order {err2:.3f} vs tol "
                        f"{SEC_TOL}, 3rd-order {err3:.3f} vs tol {THR_TOL}): the declared "
                        "build axis/base puts the part on its side or upside down or "
                        "rotated in plane")
    if problems:
        raise RegistrationError("; ".join(problems)
                                + f" [declared build_axis={build_axis}, base={base}]")
    return {"points_mm": p, "s_map": map_d["s_map"], "report": report}


def check_march_matches_stl(fields_npz, mesh: trimesh.Trimesh) -> dict:
    """Prove the densify march simulated THIS part, or raise RegistrationError.

    The march's `part` (nx, ny, nz; z = build axis) and `h` (metres) give voxel
    centres (idx + 0.5) * h in mm. Its normalised moments (translation- and
    scale-invariant) must match the STL interior's, and its physical per-axis
    extent (occupied index span * h) must match the STL bounds. Volume alone
    cannot tell parts apart: the shape library is equal-volume."""
    d = np.load(Path(fields_npz), allow_pickle=False)
    for k in ("part", "h"):
        if k not in d.files:
            raise ValueError(f"{fields_npz}: missing '{k}' (not a densify march output?)")
    part = np.asarray(d["part"], bool)
    if part.ndim != 3 or not part.any():
        raise ValueError(f"{fields_npz}: part mask must be a non-empty 3-D array")
    h_mm = float(d["h"]) * 1e3
    idx = np.argwhere(part)
    P = (idx + 0.5) * h_mm
    q = _stl_interior_points(mesh)
    err2, err3 = _signature_err(_moment_signature(P, np.ones(len(P))),
                                _moment_signature(q, np.ones(len(q))))
    ext_march = (idx.max(axis=0) - idx.min(axis=0) + 1) * h_mm
    lo, hi = mesh.bounds
    ext_stl = np.asarray(hi - lo, float)
    ext_tol = np.maximum(MARCH_EXTENT_VOX * h_mm, MARCH_EXTENT_REL * ext_stl)
    extent_ok = bool(np.all(np.abs(ext_march - ext_stl) <= ext_tol))
    moment_ok = bool(err2 < MARCH_SEC_TOL and err3 < MARCH_THR_TOL)
    report = {"source": str(fields_npz), "h_mm": round(h_mm, 4),
              "n_voxels": int(part.sum()),
              "moment_err_2nd": round(err2, 4), "moment_err_3rd": round(err3, 4),
              "moment_ok": moment_ok,
              "march_extent_mm": [round(float(x), 3) for x in ext_march],
              "stl_extent_mm": [round(float(x), 3) for x in ext_stl],
              "extent_tol_mm": [round(float(x), 3) for x in ext_tol],
              "extent_ok": extent_ok}
    problems = []
    if not moment_ok:
        problems.append(f"shape moments march vs STL differ (2nd-order {err2:.3f} vs tol "
                        f"{MARCH_SEC_TOL}, 3rd-order {err3:.3f} vs tol {MARCH_THR_TOL}): "
                        "a different shape, or the march is on its side/upside down")
    if not extent_ok:
        problems.append("physical extent march "
                        f"{np.round(ext_march, 2).tolist()} vs STL "
                        f"{np.round(ext_stl, 2).tolist()} mm exceeds "
                        f"{np.round(ext_tol, 2).tolist()} mm: a different part or size")
    if problems:
        raise RegistrationError("; ".join(problems) + f" [{fields_npz}]")
    return report


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
    # the stager centres on the UNPROCESSED triangle-vertex mean; match it exactly
    raw = trimesh.load(str(stl_path), force="mesh", process=False)
    tri = np.asarray(raw.triangles).reshape(-1, 3)
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
            "voxel_mm": [dz, dxy, dxy], "canvas_center_mm": [cx, cy],
            "registration": reg["report"],
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


def _parse_preflight(returncode: int, stdout: str, stderr: str) -> dict:
    """First preflight result from `preflight.py --json`, or a not-ready result
    naming why none could be read. Never raises."""
    def fail(reason: str) -> dict:
        tail = stderr.strip()
        return {"ready": False, "warnings": [],
                "errors": [f"{reason} (preflight exit {returncode})"
                           + (f": {tail}" if tail else "")]}
    try:
        res = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as e:
        return fail(f"preflight output is not JSON ({e})")
    if isinstance(res, list):
        res = res[0] if res else None
    if not isinstance(res, dict) or "ready" not in res:
        return fail("preflight JSON has no result")
    if returncode != 0 and res.get("ready"):       # contradictory: trust the failure
        return fail("preflight reported ready but exited non-zero")
    return res


def _deliver(staged: Path, hot_folder: Path, work: Path, ready: bool) -> tuple:
    """Move a staged job to where it belongs. READY -> the live hot folder (a
    same-volume shutil.move is an atomic rename: MetPrint never sees a partial
    job). Not ready -> <work>/_rejected. Never overwrites. Returns (path, error)."""
    dest = hot_folder / staged.name if ready else work / "_rejected" / staged.name
    if dest.exists():
        return staged, (f"{dest} already exists: refusing to overwrite it; the job "
                        f"stays in {staged}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(dest))
    return dest, None


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
    ap.add_argument("--layer-height", type=float, required=True,
                    help="the MACHINE's layer height in mm (0.1-0.3): MetPrint prints one "
                         "page per machine layer, so this must match the printer setting")
    ap.add_argument("--voxel-mm", type=float, default=0.25)
    ap.add_argument("--chamber-mm", type=float, default=None)
    ap.add_argument("--build-axis", choices=["x", "y", "z"], default="z")
    ap.add_argument("--base", choices=["min", "max"], default="min")
    ap.add_argument("--work-dir", default=None,
                    help="spec/summary/report dir (default solve3d/results/stage_inputs/<job>)")
    ap.add_argument("--meteor-tools", default=None)
    args = ap.parse_args(argv)
    # stage_job/preflight run with cwd = the tools dir, so every path handed to
    # them must be absolute -- resolve user-typed relative paths against OUR cwd.
    for name in ("map", "stl", "densify", "hot_folder", "work_dir", "meteor_tools"):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))

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
    march_rep = None
    if args.densify:
        try:
            march_rep = check_march_matches_stl(
                args.densify, trimesh.load(args.stl, force="mesh"))
        except ValueError as e:                 # includes RegistrationError
            print(f"REFUSED: densify march is not this part: {e}", file=sys.stderr)
            return 1
    reg_rep = spec["registration"]
    if not reg_rep.get("pose_unique", True):
        print(f"WARNING: {reg_rep['pose_note']}", file=sys.stderr)
    spec_npz = write_spec(spec, work / f"{args.job_name}_spec.npz")
    report_json = work / "stage_report.json"
    # stage + preflight OFF the live hot folder; only a READY job is moved in
    staging = work.resolve() / "staging"
    cmd = [sys.executable, str(tools / "stage_job.py"), str(spec_npz), "--3d",
           "--stl", str(args.stl), "--chamber-mm", repr(spec["domain_mm"]),
           "--layer-height", repr(args.layer_height),
           "--hot-folder", str(staging), "--job-name", args.job_name,
           "--report-json", str(report_json)]
    if args.densify:
        from solve3d import densify_summary as ds
        cmd += ["--densify-summary",
                str(ds.write_summary(args.densify, work / "densify_summary.json"))]
    elif args.densify_factor is not None:
        cmd += ["--z-densification", repr(args.densify_factor)]
    try:
        r = subprocess.run(cmd, cwd=str(tools), capture_output=True, text=True,
                           timeout=_SUBPROCESS_TIMEOUT_S)
        if r.returncode != 0 or not report_json.is_file():
            sys.stderr.write(r.stdout + r.stderr)
            print(f"STAGE FAILED (stage_job exit {r.returncode})", file=sys.stderr)
            return 1
        rep = json.loads(report_json.read_text())
        staged = Path(rep["out_dir"]).resolve()
        if staged.parent != staging or not staged.is_dir():
            print(f"STAGE FAILED: stage_job reported {staged}, outside {staging}; "
                  "not moving it", file=sys.stderr)
            return 1
        pf = subprocess.run([sys.executable, str(tools / "preflight.py"), str(staged),
                             "--json"], cwd=str(tools), capture_output=True, text=True,
                            timeout=_SUBPROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        print(f"STAGE FAILED (timeout after {e.timeout} s: {Path(e.cmd[1]).name})",
              file=sys.stderr)
        return 1
    pre = _parse_preflight(pf.returncode, pf.stdout, pf.stderr)
    ready = bool(rep.get("all_pass") and pre.get("ready"))
    final, move_err = _deliver(staged, Path(args.hot_folder), work.resolve(), ready)
    if move_err:
        print(f"REFUSED: {move_err}", file=sys.stderr)
    elif not ready:
        print(f"NOT READY (job moved to {final}): {pre.get('errors')}", file=sys.stderr)
    out = {"job": args.job_name, "out_dir": str(final),
           "staged_all_pass": rep.get("all_pass"),
           "print_layers": rep.get("print_layers"),
           "printed_z_mm": rep.get("printed_z_mm"),
           "z_mode": rep.get("provenance", {}).get("z", {}).get("mode"),
           "factor": rep.get("z_densification"),
           "registration": reg_rep,
           "pose_note": reg_rep.get("pose_note"),
           "densify_match": march_rep,
           "preflight_ready": pre.get("ready"),
           "preflight_errors": pre.get("errors"),
           "preflight_warnings": pre.get("warnings")}
    print(json.dumps(out, indent=2))
    return 0 if (ready and not move_err) else 1


if __name__ == "__main__":
    raise SystemExit(main())
