#!/usr/bin/env python3
"""Geometry Pre-Warp GUI worker (COMPUTE stage).

MUST be run under the analysis-3dfgm .venv312 python (py3.12 + numpy 2.1.3); the
system numpy 2.2.x has a buffer-elision bug that silently corrupts arrays. This
worker imports the FD-gated level-set ILT (`ilt_levelset.levelset_ilt`) from the
analysis-3dfgm workstream, runs the corrected nominal-target level-set pre-warp for
one geometry, and writes:
    <out_dir>/prewarp_capture.npz   — fields for the figure stage
    <out_dir>/summary.json          — IoU before/after, boundary error, metadata

PREWARP CONTRACT (held here, do not change without re-gating):
  - move only the BOUNDARY (level-set phi), hold DOPANT uniform
  - part center fixed; target = the FIXED crisp centred nominal CAD shape
  - this is a DIFFERENT correction approach from FGM (grade dopant, fixed geometry)

The level-set dJ/dphi gradient is FD-gated in ilt_levelset.run_gate; this worker
re-gates it on the requested geometry when --gate is passed and records the rel_err.

Figure rendering is a SEPARATE stage (prewarp_figure.py under a python with
matplotlib) because matplotlib is not installed in .venv312.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi


# ── geometry name map: GUI shape names -> ilt_levelset._nominal2d names ──────────
# Only these geometries have a centred analytic nominal in the prewarp engine.
# gt_logo is handled via an intake mask (not wired here yet; see follow-up note).
_GUI_TO_ENGINE = {
    "square": "square",
    "circle": "circle",
    "hexagon": "hexagon",
    "cross": "cross",
    "l_shape": "lshape",
    "lshape": "lshape",
    "t_shape": "t_shape",
    "tshape": "t_shape",
    "diamond": "diamond",
    "triangle": "triangle",
    "equilateral_triangle": "eq_triangle",
    "eq_triangle": "eq_triangle",
    "star": "star",
}

SUPPORTED_GEOMETRIES = sorted(set(_GUI_TO_ENGINE.values()))


def _boundary_rms(melt2d: np.ndarray, nominal2d: np.ndarray) -> tuple[float, float]:
    """Boundary error between a binary melt region and the binary nominal target.

    Distance (in voxels) from each melt-boundary voxel to the nearest nominal-boundary
    voxel, summarised as RMS and max. Symmetric-ish proxy of the shape mismatch the
    offline campaign reports as 'bnd-RMS'/'bnd-max'."""
    mb = melt2d > 0.5
    nb = nominal2d > 0.5
    if mb.sum() == 0 or nb.sum() == 0:
        return float("nan"), float("nan")
    # boundary = region minus its erosion
    m_edge = mb & ~ndi.binary_erosion(mb)
    n_edge = nb & ~ndi.binary_erosion(nb)
    if m_edge.sum() == 0 or n_edge.sum() == 0:
        return 0.0, 0.0
    # EDT to the nominal boundary, sampled at melt-boundary voxels
    dist_to_n = ndi.distance_transform_edt(~n_edge)
    d = dist_to_n[m_edge]
    return float(np.sqrt(np.mean(d * d))), float(d.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True, help="abs path to analysis-3dfgm")
    ap.add_argument("--out-dir", required=True, help="abs path to run output dir")
    ap.add_argument("--geometry", required=True, help="GUI shape name")
    ap.add_argument("--grid", type=int, default=56, help="grid resolution (nx=ny)")
    ap.add_argument("--iters", type=int, default=400, help="level-set iterations")
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--melt-frac", type=float, default=0.95)
    ap.add_argument("--gate", action="store_true", help="re-run the FD gate first")
    args = ap.parse_args()

    sys.path.insert(0, args.analysis_dir)
    import ilt_levelset as L  # noqa: E402  (path set above)

    g = args.geometry.strip().lower()
    if g not in _GUI_TO_ENGINE:
        print(f"[prewarp] ERROR: geometry {args.geometry!r} not supported by prewarp. "
              f"Supported: {sorted(_GUI_TO_ENGINE)}", flush=True)
        return 2
    engine_geom = _GUI_TO_ENGINE[g]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prewarp] geometry={args.geometry} -> engine='{engine_geom}'  "
          f"grid={args.grid}  iters={args.iters}  melt_frac={args.melt_frac}", flush=True)

    # ── optional FD gate on the wired path (prove the gradient is correct) ──────
    gate_rel_err = None
    if args.gate:
        print(f"[prewarp] FD-gating dJ/dphi on geometry='{engine_geom}' "
              f"(grid {args.grid}) ...", flush=True)
        gate_rel_err = float(L.run_gate(nx=args.grid, ny=args.grid,
                                        geometry=engine_geom))
        print(f"[prewarp] FD gate best rel_err = {gate_rel_err:.3e}  "
              f"({'PASS' if gate_rel_err < 1e-5 else 'FAIL'})", flush=True)

    # ── run the level-set pre-warp (capture all fields for the figure) ──────────
    t0 = time.time()
    cap = L.levelset_ilt(
        geometry=engine_geom, nx=args.grid, ny=args.grid,
        iters=args.iters, lr=args.lr, melt_frac=args.melt_frac,
        verbose=True, capture=True,
    )
    elapsed = time.time() - t0

    nominal = np.asarray(cap["nominal"]) > 0.5
    m_naive = np.asarray(cap["m_naive"]) > 0.5
    m_final = np.asarray(cap["m_final"]) > 0.5
    iou0 = float(cap["iou0"])
    iouf = float(cap["iou_final"])

    bnd_rms_naive, bnd_max_naive = _boundary_rms(m_naive, nominal)
    bnd_rms_prewarp, bnd_max_prewarp = _boundary_rms(m_final, nominal)

    # save fields for the figure stage
    npz_path = out_dir / "prewarp_capture.npz"
    np.savez_compressed(
        npz_path,
        geometry=np.array(args.geometry),
        nominal=np.asarray(cap["nominal"]).astype(float),
        theta_init=np.asarray(cap["theta_init"]).astype(float),
        theta_final=np.asarray(cap["theta_final"]).astype(float),
        m_naive=m_naive.astype(float),
        m_final=m_final.astype(float),
        phi_final=np.asarray(cap["phi_final"]).astype(float),
        iou_hist=np.asarray(cap["iou_hist"], float),
        J_hist=np.asarray(cap["J_hist"], float),
        iou0=np.array(iou0), iou_final=np.array(iouf),
    )

    summary = {
        "mode": "prewarp",
        "method": "geometry pre-warp (level-set ILT, uniform dopant)",
        "geometry_gui": args.geometry,
        "geometry_engine": engine_geom,
        "grid": args.grid,
        "iters": args.iters,
        "lr": args.lr,
        "melt_frac": args.melt_frac,
        "iou_naive": round(iou0, 4),
        "iou_prewarp": round(iouf, 4),
        "iou_gain": round(iouf - iou0, 4),
        "boundary_rms_naive_vox": round(bnd_rms_naive, 3),
        "boundary_rms_prewarp_vox": round(bnd_rms_prewarp, 3),
        "boundary_max_naive_vox": round(bnd_max_naive, 3),
        "boundary_max_prewarp_vox": round(bnd_max_prewarp, 3),
        "fd_gate_rel_err": gate_rel_err,
        "fd_gate_pass": (gate_rel_err is not None and gate_rel_err < 1e-5),
        "elapsed_s": round(elapsed, 1),
        "contract": ("moves boundary only, dopant uniform, part center fixed, fixed "
                     "nominal target; DIFFERENT approach from FGM"),
        "simplifications": ("steady thermal, theta-independent K (depth term off here), "
                            "no densification/shrinkage; IoU is the sub-unity level-set "
                            "number (coarse-grid metric saturation shown elsewhere)"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[prewarp] DONE  IoU(melt,nominal) naive -> prewarp: "
          f"{iou0:.4f} -> {iouf:.4f}  (gain {iouf - iou0:+.4f})", flush=True)
    print(f"[prewarp] boundary RMS naive -> prewarp: "
          f"{bnd_rms_naive:.3f} -> {bnd_rms_prewarp:.3f} vox  "
          f"(max {bnd_max_naive:.2f} -> {bnd_max_prewarp:.2f})", flush=True)
    print(f"[prewarp] wrote {npz_path.name} + summary.json  ({elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
