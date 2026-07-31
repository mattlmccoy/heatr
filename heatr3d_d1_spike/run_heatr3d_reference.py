"""D1 Task 2/3: heatr3d EQS-ONLY reference fields (voxel engine).

RUNS IN THE geo-prewarp VENV, NOT the spike env:
    ./.venv312/bin/python heatr3d_d1_spike/run_heatr3d_reference.py --shape cylinder --n 64 96
    ./.venv312/bin/python heatr3d_d1_spike/run_heatr3d_reference.py --shape square   --n 64 96 128

heatr3d.py is READ-ONLY here: this script imports build_gamma / solve_eqs_3d /
compute_qrf_3d and nothing else. No thermal march is run (EQS only), which is
what lifts the ceiling from n=96 (full physics) to n=128 (EQS only) -- see
heatr3d.EQS_MAX_GRID_EQS_ONLY and the EQS-01 finding.

Geometry: FULL-HEIGHT extrusions so the case is exactly z-invariant and the
mid-plane is the whole story (cylinder is full-height by construction in
heatr3d.make_geometry; the square prism is made full-height by passing
zspan = L). z-invariance is verified numerically, not assumed.

Outputs per (shape, n):
  ref_heatr3d_<shape>_n<N>.npz  -- mid-plane V (complex), |E|, Q_rf, part mask,
                                   grid coords, in-part Q histogram
  results.json[<key>]["n<N>"]   -- timing, peak RSS, field summaries, corner
                                   metrics (square), z-invariance check
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # metrics.py
sys.path.insert(0, str(HERE.parent))          # heatr3d.py (repo root, read-only)

import metrics as M                                             # noqa: E402
import heatr3d                                                  # noqa: E402

CORNER_BAND_M = 0.001          # "within 1 mm of a vertical corner edge"
PART_DIAM_M = 0.020


def _rss_gb() -> float:
    """Peak resident set size of this process [GB] (macOS: ru_maxrss in bytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(ru) / 1024.0 ** 3 if sys.platform == "darwin" else float(ru) / 1024.0 ** 2


def _emag(V: np.ndarray, h: float) -> np.ndarray:
    """|E| exactly as compute_qrf_3d forms it: E = -grad V (np.gradient,
    edge_order=1), |E|^2 = Re(E . conj(E))."""
    Ex, Ey, Ez = np.gradient(V, h, edge_order=1)
    e2 = np.real(Ex * np.conj(Ex) + Ey * np.conj(Ey) + Ez * np.conj(Ez))
    return np.sqrt(np.clip(e2, 0.0, None))


def run_one(shape: str, n: int, out_dir: Path) -> dict:
    grid = heatr3d.Grid(n=n)
    p = heatr3d.Params()
    if shape == "cylinder":
        part = heatr3d.make_geometry(grid, "cylinder", diam=PART_DIAM_M)
    elif shape == "square":
        # zspan = L -> |z| <= L/2 is true for every cell centre: a FULL-HEIGHT
        # square prism (the extruded-square analogue of the full-height cylinder)
        part = heatr3d.make_geometry(grid, "square", diam=PART_DIAM_M, zspan=grid.L)
    else:
        raise ValueError(f"unsupported shape {shape!r}")

    rec: dict = {"shape": shape, "n": n, "h_m": grid.h, "L_m": grid.L,
                 "n_unknowns": int(n ** 3),
                 "n_voxels_in_part": int(part.sum()),
                 "voxel_volume_in_part_m3": float(part.sum() * grid.dV)}

    t0 = time.perf_counter()
    gamma = heatr3d.build_gamma(part, p)              # binary path (edge_width_m=0)
    t_gamma = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        V = heatr3d.solve_eqs_3d(gamma, grid, p)
    except MemoryError as exc:                        # EQS-01 documented ceiling
        rec.update({"ok": False, "failure_mode": "MemoryError",
                    "failure_message": str(exc),
                    "wall_eqs_s": time.perf_counter() - t0,
                    "peak_rss_gb": _rss_gb()})
        return rec
    t_eqs = time.perf_counter() - t0

    t0 = time.perf_counter()
    Q = heatr3d.compute_qrf_3d(V, gamma, grid, p, doped=part, premix=False)
    t_qrf = time.perf_counter() - t0

    Emag = _emag(V, grid.h)

    # --- z-invariance check (the extrusion is uniform along z) --------------
    qin = Q[part]
    q_zvar = float(np.max(np.std(Q, axis=2)[part[:, :, 0]]) / max(qin.mean(), 1e-30))

    k = n // 2
    z_mid = float(grid.z[k])
    part_s, Q_s, V_s, E_s = part[:, :, k], Q[:, :, k], V[:, :, k], Emag[:, :, k]
    m_s = part_s

    # --- in-part Q histogram (unit-mean pattern bins) -----------------------
    pat = M.unit_mean(qin)
    counts, edges = np.histogram(pat, bins=60, range=(0.0, float(pat.max()) * 1.001))

    rec.update({
        "ok": True,
        "wall_gamma_s": t_gamma, "wall_eqs_s": t_eqs, "wall_qrf_s": t_qrf,
        "peak_rss_gb": _rss_gb(),
        "z_mid_m": z_mid,
        "z_invariance_rel_std": q_zvar,
        "q_in_part": {"mean": float(qin.mean()), "max": float(qin.max()),
                      "p99": float(np.percentile(qin, 99)),
                      "p99_over_mean": float(np.percentile(qin, 99) / qin.mean())},
        "e_in_part": {"mean": float(Emag[part].mean()), "max": float(Emag[part].max())},
        "p_total_w": float(Q.sum() * grid.dV),
        "electrode_gauge": {
            "note": "cell-centred electrodes span L-h, not L",
            "span_m": float(grid.L - grid.h),
            "expected_E_ratio_vs_fem": float(1.0 / (1.0 - 1.0 / n))},
    })

    if shape == "square":
        X, Y, _ = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
        d = M.corner_edge_distance(X.ravel(), Y.ravel(), PART_DIAM_M / 2.0)
        d = d.reshape(X.shape)
        band = part & (d <= CORNER_BAND_M)
        rec["corner"] = {
            "band_m": CORNER_BAND_M,
            "n_voxels_in_band": int(band.sum()),
            "q_corner_max": float(Q[band].max()) if band.any() else float("nan"),
            "q_corner_max_over_mean": float(Q[band].max() / qin.mean()) if band.any() else float("nan"),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"ref_heatr3d_{shape}_n{n}.npz"
    np.savez_compressed(
        npz, x=grid.x, y=grid.y, z_mid=z_mid, h=grid.h, L=grid.L, n=n,
        part_mid=m_s, V_mid=V_s, Emag_mid=E_s, Q_mid=Q_s,
        q_hist_counts=counts, q_hist_edges=edges,
        q_in_part_all=qin.astype(np.float64))
    rec["npz"] = npz.name
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=["cylinder", "square"])
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--key", default=None, help="results.json key (default by shape)")
    args = ap.parse_args()
    key = args.key or ("task2_ref" if args.shape == "cylinder" else "task3_ref")

    out: dict = {"engine": "heatr3d (voxel FV, EQS only)",
                 "shape": args.shape,
                 "part_diam_m": PART_DIAM_M,
                 "python": sys.version.split()[0],
                 "runs": {}}
    for n in args.n:
        print(f"[heatr3d] {args.shape} n={n} ...", flush=True)
        rec = run_one(args.shape, n, HERE)
        out["runs"][f"n{n}"] = rec
        print(json.dumps(rec, indent=1)[:1200], flush=True)

    p = HERE / "results.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    # MERGE, never replace: each invocation may cover only some of the
    # refinement levels (they are run separately so that hitting the EQS-01
    # memory ceiling on the largest grid cannot destroy the smaller records).
    prev = d.get(key)
    if isinstance(prev, dict) and "runs" in prev:
        merged = dict(prev["runs"])
        merged.update(out["runs"])
        out["runs"] = merged
    d[key] = out
    p.write_text(json.dumps(d, indent=1))
    print(f"wrote results.json[{key}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
