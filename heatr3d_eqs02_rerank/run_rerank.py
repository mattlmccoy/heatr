"""EQS-02 shape re-ranking: one shape, BOTH Q_rf-gradient arms, ONE EQS solve.

    ./.venv312/bin/python heatr3d_eqs02_rerank/run_rerank.py --shape square --n 64

heatr3d.py is imported READ-ONLY (nothing in it is modified, and the `diamond`
cross-section is built here as a local boolean mask rather than added to
make_geometry). Design, inventory and runtime estimate: README.md next to this file.

Both arms are post-processings of the SAME V from the SAME solve_eqs_3d call, each
renormalized by compute_qrf_3d to power_density_w_per_m3 * doped_volume, so total
absorbed power is identical and only its distribution differs. Each arm then drives
run(qrf_override=...), which skips the EQS solve entirely -- that hook is the reason
one solve can serve two marches.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # heatr3d.py at the repo root

import heatr3d  # noqa: E402

SURFACE_BAND_H = 1.5         # "surface" = within 1.5 voxels of the boundary (Task-2 rule)
PHI_TARGET = 0.90
MAX_TIME_S = 1500.0
ARMS = ("legacy", "masked")

# Full-height prisms (V is z-invariant) vs z-varying solids. Stated, not assumed:
# the campaign deliberately covers both classes.
PRISMS = ("square", "diamond", "lshape", "cross")
SOLIDS = ("sphere", "cone", "dumbbell")
SHAPES = ("cylinder",) + PRISMS + SOLIDS

# Geometry PINNED TO THE PUBLISHED RUNS so the legacy arm is a reproduction anchor,
# not a re-parameterisation:
#   cylinder/cone/sphere/dumbbell -> analysis-3dfgm/run_3d_study.py SHAPES (n=64),
#       whose baseline table is analysis-3dfgm/study_summary.csv -- THE published
#       heatr3d 4-shape sigma_T ranking this campaign re-ranks.
#   square -> EQS02_IMPACT.md (diam 20 mm, full-height extrusion).
#   diamond/lshape/cross -> same 20 mm bounding box, full height (the layerC /
#       2-D-published prism family).
# The two classes are NOT the same physical size; Params.power_density_w_per_m3 is a
# per-volume reference precisely so heating rate per volume is size-independent, but
# cross-CLASS size differences remain a stated limit on the cross-shape ranking.
GEOM = {
    "cylinder": {"diam": 0.020, "zspan": None,   "src": "run_3d_study.py"},
    "square":   {"diam": 0.020, "zspan": "L",    "src": "EQS02_IMPACT.md"},
    "diamond":  {"diam": 0.020, "zspan": "L",    "src": "local mask, 2-D-published cross-section"},
    "lshape":   {"diam": 0.020, "zspan": "L",    "src": "layerC prism family"},
    "cross":    {"diam": 0.020, "zspan": "L",    "src": "layerC prism family"},
    "sphere":   {"diam": 0.028, "zspan": None,   "src": "run_3d_study.py"},
    "cone":     {"diam": 0.024, "zspan": 0.030,  "src": "run_3d_study.py"},
    "dumbbell": {"diam": 0.018, "zspan": 0.034,  "src": "run_3d_study.py"},
}


def build_part(shape: str, grid) -> np.ndarray:
    """Part mask at the pinned geometry. `diamond` is NOT a heatr3d.make_geometry
    shape; it is the 2-D published diamond cross-section (|x|+|y| <= diam/2, i.e.
    the square rotated 45 deg on the same bounding box) extruded full height, built
    here so heatr3d.py stays untouched."""
    g = GEOM[shape]
    diam = g["diam"]
    zspan = grid.L if g["zspan"] == "L" else g["zspan"]
    if shape == "diamond":
        X, Y, _ = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
        return (np.abs(X) + np.abs(Y)) <= diam / 2.0
    if zspan is None:
        return heatr3d.make_geometry(grid, shape, diam=diam)
    return heatr3d.make_geometry(grid, shape, diam=diam, zspan=zspan)


def _depth_m(part: np.ndarray, h: float) -> np.ndarray:
    return (distance_transform_edt(part) - 0.5) * h


def _rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(ru) / 1024.0 ** 3 if sys.platform == "darwin" else float(ru) / 1024.0 ** 2


def _field_stats(q: np.ndarray, part, interior, band) -> dict:
    qin = q[part]
    mu = float(qin.mean())
    return {"mean_w_per_m3": mu,
            "max_over_mean": float(qin.max() / mu),
            "p99_over_mean": float(np.percentile(qin, 99) / mu),
            "cv": float(qin.std() / mu),
            "interior_mean_w_per_m3": float(q[interior].mean()),
            "surface_mean_w_per_m3": float(q[band].mean()),
            "power_fraction_in_surface_band": float(q[band].sum() / qin.sum())}


def _thermal_stats(res, part, interior, band, wall_s: float) -> dict:
    T = res.T_phi90
    return {"sigma_T_c": float(res.sigma_T),
            "t_phi90_s": float(res.t_phi90_s),
            "T_max_c": float(res.T_max_c),
            "T_mean_c": float(T[part].mean()),
            "T_min_c": float(T[part].min()),
            "interior_mean_c": float(T[interior].mean()),
            "interior_std_c": float(T[interior].std()),
            "surface_mean_c": float(T[band].mean()),
            "surface_std_c": float(T[band].std()),
            "surface_minus_interior_mean_c": float(T[band].mean() - T[interior].mean()),
            # ---- standing gates ------------------------------------------------
            "gates": {"reached_phi90": bool(res.reached),
                      "energy_residual_frac": float(res.energy_residual_frac),
                      "clamp_bound": bool(res.clamp_bound),
                      "cfl_violated": bool(res.cfl_violated),
                      "n_substeps_used": int(res.n_substeps_used)},
            "wall_thermal_s": wall_s}


def run_shape(shape: str, n: int) -> dict:
    grid = heatr3d.Grid(n=n)
    p = heatr3d.Params()
    part = build_part(shape, grid)
    if not part.any():
        raise RuntimeError(f"{shape}: empty part mask")

    depth = _depth_m(part, grid.h)
    interior = part & (depth > SURFACE_BAND_H * grid.h)
    band = part & ~interior

    t0 = time.perf_counter()
    gamma = heatr3d.build_gamma(part, p)
    V = heatr3d.solve_eqs_3d(gamma, grid, p)
    wall_eqs = time.perf_counter() - t0
    print(f"[rerank] {shape} n={n}: EQS solve {wall_eqs:.1f} s "
          f"({int(part.sum())} part voxels)", flush=True)

    q = {arm: heatr3d.compute_qrf_3d(V, gamma, grid, p, doped=part, premix=False,
                                     qrf_gradient=arm) for arm in ARMS}
    p_abs = {arm: float(q[arm].sum() * grid.dV) for arm in ARMS}

    rec: dict = {
        "shape": shape, "n": n, "h_m": grid.h, "L_m": grid.L,
        "geometry": dict(GEOM[shape]),
        "shape_class": "prism_full_height" if shape in PRISMS + ("cylinder",) else "solid",
        "n_voxels_in_part": int(part.sum()),
        "n_voxels_interior": int(interior.sum()),
        "n_voxels_surface_band": int(band.sum()),
        "surface_band_volume_fraction": float(band.sum() / part.sum()),
        "surface_band_rule": f"within {SURFACE_BAND_H} voxels of the part boundary",
        "wall_eqs_s": wall_eqs,
        "power_identity": {"p_legacy_w": p_abs["legacy"], "p_masked_w": p_abs["masked"],
                           "rel_diff": abs(p_abs["masked"] - p_abs["legacy"])
                           / max(p_abs["legacy"], 1e-30)},
        "qrf": {arm: _field_stats(q[arm], part, interior, band) for arm in ARMS},
    }

    p_th = dataclasses.replace(p, phase_update="enthalpy")
    thermal = {}
    for arm in ARMS:
        print(f"[rerank] {shape} n={n}: thermal march ({arm}) ...", flush=True)
        t0 = time.perf_counter()
        res = heatr3d.run(grid, part, p_th, qrf_override=q[arm],
                          max_time_s=MAX_TIME_S, phi_target=PHI_TARGET)
        thermal[arm] = _thermal_stats(res, part, interior, band,
                                      time.perf_counter() - t0)
        g = thermal[arm]["gates"]
        print(f"         sigma_T={thermal[arm]['sigma_T_c']:.3f} C  "
              f"t90={thermal[arm]['t_phi90_s']:.1f} s  "
              f"Tmax={thermal[arm]['T_max_c']:.2f} C  "
              f"reached={g['reached_phi90']} resid={g['energy_residual_frac']:.2e} "
              f"clamp={g['clamp_bound']} cfl={g['cfl_violated']}", flush=True)
    rec["thermal"] = thermal
    rec["phase_update"] = p_th.phase_update
    rec["phi_target"] = PHI_TARGET
    rec["max_time_s"] = MAX_TIME_S
    rec["peak_rss_gb"] = _rss_gb()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=SHAPES)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out-dir", default=str(HERE / "shards"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rec = run_shape(args.shape, args.n)
    rec["wall_total_s"] = time.perf_counter() - t0
    rec["python"] = sys.version.split()[0]
    rec["heatr3d_default_qrf_gradient"] = \
        heatr3d.compute_qrf_3d.__defaults__[-1] if heatr3d.compute_qrf_3d.__defaults__ else None
    path = out_dir / f"{args.shape}_n{args.n}.json"
    path.write_text(json.dumps(rec, indent=1))
    print(f"[rerank] wrote {path} ({rec['wall_total_s']:.1f} s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
