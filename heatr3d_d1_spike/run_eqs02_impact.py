"""EQS-02 impact study: what the cross-interface Q_rf artifact does to the
THERMAL answers heatr3d publishes.

RUNS IN THE geo-prewarp VENV (heatr3d), NOT the spike env:
    ./.venv312/bin/python heatr3d_d1_spike/run_eqs02_impact.py

READ-ONLY with respect to heatr3d.py. This script imports build_gamma /
solve_eqs_3d / compute_qrf_3d / run / make_geometry and changes nothing. The
corrected heating field is injected through the EXISTING validation hook
run(qrf_override=...), which is exactly what that hook is for.

THE ARTIFACT (found in Task 2, results.json["task2"]): compute_qrf_3d forms
E = -np.gradient(V) over the WHOLE domain and only afterwards zeroes Q outside
the part. The outermost in-part voxel is therefore differenced against an
OUTSIDE voxel, across the material interface where grad V jumps by the
conductivity contrast (sigma_doped/sigma_virgin = 4e6). |E| in that skin is
inflated, and because Q ~ |E|^2 the inflation is squared. The P_abs
renormalization then RESCALES THE WHOLE FIELD DOWN to hit the same total power,
so the artifact does not add energy -- it MOVES energy from the interior to the
skin.

THE CORRECTION (this script): recompute Q from the SAME V and the SAME gamma
with metrics.masked_grad_3d, a stencil that never crosses the part boundary
(second-order central where both neighbours are in-part, one-sided -- exact for
a linear field -- where only one is). Everything else in the Q definition is
byte-identical to compute_qrf_3d: Q = 0.5*Re(gamma*|E|^2), clipped at 0, zeroed
outside the part, renormalized to the SAME total absorbed power
p_target = power_density_w_per_m3 * doped_volume. So the shipped and corrected
drives carry IDENTICAL total power and differ only in its spatial distribution.

2-D vs 3-D CHOICE (documented, not assumed): both shapes here are FULL-HEIGHT
extrusions, so V is exactly z-invariant and the 2-D per-slice diagnostic would
suffice. We nevertheless use the volumetric masked_grad_3d, because (a) the
thermal march needs a full 3-D drive anyway, and (b) it lets us MEASURE the
z-invariance (recorded as ez_max_over_emag_mean) instead of assuming it.

COMPARISON BASIS: sigma_T here is the 3-D metric (std of T_phi90 over the part
voxels, heatr3d.Result.sigma_T). It is NOT comparable to the 2.5-D study's
ui_rms numbers -- different physics basis and different drive normalization.
Only shipped-vs-corrected within this script is a like-for-like comparison.
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
sys.path.insert(0, str(HERE))                 # metrics.py
sys.path.insert(0, str(HERE.parent))          # heatr3d.py (repo root, read-only)

import metrics as M                                             # noqa: E402
import heatr3d                                                  # noqa: E402

PART_DIAM_M = 0.020
SURFACE_BAND_H = 1.5      # "surface" = within 1.5 voxels of the part boundary
                          # (same rule as Task 2's run_extruded_circle.py)
PHI_TARGET = 0.90
MAX_TIME_S = 1500.0


def _rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(ru) / 1024.0 ** 3 if sys.platform == "darwin" else float(ru) / 1024.0 ** 2


def _depth_m(part: np.ndarray, h: float) -> np.ndarray:
    """Depth below the part surface [m], >0 inside (heatr3d._signed_distance_m
    convention, replicated locally so this script imports nothing private)."""
    return (distance_transform_edt(part) - 0.5) * h


def corrected_qrf(V: np.ndarray, gamma: np.ndarray, grid, p, part: np.ndarray):
    """Q_rf from the SAME V/gamma with a part-confined gradient, renormalized to
    the same fixed total power compute_qrf_3d enforces."""
    Ex, Ey, Ez = M.masked_grad_3d(V, part, grid.h)
    e2 = np.real(Ex * np.conj(Ex) + Ey * np.conj(Ey) + Ez * np.conj(Ez))
    Q = 0.5 * np.real(gamma * e2)
    Q = np.clip(np.nan_to_num(Q), 0.0, None)
    Q[~part] = 0.0
    p_target = p.power_density_w_per_m3 * (int(part.sum()) * grid.dV)
    p_now = Q.sum() * grid.dV
    if p_now > 1e-18:
        Q *= p_target / p_now
    emag = np.sqrt(np.clip(e2, 0.0, None))
    ez_rel = float(np.abs(Ez[part]).max() / max(emag[part].mean(), 1e-30))
    return Q, ez_rel


def _field_stats(q: np.ndarray, part, interior, band) -> dict:
    qin = q[part]
    mu = float(qin.mean())
    return {"mean": mu, "max": float(qin.max()),
            "p99_over_mean": float(np.percentile(qin, 99) / mu),
            "max_over_mean": float(qin.max() / mu),
            "cv": float(qin.std() / mu),
            "interior_mean": float(q[interior].mean()),
            "surface_mean": float(q[band].mean()),
            "power_fraction_in_surface_band": float(q[band].sum() / q[part].sum())}


def _thermal_stats(res, part, interior, band, wall_s: float) -> dict:
    T = res.T_phi90
    return {"sigma_T_c": float(res.sigma_T),
            "t_phi90_s": float(res.t_phi90_s),
            "reached_phi90": bool(res.reached),
            "T_max_c": float(res.T_max_c),
            "T_mean_c": float(T[part].mean()),
            "T_min_c": float(T[part].min()),
            "T_range_c": float(T[part].max() - T[part].min()),
            "interior": {"mean_c": float(T[interior].mean()),
                         "std_c": float(T[interior].std()),
                         "max_c": float(T[interior].max())},
            "surface": {"mean_c": float(T[band].mean()),
                        "std_c": float(T[band].std()),
                        "max_c": float(T[band].max())},
            "surface_minus_interior_mean_c": float(T[band].mean() - T[interior].mean()),
            "energy_residual_frac": float(res.energy_residual_frac),
            "clamp_bound": bool(res.clamp_bound),
            "n_substeps_used": int(res.n_substeps_used),
            "wall_thermal_s": wall_s}


def run_shape(shape: str, n: int) -> dict:
    grid = heatr3d.Grid(n=n)
    p = heatr3d.Params()
    if shape == "cylinder":
        part = heatr3d.make_geometry(grid, "cylinder", diam=PART_DIAM_M)
    elif shape == "square":
        part = heatr3d.make_geometry(grid, "square", diam=PART_DIAM_M, zspan=grid.L)
    else:
        raise ValueError(f"unsupported shape {shape!r}")

    depth = _depth_m(part, grid.h)
    interior = part & (depth > SURFACE_BAND_H * grid.h)
    band = part & ~interior

    # ---- (a) Q as shipped -------------------------------------------------
    t0 = time.perf_counter()
    gamma = heatr3d.build_gamma(part, p)
    V = heatr3d.solve_eqs_3d(gamma, grid, p)
    wall_eqs = time.perf_counter() - t0
    q_ship = heatr3d.compute_qrf_3d(V, gamma, grid, p, doped=part, premix=False)

    # ---- (b) Q recomputed from the SAME V, part-confined gradient ---------
    q_corr, ez_rel = corrected_qrf(V, gamma, grid, p, part)

    p_ship = float(q_ship.sum() * grid.dV)
    p_corr = float(q_corr.sum() * grid.dV)

    rec: dict = {
        "shape": shape, "n": n, "h_m": grid.h,
        "n_voxels_in_part": int(part.sum()),
        "n_voxels_interior": int(interior.sum()),
        "n_voxels_surface_band": int(band.sum()),
        "surface_band_rule": f"within {SURFACE_BAND_H} voxels of the part boundary",
        "wall_eqs_s": wall_eqs,
        "z_invariance": {"ez_max_over_emag_mean": ez_rel,
                         "note": "full-height extrusion -> exactly z-invariant V; "
                                 "a nonzero value here would invalidate the "
                                 "2-D-per-slice equivalence"},
        "power_identity": {"p_shipped_w": p_ship, "p_corrected_w": p_corr,
                           "rel_diff": abs(p_corr - p_ship) / p_ship},
        "qrf": {"shipped": _field_stats(q_ship, part, interior, band),
                "corrected": _field_stats(q_corr, part, interior, band),
                "pattern_rel_l2_corrected_vs_shipped": {
                    "all": M.rel_l2_pattern(q_corr[part], q_ship[part]),
                    "interior": M.rel_l2_pattern(q_corr[interior], q_ship[interior]),
                    "surface_band": M.rel_l2_pattern(q_corr[band], q_ship[band])},
                "interior_mean_ratio_corrected_over_shipped":
                    float(q_corr[interior].mean() / q_ship[interior].mean())},
    }

    # ---- (c) thermal march, twice, enthalpy phase update ------------------
    p_th = dataclasses.replace(p, phase_update="enthalpy")
    thermal = {}
    for tag, Q in (("shipped", q_ship), ("corrected", q_corr)):
        print(f"[eqs02] {shape} n={n}: thermal march ({tag}) ...", flush=True)
        t0 = time.perf_counter()
        res = heatr3d.run(grid, part, p_th, qrf_override=Q,
                          max_time_s=MAX_TIME_S, phi_target=PHI_TARGET)
        thermal[tag] = _thermal_stats(res, part, interior, band,
                                      time.perf_counter() - t0)
        print(f"        sigma_T={thermal[tag]['sigma_T_c']:.3f} C  "
              f"t90={thermal[tag]['t_phi90_s']:.1f} s  "
              f"Tmax={thermal[tag]['T_max_c']:.2f} C", flush=True)
    rec["thermal"] = thermal
    rec["thermal"]["phase_update"] = p_th.phase_update
    rec["thermal"]["phi_target"] = PHI_TARGET

    s, c = thermal["shipped"], thermal["corrected"]
    rec["delta_corrected_minus_shipped"] = {
        "sigma_T_c": c["sigma_T_c"] - s["sigma_T_c"],
        "sigma_T_rel": (c["sigma_T_c"] - s["sigma_T_c"]) / s["sigma_T_c"],
        "t_phi90_s": c["t_phi90_s"] - s["t_phi90_s"],
        "t_phi90_rel": (c["t_phi90_s"] - s["t_phi90_s"]) / s["t_phi90_s"],
        "T_max_c": c["T_max_c"] - s["T_max_c"],
        "interior_mean_c": c["interior"]["mean_c"] - s["interior"]["mean_c"],
        "surface_mean_c": c["surface"]["mean_c"] - s["surface"]["mean_c"],
        "surface_minus_interior_c": (c["surface_minus_interior_mean_c"]
                                     - s["surface_minus_interior_mean_c"]),
    }
    rec["peak_rss_gb"] = _rss_gb()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--shapes", nargs="+", default=["cylinder", "square"])
    args = ap.parse_args()

    out = {
        "what": "EQS-02 impact: shipped compute_qrf_3d Q vs the same V "
                "re-post-processed with a part-confined 3-D gradient, both "
                "renormalized to the same total power, each driving the "
                "thermal march via run(qrf_override=...)",
        "heatr3d_edited": False,
        "gradient_choice": "metrics.masked_grad_3d (volumetric); equivalent to "
                           "per-z-slice masked_grad_2d for these exactly "
                           "z-invariant extrusions, verified by "
                           "z_invariance.ez_max_over_emag_mean and by "
                           "test_masked_grad_3d_matches_2d_per_slice_for_a_"
                           "z_invariant_field",
        "sigma_T_basis": "3-D: std of T_phi90 over part voxels "
                         "(heatr3d.Result.sigma_T). NOT comparable to 2.5-D ui_rms.",
        "python": sys.version.split()[0],
        "shapes": {},
    }
    for shape in args.shapes:
        print(f"[eqs02] === {shape} n={args.n} ===", flush=True)
        out["shapes"][shape] = run_shape(shape, args.n)

    p = HERE / "results.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    prev = d.get("eqs02_impact")
    if isinstance(prev, dict) and "shapes" in prev:
        merged = dict(prev["shapes"])
        merged.update(out["shapes"])
        out["shapes"] = merged
    d["eqs02_impact"] = out
    p.write_text(json.dumps(d, indent=1))
    print("wrote results.json[eqs02_impact]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
