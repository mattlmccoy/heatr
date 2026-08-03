"""S2 Task 5: mechanism checks -- numbers first, statements second.

    ./.venv312/bin/python -m heatr3d_s2.mechanisms

Two anomalies inherited from heatr3d_eqs02_rerank/RERANK_REPORT.md:

  L-SHAPE OUTLIER. Under the corrected (masked) default the L-shape's sigma_T
  goes 33.204 -> 66.420, +100.0 %, by far the worst of the eight shapes.
  Hypothesis: the L has a REENTRANT corner, the only shape in the set that
  does, and a reentrant corner is a genuine field singularity. The test is
  quantitative and has a decisive form: does the corner region's share of
  absorbed power CONVERGE under refinement (a resolved feature) or keep
  growing (a singularity the grid is chasing)?

  CYLINDER NULL. The corrected cylinder field is already near-uniform, so the
  inversion heuristic finds nothing (+0.3 %). Computed here: the electromagnetic
  skin depth against the part radius, which decides whether the field CAN vary
  across the part, plus the measured flatness of Q in the part interior.

Everything is EQS-only, so this is cheap and needs no march.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import heatr3d
from heatr3d_s2 import harness

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "mechanisms.json"
CORNER_BAND_M = 0.0015


def _qfield(shape: str, n: int, p: heatr3d.Params | None = None):
    p = p or heatr3d.Params(phase_update="enthalpy")
    grid = heatr3d.Grid(n=n)
    part = harness.make_part(grid, shape)
    gamma = heatr3d.build_gamma(part, p)
    V = heatr3d.solve_eqs_3d(gamma, grid, p)
    Q = heatr3d.compute_qrf_3d(V, gamma, grid, p, doped=part, premix=False,
                               qrf_gradient="masked")
    return grid, part, Q


def lshape_corner_share(grids=(48, 64, 80)) -> dict:
    """Share of in-part absorbed power within CORNER_BAND_M of the L's
    REENTRANT corner, as a function of grid.

    The L built by heatr3d.make_geometry has its outer corner at (-half,-half)
    and its REENTRANT corner at (-half + thick, -half + thick), which is the
    only concave corner in the shape set."""
    half = harness.PART_DIAM_M / 2.0
    thick = harness.PART_DIAM_M * (5.0 / 12.0)
    cx, cy = -half + thick, -half + thick
    rows = []
    for n in grids:
        t0 = time.perf_counter()
        grid, part, Q = _qfield("lshape", n)
        X, Y, _ = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
        d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        band = part & (d <= CORNER_BAND_M)
        qin = Q[part]
        rows.append({
            "n": n, "h_m": grid.h,
            "corner_power_share": float(Q[band].sum() / Q[part].sum()),
            "corner_volume_share": float(band.sum() / part.sum()),
            "corner_max_over_mean": float(Q[band].max() / qin.mean()),
            "in_part_max_over_mean": float(qin.max() / qin.mean()),
            "in_part_cv": float(qin.std() / qin.mean()),
            "n_voxels_in_band": int(band.sum()),
            "wall_s": time.perf_counter() - t0})
    conc = [r["corner_power_share"] / r["corner_volume_share"] for r in rows]
    peak = [r["corner_max_over_mean"] for r in rows]
    return {"corner_band_m": CORNER_BAND_M,
            "reentrant_corner_xy_m": [cx, cy],
            "rows": rows,
            "concentration_ratio": conc,
            "concentration_growing": bool(all(b > a for a, b in zip(conc, conc[1:]))),
            "corner_peak_growing": bool(all(b > a for a, b in zip(peak, peak[1:]))),
            "reading": "power share divided by volume share is 1.0 for a "
                       "volume-proportional field; a value that keeps GROWING "
                       "with refinement is the grid chasing a singularity, "
                       "while one that settles is a resolved feature"}


def cylinder_uniformity(grids=(48, 64, 80)) -> dict:
    """Why the corrected cylinder field is already flat."""
    p = heatr3d.Params()
    omega = 2.0 * np.pi * p.freq_hz
    mu0 = 4.0e-7 * np.pi
    # conductive skin depth, and the loss tangent that decides which regime
    delta = float(np.sqrt(2.0 / (omega * mu0 * p.sigma_doped)))
    eps = p.eps_doped * heatr3d.EPS0
    loss_tangent = float(p.sigma_doped / (omega * eps))
    radius = harness.PART_DIAM_M / 2.0
    rows = []
    for n in grids:
        grid, part, Q = _qfield("circle", n)
        X, Y, _ = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
        r = np.sqrt(X ** 2 + Y ** 2)
        interior = part & ((radius - r) > 1.5 * grid.h)
        qi = Q[interior]
        rows.append({"n": n, "interior_cv": float(qi.std() / qi.mean()),
                     "interior_max_over_mean": float(qi.max() / qi.mean()),
                     "in_part_cv": float(Q[part].std() / Q[part].mean())})
    cv = [r["interior_cv"] for r in rows]
    return {"skin_depth_m": delta, "part_radius_m": radius,
            "skin_depth_over_radius": delta / radius,
            "loss_tangent_sigma_over_omega_eps": loss_tangent,
            "rows": rows, "interior_cv_falling":
                bool(all(b < a for a, b in zip(cv, cv[1:]))),
            "reading": "the skin depth is enormous compared with the part, and "
                       "the loss tangent places the doped part far on the "
                       "conductive side, so there is no attenuation mechanism "
                       "to make the interior field vary; the interior "
                       "coefficient of variation should be small and falling"}


def build(grids=(48, 64, 80)) -> dict:
    doc = {"what": "S2 Task 5 mechanism checks",
           "lshape_outlier": lshape_corner_share(grids),
           "cylinder_null": cylinder_uniformity(grids),
           "source_anomalies":
               "heatr3d_eqs02_rerank/RERANK_REPORT.md: lshape sigma_T "
               "33.204 -> 66.420 (+100.0 %) under the corrected default; "
               "cylinder corrected-design inversion benefit +0.3 % (null)"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    return doc


if __name__ == "__main__":
    d = build()
    print(json.dumps({"lshape_concentration": d["lshape_outlier"]["concentration_ratio"],
                      "growing": d["lshape_outlier"]["concentration_growing"],
                      "skin_over_radius": d["cylinder_null"]["skin_depth_over_radius"],
                      "interior_cv": [r["interior_cv"] for r in d["cylinder_null"]["rows"]]},
                     indent=1))
