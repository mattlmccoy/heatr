"""Phase A close-out STEP 2: the DESIGN-RELEVANT parity gate.

RUNS IN THE geo-prewarp VENV (scipy: EDT + the voxel-field interpolation):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ./.venv312/bin/python -m solve3d.shape_gate

Gates Phase A parity on the quantity Phase C actually optimizes -- the
melt/density field against the nominal shape bounds -- rather than on sigma_T.
Matt's recorded objective (spec commit d298c6d): "make every single part a
fully dense part if and only if it falls within the nominal shape bounds ...
willing to compromise a little bit on density (maybe 80-90%) if we can achieve
a better shape within the bounds."

METRICS (solve3d/shape_metrics.py) at the melt-onset read state, both engines
on the shared evaluation grid, scored against the ANALYTIC nominal shape:
  (a) IoU of {phi >= 0.8} and {phi >= 0.9}
  (b) symmetric surface distance between the phi = 0.9 fronts [mm]
  (c) out-of-part (bed) melted fraction per engine, and their difference

TOLERANCE RULE -- DECLARED BEFORE ANY NUMBER WAS COMPUTED, and the same
triangle-inequality SUM x 1.5 used in STEP 1:
  * IoU is scored through its Jaccard DISTANCE, 1 - IoU, which is a true metric
    and therefore obeys the triangle inequality exactly:
        (1 - IoU_cross)  <=  1.5 * [ (1 - IoU_heatr3d_self) + (1 - IoU_dolfinx_self) ]
  * the front distance is already a distance:
        SSD_cross  <=  1.5 * ( SSD_heatr3d_self + SSD_dolfinx_self )
  * the bed-melt fraction is scored on its absolute difference:
        |f_d - f_h|  <=  1.5 * ( |f_h64 - f_h96| + |f_dcoarse - f_dmid| )

SELF-SPREAD PROVENANCE. heatr3d's self-spread is measured on its n=64 vs n=96
extruded-circle pair; dolfinx's on its coarse-vs-mid and mid-vs-fine pairs (the
MAX, conservative). Both are the CIRCLE / coupling-off configuration, and the
resulting band is applied to all four arms -- exactly the scope Task 1 already
used for the original tolerances. Stated, not hidden.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from solve3d import gates, shape_metrics as sm

RESULTS = Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS / "phase_a_shape_gate.json"
THRESHOLDS = (0.8, 0.9)


def _phi(T):
    return gates.phase_fraction_phi(T)


def heatr3d_on_eval_grid(npz_name: str) -> np.ndarray:
    """heatr3d's melt-onset voxel field, trilinearly sampled onto the shared
    evaluation grid. Returns (nz, nx, ny)."""
    z = np.load(RESULTS / npz_name)
    interp = RegularGridInterpolator(
        (np.asarray(z["x"]), np.asarray(z["y"]), np.asarray(z["z"])),
        np.asarray(z["T_phi90"], dtype=float),
        method="linear", bounds_error=True)
    pts, shp, _ = gates.eval_grid_points()
    return interp(pts).reshape(shp)


def dolfinx_on_eval_grid(tag: str) -> np.ndarray:
    return np.asarray(np.load(RESULTS / f"eval_dolfinx_{tag}.npz")["T"],
                      dtype=float)


def compare(T_a: np.ndarray, T_b: np.ndarray, shape: str) -> dict:
    """All shape metrics for a pair of melt-onset fields on the eval grid."""
    part = gates.nominal_part_mask(shape)
    _, _, _, h = gates.eval_grid_axes()
    per_plane = [sm.plane_metrics(T_a[i], T_b[i], part, h, _phi, THRESHOLDS)
                 for i in range(T_a.shape[0])]
    return sm.aggregate_planes(per_plane)


def _self_spreads() -> dict:
    """Each engine's OWN grid/mesh spread of the shape metrics, PER SHAPE.

    ONE uniform rule: for each shape the band uses THAT SHAPE's heatr3d
    n=64-vs-n=96 spread and THAT SHAPE's dolfinx coarse-vs-mid spread. Both are
    1.5x LINEAR refinement pairs, so the two halves are commensurable, and both
    are available for both shapes -- no MAX cherry-picking.

    Why per-shape and not one circle-derived band for everything: the circle
    has ZERO out-of-part (bed) melt on every grid of both engines, so a
    circle-derived bed-melt band is identically zero and cannot bound the
    square, which DOES spill at its corners (~4% of part volume in both
    engines). Borrowing the circle band across shapes is the same class of
    scope error as borrowing a same-method band across engines. The square's
    own spread was MEASURED (heatr3d square n=64 + dolfinx square coarse) rather
    than the band being widened.

    The circle's mid-vs-fine pair is also reported, as a check that the dolfinx
    spread actually shrinks under further refinement.
    """
    out = {"circle": {}, "square": {}}
    h_c64 = heatr3d_on_eval_grid("anchor_heatr3d_circle_n64.npz")
    h_c96 = heatr3d_on_eval_grid("anchor_heatr3d_circle_n96.npz")
    out["circle"]["heatr3d_n64_vs_n96"] = compare(h_c64, h_c96, "circle")
    d_cc = dolfinx_on_eval_grid("circle_off_coarse")
    d_cm = dolfinx_on_eval_grid("circle_off_mid")
    d_cf = dolfinx_on_eval_grid("circle_off_fine")
    out["circle"]["dolfinx_coarse_vs_mid"] = compare(d_cc, d_cm, "circle")
    out["circle"]["dolfinx_mid_vs_fine"] = compare(d_cm, d_cf, "circle")

    h_s64 = heatr3d_on_eval_grid("anchor_heatr3d_square_n64.npz")
    h_s96 = heatr3d_on_eval_grid("anchor_heatr3d_square_n96.npz")
    out["square"]["heatr3d_n64_vs_n96"] = compare(h_s64, h_s96, "square")
    d_sc = dolfinx_on_eval_grid("square_off_coarse")
    d_sm = dolfinx_on_eval_grid("square_off")
    out["square"]["dolfinx_coarse_vs_mid"] = compare(d_sc, d_sm, "square")
    return out


def _extract(m: dict) -> dict:
    """The three gate quantities out of a comparison record."""
    return {
        "jaccard_dist_phi0p8": 1.0 - m["phi0p8"]["iou"],
        "jaccard_dist_phi0p9": 1.0 - m["phi0p9"]["iou"],
        "front_ssd_mm": m["front_ssd_mm_phi0p9"],
        "bed_melt_absdiff_phi0p8": m["phi0p8"]["out_of_part_frac_abs_diff"],
        "bed_melt_absdiff_phi0p9": m["phi0p9"]["out_of_part_frac_abs_diff"],
        "in_part_absdiff_phi0p8": m["phi0p8"]["in_part_melt_frac_abs_diff"],
        "in_part_absdiff_phi0p9": m["phi0p9"]["in_part_melt_frac_abs_diff"],
    }


GATE_KEYS = ("jaccard_dist_phi0p8", "jaccard_dist_phi0p9", "front_ssd_mm",
             "bed_melt_absdiff_phi0p8", "bed_melt_absdiff_phi0p9",
             "in_part_absdiff_phi0p8", "in_part_absdiff_phi0p9")


def build() -> dict:
    refs = json.loads((RESULTS / "task4_heatr3d_refs.json").read_text())["runs"]
    selves = _self_spreads()

    band_by_shape, spread_detail = {}, {}
    for shape in ("circle", "square"):
        s_h = _extract(selves[shape]["heatr3d_n64_vs_n96"])
        s_d = _extract(selves[shape]["dolfinx_coarse_vs_mid"])
        detail = {"heatr3d_n64_vs_n96": s_h, "dolfinx_coarse_vs_mid": s_d}
        if "dolfinx_mid_vs_fine" in selves[shape]:
            detail["dolfinx_mid_vs_fine"] = _extract(
                selves[shape]["dolfinx_mid_vs_fine"])
        spread_detail[shape] = detail
        b = {}
        for k in GATE_KEYS:
            b[k] = {"heatr3d_self_spread": s_h[k],
                    "dolfinx_self_spread": s_d[k],
                    "combined_spread": s_h[k] + s_d[k],
                    "tolerance": gates.combine_spreads(s_h[k], s_d[k])}
            b[k]["degenerate_zero_width"] = bool(b[k]["tolerance"] == 0.0)
        band_by_shape[shape] = b

    arms = {}
    for name, ref in refs.items():
        shape = name.rsplit("_", 1)[0]
        band = band_by_shape[shape]
        T_h = heatr3d_on_eval_grid(ref["npz"])
        T_d = dolfinx_on_eval_grid(name)
        m = compare(T_d, T_h, shape)        # a = dolfinx, b = heatr3d
        meas = _extract(m)
        checks = {}
        for k in GATE_KEYS:
            tol_k = band[k]["tolerance"]
            if tol_k > 0.0:
                margin = meas[k] / tol_k
            else:
                margin = 0.0 if meas[k] == 0.0 else float("inf")
            checks[k] = {"measured": meas[k], "tolerance": tol_k,
                         "pass": bool(meas[k] <= tol_k),
                         "margin": margin,
                         "degenerate_zero_width_band":
                             band[k]["degenerate_zero_width"]}
        checks["arm_ok"] = all(c["pass"] for c in checks.values()
                               if isinstance(c, dict))
        arms[name] = {"shape": shape, "metrics": m, "checks": checks}

    doc = {
        "what": "Phase A close-out STEP 2: parity on the SHAPE/DENSITY field, "
                "the quantity Phase C optimizes. sigma_T is not scored here.",
        "objective_source": "spec commit d298c6d (Matt, verbatim intent): "
                            "dense if and only if inside the nominal shape "
                            "bounds; 80-90% density acceptable for better shape",
        "thresholds": list(THRESHOLDS),
        "band_rule": "per SHAPE: 1.5 * (that shape's heatr3d n64-vs-n96 spread "
                     "+ that shape's dolfinx coarse-vs-mid spread); both are "
                     "1.5x linear refinement pairs. Bands are NOT borrowed "
                     "across shapes -- the circle's bed melt is identically "
                     "zero on every grid and cannot bound the square's.",
        "eval_grid": {"half_xy_m": gates.EVAL_HALF_XY_M,
                      "n_xy": gates.EVAL_N_XY,
                      "pixel_m": gates.eval_grid_axes()[3],
                      "z_planes_m": list(gates.EVAL_Z_M),
                      "nominal_part": "ANALYTIC shape, not either engine's mask"},
        "combination_rule": gates.COMBINATION_RULE,
        "safety_factor": gates.CROSS_FAMILY_SAFETY,
        "self_spread_detail": spread_detail,
        "self_spreads_full": selves,
        "band_by_shape": band_by_shape,
        "arms": arms,
        "all_arms_pass": all(a["checks"]["arm_ok"] for a in arms.values()),
    }
    gates.write_json(OUT_JSON.name, doc)
    return doc


def main() -> int:
    doc = build()
    print(json.dumps({"band_by_shape": {sh: {k: v["tolerance"] for k, v in b.items()}
                                         for sh, b in doc["band_by_shape"].items()},
                      "arms": {k: {kk: vv["pass"] for kk, vv in
                                   v["checks"].items() if isinstance(vv, dict)}
                               for k, v in doc["arms"].items()},
                      "all_arms_pass": doc["all_arms_pass"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
