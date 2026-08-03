"""S2 campaign harness: one (shape, grid) case, dual read states.

heatr3d.py is DRIVEN, never edited (the heatr3d_s4_flir isolation pattern).

COST DISCIPLINE: one EQS solve per case drives BOTH read states, because the
second march is re-driven through the documented run(qrf_override=...) hook,
which skips the internal EQS entirely. At n=96 the EQS is ~231 s and a march is
~330 s, so sharing the solve is most of the saving.

SCORING REUSES THE PHASE A/C CODE, deliberately. The shape metrics come from
solve3d/shape_metrics.py and the evaluation grid from solve3d/gates.py, both
imported UNMODIFIED. That means S2's same-engine bands and Phase A's
cross-family bands are computed by literally the same functions on literally
the same grid, so the consistency statement in the report compares like with
like instead of two conventions that happen to agree.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import heatr3d
from solve3d import gates as sg
from solve3d import shape_metrics as sm

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PART_DIAM_M = 0.020
THRESHOLDS = (0.8, 0.9)


# --------------------------------------------------------------------------- #
# Geometry: the analytic nominal, replicated from heatr3d.make_geometry
# --------------------------------------------------------------------------- #
def make_part(grid: heatr3d.Grid, shape: str) -> np.ndarray:
    """The voxel part mask, full-height extrusions so the case is z-invariant."""
    if shape == "circle":
        return heatr3d.make_geometry(grid, "cylinder", diam=PART_DIAM_M)
    if shape in ("square", "lshape"):
        return heatr3d.make_geometry(grid, shape, diam=PART_DIAM_M,
                                     zspan=grid.L)
    raise ValueError(f"unknown shape {shape!r}")


def nominal_mask_2d(shape: str, diam: float = PART_DIAM_M) -> np.ndarray:
    """The ANALYTIC nominal cross-section on the shared evaluation grid.

    Deliberately NOT any grid's voxel mask: scoring a convergence study against
    a target that itself moves with the grid would measure nothing. Replicated
    from heatr3d.make_geometry's own definitions so the nominal is the shape
    heatr3d thinks it is meshing."""
    x, y, _, _ = sg.eval_grid_axes()
    X, Y = np.meshgrid(x, y, indexing="ij")
    half = diam / 2.0
    if shape == "circle":
        return np.sqrt(X ** 2 + Y ** 2) <= half
    if shape == "square":
        return (np.abs(X) <= half) & (np.abs(Y) <= half)
    if shape == "lshape":
        thick = diam * (5.0 / 12.0)
        x0, y0 = -half, -half
        vert = (X >= x0) & (X <= x0 + thick) & (Y >= y0) & (Y <= y0 + diam)
        horiz = (Y >= y0) & (Y <= y0 + thick) & (X >= x0) & (X <= x0 + diam)
        return vert | horiz
    raise ValueError(f"unknown shape {shape!r}")


def _to_eval_grid(T: np.ndarray, grid: heatr3d.Grid) -> np.ndarray:
    """heatr3d voxel field -> the shared evaluation grid, (nz, nx, ny)."""
    interp = RegularGridInterpolator((grid.x, grid.y, grid.z),
                                     np.asarray(T, dtype=float),
                                     method="linear", bounds_error=False,
                                     fill_value=None)
    pts, shp, _ = sg.eval_grid_points()
    return interp(pts).reshape(shp)


# --------------------------------------------------------------------------- #
# Metrics for one read state
# --------------------------------------------------------------------------- #
def read_metrics(T: np.ndarray, grid: heatr3d.Grid, part: np.ndarray,
                 shape: str) -> dict:
    """Shape metrics on the shared grid + sigma_T (diagnostic) on the voxels."""
    Te = _to_eval_grid(T, grid)
    nominal = nominal_mask_2d(shape)
    per = []
    for i in range(Te.shape[0]):
        phi = sg.phase_fraction_phi(Te[i])
        row = {}
        for t in THRESHOLDS:
            m = phi >= t
            key = f"phi{t:g}".replace(".", "p")
            row[f"iou_{key}"] = sm.iou(m, nominal)
            row[f"in_part_{key}"] = sm.in_part_melt_fraction(m, nominal)
            row[f"out_of_part_{key}"] = sm.out_of_part_fraction(m, nominal)
        per.append(row)
    agg = {k: float(np.nanmean([r[k] for r in per])) for k in per[0]}
    agg.update({k + "__plane_spread":
                float(np.nanmax([r[k] for r in per]) - np.nanmin([r[k] for r in per]))
                for k in per[0]})
    tp = np.asarray(T)[part]
    agg.update({"sigma_T_c": float(tp.std()), "T_max_c": float(tp.max()),
                "T_mean_c": float(tp.mean()),
                "part_mean_phi": float(heatr3d.phase_fraction(np.asarray(T),
                                                              heatr3d.Params())[0][part].mean())})
    return agg, Te


# --------------------------------------------------------------------------- #
# One case
# --------------------------------------------------------------------------- #
def run_case(shape: str, n: int, t_ref_s: float,
             p: heatr3d.Params | None = None,
             max_time_s: float = 1500.0, save_fields: bool = True) -> dict:
    """One EQS solve + the two pre-registered read states."""
    p = p or heatr3d.Params(phase_update="enthalpy")   # corrected defaults
    grid = heatr3d.Grid(n=n)
    part = make_part(grid, shape)

    t0 = time.perf_counter()
    gamma = heatr3d.build_gamma(part, p)
    V = heatr3d.solve_eqs_3d(gamma, grid, p)
    Q = heatr3d.compute_qrf_3d(V, gamma, grid, p, doped=part, premix=False,
                               qrf_gradient="masked")
    wall_eqs = time.perf_counter() - t0

    rec = {"shape": shape, "n": n, "h_m": grid.h,
           "n_voxels_in_part": int(part.sum()),
           "part_volume_m3": float(part.sum() * grid.dV),
           "p_total_w": float(Q.sum() * grid.dV),
           "wall_eqs_s": wall_eqs, "reads": {}, "gates": {}}

    fields = {}
    # ---- READ 1: melt onset (heatr3d's native stop) --------------------- #
    t0 = time.perf_counter()
    r1 = heatr3d.run(grid, part, p, qrf_override=Q, max_time_s=max_time_s,
                     phi_target=0.90)
    w1 = time.perf_counter() - t0
    m1, Te1 = read_metrics(r1.T_phi90, grid, part, shape)
    m1.update({"t90_s": float(r1.t_phi90_s), "reached": bool(r1.reached),
               "wall_march_s": w1})
    rec["reads"]["melt_onset"] = m1
    fields["melt_onset"] = Te1
    rec["gates"]["melt_onset"] = {
        "energy_residual_frac": float(r1.energy_residual_frac),
        "clamp_bound": bool(r1.clamp_bound),
        "cfl_violated": bool(r1.cfl_violated),
        "n_substeps_used": int(r1.n_substeps_used)}

    # ---- READ 2: fixed absolute time, common to every grid --------------- #
    t0 = time.perf_counter()
    r2 = heatr3d.run(grid, part, p, qrf_override=Q, max_time_s=float(t_ref_s),
                     phi_target=2.0)
    w2 = time.perf_counter() - t0
    m2, Te2 = read_metrics(r2.T_final, grid, part, shape)
    m2.update({"t_ref_s": float(t_ref_s), "wall_march_s": w2,
               "precedes_melt_onset": bool(float(t_ref_s) < float(r1.t_phi90_s))})
    rec["reads"]["heating_fixed_time"] = m2
    fields["heating_fixed_time"] = Te2
    rec["gates"]["heating_fixed_time"] = {
        "energy_residual_frac": float(r2.energy_residual_frac),
        "clamp_bound": bool(r2.clamp_bound),
        "cfl_violated": bool(r2.cfl_violated),
        "n_substeps_used": int(r2.n_substeps_used)}

    if save_fields:
        RESULTS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(RESULTS / f"field_{shape}_n{n}.npz",
                            melt_onset=fields["melt_onset"],
                            heating_fixed_time=fields["heating_fixed_time"])
        rec["field_npz"] = f"field_{shape}_n{n}.npz"
    return rec
