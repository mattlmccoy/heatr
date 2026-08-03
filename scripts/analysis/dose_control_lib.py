#!/usr/bin/env python3
"""Analysis library for the dose-controlled three-arm 2-D grading study.

Two jobs:

1. Equal-cross-sectional-area normalization, reproducing the ESTABLISHED
   convention of ``scripts_p0_rerun/equal_ca_scale.py`` (target A* = area of a
   20 mm-diameter circle = 314.159 mm^2, solver-exact rasterized masks) but at
   an arbitrary grid resolution.
2. Reading transient sigma_T in CELSIUS at each arm's own mean-melt-fraction
   phi_bar = 0.90 crossing, from a HEATR run's ``time_series.json``.

sigma_T identity used throughout:  sigma_T(t) = ui_rms(t) * (T_bar(t) - T_amb).
``ui_rms_part`` is the solver's RMS relative non-uniformity of the part
temperature rise, so multiplying by the mean rise recovers the standard
deviation in Celsius.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rfam_eqs_coupled import _single_part_mask_and_fill  # noqa: E402

A_STAR_MM2: float = math.pi * (10.0 ** 2)  # 314.159 mm^2, the 20 mm-dia circle
CHAMBER_MM: float = 60.0
AMBIENT_C: float = 23.0
PHI_TARGET: float = 0.90


# ---------------------------------------------------------------------------
# Geometry / equal-area normalization
# ---------------------------------------------------------------------------

def make_grid(
    nx: int,
    ny: int,
    chamber_x: float = 0.06,
    chamber_y: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinate vectors matching the solver's grid construction."""
    x = np.linspace(-chamber_x / 2.0, chamber_x / 2.0, nx)
    y = np.linspace(-chamber_y / 2.0, chamber_y / 2.0, ny)
    return x, y


def _part_dict(shape: str, w_m: float, h_m: float, rot_deg: float = 0.0) -> dict:
    return {
        "shape": shape,
        "width": w_m,
        "height": h_m,
        "rotation_deg": rot_deg,
        "center_x": 0.0,
        "center_y": 0.0,
        "n_circle_pts": 800,
    }


def rasterized_mask(
    shape: str, w_m: float, h_m: float, x: np.ndarray, y: np.ndarray,
    rot_deg: float = 0.0,
) -> np.ndarray:
    _poly, mask, _fill = _single_part_mask_and_fill(
        x, y, _part_dict(shape, w_m, h_m, rot_deg)
    )
    return mask


def rasterized_area_mm2(
    shape: str, w_m: float, h_m: float, x: np.ndarray, y: np.ndarray,
    rot_deg: float = 0.0,
) -> tuple[float, int]:
    """Filled area of the solver's own mask, in mm^2, plus the cell count."""
    mask = rasterized_mask(shape, w_m, h_m, x, y, rot_deg)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    n = int(mask.sum())
    return n * dx * dy * 1e6, n


def thin_feature_mm(
    shape: str, w_m: float, h_m: float, x: np.ndarray, y: np.ndarray,
    rot_deg: float = 0.0,
) -> float:
    """Minimum filled run length across rows and columns, in mm.

    For a POINTED shape this is always about one cell at any resolution because
    an apex is a geometric point. It is a measure of POINTEDNESS, not of
    under-resolution.
    """
    mask = rasterized_mask(shape, w_m, h_m, x, y, rot_deg)
    dx_mm = float(x[1] - x[0]) * 1e3
    runs: list[int] = []
    for row in mask:
        if row.any():
            runs.append(int(row.sum()))
    for col in mask.T:
        if col.any():
            runs.append(int(col.sum()))
    return (min(runs) * dx_mm) if runs else 0.0


def solve_equal_area_scale(
    shape: str,
    w0_m: float,
    h0_m: float,
    x: np.ndarray,
    y: np.ndarray,
    rot_deg: float = 0.0,
    tol_frac: float = 0.005,
    max_iter: int = 8,
) -> dict[str, float]:
    """Solve for the linear scale whose rasterized area is closest to A*.

    Identical fixed-point iteration to scripts_p0_rerun/equal_ca_scale.py so
    that the 120-grid result reproduces equal_ca_scale.json exactly.
    """
    a0, _n0 = rasterized_area_mm2(shape, w0_m, h0_m, x, y, rot_deg)
    if a0 <= 0:
        raise ValueError(f"mask build produced zero area for shape {shape!r}")

    scale = math.sqrt(A_STAR_MM2 / a0)
    a_new = a0
    for _ in range(max_iter):
        a_new, _ = rasterized_area_mm2(
            shape, w0_m * scale, h0_m * scale, x, y, rot_deg
        )
        if abs(a_new - A_STAR_MM2) / A_STAR_MM2 < tol_frac:
            break
        scale *= math.sqrt(A_STAR_MM2 / a_new)

    w_new = w0_m * scale
    h_new = h0_m * scale
    mask = rasterized_mask(shape, w_new, h_new, x, y, rot_deg)
    dx_mm = float(x[1] - x[0]) * 1e3
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    cells_across_x = int(cols.max() - cols.min() + 1) if len(cols) else 0
    cells_across_y = int(rows.max() - rows.min() + 1) if len(rows) else 0
    thin = thin_feature_mm(shape, w_new, h_new, x, y, rot_deg)
    return {
        "shape": shape,
        "base_w_mm": w0_m * 1e3,
        "base_h_mm": h0_m * 1e3,
        "base_A_mm2": a0,
        "scale": scale,
        "new_w_m": w_new,
        "new_h_m": h_new,
        "new_w_mm": w_new * 1e3,
        "new_h_mm": h_new * 1e3,
        "new_A_mm2": a_new,
        "err_pct": 100.0 * (a_new - A_STAR_MM2) / A_STAR_MM2,
        "abs_err_pct": abs(100.0 * (a_new - A_STAR_MM2) / A_STAR_MM2),
        "n_cells": int(mask.sum()),
        "cell_mm": dx_mm,
        "thin_mm": thin,
        "thin_cells": thin / dx_mm if dx_mm else 0.0,
        "cells_across_x": cells_across_x,
        "cells_across_y": cells_across_y,
        "gap_mm": (CHAMBER_MM - w_new * 1e3) / 2.0,
        "gap_y_mm": (CHAMBER_MM - h_new * 1e3) / 2.0,
    }


# ---------------------------------------------------------------------------
# sigma_T at the phi_bar = 0.90 crossing
# ---------------------------------------------------------------------------

def sigma_T_series(ts: dict[str, Any], ambient_c: float = AMBIENT_C) -> list[float]:
    ui = ts["ui_rms_part"]
    tb = ts["mean_T_part_c"]
    return [float(u) * (float(t) - ambient_c) for u, t in zip(ui, tb)]


def read_at_phi90(
    ts: dict[str, Any],
    ambient_c: float = AMBIENT_C,
    phi_target: float = PHI_TARGET,
) -> dict[str, Any]:
    """Read the transient state at the FIRST phi_bar >= phi_target crossing.

    Any arm that never crosses is reported NOT REACHED with sigma_T_c=None and
    is excluded from ratios. It is never read at its end state.
    """
    phi = [float(v) for v in ts["mean_phi_part"]]
    sig = sigma_T_series(ts, ambient_c=ambient_c)
    idx: Optional[int] = None
    for i, p in enumerate(phi):
        if p >= phi_target:
            idx = i
            break
    if idx is None:
        return {
            "reached": False,
            "index": None,
            "t_s": None,
            "sigma_T_c": None,
            "mean_T_c": None,
            "max_T_c": None,
            "max_phi_reached": max(phi) if phi else None,
        }
    return {
        "reached": True,
        "index": idx,
        "t_s": float(ts["time_s"][idx]),
        "sigma_T_c": float(sig[idx]),
        "mean_T_c": float(ts["mean_T_part_c"][idx]),
        "max_T_c": float(ts["max_T_part_c"][idx]),
        "ui_rms": float(ts["ui_rms_part"][idx]),
        "phi_at_crossing": phi[idx],
        "max_phi_reached": max(phi),
    }


def benefit_pct(ref: Optional[float], arm: Optional[float]) -> Optional[float]:
    """Percent change of *arm* relative to *ref*. Negative = improvement."""
    if ref is None or arm is None:
        return None
    if ref == 0.0:
        return None
    return 100.0 * (arm - ref) / ref


MAX_CLIP_FRAC = 1e-6        # frac_cells_dT_clipped must be essentially zero
MAX_RESID_FRAC = 0.10       # |energy residual| / integrated dose
MAX_PHI_JUMP = 0.20         # per outer step; larger means the melt front is
                            # crossed inside one step and there is no resolved
                            # phi_bar = 0.90 crossing to read sigma_T at


def stability_gate(
    ts: dict[str, Any],
    max_clip_frac: float = MAX_CLIP_FRAC,
    max_resid_frac: float = MAX_RESID_FRAC,
    max_phi_jump: float = MAX_PHI_JUMP,
    upto_index: Optional[int] = None,
) -> dict[str, Any]:
    """Numerical-validity gate on a HEATR time series.

    Three independent failure modes, all seen in the 240-grid melt-front
    blowup that halted this study:

    dT_clip
        The explicit thermal update hit ``max_deltaT_per_step_c`` on a nonzero
        fraction of cells, so the temperature trajectory is not the solution of
        the PDE but a clipped approximation of it.
    energy_residual
        ``|energy_balance_residual| / energy_doped`` exceeded the tolerance,
        meaning energy is not conserved to a usable accuracy.
    melt_front_unresolved
        phi_bar advanced by more than ``max_phi_jump`` in a single outer step,
        so the phi_bar = 0.90 read point is a discontinuity, not a crossing.

    ``upto_index`` restricts the window to steps 0..upto_index inclusive. Pass
    the phi_bar = 0.90 crossing index to certify the read point itself. Runs
    driven far past the melt criterion accumulate clipping and residual in the
    fully dense overshoot tail, which does not invalidate a measurement taken
    before it. Both windows are reported; neither replaces the other.
    """
    def _cut(v: list) -> list:
        return v if upto_index is None else v[: int(upto_index) + 1]

    clip = _cut([float(v) for v in ts["frac_cells_dT_clipped"]])
    resid = _cut([float(v) for v in ts["energy_balance_residual_J_per_m"]])
    dose = _cut([float(v) for v in ts["energy_doped_J_per_m"]])
    phi = _cut([float(v) for v in ts["mean_phi_part"]])

    max_clip = max(clip) if clip else 0.0
    resid_frac = [
        abs(r) / max(d, 1e-12) for r, d in zip(resid, dose)
    ]
    max_resid_frac_obs = max(resid_frac) if resid_frac else 0.0
    jumps = [phi[i + 1] - phi[i] for i in range(len(phi) - 1)]
    max_jump = max(jumps) if jumps else 0.0

    failures: list[str] = []
    if max_clip > max_clip_frac:
        failures.append("dT_clip")
    if max_resid_frac_obs > max_resid_frac:
        failures.append("energy_residual")
    if max_jump > max_phi_jump:
        failures.append("melt_front_unresolved")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "max_frac_cells_dT_clipped": max_clip,
        "max_abs_resid_frac_of_dose": max_resid_frac_obs,
        "max_phi_jump_per_step": max_jump,
        "max_dT_raw_c": (max(float(v) for v in _cut(ts["max_dT_raw_c"]))
                         if "max_dT_raw_c" in ts else None),
        "upto_index": upto_index,
        "thresholds": {
            "max_clip_frac": max_clip_frac,
            "max_resid_frac": max_resid_frac,
            "max_phi_jump": max_phi_jump,
        },
    }


def peak_T_over_run(ts: dict[str, Any]) -> float:
    return float(max(float(v) for v in ts["max_T_part_c"]))
