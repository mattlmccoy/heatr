"""Melt-region shape-fidelity objective with the target indicator as an ARGUMENT.

    J(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi(x))^2

Identical to `shape_objective` in every respect except that chi is passed in
rather than taken as the solve-grid binary raster. Passing the raster back in
reproduces `shape_objective.shape_J_and_seed` to the last bit, which is the
flag-off identity that keeps the previously gated numbers comparable
(`test_raster_chi_reproduces_shape_objective_bitwise`).

The stop convention is unchanged: t_stop = argmin over the arm's OWN stored
trajectory of J, so the envelope theorem removes the dt*/ds term and the
gradient is the partial derivative at that index. `at_horizon` is flagged.

REPORTING CONVENTION. Two intersection-over-union numbers are produced and
both are always labelled:

  IoU_raster   melted region (phi >= 0.5) against the BINARY part mask. This
               is the number every earlier report quotes and it is kept so the
               comparison against `out_lib` and `out_ms` is like for like.
  IoU_area     the area-weighted overlap of the melt fraction with the
               area-fill chi, sum(min) / sum(max). It is the grid-independent
               reading and it is NOT comparable to the earlier reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MELT_LEVEL = 0.5


def _chk(chi, T):
    c = np.asarray(chi, dtype=float)
    if c.shape != np.asarray(T).shape:
        raise ValueError(f"chi shape {c.shape} does not match the field {np.shape(T)}")
    return c


def phi_of(T: np.ndarray, case) -> np.ndarray:
    p = case.pins
    return np.clip((np.asarray(T, dtype=float) - p.t_pc_c) / p.dt_pc_c + 0.5, 0.0, 1.0)


def phi_field(T: np.ndarray, case) -> tuple[np.ndarray, np.ndarray]:
    p = case.pins
    raw = (np.asarray(T, dtype=float) - p.t_pc_c) / p.dt_pc_c + 0.5
    return np.clip(raw, 0.0, 1.0), (raw > 0.0) & (raw < 1.0)


def J_and_seed(T: np.ndarray, case, chi: np.ndarray) -> tuple[float, np.ndarray]:
    """J at one time level and dJ/dT at that time level."""
    c = _chk(chi, T)
    phi, inside = phi_field(T, case)
    d = phi - c
    J = float(np.sum(d * d))
    g = 2.0 * d * inside / case.pins.dt_pc_c
    return J, np.asarray(g, dtype=float)


@dataclass(frozen=True)
class Stop:
    index: int
    J: float
    time_s: float
    at_horizon: bool
    J_first: float
    J_curve: np.ndarray


def J_curve(tr, case, chi: np.ndarray) -> np.ndarray:
    return np.asarray([J_and_seed(tr.T_at_end(n), case, chi)[0]
                       for n in range(tr.n_outer)], dtype=float)


def optimal_stop(tr, case, chi: np.ndarray) -> Stop:
    jc = J_curve(tr, case, chi)
    i = int(np.argmin(jc))
    return Stop(index=i, J=float(jc[i]), time_s=float(tr.time_s[i]),
                at_horizon=bool(i == tr.n_outer - 1), J_first=float(jc[0]),
                J_curve=jc)


def area_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Area-weighted overlap, sum(min(a, b)) / sum(max(a, b)).

    Reduces exactly to the set intersection over union when both fields are
    indicator functions, which is what makes it the right generalization to a
    fractional target.
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    den = float(np.sum(np.maximum(aa, bb)))
    return float(np.sum(np.minimum(aa, bb))) / den if den > 0 else float("nan")


def metrics(T: np.ndarray, case, chi: np.ndarray) -> dict:
    """Both readings, both labelled."""
    phi, _ = phi_field(T, case)
    melted = phi >= MELT_LEVEL
    part = case.part_mask
    n_part = int(part.sum())
    inter = int(np.sum(melted & part))
    union = int(np.sum(melted | part))
    tp = np.asarray(T, dtype=float)[part]
    tbar = float(np.mean(tp))
    return {
        "IoU": (inter / union) if union else float("nan"),
        "IoU_area": area_iou(phi, chi),
        "bed_melt_pct_of_part": 100.0 * int(np.sum(melted & ~part)) / max(n_part, 1),
        "part_under_melt_pct": 100.0 * int(np.sum(part & ~melted)) / max(n_part, 1),
        "melted_cells": int(melted.sum()),
        "part_cells": n_part,
        "mean_phi_part": float(np.mean(phi[part])),
        "mean_phi_bed": float(np.mean(phi[~part])),
        "sigma_T_at_stop_c": float(np.sqrt(np.mean((tp - tbar) ** 2))),
        "mean_T_part_c": tbar,
    }


def full_metrics(tr, case, chi: np.ndarray) -> dict:
    st = optimal_stop(tr, case, chi)
    T = tr.T_at_end(st.index)
    out = {"t_stop_index": st.index, "t_stop_s": st.time_s,
           "t_stop_at_horizon": st.at_horizon, "J": st.J,
           "J_per_part_cell": st.J / max(int(case.part_mask.sum()), 1),
           "J_at_first_step": st.J_first}
    out.update(metrics(T, case, chi))
    return out
