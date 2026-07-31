"""Shape-fidelity objective: the melted region should be the nominal part.

    J(s, t_stop) = sum over the WHOLE domain of w(x) * (phi(x, t_stop) - chi(x))^2

with phi the melt-fraction state, chi the nominal part indicator, and w = 1 by
default. Three properties the temperature-uniformity metric it replaces did not
have:

  * it reads melt fraction, not temperature, so the latent plateau cannot
    flatter it;
  * it is summed over the bed as well as the part, so melt escaping into the
    powder (part growth) is penalized;
  * under-melting a part cell and melting a bed cell cost exactly the same, so
    dose starvation is not a free move.

t_stop is a design variable chosen per arm as the argmin of J along that arm's
own trajectory. Because the stop is optimized to stationarity, the envelope
theorem gives dJ*/ds = (partial J / partial s) at the argmin: there is NO
dt*/ds term of the kind the melt-onset objective needed. That claim is checked
numerically in `gate_shape.py`, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MELT_LEVEL = 0.5          # melted region is phi >= MELT_LEVEL
WINDOW_LO_C = 175.0
WINDOW_HI_C = 185.0


def chi_part(case) -> np.ndarray:
    """Nominal part indicator: the rasterized binary part mask."""
    return case.part_mask.astype(float)


def phi_field(T: np.ndarray, case) -> tuple[np.ndarray, np.ndarray]:
    """Melt fraction and the mask of cells strictly inside the phase ramp."""
    p = case.pins
    raw = (np.asarray(T, dtype=float) - p.t_pc_c) / p.dt_pc_c + 0.5
    phi = np.clip(raw, 0.0, 1.0)
    inside = (raw > 0.0) & (raw < 1.0)
    return phi, inside


def shape_J_and_seed(T: np.ndarray, case, w: np.ndarray | None = None
                     ) -> tuple[float, np.ndarray]:
    """J at one time level, and dJ/dT at that time level."""
    phi, inside = phi_field(T, case)
    d = phi - chi_part(case)
    wt = 1.0 if w is None else np.asarray(w, dtype=float)
    J = float(np.sum(wt * d * d))
    g = 2.0 * wt * d * inside / case.pins.dt_pc_c
    return J, np.asarray(g, dtype=float)


@dataclass(frozen=True)
class Stop:
    index: int
    J: float
    time_s: float
    at_horizon: bool
    J_first: float
    J_curve: np.ndarray


def J_curve(tr, case, w: np.ndarray | None = None) -> np.ndarray:
    return np.asarray([shape_J_and_seed(tr.T_at_end(n), case, w)[0]
                       for n in range(tr.n_outer)], dtype=float)


def optimal_stop(tr, case, w: np.ndarray | None = None) -> Stop:
    """t_stop = argmin over the stored trajectory of J.

    `at_horizon` is True when the minimum sits on the last stored step, which
    means the run was truncated before the objective turned. That is reported
    loudly rather than silently accepted.
    """
    jc = J_curve(tr, case, w)
    i = int(np.argmin(jc))
    return Stop(index=i, J=float(jc[i]), time_s=float(tr.time_s[i]),
                at_horizon=bool(i == tr.n_outer - 1), J_first=float(jc[0]),
                J_curve=jc)


def region_metrics(T: np.ndarray, case, level: float = MELT_LEVEL) -> dict:
    phi, _ = phi_field(T, case)
    melted = phi >= level
    part = case.part_mask
    n_part = int(part.sum())
    inter = int(np.sum(melted & part))
    union = int(np.sum(melted | part))
    return {
        "IoU": (inter / union) if union else float("nan"),
        "bed_melt_pct_of_part": 100.0 * int(np.sum(melted & ~part)) / max(n_part, 1),
        "part_under_melt_pct": 100.0 * int(np.sum(part & ~melted)) / max(n_part, 1),
        "melted_cells": int(melted.sum()),
        "part_cells": n_part,
        "mean_phi_part": float(np.mean(phi[part])),
        "mean_phi_bed": float(np.mean(phi[~part])),
    }


def window_metrics(T: np.ndarray, case, lo: float = WINDOW_LO_C,
                   hi: float = WINDOW_HI_C) -> dict:
    """Stop-time melt-window metrics, matching `FGM_WINDOW_RESELECTION.md`.

    under_pct       part cells still below the window (unmelted powder)
    over_pct        part cells above the window
    p95_overshoot_c the 95th percentile of the part temperature minus the top of
                    the window, floored at zero
    """
    tp = np.asarray(T, dtype=float)[case.part_mask]
    n = tp.size
    return {
        "under_pct": 100.0 * float(np.sum(tp < lo)) / max(n, 1),
        "in_window_pct": 100.0 * float(np.sum((tp >= lo) & (tp <= hi))) / max(n, 1),
        "over_pct": 100.0 * float(np.sum(tp > hi)) / max(n, 1),
        "p95_overshoot_c": max(0.0, float(np.percentile(tp, 95.0)) - hi),
        "max_overshoot_c": max(0.0, float(np.max(tp)) - hi),
        "mean_T_part_c": float(np.mean(tp)),
    }


def sigma_T_diagnostic(T: np.ndarray, case) -> float:
    """The old uniformity metric, reported as a DIAGNOSTIC only."""
    tp = np.asarray(T, dtype=float)[case.part_mask]
    tbar = float(np.mean(tp))
    denom = max(tbar - case.pins.ambient_c, 1e-9)
    return float(np.sqrt(np.mean((tp - tbar) ** 2)) / denom) * (tbar - case.pins.ambient_c)


def full_metrics(tr, case, w: np.ndarray | None = None) -> dict:
    st = optimal_stop(tr, case, w)
    T = tr.T_at_end(st.index)
    out = {
        "t_stop_index": st.index,
        "t_stop_s": st.time_s,
        "t_stop_at_horizon": st.at_horizon,
        "J": st.J,
        "J_per_part_cell": st.J / max(int(case.part_mask.sum()), 1),
        "J_at_first_step": st.J_first,
        "sigma_T_at_stop_c": sigma_T_diagnostic(T, case),
        "phi_bar_part_at_stop": float(np.mean(phi_field(T, case)[0][case.part_mask])),
    }
    out.update(region_metrics(T, case))
    out.update(window_metrics(T, case))
    return out
