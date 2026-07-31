"""Objectives and read states.

sigma_T is the 2-D campaign uniformity metric
`ui_rms_part(t) * (T_bar_part(t) - T_ambient)`. With `T_bar_part > T_ambient`
that equals the part standard deviation of temperature exactly, so it is the
`Var_part(T)` functional of the assessment written in the units the campaigns
report.

Read states
-----------
heating peak   the maximum of sigma_T over the outer steps with mean part melt
               fraction below 0.90 (the FIT metric of the step-2 control)
melt onset     sigma_T at the first outer step whose mean part melt fraction is
               at or above 0.90 (the HOLD-OUT metric, production convention)
melt onset,
interpolated   sigma_T linearly interpolated to the exact crossing time t*,
               which is the differentiable version of the same read state and
               the one the L2 adjoint layer differentiates
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHI_TARGET = 0.90


class MeltNotReached(RuntimeError):
    """The part never reaches the melt-onset read state inside the horizon.

    Raised loudly. The objective is undefined there; a silent final-step
    fallback would contaminate the metric (the assessment measured exactly that
    contamination in its `corner_m` arm).
    """


@dataclass(frozen=True)
class ReadStates:
    heating_peak_index: int
    melt_onset_index: int | None
    bracket_index: int | None      # n with phi_bar[n] < 0.90 <= phi_bar[n+1]
    theta: float | None            # fractional position of the crossing in [n, n+1]


def read_states(tr, phi_target: float = PHI_TARGET) -> ReadStates:
    phi = np.asarray(tr.mean_phi_part)
    st = np.asarray(tr.sigma_T)
    pre = phi < phi_target
    if not np.any(pre):
        peak = 0
    else:
        idx = np.flatnonzero(pre)
        peak = int(idx[int(np.argmax(st[idx]))])
    reached = np.flatnonzero(phi >= phi_target)
    if reached.size == 0:
        return ReadStates(peak, None, None, None)
    n_on = int(reached[0])
    if n_on == 0:
        return ReadStates(peak, n_on, None, None)
    n = n_on - 1
    denom = phi[n + 1] - phi[n]
    theta = float((phi_target - phi[n]) / denom) if denom > 0 else 0.0
    return ReadStates(peak, n_on, n, theta)


# ---------------------------------------------------------------------------
# seeds: dJ/dT at the END of an outer step
# ---------------------------------------------------------------------------

def sigma_T_and_seed(T: np.ndarray, case) -> tuple[float, np.ndarray]:
    pm = case.part_mask
    amb = case.pins.ambient_c
    tp = T[pm]
    n = tp.size
    tbar = float(np.mean(tp))
    d = tp - tbar
    rms = float(np.sqrt(np.mean(d * d)))
    denom = max(tbar - amb, 1e-9)
    if tbar - amb <= 1e-9:  # pragma: no cover
        raise RuntimeError("mean part temperature is at ambient; sigma_T seed undefined")
    J = rms * (tbar - amb) / denom          # == rms
    g = np.zeros_like(T)
    if rms > 0.0:
        g[pm] = d / (n * rms)
    return J, g


def phi_bar_and_seed(T: np.ndarray, case) -> tuple[float, np.ndarray]:
    pm = case.part_mask
    p = case.pins
    raw = (T - p.t_pc_c) / p.dt_pc_c + 0.5
    phi = np.clip(raw, 0.0, 1.0)
    n = int(pm.sum())
    val = float(np.mean(phi[pm]))
    g = np.zeros_like(T)
    inr = (raw > 0.0) & (raw < 1.0) & pm
    g[inr] = 1.0 / (n * p.dt_pc_c)
    return val, g


def objective_fixed_horizon(tr, case, step: int) -> tuple[float, dict[int, np.ndarray]]:
    J, g = sigma_T_and_seed(tr.T_at_end(step), case)
    return J, {step: g}


def objective_heating_peak(tr, case) -> tuple[float, dict[int, np.ndarray]]:
    rs = read_states(tr)
    return objective_fixed_horizon(tr, case, rs.heating_peak_index)


def objective_melt_onset_interp(tr, case) -> tuple[float, dict[int, np.ndarray]]:
    """sigma_T at the interpolated melt-onset crossing, with the implicit
    function theorem term for the moving read state.

    With phi_bar linear between the bracketing outer steps n and n+1,
    phi_bar(t*) = 0.90 gives t* = t_n + theta*dt and

        dt*/ds = -(d phi_bar/ds) / (d phi_bar/dt)

    evaluated with the discrete slope (phi_{n+1} - phi_n)/dt. The two theta
    sensitivities below are exactly that expression written per bracketing step.
    """
    rs = read_states(tr)
    if rs.melt_onset_index is None:
        raise MeltNotReached(
            f"mean part melt fraction never reached {PHI_TARGET} in "
            f"{tr.n_outer} outer steps (final {float(tr.mean_phi_part[-1]):.4f})"
        )
    if rs.bracket_index is None:
        raise MeltNotReached("melt onset occurred at the first outer step; no bracket to interpolate")
    n = rs.bracket_index
    theta = float(rs.theta)
    T_n = tr.T_at_end(n)
    T_n1 = tr.T_at_end(n + 1)
    s_n, gs_n = sigma_T_and_seed(T_n, case)
    s_n1, gs_n1 = sigma_T_and_seed(T_n1, case)
    p_n, gp_n = phi_bar_and_seed(T_n, case)
    p_n1, gp_n1 = phi_bar_and_seed(T_n1, case)
    slope = p_n1 - p_n
    if slope <= 0.0:  # pragma: no cover
        raise MeltNotReached("melt fraction is not increasing at the crossing; IFT term undefined")

    J = (1.0 - theta) * s_n + theta * s_n1
    dJ_dtheta = s_n1 - s_n
    dtheta_dpn = (theta - 1.0) / slope
    dtheta_dpn1 = -theta / slope

    seed_n = (1.0 - theta) * gs_n + dJ_dtheta * dtheta_dpn * gp_n
    seed_n1 = theta * gs_n1 + dJ_dtheta * dtheta_dpn1 * gp_n1
    return J, {n: seed_n, n + 1: seed_n1}


def scored_metrics(tr, case) -> dict:
    """Reporting values, production read-state conventions."""
    rs = read_states(tr)
    out = {
        "heating_peak_index": rs.heating_peak_index,
        "heating_peak_sigma_T_c": float(tr.sigma_T[rs.heating_peak_index]),
        "melt_onset_index": rs.melt_onset_index,
        "final_phi_bar": float(tr.mean_phi_part[-1]),
        "max_phi_bar": float(np.max(tr.mean_phi_part)),
        "max_mean_T_part_c": float(np.max(tr.mean_T_part_c)),
        "n_outer": tr.n_outer,
    }
    if rs.melt_onset_index is None:
        out["melt_onset_sigma_T_c"] = None
        out["status"] = "NOT_REACHED"
    else:
        out["melt_onset_sigma_T_c"] = float(tr.sigma_T[rs.melt_onset_index])
        out["melt_onset_time_s"] = float(tr.time_s[rs.melt_onset_index])
        out["status"] = "OK"
    return out
