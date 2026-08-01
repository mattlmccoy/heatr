"""Density-region objective: the DENSIFIED region should be the nominal part.

    J_rho(s, t_stop) = sum over the WHOLE domain of
                       (rho_norm(x, t_stop) - chi_part(x))^2

This is the density twin of `shape_objective.J`, which reads melt fraction. It
exists because melt fraction is a transient state that relaxes when the
generator turns off, while relative density is the state that survives into the
finished part.

THE NORMALIZATION, stated exactly.

    rho_norm = (rho_rel - rho_floor) / (1 - rho_floor),    rho_floor = rho_rel_init

`rho_rel` is the relative-density state integrated by `forward.substep`
(the `physics_dual` densification model of `rfam_eqs_coupled`: a solid-state
Arrhenius branch plus a viscous-capillary liquid branch, both scaled by
(1 - rho)^e and capped per substep). `rho_floor` is the configured powder-bed
starting relative density `densification.rho_rel_initial`, 0.55 in every config
of this campaign. The forward initializes rho = rho_floor on the part, and drho
is non-negative and rho is clipped at 1, so rho_rel lies in [rho_floor, 1] and
rho_norm lies in [0, 1] with NO clip needed.

THE BED, and the asymmetry that follows. `forward.substep` integrates rho only
inside the part mask (`rho_new[pm] = clip(...)`) and leaves the bed at the
stored value 0, which is bookkeeping, not physics: undensified powder sits at
the powder floor. The bed is therefore EXTENDED at rho_floor, so rho_norm = 0
there, chi_part = 0 there, and the bed's contribution to the whole-domain sum
is identically zero at every time. Consequence, stated rather than hidden:
unlike the melt-region objective, which charges the same for a melted bed cell
as for an unmelted part cell, J_rho has NO growth term. It penalizes
under-densification of the part only. Any comparison between the two objectives
must carry that.

THE SATURATION PATHOLOGY, and the guard. Because rho only increases, J_rho(t)
is monotone non-increasing, so argmin_t J_rho is ALWAYS the last stored step,
and at full densification every per-cell term is zero, every (1 - rho)^e rate
factor is zero and the clip subgradient `rho_new_range` is False, so the
gradient is exactly zero. Reading at the argmin therefore hands the optimizer a
dead objective and, on the documented lattice-carving precedent, invites it to
carve structure that buys nothing. The read state is instead PINNED BY RULE at
the FLAT ONSET: the first stored step whose objective is already within `tol`
of the terminal value, measured as a fraction of the total decrease over the
run. Both the flat-onset index and the (always terminal) argmin are reported on
every arm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DENSIFIED_LEVEL = 0.5      # densified region is rho_norm >= DENSIFIED_LEVEL
# MEASURED: at the flat-onset read state the half level is above every cell on
# every arm, so `IoU_rho` is exactly 1.0000 and discriminates nothing there.
# Stricter levels are reported alongside it so the density census has a
# quantity that can separate arms at the read state it is actually read at.
STRICT_LEVELS = (0.5, 0.75, 0.9, 0.95)
FLAT_TOL = 0.01            # 1 percent of the total decrease
NO_PROGRESS_ABS = 1e-12


# ---------------------------------------------------------------------------
# normalization and the objective at one time level
# ---------------------------------------------------------------------------

def rho_floor(case) -> float:
    return float(case.pins.rho_rel_init)


def rho_norm(rho: np.ndarray, case) -> np.ndarray:
    """Relative density mapped from the powder floor to full density onto [0, 1]."""
    f = rho_floor(case)
    return (np.asarray(rho, dtype=float) - f) / max(1.0 - f, 1e-12)


def rho_effective(rho: np.ndarray, case) -> np.ndarray:
    """Relative density with the bed extended at the powder floor."""
    pm = np.asarray(case.part_mask, dtype=bool)
    return np.where(pm, np.asarray(rho, dtype=float), rho_floor(case))


def chi_part(case) -> np.ndarray:
    return np.asarray(case.part_mask, dtype=float)


def rho_J_and_seed(rho: np.ndarray, case) -> tuple[float, np.ndarray]:
    """J_rho at one time level, and dJ_rho/drho at that time level.

    The seed is zero in the bed, because the bed's contribution does not depend
    on the stored bed value at all (it is extended at the floor by definition).
    """
    pm = np.asarray(case.part_mask, dtype=bool)
    f = rho_floor(case)
    span = max(1.0 - f, 1e-12)
    rn = rho_norm(rho_effective(rho, case), case)
    d = rn - chi_part(case)
    J = float(np.sum(d * d))
    g = np.where(pm, 2.0 * d / span, 0.0)
    return J, np.asarray(g, dtype=float)


# ---------------------------------------------------------------------------
# the read state: flat onset, not argmin
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RhoStop:
    index: int
    J: float
    time_s: float
    argmin_index: int
    argmin_J: float
    at_horizon: bool
    no_progress: bool
    tol: float
    total_decrease: float
    frac_remaining: float
    J_curve: np.ndarray


def flat_onset_stop(curve, tol: float = FLAT_TOL, dt: float = 1.0) -> RhoStop:
    """Pin the read state at the flat onset of a monotone-decreasing curve.

    The flat onset is the FIRST index whose objective is within `tol` of the
    terminal value, measured as a fraction of the total decrease
    `J[0] - J[-1]`. Reported flags:

      no_progress  the curve never moved (nothing densified); index 0.
      at_horizon   the flat onset is the last stored step, so the curve had not
                   flattened inside the horizon and the read state is a bound.
    """
    jc = np.asarray(curve, dtype=float)
    n = int(jc.size)
    if n == 0:
        raise ValueError("empty objective curve")
    i_argmin = int(np.argmin(jc))
    total = float(jc[0] - jc[-1])
    if total <= NO_PROGRESS_ABS:
        return RhoStop(index=0, J=float(jc[0]), time_s=float(dt),
                       argmin_index=i_argmin, argmin_J=float(jc[i_argmin]),
                       at_horizon=bool(n == 1), no_progress=True, tol=float(tol),
                       total_decrease=total, frac_remaining=0.0, J_curve=jc)
    band = float(jc[-1]) + float(tol) * total
    hits = np.flatnonzero(jc <= band)
    i = int(hits[0]) if hits.size else n - 1
    return RhoStop(index=i, J=float(jc[i]), time_s=float((i + 1) * dt),
                   argmin_index=i_argmin, argmin_J=float(jc[i_argmin]),
                   at_horizon=bool(i == n - 1), no_progress=False, tol=float(tol),
                   total_decrease=total,
                   frac_remaining=float((jc[i] - jc[-1]) / total), J_curve=jc)


def rho_J_curve(tr, case) -> np.ndarray:
    """J_rho at the end of every stored outer step. Needs rho checkpoints."""
    return np.asarray([rho_J_and_seed(tr.rho_at_end(n), case)[0]
                       for n in range(tr.n_outer)], dtype=float)


def rho_stop(tr, case, tol: float = FLAT_TOL) -> RhoStop:
    return flat_onset_stop(rho_J_curve(tr, case), tol=tol, dt=float(case.pins.dt))


# ---------------------------------------------------------------------------
# region metrics
# ---------------------------------------------------------------------------

def density_region_metrics(rho: np.ndarray, case,
                           level: float = DENSIFIED_LEVEL) -> dict:
    """Densified-region metrics at one time level.

    `IoU_rho` degenerates to the densified FRACTION of the part, because the
    densification model has no bed branch and the densified region is a subset
    of the part by construction. `densified_bed_cells` is reported so that the
    degeneracy is visible in the data rather than only in this docstring.
    """
    pm = np.asarray(case.part_mask, dtype=bool)
    rn = rho_norm(rho_effective(rho, case), case)
    dense = rn >= float(level)
    n_part = int(pm.sum())
    inter = int(np.sum(dense & pm))
    union = int(np.sum(dense | pm))
    out = {f"dense_frac_{lv:.2f}".replace(".", "p"):
           (float(np.mean(rn[pm] >= lv)) if n_part else float("nan"))
           for lv in STRICT_LEVELS}
    out.update({
        "IoU_rho": (inter / union) if union else float("nan"),
        "densified_cells": inter,
        "densified_bed_cells": int(np.sum(dense & ~pm)),
        "part_under_dense_pct": 100.0 * int(np.sum(pm & ~dense)) / max(n_part, 1),
        "mean_rho_rel_part": float(np.mean(np.asarray(rho, dtype=float)[pm])) if n_part else float("nan"),
        "mean_rho_norm_part": float(np.mean(rn[pm])) if n_part else float("nan"),
        "min_rho_norm_part": float(np.min(rn[pm])) if n_part else float("nan"),
        "frac_part_fully_dense": float(np.mean(rn[pm] >= 1.0 - 1e-9)) if n_part else float("nan"),
    })
    return out


def full_metrics(tr, case, tol: float = FLAT_TOL) -> dict:
    """Every density metric at the flat-onset read state, with the guard flags."""
    st = rho_stop(tr, case, tol=tol)
    rho = tr.rho_at_end(st.index)
    out = {
        "rho_stop_index": st.index,
        "rho_stop_s": st.time_s,
        "rho_stop_rule": "flat onset: first step within tol of the terminal J_rho",
        "rho_stop_tol": st.tol,
        "rho_stop_at_horizon": st.at_horizon,
        "rho_stop_no_progress": st.no_progress,
        "rho_argmin_index": st.argmin_index,
        "rho_argmin_J": st.argmin_J,
        "rho_argmin_is_horizon": bool(st.argmin_index == tr.n_outer - 1),
        "J_rho": st.J,
        "J_rho_per_part_cell": st.J / max(int(case.part_mask.sum()), 1),
        "J_rho_at_first_step": float(st.J_curve[0]),
        "J_rho_total_decrease": st.total_decrease,
        "J_rho_frac_remaining_at_stop": st.frac_remaining,
    }
    out.update(density_region_metrics(rho, case))
    return out
