"""DENSE IF AND ONLY IF IN BOUNDS: the asymmetric shape-and-density objective.

THE SPECIFICATION THIS ENCODES, in the researcher's own terms: every part fully
dense if and only if it falls within the nominal shape bounds, with an explicit
ASYMMETRIC trade, willing to compromise in-bounds density down to about 0.80 to
0.90 relative density if that buys better shape within the bounds.

A symmetric `(phi - chi)^2` does not encode that, because it charges the same
for one melted bed cell and for one unmelted part cell and it keeps charging
for in-bounds imperfection all the way to full density. This objective splits
the two sides and treats them differently:

    J_asym(s, t) = [ w_out * sum over the BED of phi(x, t)^2
                   + w_in  * sum over the PART of h(rho_rel(x, t))^2 ] / n_part

    h(rho_rel) = max(0, floor - rho_rel) / (floor - rho_floor)      in [0, 1]

  OUT OF BOUNDS, the HARD side. Any melt fraction in a cell outside the nominal
  shape is charged at FULL amplitude: a fully melted bed cell contributes
  exactly `w_out / n_part`, whatever the part is doing. Bed melt is the
  irreversible defect (fused powder that has to be machined off), so it gets no
  tolerance band at all.

  IN BOUNDS, the SOFT side with a FLOOR. Relative density at or above `floor`
  costs EXACTLY ZERO. That zero is the whole point: the objective can never buy
  in-bounds density it does not need by paying out-of-bounds growth for it.
  Below the floor the cost rises quadratically in the normalized deficit and
  reaches `w_in / n_part` per cell at the powder floor (nothing sintered).

WHY THE STATE VARIABLES ARE WHAT THEY ARE.

  Out of bounds uses MELT FRACTION phi, and that is FORCED by the model, not
  chosen: `forward.substep` integrates relative density only inside the part
  mask, so the densification state literally cannot express bed growth. Bed
  growth is measured as melted powder, which is the physical fusing event.

  In bounds uses RELATIVE DENSITY rho_rel, and that is a choice with three
  reasons. First, "fully dense" is a density statement; melt fraction is a
  transient state that relaxes when the generator turns off, while relative
  density is what survives into the finished part. Second, the density hinge
  already subsumes under-melting: a part cell that never melts never gets the
  viscous-capillary branch and never densifies, so it sits deep below the
  floor. Third, using phi on both sides would collapse this into a hinged
  variant of the melt-region objective and would not encode density at all.

  The floor is quoted in RELATIVE DENSITY (0.85 means 85 percent of full
  density), which is the researcher's unit, not in the [0, 1] normalized
  variable of `density_objective`. The equivalent normalized value is reported
  alongside it so the two campaigns can be compared.

THE READ STATE, and why this objective does not need the flat-onset guard the
pure density objective needed. `J_rho` is monotone non-increasing, so its
argmin is always the horizon and the read state had to be pinned by a
tolerance, which turned out to be close to self-normalizing
(`DENSITY_OBJECTIVE_LIBRARY_REPORT.md` Section 1). Here the out-of-bounds term
RISES with time and the in-bounds deficit FALLS with time, so J_asym has a
genuinely interior argmin set by the physics rather than by a tolerance, and
the stop is the argmin over that arm's own trajectory, exactly the melt-region
convention. Because the stop is stationary, the envelope theorem gives
dJ*/ds = partial J / partial s at the argmin, with no dt*/ds term.

THE SATURATION PATHOLOGY IS STILL POSSIBLE and is guarded by reporting, not by
a rule change. If every in-bounds cell clears the floor before any bed melt
appears, the soft term is exactly zero, its gradient is exactly zero, and the
objective degenerates into pure growth minimization whose argmin is the first
step. `AsymStop.in_term_dead` and `AsymStop.stop_is_first_step` are set in that
case, and the flat-onset index of the combined curve is reported next to the
argmin so a read state sitting on a plateau is visible in the data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import shape_objective as so

FLOOR_RHO_REL_DEFAULT = 0.85   # relative density, the middle of the stated band
W_OUT_DEFAULT = 1.0
W_IN_DEFAULT = 1.0
FLAT_TOL = 0.01                # guard diagnostic only, not the stop rule
MELT_LEVEL = so.MELT_LEVEL


# ---------------------------------------------------------------------------
# the two sides
# ---------------------------------------------------------------------------

def rho_floor(case) -> float:
    """The configured powder-bed starting relative density."""
    return float(case.pins.rho_rel_init)


def _span(case, floor: float) -> float:
    f = float(floor)
    lo = rho_floor(case)
    if f <= lo:
        raise ValueError(
            f"floor {f!r} must exceed the powder floor {lo!r}: a floor at or "
            "below the starting relative density makes the in-bounds deficit "
            "identically zero and silently deletes the soft side of the "
            "objective")
    return f - lo


def deficit(rho: np.ndarray, case, floor: float = FLOOR_RHO_REL_DEFAULT
            ) -> np.ndarray:
    """The normalized in-bounds density deficit h, zero outside the part.

    h = max(0, floor - rho_rel) / (floor - rho_floor), clipped into [0, 1] only
    at the top by the physical fact that rho_rel starts at the powder floor.
    """
    span = _span(case, floor)
    pm = np.asarray(case.part_mask, dtype=bool)
    d = (float(floor) - np.asarray(rho, dtype=float)) / span
    return np.where(pm & (d > 0.0), d, 0.0)


def out_of_bounds(T: np.ndarray, case) -> tuple[np.ndarray, np.ndarray]:
    """Melt fraction outside the nominal shape, and the phase-ramp mask."""
    phi, inside = so.phi_field(T, case)
    pm = np.asarray(case.part_mask, dtype=bool)
    return np.where(pm, 0.0, phi), (inside & ~pm)


# ---------------------------------------------------------------------------
# the objective and its two seeds
# ---------------------------------------------------------------------------

def J_and_seeds(T: np.ndarray, rho: np.ndarray, case,
                floor: float = FLOOR_RHO_REL_DEFAULT,
                w_out: float = W_OUT_DEFAULT, w_in: float = W_IN_DEFAULT
                ) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """J_asym at one time level, dJ/dT, dJ/drho, and the decomposition.

    Both seeds travel back through the SAME single reverse sweep
    (`adjoint.gradient(..., seeds=..., seeds_rho=...)`); the substep
    vector-Jacobian product is already coupled in (T, rho).
    """
    pm = np.asarray(case.part_mask, dtype=bool)
    n_part = max(int(pm.sum()), 1)
    span = _span(case, floor)

    g_out, ramp = out_of_bounds(T, case)
    h = deficit(rho, case, floor)

    J_out = float(w_out) * float(np.sum(g_out * g_out)) / n_part
    J_in = float(w_in) * float(np.sum(h * h)) / n_part

    seed_T = (2.0 * float(w_out) / n_part) * g_out * ramp / case.pins.dt_pc_c
    seed_rho = (-2.0 * float(w_in) / (n_part * span)) * h

    parts = {"J_out": J_out, "J_in": J_in,
             "w_out": float(w_out), "w_in": float(w_in),
             "floor_rho_rel": float(floor),
             "floor_rho_norm": float((float(floor) - rho_floor(case))
                                     / max(1.0 - rho_floor(case), 1e-12)),
             "n_part": n_part}
    return J_out + J_in, np.asarray(seed_T, dtype=float), \
        np.asarray(seed_rho, dtype=float), parts


# ---------------------------------------------------------------------------
# the read state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AsymStop:
    index: int
    J: float
    J_out: float
    J_in: float
    time_s: float
    at_horizon: bool
    stop_is_first_step: bool
    in_term_dead: bool
    flat_onset_index: int
    flat_onset_gap_steps: int
    J_first: float
    J_curve: np.ndarray
    J_out_curve: np.ndarray
    J_in_curve: np.ndarray


def curves(tr, case, floor: float = FLOOR_RHO_REL_DEFAULT,
           w_out: float = W_OUT_DEFAULT, w_in: float = W_IN_DEFAULT
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """J_asym and its two parts at the end of every stored outer step."""
    n = int(tr.n_outer)
    jt = np.zeros(n)
    jo = np.zeros(n)
    ji = np.zeros(n)
    for k in range(n):
        J, _sT, _sr, p = J_and_seeds(tr.T_at_end(k), tr.rho_at_end(k), case,
                                     floor=floor, w_out=w_out, w_in=w_in)
        jt[k] = J
        jo[k] = p["J_out"]
        ji[k] = p["J_in"]
    return jt, jo, ji


def J_curve(tr, case, floor: float = FLOOR_RHO_REL_DEFAULT,
            w_out: float = W_OUT_DEFAULT, w_in: float = W_IN_DEFAULT
            ) -> np.ndarray:
    return curves(tr, case, floor, w_out, w_in)[0]


def flat_onset_index(curve: np.ndarray, tol: float = FLAT_TOL) -> int:
    """First index within `tol` of the minimum, as a fraction of the drop.

    Reported next to the argmin as a GUARD DIAGNOSTIC: a large gap between the
    two means the read state sits on a plateau and the argmin is soft.
    """
    jc = np.asarray(curve, dtype=float)
    i_min = int(np.argmin(jc))
    drop = float(jc[0] - jc[i_min])
    if drop <= 0.0:
        return i_min
    hits = np.flatnonzero(jc <= jc[i_min] + float(tol) * drop)
    return int(hits[0]) if hits.size else i_min


def asym_stop(tr, case, floor: float = FLOOR_RHO_REL_DEFAULT,
              w_out: float = W_OUT_DEFAULT, w_in: float = W_IN_DEFAULT,
              tol: float = FLAT_TOL) -> AsymStop:
    jt, jo, ji = curves(tr, case, floor, w_out, w_in)
    i = int(np.argmin(jt))
    i_flat = flat_onset_index(jt, tol)
    t = float(tr.time_s[i]) if hasattr(tr, "time_s") else float(i + 1)
    return AsymStop(index=i, J=float(jt[i]), J_out=float(jo[i]), J_in=float(ji[i]),
                    time_s=t, at_horizon=bool(i == int(tr.n_outer) - 1),
                    stop_is_first_step=bool(i == 0),
                    in_term_dead=bool(ji[i] <= 0.0),
                    flat_onset_index=i_flat, flat_onset_gap_steps=int(i - i_flat),
                    J_first=float(jt[0]), J_curve=jt, J_out_curve=jo,
                    J_in_curve=ji)


# ---------------------------------------------------------------------------
# metrics at one time level
# ---------------------------------------------------------------------------

def region_metrics(T: np.ndarray, rho: np.ndarray, case,
                   floor: float = FLOOR_RHO_REL_DEFAULT,
                   level: float = MELT_LEVEL) -> dict:
    """Everything the census reports at one read state.

    `growth_pct` and `under_pct` are melted bed cells and unmelted part cells,
    both as a percentage of the part cell count, the campaign's convention.
    The densification state is reported in BOTH units: relative density (the
    researcher's unit, what the floor is quoted in) and the [0, 1] normalized
    variable of `density_objective` (so the two campaigns are comparable).
    """
    pm = np.asarray(case.part_mask, dtype=bool)
    n_part = max(int(pm.sum()), 1)
    lo = rho_floor(case)
    r = np.asarray(rho, dtype=float)
    rn = (r - lo) / max(1.0 - lo, 1e-12)
    phi, _ = so.phi_field(T, case)
    melted = phi >= float(level)
    inter = int(np.sum(melted & pm))
    union = int(np.sum(melted | pm))
    h = deficit(r, case, floor)
    return {
        "IoU": (inter / union) if union else float("nan"),
        "growth_pct": 100.0 * int(np.sum(melted & ~pm)) / n_part,
        "under_pct": 100.0 * int(np.sum(pm & ~melted)) / n_part,
        "melted_cells": int(melted.sum()),
        "part_cells": int(pm.sum()),
        "mean_rho_rel_part": float(np.mean(r[pm])),
        "min_rho_rel_part": float(np.min(r[pm])),
        "p05_rho_rel_part": float(np.percentile(r[pm], 5.0)),
        "mean_rho_norm_part": float(np.mean(rn[pm])),
        "min_rho_norm_part": float(np.min(rn[pm])),
        "frac_part_at_or_above_floor": float(np.mean(r[pm] >= float(floor))),
        "hinge_active_frac": float(np.mean(h[pm] > 0.0)),
        "mean_deficit_part": float(np.mean(h[pm])),
        "mean_phi_part": float(np.mean(phi[pm])),
        "mean_phi_bed": float(np.mean(phi[~pm])) if int((~pm).sum()) else float("nan"),
        "sigma_T_at_stop_c": so.sigma_T_diagnostic(T, case),
    }


def full_metrics(tr, case, floor: float = FLOOR_RHO_REL_DEFAULT,
                 w_out: float = W_OUT_DEFAULT, w_in: float = W_IN_DEFAULT,
                 tol: float = FLAT_TOL) -> dict:
    """Every metric at the arm's own J_asym stop, with the guard flags."""
    st = asym_stop(tr, case, floor=floor, w_out=w_out, w_in=w_in, tol=tol)
    out = {
        "asym_stop_index": st.index,
        "asym_stop_s": st.time_s,
        "asym_stop_rule": "argmin over the arm's own stored trajectory of J_asym",
        "asym_stop_at_horizon": st.at_horizon,
        "asym_stop_is_first_step": st.stop_is_first_step,
        "asym_in_term_dead": st.in_term_dead,
        "asym_flat_onset_index": st.flat_onset_index,
        "asym_flat_onset_gap_steps": st.flat_onset_gap_steps,
        "J_asym": st.J,
        "J_asym_out": st.J_out,
        "J_asym_in": st.J_in,
        "J_asym_out_frac": st.J_out / max(st.J, 1e-30),
        "J_asym_at_first_step": st.J_first,
        "floor_rho_rel": float(floor),
        "w_out": float(w_out),
        "w_in": float(w_in),
    }
    out.update(region_metrics(tr.T_at_end(st.index), tr.rho_at_end(st.index),
                              case, floor=floor))
    return out
