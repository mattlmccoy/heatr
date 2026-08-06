"""solve3d Stage A: the thermal-ceiling observable (KS peak + TRUE max).

PURE NUMPY (imported from either environment).

Spec sec 3: the peak-temperature constraint is a max over space AND time and is
nonsmooth. For a gradient it is aggregated with a smooth KS / p-norm surrogate.
BUT the physical peak that the degradation ceiling is enforced on is the TRUE
max, and the two must never be confused:

    THE KS AGGREGATE IS THE SMOOTH GRADIENT SURROGATE ONLY. Every ceiling
    verdict (T_ceiling_ok) and every reported physical peak is computed on the
    TRUE MAX. Presenting the KS value as the physical peak is a pre-registered
    false-green class (stage_a_preregistration.json acceptance_bands.ks_vs_true_max)
    and is refused here by construction: ceiling_status() takes the true max and
    has no path to read the surrogate.

The KS aggregate uses the MEAN (volume-weighted) log-sum-exp form, so it is a
LOWER bound on the max that approaches from below as the sharpness grows -- it
can never sit ABOVE the true peak and silently certify a part that is actually
over the ceiling. It is verified to track the true max within a pre-registered
5% band (stage_a_preregistration) on a synthetic field AND a real densify march
before the constrained solve trusts it.
"""
from __future__ import annotations

import numpy as np

# KS sharpness [1/C]. Chosen so the aggregate tracks the true max within the
# pre-registered 5% band on the densify anchor (measured, recorded in
# ceiling_ks_band.json). Larger = tighter but stiffer gradient.
KS_RHO_PER_C = 1.0


def peak_temp(T, weights=None, mask=None, rho: float = KS_RHO_PER_C) -> dict:
    """Return the smooth KS aggregate AND the true max of T over the selected
    (in-part) nodes.

    KS_mean = Tmax + (1/rho) * ln( sum_i w_i exp(rho (T_i - Tmax)) / sum_i w_i )

    which is <= Tmax (equality only if every selected node is at Tmax) and
    approaches Tmax from below as rho -> inf. The shift by Tmax is the standard
    log-sum-exp stabilization (no overflow; underflow of the far-below terms is
    the correct 0).
    """
    T = np.asarray(T, dtype=float).ravel()
    if mask is not None:
        sel = np.asarray(mask, dtype=bool).ravel()
    else:
        sel = np.ones(T.size, dtype=bool)
    if not sel.any():
        raise ValueError("peak_temp: empty selection mask")
    Ts = T[sel]
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()[sel]
    else:
        w = np.ones(Ts.size, dtype=float)
    wsum = float(w.sum())
    if not (wsum > 0.0):
        raise ValueError("peak_temp: non-positive total weight")
    if not (rho > 0.0):
        raise ValueError("peak_temp: rho must be > 0")
    true_max = float(Ts.max())
    z = rho * (Ts - true_max)                 # <= 0
    ks = true_max + (1.0 / rho) * float(
        np.log(np.dot(w, np.exp(z)) / wsum))
    gap = true_max - ks                        # >= 0
    denom = abs(true_max) if abs(true_max) > 1e-12 else 1.0
    return {
        "ks_aggregate_c": float(ks),
        "true_max_c": true_max,
        "rho_per_c": float(rho),
        "n_selected": int(sel.sum()),
        "gap_c": float(gap),
        "gap_rel": float(gap / denom),
        "surrogate_is_lower_bound": bool(ks <= true_max + 1e-9),
    }


def ceiling_status(true_max_c: float, ceiling_c: float,
                   warn_c: float | None = None) -> dict:
    """The degradation-ceiling verdict, computed on the TRUE max ONLY.

    There is deliberately no KS argument: the constraint is enforced on the
    physical peak, and the false-green class (a KS below the ceiling while the
    true peak is over it) cannot be expressed through this function.
    """
    tm = float(true_max_c)
    ok = bool(tm <= float(ceiling_c))
    out = {
        "true_max_c": tm,
        "ceiling_c": float(ceiling_c),
        "T_ceiling_ok": ok,
        "over_by_c": float(tm - float(ceiling_c)),
        "peak_source": "true_trajectory_max",
        "rule": "T_ceiling_ok is true_max <= ceiling; NEVER the KS surrogate",
    }
    if warn_c is not None:
        out["warn_c"] = float(warn_c)
        out["in_warning_band"] = bool(tm >= float(warn_c))
    return out


def melt_completeness(T_peak_nodal, mask, melt_onset_c: float,
                      weights=None) -> dict:
    """Completeness requirement, reported SEPARATELY from the ceiling: every
    in-part location must have exceeded the melt onset at some point in the
    trajectory (min over the part of the per-node peak temperature >= onset).

    `T_peak_nodal` is the running per-node maximum over the trajectory (the
    solve3d densify march returns it); the end-state field is NOT sufficient,
    because a node can melt and then cool as the front moves on.
    """
    Tpk = np.asarray(T_peak_nodal, dtype=float).ravel()
    sel = np.asarray(mask, dtype=bool).ravel()
    if not sel.any():
        raise ValueError("melt_completeness: empty part mask")
    part_peaks = Tpk[sel]
    min_peak = float(part_peaks.min())
    onset = float(melt_onset_c)
    unmelted = part_peaks < onset
    frac_unmelted = float(unmelted.mean())
    return {
        "melt_onset_c": onset,
        "min_in_part_peak_c": min_peak,
        "complete": bool(min_peak >= onset),
        "n_unmelted": int(unmelted.sum()),
        "n_in_part": int(sel.sum()),
        "frac_unmelted": frac_unmelted,
        "rule": "complete iff min over the part of the per-node trajectory peak "
                ">= melt onset; a completeness requirement, not the ceiling",
    }


def observe(march_out: dict, ceiling_c: float, melt_onset_c: float,
            warn_c: float | None = None, rho: float = KS_RHO_PER_C) -> dict:
    """The Stage A ceiling read on a densify march output (forward.march_enthalpy
    with densify=True). Bundles the KS-vs-true-max surrogate check, the ceiling
    verdict (on the true max), and the melt-completeness check (separate).
    """
    if not march_out.get("densify"):
        raise ValueError("observe: needs a densify march output "
                         "(march_enthalpy(densify=True))")
    Tpk = march_out["T_peak_nodal"]
    mask = march_out["part_peak_mask"]
    weights = march_out.get("vol_nodal")
    pk = peak_temp(Tpk, weights=weights, mask=mask, rho=rho)
    status = ceiling_status(pk["true_max_c"], ceiling_c, warn_c=warn_c)
    completeness = melt_completeness(Tpk, mask, melt_onset_c, weights=weights)
    # cross-check: the spatial peak of the per-node trajectory-peak field IS the
    # scalar true (t, x) peak the march tracked incrementally.
    consistent = bool(abs(pk["true_max_c"]
                          - float(march_out["true_peak_T_c"])) <= 1e-6)
    return {
        "peak": pk,
        "ceiling": status,
        "completeness": completeness,
        "true_peak_matches_march": consistent,
        "march_true_peak_T_c": float(march_out["true_peak_T_c"]),
        "note": "ceiling.T_ceiling_ok is on the TRUE max; peak.ks_aggregate_c is "
                "the smooth gradient surrogate ONLY and is never the reported peak",
    }
