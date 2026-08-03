"""S2 convergence-band machinery.

Pure numpy. The rules implemented here are the ones frozen in
heatr3d_s2/results/s2_preregistration.json BEFORE any campaign run; this module
holds no threshold of its own beyond the structural constants the
pre-registration names (safety 1.5, non-increase tolerance 1.10, minimum 3
grids), and those are read from it by the campaign driver.

THE ONE BEHAVIOUR THAT MATTERS MOST: a diverging sequence must FAIL LOUDLY and
must never be converted into a band. A band computed from the finest pair of a
sequence whose changes are GROWING is not a convergence estimate -- it is the
last point of a divergence, and quoting it would be the most dangerous single
number this campaign could emit. `analyse` therefore refuses to return a band
in that case, and it checks the trend BEFORE it checks the magnitude, so a
small-but-growing change cannot sneak through on the ceiling.
"""
from __future__ import annotations

import numpy as np

SAFETY = 1.5              # pre-registration band_rule.safety_factor
NON_INCREASE_TOL = 1.10   # pre-registration pass_criteria.tolerance_on_non_increase
MIN_GRIDS = 3             # pre-registration convergence.min_grids_for_a_claim
L_DOMAIN = 0.060          # heatr3d.Grid default chamber size, for h = L/n


def successive_changes(grids, values, relative: bool = True) -> list[dict]:
    """Change between each consecutive pair, coarse -> fine."""
    out = []
    for (n_lo, q_lo), (n_hi, q_hi) in zip(zip(grids, values),
                                          zip(grids[1:], values[1:])):
        d = float(q_hi - q_lo)
        den = abs(float(q_hi)) if relative else 1.0
        if relative and den == 0.0:
            raise ValueError("relative change against a zero reference value; "
                             "use relative=False for quantities that can be 0")
        out.append({"n_lo": int(n_lo), "n_hi": int(n_hi),
                    "q_lo": float(q_lo), "q_hi": float(q_hi),
                    "signed_change": d / den, "change": abs(d) / den,
                    "h_pair_m": float(np.sqrt((L_DOMAIN / n_lo) * (L_DOMAIN / n_hi)))})
    return out


def classify(changes: list[dict], tol: float = NON_INCREASE_TOL) -> str:
    """Trend of the successive-change MAGNITUDES.

    monotone_convergent  every change is strictly smaller than its predecessor
    bounded_oscillatory  a change may exceed its predecessor by up to `tol`
    diverging            beyond that -- a FAIL, never a band
    """
    mags = [c["change"] for c in changes]
    if all(b < a for a, b in zip(mags, mags[1:])):
        return "monotone_convergent"
    if all(b <= tol * a for a, b in zip(mags, mags[1:])):
        return "bounded_oscillatory"
    return "diverging"


def value_trend(changes: list[dict]) -> str:
    """Whether the VALUES approach from one side or oscillate about the limit.
    Reported separately from the change-magnitude trend, because a sequence can
    oscillate in value while converging perfectly well in magnitude."""
    signs = {np.sign(c["signed_change"]) for c in changes}
    return "monotone" if len(signs) == 1 else "oscillatory"


def observed_order(grids, values) -> tuple[float, float]:
    """Observed order p, by fitting the VALUES to q(h) = q_inf + C h^p.

    A DIAGNOSTIC, never the converged truth (pre-registration
    band_rule.not_richardson).

    WHY NOT the slope of log|successive change| vs log h, which is the obvious
    thing and was the first implementation: for q = q_inf + C h^p the change
    across a pair is C h^p (r^p - 1), so the fitted slope also absorbs the
    trend in (r^p - 1). On THIS ladder the refinement ratio is not constant
    (48->64 is 1.333, 64->80 is 1.25, 80->96 is 1.2), so that factor shrinks
    monotonically along the ladder and inflates the fitted order. It read 2.98
    on an exactly second-order synthetic sequence and 1.95 on a first-order one
    -- caught by test_bands.py, which is why the tests use synthetic sequences
    with known answers.

    Fitting the values instead is free of that artifact: for a FIXED p the
    model is linear in (q_inf, C), so p is found by a 1-D scan over the
    residual. Exact on the synthetic sequences.
    """
    h = np.array([L_DOMAIN / n for n in grids], dtype=float)
    q = np.asarray(values, dtype=float)
    if h.size < 3:
        return float("nan"), float("nan")
    best = (np.inf, float("nan"), float("nan"))
    for p in np.linspace(0.25, 4.0, 1501):
        A = np.column_stack([np.ones_like(h), h ** p])
        coef, *_ = np.linalg.lstsq(A, q, rcond=None)
        res = float(np.sum((A @ coef - q) ** 2))
        if res < best[0]:
            ss_tot = float(np.sum((q - q.mean()) ** 2))
            r2 = 1.0 - res / ss_tot if ss_tot > 0 else float("nan")
            best = (res, float(p), r2)
    return best[1], best[2]


def richardson(grids, values) -> float | None:
    """Richardson estimate of the limit from the finest triple. Reported as a
    diagnostic ONLY; the pre-registration forbids presenting it as truth."""
    if len(values) < 3:
        return None
    n1, n2, n3 = grids[-3:]
    q1, q2, q3 = values[-3:]
    d1, d2 = q2 - q1, q3 - q2
    if d1 == 0.0 or d2 == 0.0 or abs(d1 - d2) < 1e-300:
        return None
    r = (L_DOMAIN / n2) / (L_DOMAIN / n3)
    with np.errstate(all="ignore"):
        p = np.log(abs(d1 / d2)) / np.log(r) if r > 1 else float("nan")
        if not np.isfinite(p) or p <= 0:
            return None
        return float(q3 + d2 / (r ** p - 1.0))


def analyse(grids, values, relative: bool = True, safety: float = SAFETY,
            ceiling: float | None = None,
            tol: float = NON_INCREASE_TOL) -> dict:
    """Full band record for one quantity on one shape."""
    grids = [int(g) for g in grids]
    values = [float(v) for v in values]
    if len(grids) != len(values):
        raise ValueError("grids and values must have the same length")
    if len(grids) < MIN_GRIDS:
        raise ValueError(f"a convergence claim needs at least {MIN_GRIDS} "
                         f"grids; got {len(grids)}")
    if not all(np.isfinite(values)):
        raise ValueError("values must all be finite; refusing to band a "
                         "sequence containing nan or inf")
    if list(grids) != sorted(grids):
        raise ValueError("grids must be ascending (coarse -> fine)")

    changes = successive_changes(grids, values, relative=relative)
    status = classify(changes, tol=tol)
    p, r2 = observed_order(grids, values)
    finest = changes[-1]["change"]
    diverging = status == "diverging"
    band = None if diverging else safety * finest

    # trend is checked BEFORE magnitude, deliberately
    if diverging:
        ok, reason = False, ("diverging: successive-pair changes grow beyond "
                             f"the {tol:g}x tolerance, so no band is emitted")
    elif ceiling is not None and finest > ceiling:
        ok, reason = False, (f"finest-pair change {finest:.6g} exceeds the "
                             f"pre-registered ceiling {ceiling:.6g}")
    else:
        ok = True if ceiling is not None else None
        reason = "" if ceiling is not None else "no ceiling supplied (diagnostic quantity)"
    return {"grids": grids, "values": values, "relative": relative,
            "changes": changes, "status": status,
            "value_trend": value_trend(changes),
            "finest_change": finest, "safety": safety, "band": band,
            "observed_order": p, "r2": r2,
            "richardson_limit_diagnostic": richardson(grids, values),
            "ceiling": ceiling, "pass": ok, "reason": reason}


def analyse_from_changes(grids, pair_values, safety: float = SAFETY,
                         ceiling: float | None = None,
                         tol: float = NON_INCREASE_TOL) -> dict:
    """Band a quantity that is ALREADY a successive-pair measurement.

    Some convergence quantities are intrinsically pairwise: the Jaccard
    distance between two grids' melt regions, or the surface distance between
    their fronts, exist only for a PAIR of grids and cannot be evaluated on one
    grid alone. `analyse` would double-difference them; this takes the pair
    values as given and applies the same trend rule, band rule and ceiling.
    """
    grids = [int(g) for g in grids]
    vals = [float(v) for v in pair_values]
    if len(vals) != len(grids) - 1:
        raise ValueError(f"expected {len(grids)-1} pair values for "
                         f"{len(grids)} grids; got {len(vals)}")
    if len(grids) < MIN_GRIDS:
        raise ValueError(f"a convergence claim needs at least {MIN_GRIDS} grids")
    if not all(np.isfinite(vals)):
        raise ValueError("pair values must all be finite")
    changes = [{"n_lo": int(a), "n_hi": int(b), "change": abs(v),
                "signed_change": float(v),
                "h_pair_m": float(np.sqrt((L_DOMAIN / a) * (L_DOMAIN / b)))}
               for a, b, v in zip(grids, grids[1:], vals)]
    status = classify(changes, tol=tol)
    finest = changes[-1]["change"]
    diverging = status == "diverging"
    band = None if diverging else safety * finest
    x = np.log([c["h_pair_m"] for c in changes])
    y = np.log([max(c["change"], 1e-300) for c in changes])
    if x.size >= 2:
        p, b0 = np.polyfit(x, y, 1)
        pred = p * x + b0
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        p, r2 = float("nan"), float("nan")
    if diverging:
        ok, reason = False, ("diverging: pairwise disagreement grows with "
                             "refinement, so no band is emitted")
    elif ceiling is not None and finest > ceiling:
        ok, reason = False, (f"finest-pair value {finest:.6g} exceeds the "
                             f"pre-registered ceiling {ceiling:.6g}")
    else:
        ok = True if ceiling is not None else None
        reason = ""
    return {"grids": grids, "pair_values": vals, "pairwise": True,
            "changes": changes, "status": status, "finest_change": finest,
            "safety": safety, "band": band, "observed_order": float(p),
            "r2": float(r2), "ceiling": ceiling, "pass": ok, "reason": reason}
