"""Pure comparison math shared by the D1 spike Tasks 2 and 3.

Kept free of dolfinx/heatr3d imports so it runs (and is unit-tested) in either
environment, and so the engine comparison cannot silently depend on one
engine's data structures.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Pattern comparison (Task 2 gate)
# --------------------------------------------------------------------------- #
def unit_mean(q: np.ndarray) -> np.ndarray:
    """Normalize a field to unit mean -- the plan's scale-free PATTERN.

    Absolute Q scale is fixed identically in both engines by the shared power
    renormalization (heatr3d.compute_qrf_3d basis), so the meaningful engine
    comparison is of the pattern; this also removes the electrode-gauge scale
    factor (heatr3d spans L-h, FEM spans L) from the Q comparison entirely."""
    q = np.asarray(q, dtype=float)
    mu = q.mean()
    if not np.isfinite(mu) or abs(mu) < 1e-300:
        raise ValueError("unit_mean: field has zero (or non-finite) mean")
    return q / mu


def rel_l2_pattern(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 difference of the unit-mean patterns of a and b.

    ||unit_mean(a) - unit_mean(b)||_2 / ||unit_mean(b)||_2 -- invariant to any
    positive rescaling of either argument."""
    pa, pb = unit_mean(a), unit_mean(b)
    return float(np.linalg.norm(pa - pb) / np.linalg.norm(pb))


# --------------------------------------------------------------------------- #
# Corner geometry (Task 3)
# --------------------------------------------------------------------------- #
def corner_edge_distance(x: np.ndarray, y: np.ndarray, half: float) -> np.ndarray:
    """Distance in the (x, y) plane to the nearest of the four vertical corner
    edges of a square prism of half-width `half` centred on the origin.

    The prism is extruded along z, so the four corner edges are the vertical
    lines (+-half, +-half); the distance is independent of z."""
    dx = np.abs(np.asarray(x, dtype=float)) - half
    dy = np.abs(np.asarray(y, dtype=float)) - half
    return np.sqrt(dx ** 2 + dy ** 2)


# --------------------------------------------------------------------------- #
# Volume-weighted statistics (Task 3: graded FEM cells are not equal-volume)
# --------------------------------------------------------------------------- #
def weighted_percentile(values: np.ndarray, weights: np.ndarray,
                        pct: float) -> float:
    """Smallest v whose cumulative weight fraction reaches pct/100.

    heatr3d voxels all have volume h^3, but a graded tetrahedral mesh does not,
    so an unweighted p99 over FEM cells would be dominated by wherever the mesh
    happens to be finest. Volume weighting makes the p99/mean ratio mean the
    same thing in both engines."""
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if v.size != w.size or v.size == 0:
        raise ValueError("weighted_percentile: size mismatch or empty input")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    target = (pct / 100.0) * cw[-1]
    i = int(np.searchsorted(cw, target * (1.0 - 1e-12), side="left"))
    return float(v[min(i, v.size - 1)])


# --------------------------------------------------------------------------- #
# Mask-aware gradient (Task 2 diagnostic)
# --------------------------------------------------------------------------- #
def masked_grad_2d(V: np.ndarray, mask: np.ndarray, h: float):
    """E = -grad V computed with a stencil that NEVER crosses the mask edge.

    heatr3d.compute_qrf_3d takes np.gradient of V over the WHOLE domain and
    only afterwards zeroes Q outside the part, so the outermost in-part voxel
    is differenced against an OUTSIDE voxel -- across the material interface,
    where grad V jumps by the conductivity contrast. This routine repeats the
    same second-order-interior stencil but falls back to a one-sided
    difference whenever the neighbour is outside the part, which is exact for
    a linear field and therefore isolates that artifact from real physics.

    Works for real or complex V (V is complex in the EQS solve)."""
    V = np.asarray(V)
    mask = np.asarray(mask, dtype=bool)
    out = []
    for ax in (0, 1):
        g = np.zeros(V.shape, dtype=V.dtype if np.iscomplexobj(V) else float)
        fwd_ok = np.zeros(V.shape, bool)
        bwd_ok = np.zeros(V.shape, bool)
        sl_all = [slice(None)] * 2
        s_lo, s_hi = list(sl_all), list(sl_all)
        s_lo[ax] = slice(0, -1)
        s_hi[ax] = slice(1, None)
        s_lo, s_hi = tuple(s_lo), tuple(s_hi)
        fwd_ok[s_lo] = mask[s_lo] & mask[s_hi]        # i -> i+1 usable
        bwd_ok[s_hi] = mask[s_lo] & mask[s_hi]        # i -> i-1 usable
        dfwd = np.zeros_like(g)
        dbwd = np.zeros_like(g)
        dfwd[s_lo] = (V[s_hi] - V[s_lo]) / h
        dbwd[s_hi] = (V[s_hi] - V[s_lo]) / h
        both = fwd_ok & bwd_ok
        g = np.where(both, 0.5 * (dfwd + dbwd),
                     np.where(fwd_ok, dfwd, np.where(bwd_ok, dbwd, 0.0)))
        g = np.where(mask, g, 0.0)
        out.append(-g)
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# Growth-law fit (Task 3 gate)
# --------------------------------------------------------------------------- #
def power_law_fit(h: np.ndarray, q: np.ndarray) -> dict:
    """Least-squares fit q = C * h**alpha in log-log space.

    Returns exponent alpha, prefactor C, and the coefficient of determination
    R^2 of the log-log fit (the plan's diagnostic-quality gate: a genuine
    integrable edge singularity gives a clean power law; an erratic voxel
    staircase does not)."""
    h = np.asarray(h, dtype=float)
    q = np.asarray(q, dtype=float)
    if h.size < 3:
        raise ValueError("power_law_fit needs at least 3 refinement points")
    if np.any(h <= 0) or np.any(q <= 0):
        raise ValueError("power_law_fit needs strictly positive h and q")
    lx, ly = np.log(h), np.log(q)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"exponent": float(slope), "prefactor": float(np.exp(intercept)),
            "r2": float(r2), "n_points": int(h.size)}
