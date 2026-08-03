"""solve3d Phase A gate math + JSON artifact helpers.

PURE NUMPY on purpose: this module is imported from BOTH environments -- the
geo-prewarp venv (heatr3d side, solve3d/cases.py) and the dolfinx spike env
(solve3d/forward.py side, which has neither heatr3d nor scipy). Keeping the
comparison math in one place is what makes the two engines provably scored by
the same yardstick.

Pattern metrics are lifted from heatr3d_d1_spike/metrics.py (unit-mean
rel-L2), re-implemented here rather than imported so the gate math has no
dependency on the spike package layout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------- #
# Field pattern comparison (heatr3d_d1_spike/metrics.py conventions)
# --------------------------------------------------------------------------- #
def unit_mean(q: np.ndarray) -> np.ndarray:
    """Normalize to unit mean -- the scale-free PATTERN.

    Both engines renormalize Q to the same fixed absorbed power, so absolute
    scale carries no information; unit-mean also removes the electrode-gauge
    factor (heatr3d spans L-h, the FEM spans L)."""
    q = np.asarray(q, dtype=float)
    mu = q.mean()
    if not np.isfinite(mu) or abs(mu) < 1e-300:
        raise ValueError("unit_mean: field has zero (or non-finite) mean")
    return q / mu


def rel_l2_pattern(a: np.ndarray, b: np.ndarray) -> float:
    """||unit_mean(a) - unit_mean(b)||_2 / ||unit_mean(b)||_2."""
    pa, pb = unit_mean(a), unit_mean(b)
    return float(np.linalg.norm(pa - pb) / np.linalg.norm(pb))


def rel_spread(a: float, ref: float) -> float:
    """|a - ref| / |ref|."""
    return float(abs(float(a) - float(ref)) / abs(float(ref)))


# --------------------------------------------------------------------------- #
# Volume-weighted statistics (FEM cells are not equal-volume; voxels are)
# --------------------------------------------------------------------------- #
def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if v.size != w.size or v.size == 0:
        raise ValueError("weighted_mean: size mismatch or empty input")
    tot = float(w.sum())
    if tot <= 0.0:
        raise ValueError("weighted_mean: non-positive total weight")
    return float(np.dot(v, w) / tot)


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """Volume-weighted POPULATION std, so it reduces to numpy's std when the
    weights are equal (which is exactly the heatr3d voxel case)."""
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    mu = weighted_mean(v, w)
    tot = float(w.sum())
    return float(np.sqrt(float(np.dot((v - mu) ** 2, w)) / tot))


# --------------------------------------------------------------------------- #
# Heating-curve comparison
# --------------------------------------------------------------------------- #
def resample_curve(t_src, y_src, t_query) -> np.ndarray:
    """Linear interpolation of a monotone-in-time heating curve."""
    return np.interp(np.asarray(t_query, dtype=float),
                     np.asarray(t_src, dtype=float),
                     np.asarray(y_src, dtype=float))


def curve_rel_l2(t_a, y_a, t_b, y_b, n_query: int = 64) -> dict:
    """Relative L2 difference of two heating curves on their COMMON time span.

    Two decisions, both load-bearing:
      * the shared query grid runs to min(t_end) so nothing is extrapolated
        past a run that stopped earlier at melt onset;
      * temperatures are compared as RISE above the shared starting value.
        Comparing absolute T would divide a real error by the 23 C preheat
        offset that both engines share by construction, deflating the metric.
    """
    t_a, y_a = np.asarray(t_a, float), np.asarray(y_a, float)
    t_b, y_b = np.asarray(t_b, float), np.asarray(y_b, float)
    t_end = min(float(t_a[-1]), float(t_b[-1]))
    tq = np.linspace(0.0, t_end, n_query)
    ya, yb = resample_curve(t_a, y_a, tq), resample_curve(t_b, y_b, tq)
    base = min(float(ya[0]), float(yb[0]))
    ra, rb = ya - base, yb - base
    return {"t_end_common_s": float(t_end), "n_query": int(n_query),
            "rel_l2": float(np.linalg.norm(ra - rb) / np.linalg.norm(rb)),
            "max_abs_diff_c": float(np.max(np.abs(ya - yb)))}


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def load_tolerances(path: Path | None = None) -> dict:
    """Read the FROZEN Task-1 tolerances. Missing file is a hard error: a gate
    with no pre-registered tolerance is not a gate."""
    p = path or (RESULTS / "parity_tolerances.json")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Phase A Task 1 (measure_self_spread) must run "
            "BEFORE any parity gate; tolerances are measured, never invented.")
    return json.loads(p.read_text())


def write_json(name: str, doc: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(doc, indent=1, default=_jsonable))
    return p


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


# --------------------------------------------------------------------------- #
# Phase A close-out: the CROSS-FAMILY tolerance rule
#
# DECLARED BEFORE ANY NUMBER WAS COMPUTED (Matt-approved close-out, 2026-08-01).
#
# The Task-1 tolerance is heatr3d's n=64-vs-n=96 spread: two grids of ONE
# method. It bounds grid refinement inside that method and nothing else, which
# is why the Task-4 gate was unachievable by any correct implementation.
#
# The cross-family band is built from BOTH engines' own same-method spreads:
#
#     |A - B|  <=  |A - A_inf| + |A_inf - B_inf| + |B_inf - B|
#
# and, if the two discretizations converge to the SAME continuum answer
# (A_inf == B_inf, which is the premise of doing parity at all), the middle
# term vanishes and the bound is the SUM of the two engines' own errors. So:
#
#     COMBINATION RULE = SUM (triangle inequality), then the SAME 1.5x safety
#     factor Task 1 used.
#
# Root-sum-square was considered and REJECTED: RSS is the right combination for
# independent RANDOM errors, but discretization errors are systematic and RSS is
# strictly smaller than the sum, i.e. it is not a bound. Recording the rejection
# so the choice cannot look like it was fitted to the answer.
#
# Honest caveat kept with the rule: a coarse-fine DIFFERENCE under-estimates the
# fine level's distance to its own limit by roughly 1/(r^p - 1) (Richardson;
# ~0.8x at r=1.5, p=2). The 1.5x safety factor is what covers that, and it is
# the same factor Task 1 already used -- not a new allowance invented here.
# --------------------------------------------------------------------------- #
COMBINATION_RULE = "sum"
CROSS_FAMILY_SAFETY = 1.5


def combine_spreads(spread_a: float, spread_b: float,
                    safety: float = CROSS_FAMILY_SAFETY) -> float:
    """Cross-family tolerance = safety * (spread_a + spread_b). See the note
    above for why the combination is a SUM and not a root-sum-square."""
    return float(safety) * (float(spread_a) + float(spread_b))


# --------------------------------------------------------------------------- #
# Shared evaluation grid for the shape metrics
# --------------------------------------------------------------------------- #
# Fine in (x, y) -- the field plane, where the melt front lives and where CAD
# tolerance is quoted -- and a handful of z-planes, because both anchors are
# FULL-HEIGHT extrusions. The planes are not there to resolve z structure; they
# are there to MEASURE that there is none (plane-to-plane spread is reported).
# The z stations stay away from the +-30 mm domain faces so point location is
# unambiguous.
EVAL_HALF_XY_M = 0.015              # +-15 mm covers the 20 mm part plus 5 mm bed
EVAL_N_XY = 200                     # -> 0.15 mm pixels
EVAL_Z_M = (-0.020, -0.010, 0.0, 0.010, 0.020)


def eval_grid_axes():
    h = 2.0 * EVAL_HALF_XY_M / EVAL_N_XY
    c = (np.arange(EVAL_N_XY) + 0.5) * h - EVAL_HALF_XY_M
    return c, c, np.asarray(EVAL_Z_M, dtype=float), h


def eval_grid_points() -> tuple[np.ndarray, tuple[int, int, int], float]:
    """(N,3) points, the (nz, nx, ny) reshape, and the (x, y) pixel size [m]."""
    x, y, z, h = eval_grid_axes()
    Z, X, Y = np.meshgrid(z, x, y, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return pts, (z.size, x.size, y.size), h


def nominal_part_mask(shape: str, diam_m: float = 0.020) -> np.ndarray:
    """The ANALYTIC nominal shape on the (x, y) evaluation grid -- deliberately
    not either engine's discretized mask, so neither engine is scored against
    its own staircase."""
    x, y, _, _ = eval_grid_axes()
    X, Y = np.meshgrid(x, y, indexing="ij")
    half = diam_m / 2.0
    if shape == "circle":
        return np.sqrt(X ** 2 + Y ** 2) <= half
    if shape == "square":
        return (np.abs(X) <= half) & (np.abs(Y) <= half)
    raise ValueError(f"unknown anchor shape {shape!r}")


def phase_fraction_phi(T, t_pc_c: float = 180.0, dt_pc_c: float = 10.0):
    """heatr3d.phase_fraction's phi, as a pure function so BOTH engines' fields
    are converted to melt fraction by ONE implementation."""
    Te = np.asarray(T, dtype=float)
    return np.clip((Te - t_pc_c) / dt_pc_c + 0.5, 0.0, 1.0)
