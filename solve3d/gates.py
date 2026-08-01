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
