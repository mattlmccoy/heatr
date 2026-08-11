"""Dynamics analysis for the constant-magnitude proportional FGM sweep.

Pure functions only (no I/O): lagged correlations and step norms on a
sequence of saturation maps, and the fixed-point / period-2 verdict.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two flattened maps (nan-safe on zero variance)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def lagged_stats(maps: Sequence[np.ndarray]) -> Dict[str, float]:
    """Lag-1/lag-2 correlations and step norms over a map sequence.

    Args:
        maps: ordered saturation maps s_k (already restricted to the pixels of
            interest, e.g. the part mask), length >= 4.

    Returns:
        dict with per-pair lists and their means:
            corr_lag1 / corr_lag1_mean   corr(s_k, s_{k+1})
            corr_lag2 / corr_lag2_mean   corr(s_k, s_{k+2})
            step_norms / step_norm_mean  RMS(s_{k+1} - s_k)
            step_norm_trend              slope of RMS step vs k (shrinking < 0)
    """
    if len(maps) < 4:
        raise ValueError("need at least 4 maps for lag-2 statistics")
    corr1 = [_pearson(maps[k], maps[k + 1]) for k in range(len(maps) - 1)]
    corr2 = [_pearson(maps[k], maps[k + 2]) for k in range(len(maps) - 2)]
    steps = [float(np.sqrt(np.mean((np.asarray(maps[k + 1], dtype=np.float64)
                                    - np.asarray(maps[k], dtype=np.float64)) ** 2)))
             for k in range(len(maps) - 1)]
    trend = float(np.polyfit(np.arange(len(steps)), steps, 1)[0])
    return {
        "corr_lag1": corr1,
        "corr_lag2": corr2,
        "step_norms": steps,
        "corr_lag1_mean": float(np.mean(corr1)),
        "corr_lag2_mean": float(np.mean(corr2)),
        "step_norm_mean": float(np.mean(steps)),
        "step_norm_trend": trend,
    }


def classify_dynamics(stats: Dict[str, float],
                      *,
                      period2_gap: float = 0.10,
                      fixed_corr: float = 0.98,
                      shrink_ratio: float = 0.5) -> str:
    """Verdict on plateau dynamics.

    period-2:    corr(s_k, s_{k+2}) exceeds corr(s_k, s_{k+1}) by > period2_gap
                 while steps stay large (non-shrinking oscillation between two maps).
    fixed-point: adjacent correlation ~1 (>= fixed_corr) AND steps shrink
                 (negative trend or final step < shrink_ratio * first step).
    neither:     anything else (e.g. wandering, slow drift, noisy plateau).
    """
    c1, c2 = stats["corr_lag1_mean"], stats["corr_lag2_mean"]
    steps: List[float] = list(stats["step_norms"])
    # Exact (quantized) fixed point: the map literally stops changing.
    if stats["step_norm_mean"] < 1e-9:
        return "fixed-point"
    steps_shrink = (stats["step_norm_trend"] < 0.0
                    and steps[-1] < shrink_ratio * max(steps[0], 1e-12))
    if c2 - c1 > period2_gap:
        return "period-2"
    if c1 >= fixed_corr and steps_shrink:
        return "fixed-point"
    return "neither"
