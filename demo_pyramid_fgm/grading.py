"""Heuristic 3-axis dopant grading law for the pyramid demo.

HONESTY SCOPE: this map is HAND-CONSTRUCTED and physically motivated, NOT solved
or optimized. Motivation: RF field concentration at exterior corners, sharp edges,
and the apex overheats those regions, so the dopant fraction is graded DOWN near
the part surface (which contains all corners/edges) and DOWN toward the apex, and
kept high in the core and base.

Law (stated on every figure):
    d_hat = depth / max depth        (Euclidean distance-to-surface inside the part)
    z_hat = (z - z_base) / (z_apex - z_base)
    sat   = clip(0.95 * (0.35 + 0.65 * d_hat) * (1 - 0.45 * z_hat), 0.20, 1.00)

The map therefore varies in x, y, AND z: radially through d_hat (the pyramid's
surface slopes in x and y) and vertically through the explicit z_hat taper.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

S_PEAK = 0.95     # core/base ceiling before clipping
D_FLOOR = 0.35    # skin fraction of the depth factor
D_GAIN = 0.65     # depth-factor gain (skin -> core)
Z_TAPER = 0.45    # fractional reduction from base to apex
S_MIN = 0.20      # clip floor (printable minimum)
S_MAX = 1.00      # clip ceiling


# Strong variant (the headline map): deeper skin/apex cuts and a lower floor so
# the redistributed heating produces a VISIBLY different outcome at the same
# exposure. Same functional form, harder constants.
S_PEAK_STRONG = 0.95
D_FLOOR_STRONG = 0.15
D_GAIN_STRONG = 0.85
Z_TAPER_STRONG = 0.75
S_MIN_STRONG = 0.10


def _graded_core(part: np.ndarray, peak: float, d_floor: float, d_gain: float,
                 z_taper: float, s_min: float) -> np.ndarray:
    m = np.asarray(part, dtype=bool)
    if not m.any():
        return np.zeros(m.shape, dtype=float)
    depth = ndimage.distance_transform_edt(m)
    d_hat = depth / max(float(depth.max()), 1e-12)
    ks = np.where(m.any(axis=(0, 1)))[0]
    k0, k1 = int(ks[0]), int(ks[-1])
    z_hat = (np.arange(m.shape[2], dtype=float) - k0) / max(k1 - k0, 1)
    z_hat = np.clip(z_hat, 0.0, 1.0)[None, None, :]
    sat = np.clip(peak * (d_floor + d_gain * d_hat) * (1.0 - z_taper * z_hat),
                  s_min, S_MAX)
    return np.where(m, sat, 0.0)


def graded_sat(part: np.ndarray) -> np.ndarray:
    """Mild heuristic graded dopant-fraction map on `part` (0 outside)."""
    return _graded_core(part, S_PEAK, D_FLOOR, D_GAIN, Z_TAPER, S_MIN)


def graded_sat_strong(part: np.ndarray) -> np.ndarray:
    """Strong heuristic graded map: deeper skin/apex cuts, lower floor."""
    return _graded_core(part, S_PEAK_STRONG, D_FLOOR_STRONG, D_GAIN_STRONG,
                        Z_TAPER_STRONG, S_MIN_STRONG)


LAW_TEXT = ("sat = clip(0.95 * (0.35 + 0.65 * d_hat) * (1 - 0.45 * z_hat), "
            "0.20, 1.00)")
LAW_TEXT_STRONG = ("sat = clip(0.95 * (0.15 + 0.85 * d_hat) * (1 - 0.75 * z_hat), "
                   "0.10, 1.00)")
HONESTY_TEXT = "heuristic graded map (not solved), heatr3d n=48, simulation-only"
