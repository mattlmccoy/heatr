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


def graded_sat(part: np.ndarray) -> np.ndarray:
    """Return the heuristic graded dopant-fraction map on `part` (0 outside)."""
    m = np.asarray(part, dtype=bool)
    if not m.any():
        return np.zeros(m.shape, dtype=float)
    depth = ndimage.distance_transform_edt(m)
    d_hat = depth / max(float(depth.max()), 1e-12)
    ks = np.where(m.any(axis=(0, 1)))[0]
    k0, k1 = int(ks[0]), int(ks[-1])
    z_hat = (np.arange(m.shape[2], dtype=float) - k0) / max(k1 - k0, 1)
    z_hat = np.clip(z_hat, 0.0, 1.0)[None, None, :]
    sat = np.clip(S_PEAK * (D_FLOOR + D_GAIN * d_hat) * (1.0 - Z_TAPER * z_hat),
                  S_MIN, S_MAX)
    sat = np.where(m, sat, 0.0)
    return sat


LAW_TEXT = ("sat = clip(0.95 * (0.35 + 0.65 * d_hat) * (1 - 0.45 * z_hat), "
            "0.20, 1.00)")
HONESTY_TEXT = "heuristic graded map (not solved), heatr3d n=48, simulation-only"
