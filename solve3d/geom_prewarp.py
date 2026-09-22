"""Green-geometry pre-warp backsolve (Z densification compensation).

Pure per-column geometry logic + the outer-loop driver. The heavy heatr3d
densify march is injected as a callable so this module has NO solver dependency
and its tests run in the pure-numpy venv. Convention: arrays are (nx, ny, nz)
with z = axis 2 (heatr3d build axis); column heights are (nx, ny) in metres.
"""
from __future__ import annotations

import numpy as np


def resample_column(src: np.ndarray, n_out: int) -> np.ndarray:
    """Linear, layer-centre-aligned 1-D resample of one column's values to
    n_out samples (output i samples input at (i+0.5)*n_in/n_out - 0.5)."""
    src = np.asarray(src, float)
    n_in = src.shape[0]
    if n_out <= 0:
        return np.zeros(0)
    if n_in == 0:
        return np.zeros(n_out)
    if n_out == n_in:
        return src.copy()
    zi = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0.0, n_in - 1.0)
    lo = np.floor(zi).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    w = zi - lo
    return src[lo] * (1.0 - w) + src[hi] * w
