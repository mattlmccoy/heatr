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


def target_column_heights(mask0: np.ndarray, h: float) -> np.ndarray:
    """Per-(x,y) nominal (target dense) column height in metres = occupied
    voxel count along z (axis 2) times h."""
    return np.asarray(mask0, bool).sum(axis=2).astype(float) * float(h)


def column_height_update(H_green: np.ndarray, H_target: np.ndarray,
                         H_measured: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Multiplicative green-height update: H_green *= H_target / H_measured.
    Non-part columns (H_target == 0) stay 0. Robust form (matches the shrinkage
    compensation convention f = target/built)."""
    H_green = np.asarray(H_green, float)
    H_target = np.asarray(H_target, float)
    H_measured = np.asarray(H_measured, float)
    gain = H_target / np.maximum(H_measured, eps)
    return np.where(H_target > 0, H_green * gain, 0.0)


def max_rel_error(H_target: np.ndarray, H_measured: np.ndarray,
                  cols: np.ndarray | None = None) -> float:
    """Max over part columns of |H_target - H_measured| / H_target."""
    H_target = np.asarray(H_target, float)
    H_measured = np.asarray(H_measured, float)
    if cols is None:
        cols = H_target > 0
    if not np.any(cols):
        return 0.0
    return float(np.max(np.abs(H_target[cols] - H_measured[cols])
                        / np.maximum(H_target[cols], 1e-9)))
