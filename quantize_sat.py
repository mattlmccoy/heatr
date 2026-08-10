"""Printable-bpp quantization of a continuous dopant saturation map.

RFAM prints at 2 or 4 bits-per-pixel (2 bpp -> 4 levels, 4 bpp -> 16 levels).
sat > 1 is realized by a double pass, so levels span [0, sat_max]. Every solved
map must be quantized and re-scored (printing-constraints-and-ink).
"""
from __future__ import annotations

import numpy as np


def quantize_sat(sat, n_bits: int, sat_max: float = 1.5):
    """Snap a continuous sat map to 2**n_bits evenly-spaced levels over [0, sat_max]."""
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    n_levels = 2 ** int(n_bits)
    s = np.clip(np.asarray(sat, dtype=float), 0.0, sat_max)
    levels = np.linspace(0.0, sat_max, n_levels)
    idx = np.abs(s[..., None] - levels).argmin(axis=-1)
    return levels[idx].astype(np.float32)
