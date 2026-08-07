"""Pure metrics for the premix 0->15 wt% sweep (no solve).

bed_absorption_fraction  -- how much of the absorbed RF power lands in the bed
                            (outside the doped part). Rises with premix; the
                            unambiguous 'RF coupling redistribution' signal.
peak_to_mean             -- part-field uniformity (1.0 = perfectly uniform). The
                            mechanism by which premix could help under a ceiling.
peak_location            -- argmax cell within a mask, for peak-relocation deltas.
"""
from __future__ import annotations

import numpy as np


def bed_absorption_fraction(Qrf, doped_mask):
    """Fraction of total absorbed Qrf that lands OUTSIDE the doped part."""
    Q = np.asarray(Qrf, dtype=float)
    bed = ~np.asarray(doped_mask, dtype=bool)
    total = float(Q.sum())
    if total <= 0.0:
        return 0.0
    return float(Q[bed].sum()) / total


def peak_to_mean(field, mask):
    """max/mean of `field` over `mask` (part-field uniformity; 1.0 = uniform)."""
    f = np.asarray(field, dtype=float)
    m = np.asarray(mask, dtype=bool)
    vals = f[m]
    mean = float(vals.mean())
    if mean == 0.0:
        return 0.0
    return float(vals.max()) / mean


def peak_location(field, mask):
    """(row, col) of the max of `field` within `mask`."""
    f = np.asarray(field, dtype=float)
    m = np.asarray(mask, dtype=bool)
    masked = np.where(m, f, -np.inf)
    idx = np.unravel_index(int(np.argmax(masked)), f.shape)
    return (int(idx[0]), int(idx[1]))
