"""Voxel shape-fidelity metrics for heatr3d runs (solver-venv side; numpy ok).

Definitions ported from solve3d/shape_metrics.py (Matt's asymmetric shape
objective): melt-region IoU vs the nominal part mask at phi >= 0.8 / 0.9,
out-of-bounds melt fraction (the hard side of "dense iff in-bounds"),
in-part melt fraction, and the symmetric front distance in mm. iou() keeps
the NaN-on-empty rule: two empty regions are no information, not agreement.

Nominal reference = the run-grid part mask (approved call Q3).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:  # front distance needs scipy; absent -> reported as None, never faked
    from scipy.ndimage import binary_erosion, distance_transform_edt
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union; empty-vs-empty returns NaN (no false-green)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(a & b) / union)


def out_of_part_fraction(melt: np.ndarray, part: np.ndarray) -> float:
    """Melted volume OUTSIDE the part, as a fraction of the part volume."""
    part = np.asarray(part, dtype=bool)
    n_part = int(np.count_nonzero(part))
    if n_part == 0:
        return float("nan")
    spill = int(np.count_nonzero(np.asarray(melt, dtype=bool) & ~part))
    return float(spill / n_part)


def in_part_melt_fraction(melt: np.ndarray, part: np.ndarray) -> float:
    """Fraction of part voxels that melted."""
    part = np.asarray(part, dtype=bool)
    n_part = int(np.count_nonzero(part))
    if n_part == 0:
        return float("nan")
    return float(np.count_nonzero(np.asarray(melt, dtype=bool) & part) / n_part)


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask)


def front_distance_mm(a: np.ndarray, b: np.ndarray, h_mm: float) -> Optional[float]:
    """Mean symmetric surface distance between two mask boundaries, in mm.

    None when scipy is unavailable or either mask is empty (reported as
    not-computed, never fabricated).
    """
    if not _HAVE_SCIPY:
        return None
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if not a.any() or not b.any():
        return None
    if a.all() or b.all():
        return None                       # no boundary exists inside the grid
    ba, bb = _boundary(a), _boundary(b)
    if not ba.any() or not bb.any():
        return None
    da = distance_transform_edt(~ba) * h_mm    # distance TO a's boundary
    db = distance_transform_edt(~bb) * h_mm
    return float(0.5 * (db[ba].mean() + da[bb].mean()))


def shape_metrics(phi: np.ndarray, part: np.ndarray,
                  h_mm: float = 0.0) -> Dict[str, Optional[float]]:
    """The workbench headline strip for one run.

    Keys: iou_phi80, iou_phi90, oob_melt_frac_phi80, oob_melt_frac_phi90,
    in_part_melt_frac_phi90, front_dist_phi90_mm (None if not computable).
    """
    phi = np.asarray(phi, dtype=float)
    part = np.asarray(part, dtype=bool)
    m80 = phi >= 0.8
    m90 = phi >= 0.9
    out: Dict[str, Optional[float]] = {
        "iou_phi80": iou(m80, part),
        "iou_phi90": iou(m90, part),
        "oob_melt_frac_phi80": out_of_part_fraction(m80, part),
        "oob_melt_frac_phi90": out_of_part_fraction(m90, part),
        "in_part_melt_frac_phi90": in_part_melt_fraction(m90, part),
        "front_dist_phi90_mm": front_distance_mm(m90, part, h_mm) if h_mm else None,
    }
    return out
