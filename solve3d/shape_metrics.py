"""Phase A close-out: SHAPE-fidelity metrics for the melt/density field.

Runs in the geo-prewarp venv (scipy is needed for the Euclidean distance
transform; the dolfinx spike env has no scipy). The dolfinx side therefore
EXPORTS its melt-onset temperature field sampled on the shared evaluation grid
(solve3d/gates.eval_grid_points) and this module scores both engines on that
one grid.

WHY THESE METRICS
-----------------
Matt's recorded objective (spec, commit d298c6d): "our goal is to make every
single part a fully dense part if and only if it falls within the nominal shape
bounds ... It must be dense and match the shape, and we are willing to
compromise a little bit on density (maybe 80-90% density) if we can achieve a
better shape within the bounds."

That is an ASYMMETRIC shape statement, so it is scored as one:
  (a) melt-region agreement  -- IoU of {phi >= t} for t in {0.8, 0.9}, the band
      Matt named. Symmetric agreement on where the part melts.
  (b) melt-front position    -- symmetric surface distance between the two
      engines' phi = 0.9 fronts, in millimetres. IoU saturates; a front that is
      0.2 mm off and a front that is 2 mm off both read "high IoU" on a 20 mm
      part, so the distance is reported in the units the CAD tolerance lives in.
  (c) bed melt / out-of-bounds -- melted volume OUTSIDE the nominal part, as a
      fraction of the nominal part volume. This is the hard side of the
      objective ("if and only if it falls within the nominal shape bounds"), so
      the two engines must agree here or the forward model cannot be trusted to
      score a candidate design's spill.

sigma_T is NOT here. It stays a reported flatness diagnostic.

EVALUATION GEOMETRY. Both anchors are FULL-HEIGHT extrusions, so the fields are
z-invariant and every metric is computed per z-plane on a fine (x, y) grid and
averaged, with the plane-to-plane spread reported as a measured z-invariance
check rather than an assumption. For an extrusion, area fraction == volume
fraction, which is why (c) is reported as a volume fraction from planar data.
The "nominal part" is the ANALYTIC shape (exact circle / square), not either
engine's discretization of it, so neither engine is scored on its own mask.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks.

    An empty-vs-empty pair returns NaN, NOT 1.0. Two regions that do not exist
    are not perfect agreement, they are no information -- rendering that as a
    perfect score would be a false-green."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(a & b) / union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    """The inner boundary layer of a mask (mask minus its erosion)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask)
    return mask & ~binary_erosion(mask, border_value=1)


def symmetric_surface_distance_mm(a: np.ndarray, b: np.ndarray,
                                  h_m: float) -> float:
    """Symmetric (mean) surface distance between two regions' boundaries [mm].

        SSD = 0.5 * ( mean_{x in dA} dist(x, dB) + mean_{y in dB} dist(y, dA) )

    Symmetrized on purpose: a one-sided distance is small whenever one front is
    a subset of a neighbourhood of the other, which hides an engine that
    systematically under- or over-melts. NaN if either region is empty."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if not a.any() or not b.any():
        return float("nan")
    ba, bb = _boundary(a), _boundary(b)
    if not ba.any() or not bb.any():
        return float("nan")
    d_to_b = distance_transform_edt(~bb, sampling=h_m)
    d_to_a = distance_transform_edt(~ba, sampling=h_m)
    ab = float(d_to_b[ba].mean())
    ba_ = float(d_to_a[bb].mean())
    return 1000.0 * 0.5 * (ab + ba_)


def out_of_part_fraction(melt: np.ndarray, part: np.ndarray) -> float:
    """Melted area/volume OUTSIDE the nominal part, as a fraction of the
    nominal part's area/volume. 0.0 = no spill into the bed."""
    melt = np.asarray(melt, dtype=bool)
    part = np.asarray(part, dtype=bool)
    n_part = int(np.count_nonzero(part))
    if n_part == 0:
        return float("nan")
    return float(np.count_nonzero(melt & ~part) / n_part)


def in_part_melt_fraction(melt: np.ndarray, part: np.ndarray) -> float:
    """Fraction of the nominal part that reached the melt threshold -- the
    'is it dense inside the bounds' half of the objective."""
    melt = np.asarray(melt, dtype=bool)
    part = np.asarray(part, dtype=bool)
    n_part = int(np.count_nonzero(part))
    if n_part == 0:
        return float("nan")
    return float(np.count_nonzero(melt & part) / n_part)


def plane_metrics(T_a: np.ndarray, T_b: np.ndarray, part: np.ndarray,
                  h_m: float, phi_of_T, thresholds=(0.8, 0.9)) -> dict:
    """All Phase-A shape metrics for ONE (x, y) plane.

    T_a / T_b are the two engines' melt-onset temperature fields on the SAME
    grid; `part` is the analytic nominal-shape mask on that grid."""
    out: dict = {}
    pa, pb = phi_of_T(T_a), phi_of_T(T_b)
    for t in thresholds:
        ma, mb = pa >= t, pb >= t
        key = f"phi{t:g}".replace(".", "p")
        out[key] = {
            "iou": iou(ma, mb),
            "in_part_melt_frac_a": in_part_melt_fraction(ma, part),
            "in_part_melt_frac_b": in_part_melt_fraction(mb, part),
            "out_of_part_frac_a": out_of_part_fraction(ma, part),
            "out_of_part_frac_b": out_of_part_fraction(mb, part),
        }
        o = out[key]
        o["out_of_part_frac_abs_diff"] = abs(
            o["out_of_part_frac_a"] - o["out_of_part_frac_b"])
        o["in_part_melt_frac_abs_diff"] = abs(
            o["in_part_melt_frac_a"] - o["in_part_melt_frac_b"])
    out["front_ssd_mm_phi0p9"] = symmetric_surface_distance_mm(
        pa >= 0.9, pb >= 0.9, h_m)
    return out


def aggregate_planes(per_plane: list[dict]) -> dict:
    """Mean over z-planes, with the plane-to-plane spread kept as the measured
    z-invariance check (these anchors are extrusions, so a large spread would
    mean the extrusion assumption failed and the planar reduction is invalid)."""
    def _walk(items):
        first = items[0]
        agg: dict = {}
        for k, v in first.items():
            if isinstance(v, dict):
                agg[k] = _walk([it[k] for it in items])
            else:
                vals = np.array([float(it[k]) for it in items], dtype=float)
                agg[k] = float(np.nanmean(vals))
                agg[k + "__plane_spread"] = float(np.nanmax(vals) - np.nanmin(vals))
        return agg
    return _walk(per_plane)
