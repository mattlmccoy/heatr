"""Symmetry-consistency gate for solved dopant maps (3-D lane) -- PRIMITIVE ONLY.

A budget-limited solve stopped still-descending on a coarse mesh fits
DISCRETIZATION-FRAME noise, so the published map's spatial structure is mostly
numerical RESIDUE even though the peak/density/hold-out gates PASS (they bound
the consequence, not the map). The remedy is to project the solved map onto the
symmetry group the problem REQUIRES and check how much of the map's variance
survives: residue is asymmetric and washes out, real structure survives.

WHAT IS HERE: the spec-independent linear-algebra core -- given a map, node
coords, weights, and an EXPLICIT group (list of coord->coord ops), the
volume-weighted fraction of the map's variance in the group-symmetric subspace.

WHAT IS NOT HERE (conforms to the campaign lane's SYMMETRY_GATE_REPORT.md when it
lands, then wired into studio_solve acceptance + the is_sendable emission, per
Matt's go): (1) group DETECTION -- part INTERSECT field INTERSECT objective
INTERSECT convection-BC symmetry. The convection-BC term is load-bearing: RFAM's
convection is ONE-SIDED on the top face y=+L/2 (forward.py), so the objective is
NOT y-mirror symmetric even though the EQS |E|^2 is; the required group for a
centered part is {x-mirror, z-mirror} (+ build-axis rotation), y EXCLUDED. A
y-inclusive group false-rejects the square (0.860 correct vs 0.699 full).
(2) the vacuous-pass rule for asymmetric parts. (3) the per-run results-JSON
contract (no frozen-schema change).

Retro numbers that motivated this (solve3d/results/symmetry_retro_3d.json):
cube 0.912 PASS, square 0.860 PASS, pyramid 0.354 FAIL (residue, like the
Phase C cylinder).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree

SYMMETRY_THRESHOLD = 0.8            # PASS iff symmetric-variance fraction >= this

CoordOp = Callable[[np.ndarray], np.ndarray]


def symmetric_variance_fraction(s: np.ndarray, coords: np.ndarray,
                                weights: np.ndarray,
                                group_ops: Sequence[CoordOp]) -> dict:
    """Volume-weighted fraction of `s`'s variance in the subspace symmetric under
    `group_ops`.

    Each op maps coords -> coords; the acting-on-the-map permutation is recovered
    by matching each transformed node to its nearest original node (KDTree). The
    group-symmetric component is the orbit average s_sym = mean_g s[perm_g], and
    the fraction is ||s_sym - sbar||^2_w / ||s - sbar||^2_w (weighted, mean
    removed). 1.0 = fully symmetric, 0.0 = pure residue/anti-symmetric.

    Returns {"fraction", "max_match_dist", "n"}. `max_match_dist` is the largest
    node->mirror match distance: if it is not ~0 the mesh itself is not
    group-symmetric and the fraction is a slight UNDER-estimate (matching slop
    injects apparent asymmetry) -- callers should surface it, not hide it.
    """
    s = np.asarray(s, float)
    coords = np.asarray(coords, float)
    w = np.asarray(weights, float)
    if not (len(s) == len(coords) == len(w)):
        raise ValueError("s, coords, weights must have the same length")

    tree = cKDTree(coords)
    orbit = np.zeros_like(s)
    max_match = 0.0
    for op in group_ops:
        moved = np.asarray(op(coords), float)
        dist, idx = tree.query(moved, k=1)
        orbit += s[idx]
        max_match = max(max_match, float(np.max(dist)))
    s_sym = orbit / float(len(group_ops))

    wsum = float(w.sum())
    sbar = float(np.dot(w, s) / wsum)
    den = float(np.dot(w, (s - sbar) ** 2))
    if den <= 0.0:                              # constant map: trivially symmetric
        frac = 1.0
    else:
        num = float(np.dot(w, (s_sym - sbar) ** 2))
        frac = num / den
    return {"fraction": frac, "max_match_dist": max_match, "n": int(len(s))}


def symmetry_verdict(fraction: float,
                     threshold: float = SYMMETRY_THRESHOLD) -> str:
    """PASS iff the symmetric-variance fraction meets the threshold (default 0.8).
    The projection-cost alternative (<= 1% of the solve margin) and the
    vacuous-pass for asymmetric parts are added when conforming to the report."""
    return "PASS" if float(fraction) >= float(threshold) else "FAIL"
