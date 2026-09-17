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

# Group-DETECTION containment criterion (calibrated 2026-08-10 on the real
# stage_b4 cube/square/pyramid + Phase C cylinder meshes). A candidate mirror is
# accepted iff the fraction of in-part nodes whose reflected image lands within
# CONTAIN_K * median-NN of an in-part node is >= ACCEPT_THRESHOLD. On EVERY real
# mesh a VALID mirror reaches full containment (1.0000) at k = 2.0 -- its max
# match distance is 1.65..1.91 NN, so a 2.0-NN ball captures every mirrored node
# -- while a GENUINE non-symmetry (the pyramid's apex-along-z z-mirror, whose
# cross-section varies along the axis) sits at 0.314. ACCEPT_THRESHOLD = 0.90
# lives in that wide empty gap: high enough to be a real symmetry test, with
# headroom for mesh rattiness on a valid mirror (worst valid is 0.995 at k=1.5).
# The old 0.5*median-NN max-distance rule was TIGHTER than the mesh's inherent
# mirror-match slop (1.7-1.9 NN) and false-rejected every real mirror -> vacuous.
CONTAIN_K = 2.0                    # match radius in median-NN units
ACCEPT_THRESHOLD = 0.90           # min containment fraction to accept a mirror

CoordOp = Callable[[np.ndarray], np.ndarray]

_AXIS = {"x": 0, "y": 1, "z": 2}
_LETTER = {0: "x", 1: "y", 2: "z"}


def _group_average(s: np.ndarray, coords: np.ndarray,
                   group_ops: Sequence[CoordOp], interp: bool = False):
    """Orbit-average s over the group. Returns (s_sym, max_match_dist).

    matcher (`interp`):
      * False (default): recover each op's acting permutation by NEAREST-NODE
        match -- on a coarse/unstructured mesh the mirror image lands off-node, so
        s[idx] grabs a neighbour's value and injects apparent asymmetry (the
        fraction is then a LOWER bound). The calibrated retro cross-checks use this.
      * True: evaluate s AT the mirror image by LINEAR interpolation on the node
        cloud (hull-exterior images fall back to nearest). This removes the
        mesh-frame slop -- unbiased, so it cannot turn a genuinely asymmetric map
        symmetric (no false-pass) -- and makes the fraction mesh-resolution robust.
    `max_match_dist` (the nearest-node mirror distance) is still reported either
    way as the exactness/slop indicator.
    """
    tree = cKDTree(coords)
    orbit = np.zeros_like(s)
    max_match = 0.0

    sample = None
    if interp:
        # Interpolate in the cloud's INTRINSIC dimensions: a mesh confined to a
        # plane (or line) is degenerate for a full-3D triangulation, so drop the
        # axes with no range. Hull-exterior images fall back to nearest-node.
        rng = coords.max(0) - coords.min(0)
        scale = float(np.max(rng)) or 1.0
        vary = [k for k in range(coords.shape[1]) if rng[k] > 1e-9 * scale]
        if len(vary) >= 2:
            from scipy.interpolate import (LinearNDInterpolator,
                                           NearestNDInterpolator)
            lin = LinearNDInterpolator(coords[:, vary], s)
            nrst = NearestNDInterpolator(coords[:, vary], s)

            def sample(moved):
                q = moved[:, vary]
                vals = np.asarray(lin(q), float)
                miss = ~np.isfinite(vals)
                if miss.any():
                    vals[miss] = np.asarray(nrst(q[miss]), float)
                return vals
        elif len(vary) == 1:
            k = vary[0]
            order = np.argsort(coords[:, k])
            xp, fp = coords[order, k], s[order]

            def sample(moved):
                return np.interp(moved[:, k], xp, fp)     # clamps outside range
        else:                                             # constant cloud
            def sample(moved):
                return s.copy()

    for op in group_ops:
        moved = np.asarray(op(coords), float)
        dist, idx = tree.query(moved, k=1)
        max_match = max(max_match, float(np.max(dist)))
        orbit += sample(moved) if interp else s[idx]
    return orbit / float(len(group_ops)), max_match


def _mirror_op(center: np.ndarray, flip_axes: Sequence[int]) -> CoordOp:
    """Reflection about `center` in the given axes (a sign flip per axis)."""
    sign = np.ones(3)
    for a in flip_axes:
        sign[a] = -1.0
    c = np.asarray(center, float)
    return lambda p: c + (np.asarray(p, float) - c) * sign


def _median_nn(coords: np.ndarray) -> float:
    """Median nearest-neighbour spacing (the length scale for the match tol)."""
    if len(coords) < 2:
        return 1.0
    d, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(d[:, 1]))


def _mirror_containment(c_ip: np.ndarray, ip_tree: cKDTree,
                        center: np.ndarray, axis: int, tol: float):
    """Containment fraction of a single-axis mirror: the fraction of in-part
    nodes whose reflected image lies within `tol` of an in-part node. A part
    symmetry sends every in-part node onto (near) another in-part node, so the
    fraction is ~1; an axis along which the cross-section varies (an apex, a
    wedge) sends a macroscopic share of nodes off the part and the fraction
    drops. Returns (containment_fraction, max_match_dist)."""
    moved = _mirror_op(center, [axis])(c_ip)
    dist, _ = ip_tree.query(moved, k=1)
    return float(np.mean(dist <= tol)), float(np.max(dist))


def symmetry_gate_record(s: np.ndarray, coords: np.ndarray, weights: np.ndarray,
                         in_part: np.ndarray, build_axis: str = "y",
                         convective_faces: Sequence[str] = ("y=+L/2",),
                         threshold: float = SYMMETRY_THRESHOLD,
                         price_threshold: float = 0.01,
                         contain_k: float = CONTAIN_K,
                         accept_threshold: float = ACCEPT_THRESHOLD,
                         interp: bool = False,
                         scorer: Callable[[np.ndarray], float] | None = None,
                         j_uniform: float | None = None,
                         j_solved: float | None = None) -> dict:
    """Assemble the conformant `symmetry_gate` record (SYMMETRY_GATE_REPORT.md
    3.4/3.5/4) for a solved map on a 3-D mesh.

    Group detection: the two mirrors PERPENDICULAR to the build axis are each
    accepted ONLY if the mirror maps the in-part node set onto itself, verified by
    a CONTAINMENT criterion (`_mirror_containment`): at least `accept_threshold`
    of the in-part nodes' reflected images must land within `contain_k` * median
    NN of an in-part node. This is robust to a real mesh's inherent mirror-match
    slop (1.7-1.9 NN); a nearest-max-distance rule at 0.5 NN was tighter than that
    slop and false-rejected every real mirror. Rejected mirrors and the
    build-axis mirror (excluded a priori by the one-sided top convection) are
    recorded in `reductions`. Rotation is NOT
    composed here (all current solves are static; a turntable map must first prove
    its rotation axis is parallel to the convective-face normal).

    Verdict (either sufficient): PASS iff symmetric-variance fraction >= threshold
    OR projection price <= price_threshold; price is bought (one forward via
    `scorer`) ONLY when the fraction has already failed. A trivial group is a
    stated VACUOUS_PASS. `sendable` is True unless the verdict is FAIL.

    EXACTNESS: mesh-frame nearest-node matching injects apparent asymmetry, so the
    reported fraction is a LOWER bound; `max_match_dist` is reported so a
    slop-limited number is visible, never hidden.
    """
    s = np.asarray(s, float); coords = np.asarray(coords, float)
    w = np.asarray(weights, float); in_part = np.asarray(in_part, bool)
    b = _AXIS[build_axis]
    ip = in_part
    s_ip, c_ip, w_ip = s[ip], coords[ip], w[ip]
    center = 0.5 * (c_ip.min(0) + c_ip.max(0))
    nn = _median_nn(c_ip)
    tol = float(contain_k) * nn
    ip_tree = cKDTree(c_ip)

    accepted_axes: list[int] = []
    reductions: list[dict] = []
    max_match = 0.0
    for a in (0, 1, 2):
        if a == b:
            reductions.append({
                "element": f"mirror_{_LETTER[a]}",
                "reason": ("excluded a priori: one-sided convection on "
                           f"{list(convective_faces)} breaks the build-axis "
                           "mirror in the thermal/objective field")})
            continue
        contain, md = _mirror_containment(c_ip, ip_tree, center, a, tol)
        if contain >= accept_threshold:
            accepted_axes.append(a)
            max_match = max(max_match, md)
        else:
            reductions.append({
                "element": f"mirror_{_LETTER[a]}",
                "reason": (f"part not symmetric under it: only {contain:.3f} of "
                           f"in-part nodes' mirror images fall within "
                           f"{float(contain_k):g}x median NN ({tol:.2e}) of an "
                           f"in-part node (< accept {float(accept_threshold):g}); "
                           f"cross-section varies along this axis, max match dist "
                           f"{md:.2e} ({md / nn:.1f} NN)")})

    # group = all sign-flip subsets of the accepted axes
    from itertools import combinations
    group_ops = [lambda p: p]                      # identity
    group_names = ["identity"]
    for r in range(1, len(accepted_axes) + 1):
        for combo in combinations(accepted_axes, r):
            group_ops.append(_mirror_op(center, combo))
            group_names.append("mirror_" + "".join(_LETTER[a] for a in combo))

    vacuous = len(accepted_axes) == 0
    frac_out = symmetric_variance_fraction(s_ip, c_ip, w_ip, group_ops,
                                           interp=interp)
    fraction = frac_out["fraction"]

    price = None
    if not vacuous and fraction < threshold and scorer is not None \
            and j_uniform is not None and j_solved is not None:
        s_sym, _ = _group_average(s_ip, c_ip, group_ops, interp=interp)
        proj = s.copy(); proj[ip] = s_sym          # projected map, in-part only
        margin = abs(float(j_uniform) - float(j_solved))
        price = (abs(float(scorer(proj)) - float(j_solved)) / margin
                 if margin > 0 else float("inf"))

    if vacuous:
        verdict = "VACUOUS_PASS"
    elif fraction >= threshold or (price is not None and price <= price_threshold):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "fraction": fraction,
        "threshold": float(threshold),
        "projection_price": price,
        "price_threshold": float(price_threshold),
        "group": group_names,
        "reductions": reductions,
        "build_axis": build_axis,
        "convective_faces": list(convective_faces),
        "vacuous": bool(vacuous),
        "matcher": "interp" if interp else "nearest",
        "max_match_dist": float(max_match),
        "match_dist_note": ("nearest-node matching injects apparent asymmetry; "
                            "the fraction is a LOWER bound (can false-fail, never "
                            "false-pass). matcher='interp' evaluates the field at "
                            "the mirror image by linear interpolation and removes "
                            "that mesh-frame slop."),
        "verdict": verdict,
        "sendable": verdict != "FAIL",
    }


def symmetric_variance_fraction(s: np.ndarray, coords: np.ndarray,
                                weights: np.ndarray,
                                group_ops: Sequence[CoordOp],
                                interp: bool = False) -> dict:
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

    s_sym, max_match = _group_average(s, coords, group_ops, interp=interp)

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
