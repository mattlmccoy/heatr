#!/usr/bin/env python3
"""Parametric outlines that are NOT in the eighteen-shape standardized library.

These exist so the generalization claim can be tested on geometry the campaign
has never seen. Each is a plain vertex list in metres; nothing here knows
anything about heating, symmetry or dopant, and no downstream module is told
which shape it is looking at.
"""
from __future__ import annotations

import numpy as np

__all__ = ["gear", "keyhole", "arrow", "NOVEL"]


def gear(n_teeth: int = 8, r_tip: float = 0.011, root_ratio: float = 0.72,
         tooth_fraction: float = 0.45, flank_fraction: float = 0.12,
         n_arc: int = 6) -> np.ndarray:
    """A spur-gear silhouette: `n_teeth` trapezoidal teeth on a round body."""
    r_root = float(r_tip) * float(root_ratio)
    pts: list[tuple[float, float]] = []
    step = 2.0 * np.pi / int(n_teeth)
    half_tip = 0.5 * float(tooth_fraction) * step
    half_root = half_tip + float(flank_fraction) * step
    for k in range(int(n_teeth)):
        c = k * step
        for t in np.linspace(c - step / 2.0, c - half_root, n_arc, endpoint=False):
            pts.append((r_root * np.cos(t), r_root * np.sin(t)))
        pts.append((r_tip * np.cos(c - half_tip), r_tip * np.sin(c - half_tip)))
        pts.append((r_tip * np.cos(c + half_tip), r_tip * np.sin(c + half_tip)))
        pts.append((r_root * np.cos(c + half_root), r_root * np.sin(c + half_root)))
    return np.asarray(pts, dtype=float)


def keyhole(r_head: float = 0.0080, stem_top: float = 0.0092,
            stem_bottom: float = 0.0055, stem_len: float = 0.016,
            n_arc: int = 96) -> np.ndarray:
    """A round head over a tapered stem: one mirror axis, no rotational symmetry.

    The head arc is TRIMMED to the angular sector that lies outside the stem
    width, so the outline never crosses itself. An earlier version of this
    function walked the whole circle and then appended the stem corners, which
    crossed at the join; the even-odd fill turned that crossing into a notch cut
    out of the head, and the intake now refuses such an outline outright
    (`geometry_intake.find_self_intersection`).
    """
    y0 = 0.5 * float(stem_len)
    half = 0.5 * float(stem_top)
    if half >= r_head:
        raise ValueError("stem_top must be narrower than the head diameter")
    a = float(np.arcsin(half / float(r_head)))        # where the stem meets it
    t = np.linspace(-np.pi / 2.0 + a, 1.5 * np.pi - a, n_arc)
    head = np.column_stack([r_head * np.cos(t), r_head * np.sin(t) + y0])
    # the arc already ENDS at the two stem shoulders, so only the two bottom
    # corners are added; repeating the shoulders would make a zero-length edge
    stem = np.array([
        [-0.5 * stem_bottom, y0 - stem_len],
        [0.5 * stem_bottom, y0 - stem_len],
    ])
    poly = np.vstack([head, stem])
    poly[:, 1] -= 0.5 * (poly[:, 1].min() + poly[:, 1].max())
    return poly


def arrow(length: float = 0.026, head_w: float = 0.019, shaft_w: float = 0.0085,
          head_len: float = 0.011) -> np.ndarray:
    """A block arrow: one mirror axis, strongly directional."""
    hl = float(head_len)
    L = float(length)
    poly = np.array([
        [0.5 * L, 0.0],
        [0.5 * L - hl, 0.5 * head_w],
        [0.5 * L - hl, 0.5 * shaft_w],
        [-0.5 * L, 0.5 * shaft_w],
        [-0.5 * L, -0.5 * shaft_w],
        [0.5 * L - hl, -0.5 * shaft_w],
        [0.5 * L - hl, -0.5 * head_w],
    ])
    poly[:, 0] -= poly[:, 0].mean()
    return poly


NOVEL = {"gear8": lambda: gear(8), "keyhole": keyhole, "arrow": arrow}
