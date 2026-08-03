"""Canonical constants for the 3-D primitive shape library.

V_STAR_MM3 is the equal-volume reference (Matt, 2026-08-01): the 20 mm-diameter
sphere, the 3-D analog of the 2-D equal-area reference A* = pi*10^2 (the 20 mm
circle). Every Tier-1/2 primitive is uniform-scaled to this solid volume.
"""
from __future__ import annotations

import math

V_STAR_MM3: float = (4.0 / 3.0) * math.pi * 10.0 ** 3  # 4188.7902 mm^3, 20mm sphere
CHAMBER_L_MM: float = 60.0        # heatr3d cubic domain (heatr3d.Grid.L = 0.060 m)
VOL_REL_TOL: float = 1e-4         # equal-volume acceptance |v - V*| / V*
VOL_EPS_MM3: float = 1e-6         # zero-volume rejection threshold
EXTENT_EPS_MM: float = 1e-6       # degenerate (planar) bbox-axis threshold

__all__ = [
    "V_STAR_MM3",
    "CHAMBER_L_MM",
    "VOL_REL_TOL",
    "VOL_EPS_MM3",
    "EXTENT_EPS_MM",
]
