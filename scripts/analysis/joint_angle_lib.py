"""Pure logic for the JOINT per-angle dopant-map re-solve.

The orientation sweep (`ORIENTATION_OPTIMIZATION_REPORT.md`) scanned angles with
the dopant map SOLVED ONCE at zero degrees and rigidly rotated. Its limit 2 says
so explicitly. This campaign re-solves the map at every angle, which makes the
angle scan a joint optimization over orientation and dopant map, and tests
whether the best angle moves.

Everything here is deterministic and side-effect free so it can be tested before
the driver runs. The rotation itself is delegated to the already proven
`orientation_map_rotation.rotate_sat_map` (production convention
`rotation_deg = +90` equals `np.rot90(k=-1)`, zero mismatched cells).
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .orientation_map_rotation import rotate_sat_map

# Symmetry-aware angle sets, identical to the fixed-map orientation sweep so the
# two curves are read at the same abscissae.
ANGLES: dict[str, tuple[float, ...]] = {
    "T_shape": (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0),
    "L_shape": (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0),
    "cross": (0.0, 15.0, 30.0, 45.0),
    "star": (0.0, 9.0, 18.0, 27.0, 36.0),
}

# The period an angle difference is quoted modulo. T_shape and L_shape use 180
# because the y-flip equivalence theta ~ theta + 180 is ASSUMED up to the
# top-only convection term (the orientation sweep observed it to hold to better
# than 0.01 percent of J); the cross is 4-fold and the five-point star 5-fold.
SYMMETRY_PERIOD_DEG: dict[str, float] = {
    "T_shape": 180.0, "L_shape": 180.0, "cross": 90.0, "star": 72.0,
}

# The operating ceiling the campaign quotes; a strict exceedance is flagged.
CEILING_C = 250.0


def rotated_warm_start(sat0: np.ndarray, rotation_deg: float,
                       part_mask_rot: np.ndarray,
                       box: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """A design-variable start built from a zero-degree map at `rotation_deg`.

    Two steps, both already conventions of this campaign: rotate the lab-frame
    map by the production part-rotation convention, then project it into the
    design box inside the rotated part while holding saturation 1 outside
    (`multistart.start_from_map`).
    """
    rot = rotate_sat_map(sat0, rotation_deg, part_mask_rot=part_mask_rot,
                         outside=1.0)
    pm = np.asarray(part_mask_rot, dtype=bool)
    return np.where(pm, np.clip(rot, float(box[0]), float(box[1])), 1.0)


def best_row(rows: Sequence[Mapping]) -> Mapping:
    """The lowest-J row; the first one wins a tie."""
    if not rows:
        raise ValueError("best_row needs at least one row")
    return min(rows, key=lambda r: float(r["J"]))


def argmin_angle(rows: Iterable[Mapping]) -> float:
    """The angle of the lowest-J row; the smaller angle wins a tie."""
    rs = list(rows)
    if not rs:
        raise ValueError("argmin_angle needs at least one row")
    return float(min(rs, key=lambda r: (float(r["J"]), float(r["angle_deg"])))
                 ["angle_deg"])


def angle_delta_deg(angle: float, reference: float, period: float) -> float:
    """Signed move from `reference` to `angle`, wrapped into (-period/2, period/2].

    Wrapping by the shape's own symmetry period is what makes the number
    meaningful: a cross at 90 degrees is the same orientation as a cross at 0.
    """
    p = float(period)
    if p <= 0.0:
        raise ValueError(f"period must be positive, got {period!r}")
    d = (float(angle) - float(reference)) % p
    if d > p / 2.0:
        d -= p
    return float(d)


def exceeds_ceiling(max_T_c: float, ceiling: float = CEILING_C) -> bool:
    """True when a maximum temperature is strictly above the operating ceiling."""
    return bool(float(max_T_c) > float(ceiling))
