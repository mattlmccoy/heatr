"""Rotate a lab-frame dopant saturation map by the production part-rotation
convention.

Contract (probed on the real engine, see test_orientation_map_rotation.py):
`geometry.part.rotation_deg = +90` in `rfam_eqs_coupled.make_domain` equals
`np.rot90(mask, k=-1)` in array coordinates, i.e. `scipy.ndimage.rotate`
with `angle = -rotation_deg`, `reshape=False`.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate as _ndrotate


def rotate_sat_map(
    sat: np.ndarray,
    rotation_deg: float,
    part_mask_rot: np.ndarray | None = None,
    outside: float = 1.0,
) -> np.ndarray:
    """Rotate a (ny, nx) saturation map to follow the part at rotation_deg.

    Args:
        sat: lab-frame saturation map solved at rotation 0.
        rotation_deg: the production geometry.part.rotation_deg of the run
            the map will be injected into.
        part_mask_rot: optional part mask rasterized at rotation_deg; when
            given, cells outside it are reset to `outside` (prototype
            convention: s = 1 outside the part so only the print changes).
        outside: fill value for cells rotated in from beyond the array and
            for the outside re-mask.

    Returns:
        Rotated map, clipped to [0, 1].
    """
    out = _ndrotate(
        np.asarray(sat, dtype=float),
        angle=-float(rotation_deg),
        reshape=False,
        order=1,
        mode="constant",
        cval=float(outside),
    )
    out = np.clip(out, 0.0, 1.0)
    if part_mask_rot is not None:
        out = np.where(np.asarray(part_mask_rot, dtype=bool), out, float(outside))
    return out
