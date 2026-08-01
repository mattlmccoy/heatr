"""Red-first tests for the PART-frame <-> LAB-frame transform operator.

The continuous-rotation averaged-kernel solve needs the rotation as a LINEAR
OPERATOR with an exact transpose, not as a call into `scipy.ndimage.rotate`:
the averaged heating kernel rotates the design map into the lab frame at every
sampled angle and rotates the resulting heating back into the part frame, and
the adjoint has to walk both of those backwards. An assembled sparse matrix
gives the transpose for free and makes forward and adjoint share one operator.

The contract the operator has to reproduce is already PROVEN elsewhere and is
inherited here rather than re-derived:

  `test_orientation_map_rotation.py` established that the production
  `geometry.part.rotation_deg = +90` equals `np.rot90(k=-1)` in array
  coordinates with ZERO mismatched cells on the T_shape at grid 120, which is
  `scipy.ndimage.rotate(angle = -rotation_deg, reshape=False, order=1)`.

So the operator must (a) reduce to that exact pixel permutation at 90 degrees,
(b) agree with `scripts.analysis.orientation_map_rotation.rotate_sat_map` in
the array interior at a general angle, and (c) satisfy the dot-product
transpose identity to machine precision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _ramp(n: int = 24) -> np.ndarray:
    """A smooth non-symmetric test pattern; no symmetry can hide a sign error."""
    j, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (0.3 + 0.4 * np.sin(2.1 * i / n) + 0.2 * np.cos(3.3 * j / n)
            + 0.05 * (i * j) / (n * n))


def test_identity_at_zero_degrees():
    from adjoint2d.rot_frame import rotation_operator
    m = _ramp()
    R = rotation_operator(m.shape, 0.0)
    out = R.apply(m, outside=1.0)
    assert np.allclose(out, m, atol=1e-12)


def test_90_degrees_is_the_exact_pixel_permutation():
    """Grid 120 (and any even n) puts the rotation centre at index (n-1)/2, so a
    90-degree rotation maps grid points onto grid points and the order-1
    interpolation must collapse to `np.rot90(k=-1)` exactly."""
    from adjoint2d.rot_frame import rotation_operator
    m = _ramp()
    R = rotation_operator(m.shape, 90.0)
    out = R.apply(m, outside=0.0)
    assert np.allclose(out, np.rot90(m, k=-1), atol=1e-9)


def test_180_and_270_are_exact_permutations():
    from adjoint2d.rot_frame import rotation_operator
    m = _ramp()
    for deg, k in ((180.0, -2), (270.0, -3)):
        out = rotation_operator(m.shape, deg).apply(m, outside=0.0)
        assert np.allclose(out, np.rot90(m, k=k), atol=1e-9), deg


def test_matches_production_rotate_sat_map_in_the_interior():
    """Same convention as the already-proven scipy helper, general angle."""
    from scipy.ndimage import binary_erosion

    from adjoint2d.rot_frame import rotation_operator
    from scripts.analysis.orientation_map_rotation import rotate_sat_map
    m = _ramp(40)
    ref = rotate_sat_map(m, 37.0, outside=0.0)
    out = rotation_operator(m.shape, 37.0).apply(m, outside=0.0)
    core = binary_erosion(np.ones(m.shape, dtype=bool), iterations=3)
    assert float(np.max(np.abs(out[core] - ref[core]))) < 1e-9


def test_transpose_dot_product_identity():
    """<R x, y> == <x, R^T y> to machine precision, the adjoint gate."""
    from adjoint2d.rot_frame import rotation_operator
    rng = np.random.default_rng(20260801)
    n = 32
    R = rotation_operator((n, n), 23.0)
    x = rng.standard_normal((n, n))
    y = rng.standard_normal((n, n))
    lhs = float(np.sum(R.apply(x, outside=0.0) * y))
    rhs = float(np.sum(x * R.apply_T(y)))
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), 1.0)


def test_outside_fill_is_affine_and_only_touches_the_border():
    """`outside` enters as a constant offset carrying no design sensitivity.

    R.apply(x, outside=c) == R.apply(x, outside=0) + c * (1 - rowsum), which is
    what lets the adjoint use `apply_T` unchanged whatever the fill value is.
    """
    from adjoint2d.rot_frame import rotation_operator
    rng = np.random.default_rng(7)
    n = 20
    R = rotation_operator((n, n), 31.0)
    x = rng.standard_normal((n, n))
    a = R.apply(x, outside=0.0)
    b = R.apply(x, outside=1.0)
    assert np.allclose(b - a, R.deficit, atol=1e-12)
    assert float(np.max(np.abs(R.deficit[3:-3, 3:-3]))) < 1e-12


def test_round_trip_recovers_the_interior():
    from scipy.ndimage import binary_erosion

    from adjoint2d.rot_frame import rotation_operator
    m = _ramp(48)
    fwd = rotation_operator(m.shape, 41.0)
    back = rotation_operator(m.shape, -41.0)
    out = back.apply(fwd.apply(m, outside=0.0), outside=0.0)
    core = binary_erosion(np.ones(m.shape, dtype=bool), iterations=10)
    assert float(np.mean(np.abs(out[core] - m[core]))) < 0.02


def test_partition_of_unity_in_the_interior():
    """Rows sum to 1 away from the border, so a constant map rotates to itself."""
    from adjoint2d.rot_frame import rotation_operator
    n = 30
    R = rotation_operator((n, n), 17.0)
    out = R.apply(np.ones((n, n)), outside=0.0)
    # Eroded by 8 cells so every sampled source point stays inside the array
    # for this angle; the corners of a square necessarily leave it.
    assert float(np.max(np.abs(out[8:-8, 8:-8] - 1.0))) < 1e-12


def test_angle_set_covers_the_full_circle_without_duplicating_the_endpoint():
    from adjoint2d.rot_frame import averaging_angles
    a = averaging_angles(15.0)
    assert len(a) == 24
    assert a[0] == 0.0 and a[-1] == 345.0
    assert np.allclose(np.diff(a), 15.0)
    with pytest.raises(ValueError):
        averaging_angles(0.0)
    with pytest.raises(ValueError):
        averaging_angles(7.0)      # does not divide 360


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
