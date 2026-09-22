import numpy as np
import pytest
from solve3d import geom_prewarp as gp


def test_resample_column_stretches_and_preserves_endpoints():
    src = np.linspace(0.0, 1.0, 4)          # 4 -> 9, monotone ramp
    out = gp.resample_column(src, 9)
    assert out.shape == (9,)
    assert abs(out[0] - 0.0) < 0.05 and abs(out[-1] - 1.0) < 0.05
    assert np.all(np.diff(out) >= -1e-9)


def test_resample_column_identity_and_empty():
    src = np.array([2.0, 5.0, 7.0])
    assert np.allclose(gp.resample_column(src, 3), src)
    assert gp.resample_column(np.zeros(0), 4).shape == (4,)   # empty src -> zeros
    assert gp.resample_column(src, 0).shape == (0,)
