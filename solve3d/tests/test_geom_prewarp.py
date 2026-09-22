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


def test_target_column_heights_counts_occupied_times_h():
    mask = np.zeros((2, 2, 5), bool)
    mask[0, 0, :3] = True          # column (0,0): 3 voxels
    mask[1, 1, :5] = True          # column (1,1): 5 voxels
    H = gp.target_column_heights(mask, h=0.2)
    assert H.shape == (2, 2)
    assert np.isclose(H[0, 0], 0.6) and np.isclose(H[1, 1], 1.0)
    assert H[0, 1] == 0.0


def test_column_height_update_multiplicative_and_masks_empty():
    Hg = np.array([[1.0, 1.0]])
    Ht = np.array([[2.0, 0.0]])     # 2nd column not part
    Hm = np.array([[1.0, 0.0]])     # measured half of target -> gain 2x
    out = gp.column_height_update(Hg, Ht, Hm)
    assert np.isclose(out[0, 0], 2.0)   # 1.0 * 2.0/1.0
    assert out[0, 1] == 0.0             # non-part column stays 0


def test_max_rel_error_over_part_columns():
    Ht = np.array([[10.0, 0.0], [10.0, 10.0]])
    Hm = np.array([[9.5, 0.0], [10.0, 8.0]])   # errors 5%, -, 0%, 20%
    assert abs(gp.max_rel_error(Ht, Hm) - 0.20) < 1e-9
