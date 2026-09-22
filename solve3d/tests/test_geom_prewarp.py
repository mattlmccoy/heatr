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


def test_build_green_volume_stretches_columns_and_conserves_mask():
    # nominal: column A 3 voxels tall, column B 2 voxels; dopant ramps in z
    nx, ny, nz0 = 2, 1, 4
    mask0 = np.zeros((nx, ny, nz0), bool)
    dop0 = np.zeros((nx, ny, nz0))
    mask0[0, 0, :3] = True; dop0[0, 0, :3] = [0.2, 0.5, 0.8]
    mask0[1, 0, :2] = True; dop0[1, 0, :2] = [0.4, 0.6]
    h = 0.2
    # ask column A to be 6 voxels tall (double), B to be 2 (unchanged)
    H_green = np.array([[6 * h], [2 * h]])
    gm, gd = gp.build_green_volume(mask0, dop0, H_green, h)
    assert gm.shape == (nx, ny, 6) and gd.shape == (nx, ny, 6)
    # column A: 6 occupied, dopant nonzero within, zero above
    assert gm[0, 0].sum() == 6 and gm[0, 0, :6].all()
    assert np.all(gd[0, 0, :6] > 0)
    # column B: 2 occupied, rest empty
    assert gm[1, 0].sum() == 2 and not gm[1, 0, 2:].any()
    assert np.all(gd[1, 0, 2:] == 0.0)
    # dopant range preserved (resample stays within source min/max)
    assert 0.2 - 1e-9 <= gd[0, 0, :6].min() and gd[0, 0, :6].max() <= 0.8 + 1e-9
