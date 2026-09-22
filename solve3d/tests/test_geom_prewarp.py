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


def test_prewarp_solve_converges_against_uniform_forward():
    nx, ny, nz0 = 3, 3, 8
    mask0 = np.zeros((nx, ny, nz0), bool); mask0[:, :, :4] = True
    dop0 = np.where(mask0, 0.5, 0.0)
    h = 0.2
    LAM_Z = 0.5   # 4 voxels / 0.5 = 8 voxels exactly -> reachable to tol

    def forward(green_mask, green_dop):
        occ = green_mask.sum(axis=2).astype(float)
        return occ * h * LAM_Z, 0.0, {}

    res = gp.prewarp_solve(mask0, dop0, h, forward,
                           bulk_factor=1.0, tol=0.01, k_max=8)
    assert res["converged"] is True
    Ht = gp.target_column_heights(mask0, h)
    expect = Ht / LAM_Z
    assert np.allclose(res["H_green"][Ht > 0], expect[Ht > 0], rtol=0.02)
    assert res["err_history"][-1] < 0.01
    assert res["iters"] >= 1


def test_prewarp_solve_stall_breaks_on_voxel_quantization():
    nx, ny, nz0 = 3, 3, 8
    mask0 = np.zeros((nx, ny, nz0), bool); mask0[:, :, :4] = True
    dop0 = np.where(mask0, 0.5, 0.0)
    h = 0.2

    def forward(green_mask, green_dop):
        occ = green_mask.sum(axis=2).astype(float)
        return occ * h * 0.6, 0.0, {}   # 6.67 voxels -> unreachable to 1%

    res = gp.prewarp_solve(mask0, dop0, h, forward,
                           bulk_factor=1.0, tol=0.01, k_max=20)
    assert res["converged"] is False
    assert res["iters"] < 8
    assert min(res["err_history"]) < 0.06
    assert res["warp_std"] is not None


def test_column_height_update_clamps_runaway_gain():
    Hg = np.array([[1.0]]); Ht = np.array([[2.0]]); Hm = np.array([[1e-12]])  # ~0 measured
    out = gp.column_height_update(Hg, Ht, Hm, gain_cap=8.0)
    assert out[0, 0] <= 8.0 + 1e-9        # bounded by gain_cap, not 2e12


def test_build_green_volume_raises_on_divergence():
    mask0 = np.zeros((2, 2, 4), bool); mask0[:, :, :4] = True
    dop0 = np.where(mask0, 0.5, 0.0)
    Hgiant = np.full((2, 2), 1000.0)      # 1000/0.2 = 5000 voxels >> 5x*4
    with pytest.raises(ValueError):
        gp.build_green_volume(mask0, dop0, Hgiant, h=0.2)


def test_build_green_volume_rejects_nonpositive_h():
    mask0 = np.zeros((1, 1, 2), bool); mask0[0, 0, :2] = True
    with pytest.raises(AssertionError):
        gp.build_green_volume(mask0, np.zeros((1, 1, 2)), np.array([[0.4]]), h=0.0)


def test_build_green_volume_mixed_part_and_empty_columns():
    mask0 = np.zeros((2, 2, 4), bool)
    mask0[0, 0, :3] = True                 # only one column is part
    dop0 = np.where(mask0, 0.5, 0.0)
    Ht = gp.target_column_heights(mask0, 0.2)
    gm, gd = gp.build_green_volume(mask0, dop0, Ht, h=0.2)
    assert gm[0, 0].sum() == 3
    assert gm[0, 1].sum() == 0 and gm[1, 0].sum() == 0 and gm[1, 1].sum() == 0
    assert np.all(gd[~gm] == 0.0)


def test_prewarp_solve_rejects_bad_kmax():
    mask0 = np.zeros((2, 2, 4), bool); mask0[:, :, :4] = True
    with pytest.raises(ValueError):
        gp.prewarp_solve(mask0, np.zeros((2, 2, 4)), 0.2, lambda gm, gd: (None, 0, {}), k_max=0)


def test_emit_prewarped_spec_roundtrips(tmp_path):
    gm = np.zeros((2, 2, 3), bool); gm[:, :, :2] = True
    gd = np.where(gm, 0.5, 0.0)
    prov = {"enabled": True, "iters": 3, "converged": True, "tol": 0.01,
            "warp_std_before": 9.5, "warp_std_after": 2.0,
            "bulk_factor": 1.58, "source_densify": "densify_pyramid"}
    out = tmp_path / "pyr_prewarped_green_spec.npz"
    gp.emit_prewarped_spec(gm, gd, out, prov)
    d = np.load(out, allow_pickle=True)
    assert str(d["proxy_field"]) == "solve"
    assert d["SOLVE_cont"].shape == gd.shape
    assert d["part_mask"].shape == gm.shape and d["part_mask"].dtype == bool
    rec = d["prewarp"].item()          # dict round-trips via object array
    assert rec["enabled"] is True and rec["converged"] is True
    # dopant zero outside the mask (staging validity)
    assert np.all(d["SOLVE_cont"][~d["part_mask"]] == 0.0)
