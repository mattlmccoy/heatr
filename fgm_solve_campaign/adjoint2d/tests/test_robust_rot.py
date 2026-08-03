"""Red-first tests for the ROTATING / DWELL grid hold-out harness.

`adjoint2d.robust.py` moved a STATIC solved dopant map from the 120 grid to the
160 grid. A rotating or indexed arm needs three more things to be right before
any number it produces means anything:

  1. the rotation operator must still be an EXACT pixel permutation at the new
     grid for the 90-degree indexing schedules, otherwise the hold-out measures
     the bilinear remap error of `rot_frame` rather than the map transfer
     (`CONTINUOUS_ROTATION_REPORT.md` Section 6.1 measured that error at 0.007
     to 0.010 percent of dose per rotation event, which is enough to break the
     standing energy gate at 375 events);
  2. resampling the part-frame map 120 -> 160 and co-rotating it must COMMUTE
     at the exact-permutation angles, so that "resample then co-rotate" and
     "co-rotate then resample" are the same lab-frame map and the transfer
     convention is unambiguous;
  3. the stored turntable program must expand to the same per-outer-step
     position sequence at the new grid, because the program is a wall-clock
     object and the grid must not touch it.

Everything here is grid-160 unless it says otherwise. No physics is solved in
this file except one electro-quasi-static solve for the drive recalibration.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import robust as rb
from adjoint2d import robust_rot as rr
from adjoint2d.rot_frame import rotation_operator

N_HOLDOUT = 160
N_SOLVED_AT = 120


def _bumpy(n: int, seed: int = 0) -> np.ndarray:
    """A map with structure at every scale, so a permutation error shows."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / float(n - 1)
    smooth = 0.5 + 0.3 * np.sin(6.0 * np.pi * xx) * np.cos(4.0 * np.pi * yy)
    return np.clip(smooth + 0.05 * rng.standard_normal((n, n)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 1. the exact-permutation property, re-established at the hold-out grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deg,k", [(90.0, -1), (180.0, -2), (270.0, -3)])
def test_quarter_turns_are_exact_pixel_permutations_at_grid_160(deg, k):
    m = _bumpy(N_HOLDOUT)
    got = rotation_operator((N_HOLDOUT, N_HOLDOUT), deg).apply(m, outside=1.0)
    want = np.rot90(m, k=k)
    assert np.array_equal(got, want), (
        f"{deg} degrees at grid {N_HOLDOUT} is not an exact permutation: "
        f"{int(np.sum(got != want))} cells differ, max |delta| "
        f"{float(np.max(np.abs(got - want))):.3e}")


def test_quarter_turn_round_trip_at_grid_160_is_the_identity():
    m = _bumpy(N_HOLDOUT, seed=3)
    fwd_op = rotation_operator((N_HOLDOUT, N_HOLDOUT), 90.0)
    back = rotation_operator((N_HOLDOUT, N_HOLDOUT), -90.0)
    assert np.array_equal(back.apply(fwd_op.apply(m)), m)


def test_the_new_grid_loses_no_weight_at_a_quarter_turn():
    """`deficit` is the weight that fell outside the array extent. At an exact
    permutation it must be zero everywhere, at grid 160 as at grid 120."""
    for n in (N_SOLVED_AT, N_HOLDOUT):
        op = rotation_operator((n, n), 90.0)
        assert float(np.max(np.abs(op.deficit))) == 0.0


# ---------------------------------------------------------------------------
# 2. resample and co-rotate must commute
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deg", [90.0, 180.0, 270.0])
def test_resample_and_corotation_commute_at_the_holdout_grid(deg):
    s120 = _bumpy(N_SOLVED_AT, seed=1)
    op160 = rotation_operator((N_HOLDOUT, N_HOLDOUT), deg)
    op120 = rotation_operator((N_SOLVED_AT, N_SOLVED_AT), deg)

    resample_then_rotate = op160.apply(
        rb.resample_map(s120, N_HOLDOUT, N_HOLDOUT))
    rotate_then_resample = rb.resample_map(
        op120.apply(s120), N_HOLDOUT, N_HOLDOUT)

    assert np.allclose(resample_then_rotate, rotate_then_resample, atol=1e-12), (
        "resampling and co-rotation do not commute at "
        f"{deg} degrees: max |delta| "
        f"{float(np.max(np.abs(resample_then_rotate - rotate_then_resample))):.3e}")


def test_transfer_map_holds_the_nominal_value_outside_the_part_and_quantizes_inside():
    s120 = _bumpy(N_SOLVED_AT, seed=2)
    pm = np.zeros((N_HOLDOUT, N_HOLDOUT), dtype=bool)
    pm[40:120, 40:120] = True

    out = rr.transfer_map(s120, pm, bpp=4)

    assert out.shape == (N_HOLDOUT, N_HOLDOUT)
    assert np.all(out[~pm] == 1.0), "outside the part must hold the nominal 1"
    levels = np.round(out[pm] * 15.0)
    assert np.allclose(out[pm], levels / 15.0, atol=1e-12), (
        "in-part values are not on the 4-bits-per-pixel printer level grid")
    assert out[pm].min() >= 0.0 and out[pm].max() <= 1.0


def test_transfer_map_at_the_same_grid_is_only_a_quantization():
    """The 1 percent dead band of the production loader means a same-grid
    transfer must not resample at all."""
    s = _bumpy(N_HOLDOUT, seed=5)
    pm = np.ones((N_HOLDOUT, N_HOLDOUT), dtype=bool)
    out = rr.transfer_map(s, pm, bpp=4)
    assert np.allclose(out, np.round(s * 15.0) / 15.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. the stored program is a wall-clock object and must not move with the grid
# ---------------------------------------------------------------------------

def test_index_program_expands_to_four_outer_steps_per_position():
    """90-degree indexing every 2.0 s at the pinned 0.5 s outer step."""
    pos = rr.index_program_positions(n_positions=4, interval_s=2.0, dt_s=0.5,
                                     n_steps=20)
    assert pos.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                            3, 3, 3, 3, 0, 0, 0, 0]


def test_stored_moves_expand_identically_whatever_the_grid():
    moves = [{"position_deg": 0.0, "dwell_s": 5.0, "move_at_s": 0.0},
             {"position_deg": 90.0, "dwell_s": 5.0, "move_at_s": 5.0},
             {"position_deg": 180.0, "dwell_s": 5.0, "move_at_s": 10.0}]
    a = rr.moves_to_positions(moves, [0.0, 90.0, 180.0, 270.0], dt_s=0.5,
                              n_steps=30)
    b = rr.moves_to_positions(moves, [0.0, 90.0, 180.0, 270.0], dt_s=0.5,
                              n_steps=30)
    assert a.tolist() == b.tolist()
    assert a[:10].tolist() == [0] * 10
    assert a[10:20].tolist() == [1] * 10
    assert a[20:30].tolist() == [2] * 10


def test_program_positions_reject_a_position_absent_from_the_angle_set():
    moves = [{"position_deg": 37.0, "dwell_s": 1.0, "move_at_s": 0.0}]
    with pytest.raises(ValueError, match="37"):
        rr.moves_to_positions(moves, [0.0, 90.0, 180.0, 270.0], dt_s=0.5,
                              n_steps=4)


# ---------------------------------------------------------------------------
# 4. the drive recalibration at the hold-out grid, exact by the quadratic law
# ---------------------------------------------------------------------------

def test_recalibrated_drive_makes_the_uniform_arm_absorb_the_target_at_grid_160():
    """One electro-quasi-static solve, one exact rescale, then VERIFY by a
    second solve. Absorbed power is quadratic in the drive because the solve is
    linear in the applied potential and neither conductivity nor permittivity
    depends on it."""
    from adjoint2d.library_solve import shape_config
    from adjoint2d.pins import load_cfg

    cfg = load_cfg(shape_config("cross"))
    cal = rr.recalibrate_at_grid(cfg, N_HOLDOUT, target_w_per_m=500.0)

    assert cal["p_at_pinned_v_w_per_m"] > 0.0
    assert abs(cal["p_verified_w_per_m"] - 500.0) < 1e-6, cal
    assert cal["n_grid"] == N_HOLDOUT
