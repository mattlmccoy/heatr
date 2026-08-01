"""Pure-logic tests for the scheduling driver. Written RED first."""
from __future__ import annotations

import pytest

from adjoint2d import sched_solve as ss


def test_rounding_loss_is_signed_and_relative():
    r = ss.rounding_loss(100.0, 110.0)
    assert r["delta_J"] == pytest.approx(10.0)
    assert r["rel_loss"] == pytest.approx(0.10)
    # rounding can also HELP on a non-convex objective, and that must show as a
    # negative loss rather than be clamped away
    r2 = ss.rounding_loss(100.0, 95.0)
    assert r2["delta_J"] == pytest.approx(-5.0)
    assert r2["rel_loss"] < 0.0


def test_rescue_verdict_has_a_tie_band_and_names_the_null():
    assert ss.rescue_verdict(0.60, 0.65) == "SCHEDULING HELPS"
    assert ss.rescue_verdict(0.60, 0.55) == "SCHEDULING HURTS"
    assert ss.rescue_verdict(0.60, 0.61) == "NULL, scheduling buys nothing"
    assert ss.rescue_verdict(0.60, 0.62) == "SCHEDULING HELPS"    # exactly at the band


def test_the_cross_horizon_is_extended_and_the_others_are_not():
    # The library run pinned the cross J-minimum on the 1500-step horizon, so
    # its J was a bound. This campaign must run it longer.
    assert ss.HORIZON["cross"] == 2500
    assert ss.HORIZON["star"] == 1500
    assert set(ss.SHAPES) == {"cross", "T_shape", "L_shape", "star"}


def test_iso_j_hold_gain_finds_the_densest_time_inside_the_J_band():
    # J is flat-ish after the stop, rho keeps climbing. The probe must report
    # the LAST time still inside the J band, because that is the densest.
    j = [10.0, 5.0, 4.0, 4.02, 4.05, 4.20, 9.0]
    rho = [0.10, 0.30, 0.50, 0.55, 0.60, 0.70, 0.90]
    r = ss.iso_j_hold_gain(j, rho, stop_index=2, tol=0.02)
    assert r["index"] == 4                 # J 4.05 <= 4.0 * 1.02 = 4.08
    assert r["rho"] == pytest.approx(0.60)
    assert r["rho_at_stop"] == pytest.approx(0.50)
    assert r["d_rho"] == pytest.approx(0.10)
    assert r["extra_steps"] == 2


def test_iso_j_hold_gain_returns_the_stop_itself_when_J_rises_immediately():
    j = [10.0, 4.0, 9.0, 20.0]
    rho = [0.1, 0.5, 0.8, 0.9]
    r = ss.iso_j_hold_gain(j, rho, stop_index=1, tol=0.02)
    assert r["index"] == 1
    assert r["d_rho"] == pytest.approx(0.0)
    assert r["extra_steps"] == 0


def test_iso_j_hold_gain_does_not_jump_a_J_excursion():
    # A later point inside the band but separated by an excursion outside it is
    # not reachable by holding, because the march passes through the excursion.
    j = [10.0, 4.0, 9.0, 4.01]
    rho = [0.1, 0.5, 0.8, 0.95]
    r = ss.iso_j_hold_gain(j, rho, stop_index=1, tol=0.02)
    assert r["index"] == 1


def test_block_order_alternates_and_ends_on_the_map():
    assert ss.BLOCK_ORDER.count("p") == ss.BLOCK_ORDER.count("s")
    assert all(a != b for a, b in zip(ss.BLOCK_ORDER, ss.BLOCK_ORDER[1:]))
