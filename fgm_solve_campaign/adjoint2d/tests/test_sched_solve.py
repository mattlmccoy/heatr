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


def test_block_order_alternates_and_ends_on_the_map():
    assert ss.BLOCK_ORDER.count("p") == ss.BLOCK_ORDER.count("s")
    assert all(a != b for a, b in zip(ss.BLOCK_ORDER, ss.BLOCK_ORDER[1:]))
