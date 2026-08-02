"""RED-first tests for the pure logic of the asymmetric-objective driver.

Three pieces, all pure and all cheap:

  * `better_of`, which start or arm wins;
  * `spec_verdict`, the DENSE-IF-AND-ONLY-IF-IN-BOUNDS acceptance test, stated
    once in code so no table can quietly use a different rule;
  * `trade_curve`, the exchange-rate sweep. Because J_asym = w_out * J_out +
    w_in * J_in with both parts read at the SAME index, the whole sweep over
    w_out is recoverable EXACTLY from the two stored curves of a single forward
    run. That identity is what makes the trade curve free, so it is pinned by a
    test rather than trusted.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import asym_solve as asol


def test_better_of_keeps_the_lower_J_asym():
    a = {"J_asym": 0.30}
    b = {"J_asym": 0.25}
    assert asol.better_of(a, b) is b
    assert asol.better_of(b, a) is b
    assert asol.better_of(None, a) is a
    assert asol.better_of(a, None) is a
    assert asol.better_of(None, None) is None


def test_spec_verdict_needs_both_sides():
    ok = asol.spec_verdict(growth_pct=0.5, frac_at_or_above_floor=0.99)
    assert ok["in_bounds_ok"] and ok["no_growth_ok"] and ok["PASS"]
    grew = asol.spec_verdict(growth_pct=6.0, frac_at_or_above_floor=0.99)
    assert grew["no_growth_ok"] is False and grew["PASS"] is False
    thin = asol.spec_verdict(growth_pct=0.5, frac_at_or_above_floor=0.10)
    assert thin["in_bounds_ok"] is False and thin["PASS"] is False


def test_spec_verdict_tolerances_are_explicit_and_recorded():
    v = asol.spec_verdict(growth_pct=0.5, frac_at_or_above_floor=0.99,
                          growth_tol_pct=0.1, floor_frac_tol=1.0)
    assert v["growth_tol_pct"] == 0.1
    assert v["floor_frac_tol"] == 1.0
    assert v["PASS"] is False


def test_trade_curve_reproduces_the_argmin_of_the_weighted_sum():
    rng = np.random.default_rng(3)
    jo = np.cumsum(rng.random(40)) / 40.0          # rises with time
    ji = np.linspace(1.0, 0.05, 40)                # falls with time
    rows = asol.trade_curve(jo, ji, (0.5, 1.0, 4.0))
    assert [r["w_out"] for r in rows] == [0.5, 1.0, 4.0]
    for r in rows:
        expect = int(np.argmin(r["w_out"] * jo + ji))
        assert r["index"] == expect
        assert r["J_asym"] == pytest.approx(float(np.min(r["w_out"] * jo + ji)))
        assert r["J_out_unweighted"] == pytest.approx(float(jo[expect]))
        assert r["J_in"] == pytest.approx(float(ji[expect]))


def test_a_harder_out_of_bounds_weight_never_stops_later():
    """Monotonicity, and it is the reason the trade curve is a curve: raising
    the price of growth can only move the read state earlier or leave it."""
    rng = np.random.default_rng(5)
    jo = np.cumsum(rng.random(60)) / 60.0
    ji = np.linspace(1.0, 0.02, 60)
    idx = [r["index"] for r in asol.trade_curve(jo, ji, (0.25, 0.5, 1, 2, 4, 8, 16))]
    assert idx == sorted(idx, reverse=True)


def test_trade_curve_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        asol.trade_curve(np.zeros(5), np.zeros(6), (1.0,))
