"""S2 Task 1: band machinery on SYNTHETIC convergence sequences.

The band code is tested against sequences whose answer is known before it is
pointed at the campaign -- if the band code is wrong, every S2 verdict is wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from heatr3d_s2 import bands


GRIDS = [48, 64, 80, 96]


def _second_order(grids, q_inf=10.0, c=1000.0):
    return [q_inf + c * (1.0 / n) ** 2 for n in grids]


def test_second_order_sequence_recovers_order_two():
    v = _second_order(GRIDS)
    out = bands.analyse(GRIDS, v, relative=True)
    assert out["status"] == "monotone_convergent"
    assert abs(out["observed_order"] - 2.0) < 0.25, out["observed_order"]
    assert out["r2"] > 0.99


def test_first_order_sequence_recovers_order_one():
    v = [10.0 + 50.0 / n for n in GRIDS]
    out = bands.analyse(GRIDS, v, relative=True)
    assert abs(out["observed_order"] - 1.0) < 0.25, out["observed_order"]


def test_band_is_the_frozen_multiple_of_the_finest_pair_change():
    v = _second_order(GRIDS)
    out = bands.analyse(GRIDS, v, relative=True, safety=1.5)
    assert out["band"] == pytest.approx(1.5 * out["finest_change"])
    assert out["finest_change"] == pytest.approx(out["changes"][-1]["change"])


def test_bounded_oscillatory_sequence_still_yields_a_band():
    """Successive changes that wobble slightly but do not grow are convergent."""
    v = [10.5, 10.2, 10.28, 10.25]
    out = bands.analyse(GRIDS, v, relative=True)
    assert out["status"] in ("monotone_convergent", "bounded_oscillatory")
    assert out["band"] is not None
    assert out["value_trend"] == "oscillatory"


def test_diverging_sequence_FAILS_LOUDLY_and_yields_no_band():
    """The one behaviour that must never be smoothed into a band."""
    v = [10.0, 10.1, 10.4, 11.5]           # successive changes growing
    out = bands.analyse(GRIDS, v, relative=True)
    assert out["status"] == "diverging"
    assert out["band"] is None
    assert out["pass"] is False
    assert "diverg" in out["reason"].lower()


def test_diverging_is_still_diverging_when_a_ceiling_would_have_passed_it():
    """A tiny but GROWING change must fail on the trend, not sneak through on
    magnitude. This is the trap the pre-registration's 'diverging_is_FAIL'
    clause exists to close."""
    v = [10.0, 10.000001, 10.000004, 10.00002]
    out = bands.analyse(GRIDS, v, relative=True, ceiling=0.01)
    assert out["finest_change"] < 0.01
    assert out["status"] == "diverging"
    assert out["pass"] is False


def test_pass_requires_both_the_trend_and_the_ceiling():
    v = _second_order(GRIDS)
    out = bands.analyse(GRIDS, v, relative=True, ceiling=1.0)
    assert out["pass"] is True
    tight = bands.analyse(GRIDS, v, relative=True, ceiling=1e-9)
    assert tight["status"] == "monotone_convergent"
    assert tight["pass"] is False
    assert "ceiling" in tight["reason"].lower()


def test_absolute_mode_does_not_divide_by_the_value():
    """Shape metrics are already fractions or millimetres; dividing them by
    their own magnitude would be meaningless."""
    v = [0.50, 0.40, 0.36, 0.34]
    rel = bands.analyse(GRIDS, v, relative=True)
    ab = bands.analyse(GRIDS, v, relative=False)
    assert ab["changes"][-1]["change"] == pytest.approx(0.02)
    assert rel["changes"][-1]["change"] == pytest.approx(0.02 / 0.34)


def test_three_grids_is_the_minimum_and_two_is_refused():
    with pytest.raises(ValueError, match="at least 3"):
        bands.analyse([48, 64], [1.0, 2.0])


def test_nan_input_is_refused_rather_than_producing_a_band():
    with pytest.raises(ValueError, match="finite"):
        bands.analyse(GRIDS, [1.0, float("nan"), 1.0, 1.0])
