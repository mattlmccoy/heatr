"""solve3d gate math -- pure numpy, so it runs in BOTH environments
(geo-prewarp venv AND the dolfinx spike env, which has no scipy)."""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import gates


def test_curve_rel_l2_is_zero_for_identical_curves_on_different_samplings():
    """The two engines sample the heating curve at different times and stop at
    different melt-onset times, so the metric must interpolate onto a common
    span. An identical underlying curve read on two samplings must score 0."""
    # LINEAR in t so linear interpolation onto the shared grid is exact and
    # the test measures the metric, not the interpolator.
    f = lambda t: 23.0 + 0.4 * t
    ta = np.linspace(0.0, 300.0, 31)
    tb = np.linspace(0.0, 280.0, 57)
    out = gates.curve_rel_l2(ta, f(ta), tb, f(tb))
    assert out["t_end_common_s"] == pytest.approx(280.0)
    assert out["rel_l2"] < 1e-12
    assert out["max_abs_diff_c"] < 1e-9


def test_curve_rel_l2_measures_rise_not_offset():
    """A 1 % error on the RISE must read 1 %. Comparing absolute temperatures
    would deflate it by the shared 23 C preheat offset (here by 23/123).

    The SECOND curve is the reference (the denominator), so the candidate is
    the 1.01x one."""
    t = np.linspace(0.0, 100.0, 51)
    base, rise = 23.0, 100.0 * (t / 100.0)
    out = gates.curve_rel_l2(t, base + 1.01 * rise, t, base + rise)
    assert out["rel_l2"] == pytest.approx(0.01, rel=1e-9)
    # the same comparison done on ABSOLUTE temperatures deflates the error by
    # the shared-offset factor (measured 0.00737 vs the true 0.01 here)
    naive = float(np.linalg.norm(0.01 * rise) / np.linalg.norm(base + rise))
    assert naive < 0.75 * out["rel_l2"]


def test_weighted_std_reduces_to_plain_std_for_equal_weights():
    """heatr3d voxels are equal-volume; FEM cells are not. The volume-weighted
    std must agree with numpy's population std when weights are equal."""
    rng = np.random.default_rng(3)
    v = rng.normal(size=257)
    assert gates.weighted_std(v, np.ones_like(v)) == pytest.approx(float(v.std()))


def test_weighted_std_ignores_zero_weight_samples():
    v = np.array([1.0, 2.0, 3.0, 1000.0])
    w = np.array([1.0, 1.0, 1.0, 0.0])
    assert gates.weighted_std(v, w) == pytest.approx(float(np.std([1.0, 2.0, 3.0])))


def test_rel_l2_pattern_is_scale_invariant():
    rng = np.random.default_rng(11)
    a = rng.random(500) + 0.5
    assert gates.rel_l2_pattern(a, 7.3 * a) < 1e-14


def test_rel_spread_is_relative_to_the_reference():
    assert gates.rel_spread(1.1, 1.0) == pytest.approx(0.1)
    assert gates.rel_spread(0.9, 1.0) == pytest.approx(0.1)
