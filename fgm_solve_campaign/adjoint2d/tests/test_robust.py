"""Tests for the robustness-validation helpers (grid transfer and rim smoothing).

Written red first: `adjoint2d.robust` does not exist when these are first run.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import robust as rb


# ---------------------------------------------------------------------------
# resample_map: the production map-injection resampling convention
# ---------------------------------------------------------------------------

def test_resample_identity_shape_is_a_noop():
    rng = np.random.default_rng(0)
    a = rng.random((12, 12))
    out = rb.resample_map(a, 12, 12)
    assert out.shape == (12, 12)
    assert np.array_equal(out, a)


def test_resample_changes_shape_and_stays_in_box():
    a = np.linspace(0.0, 1.0, 16)[None, :] * np.ones((16, 1))
    out = rb.resample_map(a, 24, 24)
    assert out.shape == (24, 24)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # a horizontal ramp stays a horizontal ramp: every row identical
    assert np.allclose(out, out[0][None, :])
    # monotone increasing along x
    assert np.all(np.diff(out[0]) > 0)


def test_resample_matches_scipy_zoom_order1_clipped():
    """Contract: same call the production loader makes (rfam_eqs_coupled.py:374-380)."""
    from scipy.ndimage import zoom
    rng = np.random.default_rng(1)
    a = rng.random((10, 10)) * 1.4 - 0.2
    ref = np.clip(zoom(a, (20 / 10, 20 / 10), order=1), 0.0, 1.0)
    assert np.allclose(rb.resample_map(a, 20, 20), ref)


# ---------------------------------------------------------------------------
# smooth_in_part: part-masked Gaussian smoothing of the continuous map
# ---------------------------------------------------------------------------

def _mask(n=20):
    pm = np.zeros((n, n), dtype=bool)
    pm[5:15, 5:15] = True
    return pm


def test_smooth_zero_radius_is_identity_in_part():
    pm = _mask()
    rng = np.random.default_rng(2)
    s = np.where(pm, rng.random(pm.shape), 1.0)
    out = rb.smooth_in_part(s, pm, 0.0)
    assert np.allclose(out[pm], s[pm])


def test_smooth_holds_outside_at_nominal_one():
    pm = _mask()
    s = np.where(pm, 0.2, 1.0)
    out = rb.smooth_in_part(s, pm, 2.0)
    assert np.all(out[~pm] == 1.0)


def test_smooth_does_not_bleed_the_outside_value_into_the_part():
    """A part that is uniformly 0.2 must stay 0.2 even though outside is 1.0."""
    pm = _mask()
    s = np.where(pm, 0.2, 1.0)
    out = rb.smooth_in_part(s, pm, 2.0)
    assert np.allclose(out[pm], 0.2, atol=1e-12)


def test_smooth_reduces_in_part_roughness():
    pm = _mask()
    rng = np.random.default_rng(3)
    s = np.where(pm, rng.random(pm.shape), 1.0)
    r0 = float(np.std(s[pm]))
    r1 = float(np.std(rb.smooth_in_part(s, pm, 1.0)[pm]))
    r2 = float(np.std(rb.smooth_in_part(s, pm, 2.0)[pm]))
    assert r2 < r1 < r0


def test_smooth_stays_in_the_unit_box():
    pm = _mask()
    rng = np.random.default_rng(4)
    s = np.where(pm, rng.random(pm.shape), 1.0)
    out = rb.smooth_in_part(s, pm, 2.0)
    assert out[pm].min() >= 0.0 and out[pm].max() <= 1.0


def test_smooth_rejects_negative_radius():
    pm = _mask()
    with pytest.raises(ValueError):
        rb.smooth_in_part(np.ones(pm.shape), pm, -1.0)


# ---------------------------------------------------------------------------
# recalibrated_voltage: absorbed power is quadratic in the drive voltage
# ---------------------------------------------------------------------------

def test_recalibrated_voltage_is_the_square_root_law():
    assert rb.recalibrated_voltage(100.0, 250.0, 1000.0) == pytest.approx(200.0)
    assert rb.recalibrated_voltage(2428.17, 500.0, 500.0) == pytest.approx(2428.17)


def test_recalibrated_voltage_rejects_nonpositive_measured_power():
    with pytest.raises(ValueError):
        rb.recalibrated_voltage(100.0, 0.0, 500.0)
