"""RED-first tests for the physical-length design filter.

The robustness validation (`SOLVE_ROBUSTNESS_VALIDATION.md`, Task B) measured
that the solved dopant maps carry load-bearing structure at the single-cell
scale: a one-cell part-masked blur costs between +37 and +892 percent of the
shape objective on five of six shapes. That is the optimizer buying fidelity
with sub-resolution rim sculpture. The fix is to filter the DESIGN VARIABLE
inside the solve, so that structure finer than the filter radius is not
expressible at all.

The filter must be:
  * linear in the design variable (so its transpose is exact and cheap),
  * part-masked with no bleed from the nominal value held outside,
  * box preserving without a clip, because a normalized convolution is a
    convex combination of in-part values,
  * self-adjoint up to the part mask, checked by the dot-product identity.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import design_filter as df


def _mask(ny=12, nx=14):
    m = np.zeros((ny, nx), dtype=bool)
    m[3:9, 4:11] = True
    return m


def test_zero_radius_is_the_identity_inside_the_part():
    pm = _mask()
    rng = np.random.default_rng(0)
    v = rng.random(pm.shape)
    s = df.apply_filter(v, pm, 0.0)
    assert np.allclose(s[pm], v[pm])


def test_outside_the_part_is_held_at_the_nominal_value():
    pm = _mask()
    rng = np.random.default_rng(1)
    v = rng.random(pm.shape)
    s = df.apply_filter(v, pm, 1.5, outside=1.0)
    assert np.all(s[~pm] == 1.0)


def test_no_bleed_from_outside_into_the_part():
    """A part that is uniformly 0.2 stays 0.2 even with 1.0 held outside."""
    pm = _mask()
    v = np.where(pm, 0.2, 1.0)
    s = df.apply_filter(v, pm, 2.0, outside=1.0)
    assert np.allclose(s[pm], 0.2, atol=1e-12)


def test_box_is_preserved_without_a_clip():
    pm = _mask()
    rng = np.random.default_rng(2)
    v = np.where(pm, rng.random(pm.shape), 1.0)
    s = df.apply_filter(v, pm, 1.75)
    assert s[pm].min() >= v[pm].min() - 1e-12
    assert s[pm].max() <= v[pm].max() + 1e-12


def test_filter_reduces_in_part_roughness():
    pm = _mask()
    rng = np.random.default_rng(3)
    v = np.where(pm, rng.random(pm.shape), 1.0)
    rough = [float(np.std(df.apply_filter(v, pm, r)[pm])) for r in (0.0, 1.0, 2.0)]
    assert rough[0] > rough[1] > rough[2]


def test_transpose_satisfies_the_dot_product_identity():
    """<F v, w> == <v, F^T w> for every v, w supported on the part."""
    pm = _mask()
    rng = np.random.default_rng(4)
    v = np.zeros(pm.shape)
    w = np.zeros(pm.shape)
    v[pm] = rng.standard_normal(int(pm.sum()))
    w[pm] = rng.standard_normal(int(pm.sum()))
    # apply_filter with outside=0 is the pure linear map; the nominal outside
    # value is an affine offset that carries no design sensitivity.
    lhs = float(np.sum(df.apply_filter(v, pm, 1.5, outside=0.0) * w))
    rhs = float(np.sum(v * df.filter_vjp(w, pm, 1.5)))
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-14)


def test_transpose_is_supported_on_the_part_only():
    pm = _mask()
    rng = np.random.default_rng(5)
    w = rng.standard_normal(pm.shape)
    g = df.filter_vjp(w, pm, 1.5)
    assert np.all(g[~pm] == 0.0)


def test_zero_radius_transpose_is_the_masked_identity():
    pm = _mask()
    rng = np.random.default_rng(6)
    w = rng.standard_normal(pm.shape)
    g = df.filter_vjp(w, pm, 0.0)
    assert np.allclose(g[pm], w[pm])


def test_negative_radius_raises():
    with pytest.raises(ValueError):
        df.apply_filter(np.ones((4, 4)), np.ones((4, 4), dtype=bool), -1.0)
