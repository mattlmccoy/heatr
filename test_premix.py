"""TDD ladder for the premix baseline dopant material law (premix.py).

Mirrors heatr3d.build_gamma premix semantics (heatr3d.py:591-624).
premix_frac=0.0 must reproduce the reference inline blend BIT-FOR-BIT.
"""
import numpy as np

from premix import apply_premix, premix_frac_from_wtpct, PREMIX_WTPCT_FULL

# Material endpoints matching the jared config family (effective composite sigma).
V = dict(sigma_v=1e-8, sigma_d0=0.04, eps_v=2.7, eps_d=20.0)


def test_wtpct_bridge_is_linear_over_doped_label():
    assert premix_frac_from_wtpct(0.0) == 0.0
    assert np.isclose(premix_frac_from_wtpct(PREMIX_WTPCT_FULL), 1.0)
    assert np.isclose(premix_frac_from_wtpct(15.0), 15.0 / PREMIX_WTPCT_FULL)  # 0.6


def test_premix_off_is_bit_identical_to_inline_formula():
    rng = np.random.default_rng(0)
    blend = rng.random((16, 16))  # fill_frac / _eff_fill in [0, 1]
    ref_sigma = V["sigma_v"] + blend * (V["sigma_d0"] - V["sigma_v"])
    ref_eps = V["eps_v"] + blend * (V["eps_d"] - V["eps_v"])
    sigma, eps_r = apply_premix(blend, blend, premix_frac=0.0, **V)
    # Bit-for-bit: identical float operations, identical order.
    assert np.array_equal(sigma, ref_sigma)
    assert np.array_equal(eps_r, ref_eps)


def test_floor_added_bed_and_part_values():
    # bed (blend=0) rises to sigma_premix; part (blend=1) = sigma_premix + full span.
    blend = np.array([[0.0, 1.0]])
    f = 0.5
    sigma, eps_r = apply_premix(blend, blend, premix_frac=f,
                                premix_budget="floor_added", **V)
    sigma_premix = V["sigma_v"] + f * (V["sigma_d0"] - V["sigma_v"])
    span = V["sigma_d0"] - V["sigma_v"]
    assert np.isclose(sigma[0, 0], sigma_premix)               # bed
    assert np.isclose(sigma[0, 1], sigma_premix + 1.0 * span)  # part boosted above doped
    eps_premix = V["eps_v"] + f * (V["eps_d"] - V["eps_v"])
    assert np.isclose(eps_r[0, 0], eps_premix)


def test_budget_fixed_part_pinned_to_doped():
    # budget_fixed: part (blend=1) pinned at sigma_d0; bed at sigma_premix.
    blend = np.array([[0.0, 1.0]])
    f = 0.5
    sigma, _ = apply_premix(blend, blend, premix_frac=f,
                            premix_budget="budget_fixed", **V)
    sigma_premix = V["sigma_v"] + f * (V["sigma_d0"] - V["sigma_v"])
    assert np.isclose(sigma[0, 0], sigma_premix)   # bed
    assert np.isclose(sigma[0, 1], V["sigma_d0"])  # part exactly doped (total ~const)


def test_unknown_budget_raises():
    import pytest
    with pytest.raises(ValueError):
        apply_premix(np.zeros((2, 2)), np.zeros((2, 2)),
                     premix_frac=0.5, premix_budget="bogus", **V)
