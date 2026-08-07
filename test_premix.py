"""TDD ladder for the premix baseline dopant material law (premix.py).

Mirrors heatr3d.build_gamma premix semantics (heatr3d.py:591-624).
premix_frac=0.0 must reproduce the reference inline blend BIT-FOR-BIT.
"""
import numpy as np

from premix import apply_premix

# Material endpoints matching the jared config family (effective composite sigma).
V = dict(sigma_v=1e-8, sigma_d0=0.04, eps_v=2.7, eps_d=20.0)


def test_premix_off_is_bit_identical_to_inline_formula():
    rng = np.random.default_rng(0)
    blend = rng.random((16, 16))  # fill_frac / _eff_fill in [0, 1]
    ref_sigma = V["sigma_v"] + blend * (V["sigma_d0"] - V["sigma_v"])
    ref_eps = V["eps_v"] + blend * (V["eps_d"] - V["eps_v"])
    sigma, eps_r = apply_premix(blend, blend, premix_frac=0.0, **V)
    # Bit-for-bit: identical float operations, identical order.
    assert np.array_equal(sigma, ref_sigma)
    assert np.array_equal(eps_r, ref_eps)
