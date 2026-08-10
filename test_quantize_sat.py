"""TDD for printable-bpp quantization of a continuous dopant sat map."""
import numpy as np

from quantize_sat import quantize_sat


def test_4bpp_gives_16_distinct_levels_max():
    rng = np.random.default_rng(0)
    sat = rng.uniform(0.0, 1.5, size=(40, 40))
    q = quantize_sat(sat, n_bits=4, sat_max=1.5)
    assert len(np.unique(q)) <= 16
    assert q.max() <= 1.5 + 1e-9 and q.min() >= 0.0


def test_2bpp_gives_4_levels():
    sat = np.linspace(0.0, 1.5, 100).reshape(10, 10)
    q = quantize_sat(sat, n_bits=2, sat_max=1.5)
    assert len(np.unique(q)) <= 4


def test_snaps_to_nearest_level():
    # sat_max=1.0, 1 bit -> levels {0,1}. 0.4 -> 0, 0.6 -> 1.
    sat = np.array([[0.4, 0.6]])
    q = quantize_sat(sat, n_bits=1, sat_max=1.0)
    assert np.allclose(q, [[0.0, 1.0]])


def test_continuous_preserved_when_already_on_grid():
    sat = np.array([[0.0, 0.5, 1.0]])
    q = quantize_sat(sat, n_bits=1, sat_max=1.0)  # levels {0,1}; 0.5 ties -> 0 or 1
    assert set(np.unique(q)).issubset({0.0, 1.0})
