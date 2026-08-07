"""TDD for premix-sweep metrics (pure functions, no solve)."""
import numpy as np

from premix_study_metrics import (
    bed_absorption_fraction,
    peak_to_mean,
    peak_location,
)


def test_bed_absorption_fraction_zero_when_bed_empty():
    Qrf = np.array([[0.0, 0.0], [2.0, 4.0]])
    doped = np.array([[False, False], [True, True]])  # bed rows all zero
    assert bed_absorption_fraction(Qrf, doped) == 0.0


def test_bed_absorption_fraction_half():
    Qrf = np.array([[3.0, 0.0], [0.0, 3.0]])
    doped = np.array([[False, False], [False, True]])  # one doped cell (3), bed=3
    # bed = ~doped -> cells (0,0)=3,(0,1)=0,(1,0)=0 sum=3; total=6 -> 0.5
    assert np.isclose(bed_absorption_fraction(Qrf, doped), 0.5)


def test_peak_to_mean_uniform_is_one():
    f = np.full((4, 4), 5.0)
    mask = np.ones((4, 4), dtype=bool)
    assert np.isclose(peak_to_mean(f, mask), 1.0)


def test_peak_to_mean_masked():
    f = np.array([[10.0, 2.0], [2.0, 2.0]])
    mask = np.ones((2, 2), dtype=bool)
    # peak 10, mean (10+2+2+2)/4 = 4 -> 2.5
    assert np.isclose(peak_to_mean(f, mask), 2.5)


def test_peak_location_returns_argmax_within_mask():
    f = np.array([[1.0, 9.0], [5.0, 2.0]])
    mask = np.array([[True, False], [True, True]])  # (0,1)=9 excluded by mask
    assert peak_location(f, mask) == (1, 0)  # 5 is the masked max
