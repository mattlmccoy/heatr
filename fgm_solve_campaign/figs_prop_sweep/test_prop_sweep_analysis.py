"""Red-green tests for the proportional-sweep dynamics analysis.

Run:
    ./.venv312/bin/python -m pytest fgm_solve_campaign/figs_prop_sweep/test_prop_sweep_analysis.py -q
"""
from __future__ import annotations

import numpy as np

from prop_sweep_analysis import lagged_stats, classify_dynamics


def _seq_period2(n_iter: int = 12, n_pix: int = 500) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    a = rng.uniform(0.2, 0.8, n_pix)
    b = 1.0 - a
    return [a if k % 2 == 0 else b for k in range(n_iter)]


def _seq_fixed_point(n_iter: int = 12, n_pix: int = 500) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    target = rng.uniform(0.2, 0.8, n_pix)
    start = rng.uniform(0.2, 0.8, n_pix)
    return [target + (start - target) * (0.5 ** k) for k in range(n_iter)]


def test_lagged_stats_period2() -> None:
    stats = lagged_stats(_seq_period2())
    assert stats["corr_lag2_mean"] > 0.99
    assert stats["corr_lag1_mean"] < stats["corr_lag2_mean"] - 0.5
    assert stats["step_norm_mean"] > 0.1  # steps do not shrink


def test_lagged_stats_fixed_point() -> None:
    # Analysis contract: stats are computed on the PLATEAU slice, so drop
    # the transient (first 3 iterates) exactly as the real pipeline does.
    stats = lagged_stats(_seq_fixed_point()[3:])
    assert stats["corr_lag1_mean"] > 0.99
    assert stats["step_norm_trend"] < 0.0  # steps shrink


def test_classify_period2() -> None:
    assert classify_dynamics(lagged_stats(_seq_period2())) == "period-2"


def test_classify_fixed_point() -> None:
    assert classify_dynamics(lagged_stats(_seq_fixed_point()[3:])) == "fixed-point"


def test_classify_exact_quantized_fixed_point() -> None:
    # 4bpp quantization can freeze the map EXACTLY: every plateau map identical,
    # steps identically zero. That is a fixed point, not "neither".
    rng = np.random.default_rng(2)
    frozen = rng.uniform(0.2, 0.8, 500)
    seq = [frozen.copy() for _ in range(10)]
    assert classify_dynamics(lagged_stats(seq)) == "fixed-point"
