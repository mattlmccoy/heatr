"""Dithered 4bpp quantization at printer DPI (Matt 2026-08-03: cash in the
720 DPI printhead for DOSE fidelity).

Plain per-pixel rounding biases the printed dose by up to half a level
(1/30 of full scale) uniformly over a region; ordered dithering makes the
LOCAL AVERAGE dose track the continuous solved value, and the physics
(millimeter thermal averaging) reads the average.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stl_compensation_tool" / "webapp"))

from meteor_bridge import apply_sat_to_levels  # noqa: E402


def test_dithered_local_average_tracks_the_continuous_dose():
    h = w = 256
    levels = np.full((h, w), 15, np.uint8)
    sat = np.tile(np.linspace(0.05, 0.95, w), (h, 1))
    out = apply_sat_to_levels(levels, sat, bpp=4)
    assert out.dtype == np.uint8
    assert out.max() <= 15
    # 16x16 block averages of printed dose vs the continuous target
    blocks = out.reshape(h // 16, 16, w // 16, 16).mean(axis=(1, 3)) / 15.0
    target = sat.reshape(h // 16, 16, w // 16, 16).mean(axis=(1, 3))
    assert np.abs(blocks - target).max() < 0.02


def test_dithering_beats_rounding_on_a_constant_field():
    levels = np.full((128, 128), 15, np.uint8)
    sat = np.full((128, 128), 0.633)
    dithered = apply_sat_to_levels(levels, sat, bpp=4)
    rounded = apply_sat_to_levels(levels, sat, bpp=4, dither=None)
    err_d = abs(dithered.mean() / 15.0 - 0.633)
    err_r = abs(rounded.mean() / 15.0 - 0.633)
    assert err_d < 0.005
    assert err_d < err_r / 3


def test_zeros_and_determinism_preserved():
    rng = np.random.default_rng(3)
    levels = np.where(rng.random((64, 64)) > 0.4, 15, 0).astype(np.uint8)
    sat = 0.3 + 0.5 * rng.random((64, 64))
    a = apply_sat_to_levels(levels, sat, bpp=4)
    b = apply_sat_to_levels(levels, sat, bpp=4)
    assert np.array_equal(a, b), "dither must be deterministic (no RNG)"
    assert (a[levels == 0] == 0).all(), "shell/infill zeros must survive"
