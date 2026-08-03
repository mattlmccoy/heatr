"""Red-green tests for webapp/meteor_bridge pure logic.

Contracts probed from real code/data (data-contract rule):
- Meteor TIFF levels: uint8 in [0, 2^bpp - 1]; rasterize_zones fills
  shell/infill values, 0 = no ink (slicer.py:498).
- Print canvas: pixels[r, c], PIL origin top-left, y flipped
  (mm_to_pix: r = height_px - y_mm * ppm), part auto-centered + margin.
- Dopant grid: sat[iy, ix] over the 60 mm chamber, x/y ascending, part
  centered at origin (fields.npz x = linspace(-0.03, 0.03, 120)).
- dopant_volume.npz sat already includes the z-gain (pipeline.py).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

from meteor_bridge import (
    layer_to_slice_index, apply_sat_to_levels, sample_sat_on_canvas,
    decode_tiff_gray_to_levels,
)


def test_decode_tiff_gray_matches_meteor_encoding():
    # Probed 2026-07-30: write_tiff(levels 0..15, bpp=4) read back by PIL as
    # gray = 255 - level*17 (black = max ink). Decode must invert exactly.
    gray = 255 - np.arange(16, dtype=np.uint8) * 17
    levels = decode_tiff_gray_to_levels(gray, bpp=4)
    assert (levels == np.arange(16)).all()
    # 2 bpp: levels 0..3 -> gray 255, 170, 85, 0
    gray2 = np.array([255, 170, 85, 0], dtype=np.uint8)
    assert (decode_tiff_gray_to_levels(gray2, bpp=2) == [0, 1, 2, 3]).all()


def test_layer_to_slice_index_nearest_mid_layer():
    # analysis slices at z = 0.5, 1.5, 2.5 mm (layer_mm = 1.0);
    # print layers 0.1 mm: layer k mid-z = (k + 0.5) * 0.1
    z_slices = np.array([0.5, 1.5, 2.5])
    assert layer_to_slice_index(0, 0.1, z_slices) == 0      # z = 0.05
    assert layer_to_slice_index(9, 0.1, z_slices) == 0      # z = 0.95
    assert layer_to_slice_index(10, 0.1, z_slices) == 1     # z = 1.05
    assert layer_to_slice_index(24, 0.1, z_slices) == 2     # z = 2.45
    assert layer_to_slice_index(500, 0.1, z_slices) == 2    # clamped


def test_apply_sat_to_levels_scales_and_preserves_zeros():
    levels = np.array([[0, 15], [15, 8]], dtype=np.uint8)   # 4 bpp
    sat = np.array([[1.0, 1.0], [0.5, 0.0]])
    out = apply_sat_to_levels(levels, sat, bpp=4)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0          # no ink stays no ink
    assert out[0, 1] == 15         # sat 1.0 unchanged
    # dithered quantization (spec 7b amendment 2026-08-03): a
    # half-level dose is position-dependent {7, 8}; the legacy
    # rounding path is dither=None and still rounds to 8
    assert out[1, 0] in (7, 8)
    assert apply_sat_to_levels(levels, sat, bpp=4,
                               dither=None)[1, 0] == 8
    assert out[1, 1] == 0          # sat 0 removes ink
    # never exceeds the bpp ceiling even for sat > 1
    out2 = apply_sat_to_levels(levels, np.full((2, 2), 1.4), bpp=4)
    assert out2.max() == 15


def test_sample_sat_on_canvas_center_alignment():
    # dopant grid: 120x120 over 60 mm, part = central disk sat 0.8, bg 1.0
    x = np.linspace(-0.03, 0.03, 120)
    yy, xx = np.meshgrid(x, x, indexing="ij")
    sat = np.where(np.hypot(xx, yy) < 0.010, 0.8, 1.0)
    # print canvas: 24 x 24 mm at 10 px/mm, part center at canvas center
    out = sample_sat_on_canvas(
        sat, canvas_w_px=240, canvas_h_px=240, ppm=10.0,
        part_center_canvas_mm=(12.0, 12.0))
    # canvas center pixel maps to dopant origin (part center)
    assert out[120, 120] == pytest.approx(0.8, abs=0.02)
    # 12 mm right of center is outside the 10 mm part radius -> background
    assert out[120, 239] == pytest.approx(1.0, abs=0.02)
    # y-flip: canvas row 0 is TOP (y = +12 mm from part center)
    # make sat asymmetric in y to verify orientation
    sat_asym = np.where(yy > 0.005, 0.2, 1.0)   # low sat in dopant +y half
    out2 = sample_sat_on_canvas(
        sat_asym, canvas_w_px=240, canvas_h_px=240, ppm=10.0,
        part_center_canvas_mm=(12.0, 12.0))
    assert out2[20, 120] == pytest.approx(0.2, abs=0.05)    # top rows = +y
    assert out2[220, 120] == pytest.approx(1.0, abs=0.05)   # bottom rows = -y
