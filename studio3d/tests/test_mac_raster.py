"""Mac-native 3D printable-raster emitter (studio3d.mac_raster): the pure
transforms that turn a solved dopant map into per-layer 720-dpi dithered printer
level maps WITHOUT the Windows Meteor slicer. Reuses the trusted 2D
printer_level_map dither and the studio3d dg0_to_voxel resampler.

Pure logic only here (voxel->stack convention, layer->levels, WhiteIsZero); the
IO orchestrator emit_printable_package is covered by an end-to-end run.
"""
from __future__ import annotations

import numpy as np
import pytest

from studio3d import mac_raster as mr


def test_voxel_to_layer_stack_convention():
    """(nx,ny,nz) voxel sat+part -> (nz,ny,nx) Meteor stack over occupied z-layers;
    sat=1 outside the part (undoped baseline); z_mm cell-centered."""
    nx = ny = nz = 6
    part = np.zeros((nx, ny, nz), bool)
    part[1:4, 1:5, 2:5] = True                   # asymmetric in x,y to catch transpose
    sat = np.zeros((nx, ny, nz), float)
    sat[part] = 0.7
    h = 1.0e-3
    sat_s, mask_s, z_mm = mr.voxel_to_layer_stack(sat, part, h)
    assert mask_s.shape == (3, ny, nx)           # 3 occupied z-layers, (nz,ny,nx)
    assert sat_s.shape == (3, ny, nx)
    assert np.array_equal(mask_s[0], part[:, :, 2].T)     # the (2,1,0) transpose
    assert np.all(sat_s[~mask_s] == 1.0)         # undoped baseline outside part
    assert np.allclose(sat_s[mask_s], 0.7)
    assert np.allclose(z_mm, (np.arange(3) + 0.5) * h * 1e3)


def test_layer_to_levels_range_and_outside_zero():
    """One sat layer -> uint8 level map in [0, mv]; 0 ink far outside the part."""
    h = 2.121e-4                                 # ~6x upsample to 720 dpi
    mask = np.zeros((10, 10), bool); mask[2:8, 2:8] = True
    sat = np.full((10, 10), 0.5)
    lv = mr.layer_to_levels(sat, mask, h, dpi=720, bpp=4, grey_levels=8)
    assert lv.dtype == np.uint8
    assert lv.max() <= 7 and lv.min() >= 0       # mv = min(15,7) = 7
    assert lv[0, 0] == 0                          # far outside footprint -> no ink


def test_layer_to_levels_dithers_partial_saturation():
    """Half-dose over the whole layer -> ordered dither yields MULTIPLE levels
    (a smooth grade out of 8 doses), not one flat level."""
    h = 2.121e-4
    mask = np.ones((20, 20), bool)
    sat = np.full((20, 20), 0.5)
    lv = mr.layer_to_levels(sat, mask, h)
    assert len(np.unique(lv)) >= 2


def test_levels_to_whiteiszero_black_is_max_ink():
    """Meteor WhiteIsZero: level 0 -> 255 (white/no ink), level mv -> 0 (black/max
    ink), mid -> gray, monotonic decreasing in level."""
    mv = 7
    lv = np.array([[0, mv], [3, mv]], np.uint8)
    g8 = mr.levels_to_whiteiszero(lv, bpp=4, grey_levels=8)
    assert g8.dtype == np.uint8
    assert g8[0, 0] == 255
    assert g8[0, 1] == 0
    assert 0 < g8[1, 0] < 255
