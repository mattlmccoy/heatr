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


def test_centered_axis_mm_matches_2d_convention():
    """Placement georeferencing: n cell-centered chamber columns -> centered mm
    coords (symmetric about 0), spacing h, mirroring the 2D x_mm/y_mm contract
    (scripts/solve_fgm.py:409, x_mm = centered domain coords * 1000)."""
    n, h = 6, 1.0e-3
    ax = mr.centered_axis_mm(n, h)
    assert ax.shape == (n,)
    assert np.isclose(ax.mean(), 0.0)                      # centered about the origin
    assert np.allclose(np.diff(ax), h * 1e3)               # spacing = h in mm
    assert np.isclose(ax[-1], -ax[0])                      # symmetric
    # cell-centered chamber: half-cell inset from the edges
    assert np.isclose(ax[0], -(n / 2 - 0.5) * h * 1e3)


def test_emit_level_map_npz_is_fgm_to_rip_compatible(tmp_path):
    """The fold: emit_printable_package writes a level_map npz in the exact format
    software/meteor/tools/fgm_to_rip.fgm_to_tiff_stack consumes -- level_map
    (nz,ny,nx) uint8 in [0, head_ceiling=7] at printer resolution, plus bpp/dpi/
    width_mm/height_mm -- instead of writing its own (wrong 8-bit) TIFFs. The real
    MetPrint TIFFs are written by meteor_rip (4bpp/LZW/WhiteIsZero), not here."""
    import numpy as np
    map_npz, part_npz = _tiny_map_and_part(tmp_path)
    man = mr.emit_printable_package(str(map_npz), str(part_npz), str(tmp_path / "pkg"),
                                    dpi=720, bpp=4, grey_levels=8)
    d = np.load(tmp_path / "pkg" / "fgm_level_map.npz")
    lm = d["level_map"]
    assert lm.ndim == 3                              # (nz,ny,nx) -> one TIFF page per z
    assert lm.dtype == np.uint8
    assert int(lm.max()) <= 7                        # head ceiling, NOT container 15
    assert int(d["bpp"]) == 4 and int(d["dpi"]) == 720
    assert "width_mm" in d.files and "height_mm" in d.files
    assert man["level_map_npz"].endswith("fgm_level_map.npz")


def _tiny_map_and_part(tmp_path):
    """A tiny solved-map npz + matching voxel part for the emit integration test."""
    import numpy as np
    n = 8
    part = np.zeros((n, n, n), bool); part[2:6, 2:6, 1:7] = True
    h = 2.121e-4
    pnp = tmp_path / "part.npz"; np.savez(pnp, part=part, h=h, n=n)
    # a FEM-ish map: cell centroids at the in-part voxels, graded s_map
    idx = np.argwhere(part)
    cen = (idx + 0.5) * h - 0.5 * n * h
    s = np.linspace(0.1, 0.95, len(idx)).astype(float)
    vol = np.full(len(idx), h ** 3)
    mnp = tmp_path / "map.npz"; np.savez(mnp, centroids=cen, s_map=s, volumes=vol)
    return mnp, pnp
