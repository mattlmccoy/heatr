"""STL -> heatr3d Grid boolean mask bridge, with Tier-3 rejection first."""
import numpy as np
import pytest

from shape_library_3d import generators as G
from shape_library_3d.constants import V_STAR_MM3
from shape_library_3d.normalize import scale_to_volume
from shape_library_3d.validate import ZeroVolumeError
from shape_library_3d.voxelize import stl_to_mask, voxel_volume_report


class _Grid:
    """Minimal heatr3d.Grid stand-in (same centered-cell convention, metres)."""

    def __init__(self, n, L=0.060):
        self.n = n
        self.L = L
        self.h = L / n
        c = (np.arange(n) + 0.5) * self.h - L / 2.0
        self.x = self.y = self.z = c


def test_cube_mask_matches_exact_analytic_occupancy():
    """The voxel mask must equal the exact cube occupancy on the same grid
    (catches unit/scale/centering bugs); deviation from smooth (side/L)^3 is
    pure staircase and is NOT what we assert."""
    m = G.make_cube()
    scale_to_volume(m, V_STAR_MM3)              # side 16.120 mm
    grid = _Grid(64)
    mask = stl_to_mask(m, grid)
    assert mask.shape == (64, 64, 64)
    half_m = float(m.extents[0]) / 2.0 / 1000.0  # cube half-extent in metres
    xx, yy, zz = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    expected = (np.abs(xx) <= half_m) & (np.abs(yy) <= half_m) & (np.abs(zz) <= half_m)
    assert mask.sum() > 0
    assert int((mask != expected).sum()) == 0   # exact agreement


def test_voxel_volume_converges_toward_Vstar_with_resolution():
    """Staircase error must shrink as the grid refines."""
    m = G.make_cube()
    scale_to_volume(m, V_STAR_MM3)
    errs = []
    for n in (48, 96):
        grid = _Grid(n)
        rep = voxel_volume_report(stl_to_mask(m, grid), grid, V_STAR_MM3)
        errs.append(abs(rep["voxel_vs_Vstar_frac"]))
    assert errs[1] < errs[0]                     # finer grid -> smaller error


def test_voxel_volume_report_near_Vstar():
    m = G.make_sphere()
    scale_to_volume(m, V_STAR_MM3)
    grid = _Grid(48)
    mask = stl_to_mask(m, grid)
    rep = voxel_volume_report(mask, grid, V_STAR_MM3)
    assert abs(rep["voxel_vs_Vstar_frac"]) < 0.10  # within 10% at n=48


def test_tier3_mesh_rejected_before_voxelizing():
    with pytest.raises(ZeroVolumeError):
        stl_to_mask(G.make_flat_plane(), _Grid(24))
