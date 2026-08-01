"""Equal-volume normalization: uniform scale to V*."""
import pytest
import trimesh

from shape_library_3d.constants import V_STAR_MM3
from shape_library_3d.normalize import scale_to_volume


def test_unit_cube_scales_to_Vstar():
    m = trimesh.creation.box(extents=[1.0, 1.0, 1.0])  # volume 1 mm^3
    scaled, s = scale_to_volume(m, V_STAR_MM3)
    assert scaled.volume == pytest.approx(V_STAR_MM3, rel=1e-9)
    assert s == pytest.approx(V_STAR_MM3 ** (1.0 / 3.0), rel=1e-9)


def test_nonunit_mesh_hits_Vstar_regardless_of_start():
    m = trimesh.creation.icosphere(subdivisions=3, radius=7.3)
    scaled, _ = scale_to_volume(m, V_STAR_MM3)
    assert scaled.volume == pytest.approx(V_STAR_MM3, rel=1e-6)


def test_raises_on_nonpositive_volume():
    m = trimesh.creation.box(extents=[1, 1, 1])
    m.invert()  # flips winding -> negative signed volume
    with pytest.raises(ValueError):
        scale_to_volume(m, V_STAR_MM3)
