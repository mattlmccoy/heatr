"""Tier-3 rejection contract: non-parts must be rejected loudly with typed errors."""
import numpy as np
import pytest
import trimesh

from shape_library_3d.validate import (
    InvalidPartGeometryError,
    NonWatertightMeshError,
    ZeroVolumeError,
    validate_part_mesh,
)


def _open_cylinder():
    """Side wall only, no end caps -> naked edges, but a real 3-D bounding box."""
    n, r, h = 48, 10.0, 20.0
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bot = np.c_[r * np.cos(th), r * np.sin(th), np.full(n, -h / 2)]
    top = np.c_[r * np.cos(th), r * np.sin(th), np.full(n, h / 2)]
    v = np.vstack([bot, top])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + i], [j, n + j, n + i]]
    return trimesh.Trimesh(vertices=v, faces=np.array(f))


def _flat_plane(k=5, w=20.0):
    """z=0 triangulated sheet -> zero z-extent, zero enclosed volume."""
    xs = np.linspace(-w / 2, w / 2, k)
    ys = np.linspace(-w / 2, w / 2, k)
    v = np.array([[x, y, 0.0] for y in ys for x in xs])
    f = []
    for r in range(k - 1):
        for c in range(k - 1):
            a = r * k + c
            f += [[a, a + 1, a + k], [a + 1, a + k + 1, a + k]]
    return trimesh.Trimesh(vertices=v, faces=np.array(f))


def test_open_cylinder_rejected_not_watertight():
    with pytest.raises(NonWatertightMeshError) as e:
        validate_part_mesh(_open_cylinder())
    assert "watertight" in str(e.value).lower()


def test_flat_plane_rejected_zero_volume():
    with pytest.raises(ZeroVolumeError) as e:
        validate_part_mesh(_flat_plane())
    assert "volume" in str(e.value).lower()


def test_error_types_subclass_base():
    assert issubclass(NonWatertightMeshError, InvalidPartGeometryError)
    assert issubclass(ZeroVolumeError, InvalidPartGeometryError)


def test_valid_solid_passes():
    validate_part_mesh(trimesh.creation.box(extents=[5.0, 5.0, 5.0]))  # no raise
    validate_part_mesh(trimesh.creation.icosphere(subdivisions=3, radius=6.0))
