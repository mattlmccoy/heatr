"""Every Tier-1/2 generator yields a watertight solid that normalizes to V*;
genus is asserted where analytic and recorded where emergent (lattice)."""
import numpy as np
import pytest

from shape_library_3d import generators as G
from shape_library_3d.constants import V_STAR_MM3
from shape_library_3d.normalize import scale_to_volume

VALID = [
    "cube", "cylinder", "sphere", "toroid", "cone", "pyramid", "pipe",
    "lattice", "l_extrusion", "trunc_octahedron", "icosphere_coarse", "uv_sphere",
]
EXPECT_GENUS = {"toroid": 1, "pipe": 1}  # solids are genus 0; lattice asserted >=1 separately


def _genus(mesh):
    return (2 - int(mesh.euler_number)) // 2


@pytest.mark.parametrize("name", VALID)
def test_generator_watertight_winding_and_Vstar(name):
    m = getattr(G, f"make_{name}")()
    assert m.is_watertight, f"{name} not watertight"
    assert m.is_winding_consistent, f"{name} winding inconsistent"
    scale_to_volume(m, V_STAR_MM3)
    assert m.volume == pytest.approx(V_STAR_MM3, rel=1e-4), f"{name} volume off"


@pytest.mark.parametrize("name,g", list(EXPECT_GENUS.items()))
def test_expected_genus(name, g):
    m = getattr(G, f"make_{name}")()
    assert _genus(m) == g, f"{name} genus {_genus(m)} != {g}"


def test_solids_are_genus_zero():
    for name in ["cube", "cylinder", "sphere", "cone", "pyramid",
                 "l_extrusion", "trunc_octahedron", "icosphere_coarse", "uv_sphere"]:
        assert _genus(getattr(G, f"make_{name}")()) == 0, f"{name} not genus 0"


def test_lattice_is_multiply_connected():
    assert _genus(G.make_lattice()) >= 1


def test_sphere_family_same_nominal_radius_after_Vstar():
    for nm in ("sphere", "icosphere_coarse", "uv_sphere"):
        m = getattr(G, f"make_{nm}")()
        scale_to_volume(m, V_STAR_MM3)
        r = np.linalg.norm(m.vertices - m.bounds.mean(axis=0), axis=1).mean()
        assert r == pytest.approx(10.0, rel=0.02), f"{nm} mean radius {r} != ~10"


def test_uv_sphere_face_count_matches_icosphere_coarse():
    target = len(G.make_icosphere_coarse().faces)
    got = len(G.make_uv_sphere().faces)
    assert abs(got - target) <= 0.10 * target, f"uv {got} not within 10% of {target}"


def test_tier3_generators_are_invalid_by_construction():
    from shape_library_3d.validate import validate_part_mesh, InvalidPartGeometryError
    for nm in ("open_cylinder", "flat_plane"):
        with pytest.raises(InvalidPartGeometryError):
            validate_part_mesh(getattr(G, f"make_{nm}")())
