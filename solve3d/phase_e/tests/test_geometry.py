"""Phase E: the OCC constructions must BE the library shapes.

RUNS IN THE SPIKE ENV (gmsh + dolfinx).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def _meta(shape: str) -> dict:
    return json.loads((ROOT / "shape_library_3d" / "meta" / f"{shape}.json").read_text())


@pytest.mark.parametrize("shape", ["pyramid", "cube", "cone", "sphere",
                                   "cylinder"])
def test_occ_solid_volume_matches_the_library_stl(shape):
    """The check that proves the analytic solid IS the library shape, at the
    pre-registered 1e-9 relative tolerance.

    Polyhedra (cube/pyramid) match the STL exactly because the facets ARE the
    surface. Curved primitives (cone/sphere/cylinder) are sized to the library's
    EQUAL-VOLUME invariant (4188.79 mm^3), so the analytic volume still matches
    the recorded volume at 1e-9; only the bbox differs from the faceted STL by
    the tessellation artifact (~0.07 %, sub-voxel, measured by the fidelity gate).
    """
    from solve3d.phase_e import geometry
    v = geometry.occ_volume_m3(shape)
    want = _meta(shape)["actual_volume_mm3"] * 1e-9
    assert abs(v - want) / want <= 1e-9, (shape, v, want)


@pytest.mark.parametrize("shape", ["pyramid", "cube", "cone", "sphere",
                                   "cylinder"])
def test_predicate_volume_agrees_with_the_solid(shape):
    """The in-part predicate feeds materials, chi and the nominal masks. If it
    disagrees with the meshed solid the whole campaign scores the wrong body.

    Estimated by MONTE CARLO, not by counting a midpoint voxel grid. A midpoint
    grid has a SYSTEMATIC staircase bias for an axis-aligned boundary -- the
    first version of this test used one and read the cube 2.2 % low, which is
    exactly the effect S2 measured and the reason Phase E is on the conforming
    engine at all (heatr3d_s2/S2_GATE_REPORT.md close-out C2). Monte Carlo is
    unbiased, so its error is statistical and shrinks with the sample count.
    """
    from solve3d.phase_e import geometry
    rng = np.random.default_rng(11)
    L, N = 0.060, 4_000_000
    pts = (rng.random((3, N)) - 0.5) * L
    frac = float(geometry.in_part_predicate(shape)(pts).mean())
    v = frac * L ** 3
    want = geometry.occ_volume_m3(shape)
    # 1-sigma of the estimator is ~0.4 % at this N and volume fraction
    assert abs(v - want) / want < 0.015, (shape, v, want)


def test_pyramid_cross_section_shrinks_to_a_point_at_the_apex():
    """Apex up: the cross-section must vanish at the top and be full at the
    base. A pyramid built upside down would still have the right volume."""
    from solve3d.phase_e import geometry
    h = geometry.PYR_H_M
    pred = geometry.in_part_predicate("pyramid")
    eps = 1e-4
    base = np.array([[0.0], [0.0], [-h / 2 + eps]])
    apex = np.array([[0.0], [0.0], [+h / 2 - eps]])
    assert pred(base)[0] and pred(apex)[0]           # axis is inside at both
    wide = geometry.PYR_B_M * 0.45
    assert pred(np.array([[wide], [0.0], [-h / 2 + eps]]))[0]      # wide at base
    assert not pred(np.array([[wide], [0.0], [+h / 2 - eps]]))[0]  # narrow at apex


def test_cone_cross_section_shrinks_to_a_point_at_the_apex():
    """Axis +Y, apex UP (the build/field axis is Y): full circular section at the
    base (y=-h/2), vanishing at the apex (y=+h/2). A cone built along z or apex
    down would pass the volume check but solve sideways / mirrored."""
    from solve3d.phase_e import geometry
    h = geometry.CONE_H_M
    R = geometry.CONE_R_M
    pred = geometry.in_part_predicate("cone")
    eps = 1e-4
    assert pred(np.array([[0.0], [-h / 2 + eps], [0.0]]))[0]       # axis@base in
    assert pred(np.array([[0.0], [+h / 2 - eps], [0.0]]))[0]       # axis@apex in
    assert pred(np.array([[R * 0.9], [-h / 2 + eps], [0.0]]))[0]   # wide at base
    assert not pred(np.array([[R * 0.9], [+h / 2 - eps], [0.0]]))[0]  # narrow apex
    # and NOT wide along z at the apex either (true body of revolution about Y)
    assert not pred(np.array([[0.0], [+h / 2 - eps], [R * 0.9]]))[0]


def test_cylinder_is_a_y_axis_body_of_revolution():
    from solve3d.phase_e import geometry
    h, R = geometry.CYL_H_M, geometry.CYL_R_M
    pred = geometry.in_part_predicate("cylinder")
    eps = 1e-4
    # full radius (in x-z) at both flat caps; nothing past the wall or the caps
    assert pred(np.array([[R * 0.99], [-h / 2 + eps], [0.0]]))[0]
    assert pred(np.array([[0.0], [+h / 2 - eps], [R * 0.99]]))[0]
    assert not pred(np.array([[R * 1.01], [0.0], [0.0]]))[0]        # past wall
    assert not pred(np.array([[0.0], [h / 2 + eps], [0.0]]))[0]     # past cap


def test_sphere_predicate_is_the_ball():
    from solve3d.phase_e import geometry
    R = geometry.SPH_R_M
    pred = geometry.in_part_predicate("sphere")
    assert pred(np.array([[0.0], [0.0], [0.0]]))[0]
    assert pred(np.array([[R * 0.99], [0.0], [0.0]]))[0]
    assert not pred(np.array([[R * 0.99], [R * 0.99], [0.0]]))[0]   # corner out


@pytest.mark.parametrize("shape", ["pyramid", "cube", "cone", "sphere",
                                   "cylinder"])
def test_mesh_conforms_and_carries_the_right_part_volume(shape):
    from solve3d.phase_e import geometry
    msh, info = geometry.build_mesh(shape, lc_part=0.0020)
    want = geometry.occ_volume_m3(shape)
    # curved primitives lose a little volume to the linear tets that facet the
    # surface at this lc; polyhedra are exact. 0.5 % covers the curved case.
    tol = 1e-6 if shape in ("cube", "pyramid") else 5e-3
    assert abs(info.part_volume_m3 - want) / want < tol, (shape, info.part_volume_m3)
    assert info.n_nodes_in_part > 200
