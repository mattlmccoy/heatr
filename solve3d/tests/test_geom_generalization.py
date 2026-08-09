"""Geometry generalization of the B1-B4 ceiling-coupled dopant solve.

The square loop is closed cross-engine; these tests pin the ADDITIVE geometry
parameter that lets the SAME machinery (density co-state, AL, drive probe) run on
the Phase E conforming cube and pyramid meshes -- without forking the adjoint and
without changing the square path.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest \
        solve3d/tests/test_geom_generalization.py -x -q

The two heavy tests build real gmsh+dolfinx conforming meshes (spike env only)
and are the contract that `build_coarse_case(shape=...)` meshes the RIGHT solid.
"""
import inspect

import numpy as np
import pytest

from solve3d import density_adjoint as da


# --------------------------------------------------------------------------- #
# Pure-logic: the geometry-family dispatch + the byte-identical square default
# --------------------------------------------------------------------------- #
def test_geometry_family_classifies_shapes():
    # square/circle stay on the Phase A anchor path; cube/pyramid route to the
    # Phase E conforming mesh. An unknown shape is a hard error (never a silent
    # fallthrough to the square mesh).
    assert da.geometry_family("square") == "anchor"
    assert da.geometry_family("circle") == "anchor"
    assert da.geometry_family("cube") == "phase_e"
    assert da.geometry_family("pyramid") == "phase_e"
    with pytest.raises(ValueError):
        da.geometry_family("dodecahedron")


def test_build_coarse_case_default_shape_is_square():
    # The geometry parameter is ADDITIVE: its default is "square", so every
    # existing (B1/B2/B3/B4) call site is byte-identical.
    sig = inspect.signature(da.build_coarse_case)
    assert "shape" in sig.parameters
    assert sig.parameters["shape"].default == "square"


# --------------------------------------------------------------------------- #
# Heavy: the cube/pyramid coarse cases actually mesh the right solid
# --------------------------------------------------------------------------- #
def _meshed_part_volume_m3(case) -> float:
    tc = case.tc
    return float(tc.eqs.vol[tc.eqs.part].sum())


@pytest.mark.parametrize("shape", ["cube", "pyramid"])
def test_build_coarse_case_meshes_the_phase_e_solid(shape):
    from solve3d.phase_e import geometry as geo
    case = da.build_coarse_case(shape=shape)
    # a real part with design dofs
    assert case.chain.n_design > 0
    assert case.tc.eqs.part.size == case.chain.n_design
    # the meshed part volume matches the analytic nominal solid (coarse mesh, so
    # a few percent; this is the proof it meshed the CUBE/PYRAMID, not a square)
    meshed = _meshed_part_volume_m3(case)
    nominal = geo.nominal_volume_m3(shape)
    rel = abs(meshed - nominal) / nominal
    assert rel < 0.05, f"{shape} meshed vol {meshed:.3e} vs nominal {nominal:.3e} rel={rel:.3f}"


def test_square_default_and_explicit_agree():
    # Building with the default and with shape="square" must give the SAME coarse
    # square case (same design dof count) -- the additive parameter cannot perturb
    # the anchor path.
    a = da.build_coarse_case()
    b = da.build_coarse_case(shape="square")
    assert a.chain.n_design == b.chain.n_design
