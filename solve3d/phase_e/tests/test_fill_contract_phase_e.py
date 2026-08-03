"""Phase E: the target indicator must satisfy the shared cross-lane fill
contract ON THESE SHAPES, before any solve (registration: conventions.chi).

The contract's own entry point assumes an EXTRUSION. The cube is one (a square
extruded over |z| <= a/2) and runs it directly. The pyramid is NOT -- its
cross-section tapers -- so the contract is applied to its LOCAL cross-section at
each height, which is the strongest form of the statement that still means
something for a tapered solid. The adaptation is recorded rather than skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fgm_solve_campaign" / "adjoint2d" / "tests"))
import fill_contract as fc            # noqa: E402  THEIR file, unmodified

from solve3d import fill              # noqa: E402
from solve3d.phase_e import geometry  # noqa: E402


def _axes(n: int = 120, L: float = 0.060):
    h = L / n
    c = (np.arange(n) + 0.5) * h - L / 2.0
    return c, c


def _square_poly(half: float) -> np.ndarray:
    return np.array([[-half, -half], [half, -half], [half, half], [-half, half]])


def test_cube_passes_the_contract_extrusion_reduction_directly():
    """The cube IS an extrusion, so the contract applies unmodified."""
    x, y = _axes()
    poly = _square_poly(geometry.CUBE_A_M / 2.0)
    fc.assert_extrusion_slice_reduction(fill.volume_fill_slice, fill.area_fill_2d,
                                        x, y, poly=poly)
    fc.assert_winding_invariance(fill.area_fill_2d, x, y, poly=poly)


def test_cube_fill_area_matches_the_closed_form():
    x, y = _axes()
    a = geometry.CUBE_A_M
    f = fill.area_fill_2d(_square_poly(a / 2.0), x, y)
    h = float(x[1] - x[0])
    assert abs(float(f.sum()) * h * h - a * a) / (a * a) < fc.AREA_REL_TOL


@pytest.mark.parametrize("z_frac", [-0.4, 0.0, 0.4])
def test_pyramid_local_cross_section_passes_the_contract(z_frac):
    """The pyramid tapers, so the contract is applied to its cross-section at
    each height -- there the solid IS locally an extrusion.

    The grid is RESOLUTION-MATCHED to the cross-section (~40 cells across)
    rather than fixed. On a fixed 120-cell grid the z = +0.4h section is only
    ~2.3 cells wide and misses the contract's 0.2 percent area tolerance, which
    is calibrated for a 20 mm shape. That is not a fill error -- it is the apex
    being under-resolved -- and it is measured separately in
    test_apex_cross_section_is_under_resolved_on_a_fixed_grid rather than
    hidden by widening a tolerance that belongs to the other lane.
    """
    h = geometry.PYR_H_M
    z = z_frac * h
    half = (geometry.PYR_B_M / 2.0) * (h / 2.0 - z) / h
    L = 0.060
    n = int(np.clip(round(40.0 * L / (2 * half)), 60, 900))
    x, y = _axes(n=n, L=L)
    poly = _square_poly(half)
    fc.assert_extrusion_slice_reduction(fill.volume_fill_slice, fill.area_fill_2d,
                                        x, y, poly=poly)
    fc.assert_winding_invariance(fill.area_fill_2d, x, y, poly=poly)
    ff = fill.area_fill_2d(poly, x, y)
    hx = float(x[1] - x[0])
    want = (2 * half) ** 2
    assert abs(float(ff.sum()) * hx * hx - want) / want < fc.AREA_REL_TOL


def test_apex_cross_section_is_under_resolved_on_a_fixed_grid():
    """MEASURED and reported as a Phase E finding rather than worked around: on
    the shared evaluation grid the pyramid's cross-section near the apex spans
    only a couple of cells, so any metric read there is resolution limited.
    This is the concrete form of the registration's warning that ABSOLUTE
    fidelity numbers on this shape are ungated."""
    from solve3d import gates as sg
    _, _, _, h_eval = sg.eval_grid_axes()
    h = geometry.PYR_H_M
    cells = {}
    for zf in (-0.4, 0.0, 0.4, 0.49):
        half = (geometry.PYR_B_M / 2.0) * (h / 2.0 - zf * h) / h
        cells[zf] = 2 * half / h_eval
    # MEASURED on the shared 0.15 mm evaluation grid: the pyramid is well
    # resolved over most of its height (139.5 cells across at z/h = -0.4, 77.5
    # at mid-height, 15.5 at +0.4) and becomes marginal only in the last few
    # percent (1.55 cells at z/h = +0.49). So apex-limited resolution is a real
    # but LOCALISED effect, not a whole-shape one -- which is why the campaign
    # scores solved-vs-uniform at matched read rather than absolute fidelity.
    assert cells[-0.4] > 100, cells
    assert cells[0.4] > 10, cells
    assert cells[0.49] < 3, cells


@pytest.mark.parametrize("shape", ["pyramid", "cube"])
def test_chi_on_the_solve_mesh_is_exactly_the_conforming_indicator(shape):
    """The practical statement for Phase E: on a conforming mesh no cell is
    partial, so chi is exact and the sub-cell machinery is provably inert here.
    Measured, not assumed -- and it is what lets the campaign use the doped
    indicator directly."""
    from solve3d import forward as fwd
    msh, info = geometry.build_mesh(shape, lc_part=0.0020)
    p = fwd.ForwardParams()
    mats = fwd.build_materials(msh, geometry.in_part_predicate(shape), p)
    d = np.real(mats.doped.x.array)
    assert np.all((d == 0.0) | (d == 1.0)), shape
    import femutils as fu
    vol = np.real(fu.cell_volumes(msh, mats.dg0)).astype(float)
    v_chi = float(np.dot(d.astype(float), vol))
    want = geometry.occ_volume_m3(shape)
    assert abs(v_chi - want) / want < 0.02, (shape, v_chi, want)
