"""S2 Task 1: harness conventions."""
from __future__ import annotations

import numpy as np
import pytest

import heatr3d
from heatr3d_s2 import harness


def test_nominal_masks_are_analytic_and_match_heatr3d_geometry():
    """The nominal must be the shape heatr3d thinks it is meshing. Checked by
    area against a fine voxelization of heatr3d's OWN make_geometry."""
    for shape, kw in (("circle", {}), ("square", {}), ("lshape", {})):
        nom = harness.nominal_mask_2d(shape)
        _, _, _, h = __import__("solve3d.gates", fromlist=["x"]).eval_grid_axes()
        area_nom = float(nom.sum()) * h * h
        g = heatr3d.Grid(n=192)
        part = harness.make_part(g, shape)
        k = g.n // 2
        area_vox = float(part[:, :, k].sum()) * g.h * g.h
        assert abs(area_nom - area_vox) / area_vox < 0.02, shape


def test_dual_read_extraction_matches_heatr3d_native_conventions():
    """The melt-onset read must be heatr3d's OWN t_phi90/T_phi90, not a
    re-derivation, and the fixed-time read must genuinely precede it."""
    n, shape, t_ref = 24, "circle", 200.0
    p = heatr3d.Params(phase_update="enthalpy")
    rec = harness.run_case(shape, n, t_ref, p=p, save_fields=False)
    grid = heatr3d.Grid(n=n)
    part = harness.make_part(grid, shape)
    gamma = heatr3d.build_gamma(part, p)
    V = heatr3d.solve_eqs_3d(gamma, grid, p)
    Q = heatr3d.compute_qrf_3d(V, gamma, grid, p, doped=part, premix=False,
                               qrf_gradient="masked")
    ref = heatr3d.run(grid, part, p, qrf_override=Q, max_time_s=1500.0,
                      phi_target=0.90)
    mo = rec["reads"]["melt_onset"]
    assert mo["t90_s"] == ref.t_phi90_s
    assert mo["sigma_T_c"] == pytest.approx(ref.sigma_T, rel=1e-12)
    assert mo["reached"] is True
    assert rec["reads"]["heating_fixed_time"]["precedes_melt_onset"] is True


def test_standing_gates_are_recorded_for_both_reads():
    rec = harness.run_case("circle", 24, 150.0, save_fields=False)
    for read in ("melt_onset", "heating_fixed_time"):
        g = rec["gates"][read]
        assert abs(g["energy_residual_frac"]) < 1e-2, read
        assert g["clamp_bound"] is False and g["cfl_violated"] is False


def test_metrics_are_z_invariant_for_an_extrusion():
    """These are full-height extrusions; a large plane spread would mean the
    planar reduction the metrics rely on is invalid."""
    rec = harness.run_case("square", 24, 150.0, save_fields=False)
    m = rec["reads"]["melt_onset"]
    assert m["iou_phi0p9__plane_spread"] < 0.02
