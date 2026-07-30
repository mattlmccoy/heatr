"""S1 numerical-integrity tests for heatr3d (spec: docs/superpowers/specs/
2026-07-30-heatr3d-graduation-design.md, Gate S1)."""
import numpy as np
import pytest

from heatr3d import Grid, Params, make_geometry, run


def small_sphere_case(n=32):
    grid = Grid(n=n, L=0.060)
    part = make_geometry(grid, "sphere", diam=0.020)
    p = Params()
    return grid, part, p


def test_energy_audit_fields_present_and_small_on_benign_run():
    grid, part, p = small_sphere_case()
    res = run(grid, part, p, max_time_s=30.0, verbose=False)
    # new provenance fields
    assert hasattr(res, "energy_in_j")
    assert hasattr(res, "energy_stored_j")
    assert hasattr(res, "energy_loss_j")
    assert hasattr(res, "energy_residual_frac")
    assert res.energy_in_j > 0
    # benign pre-melt run must conserve energy to a few percent
    assert abs(res.energy_residual_frac) < 0.05
