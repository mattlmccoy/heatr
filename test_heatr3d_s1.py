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


def spike_case(n=32, spike_mult=400.0):
    """Uniform mild heating plus one interior hot column whose raw per-step dT
    exceeds the full melt window (dt_pc_c), forcing a window-crossing step."""
    grid, part, p = small_sphere_case(n)
    q = np.zeros((n, n, n))
    q[part] = p.power_density_w_per_m3
    ii = np.argwhere(part)
    c = ii[len(ii) // 2]
    q[c[0], c[1], c[2]] *= spike_mult
    return grid, part, p, q


def test_legacy_phase_update_skips_latent_on_window_crossing():
    """Legacy failure signature, calibrated at spike_mult=400.0 (n=32, 120 s).

    Measured (2026-07-30, ./.venv312): clamp_bound=True,
    energy_in=1287.9 J, stored=899.3 J, loss=0.023 J,
    energy_residual_frac=+0.3017. The spiked voxel (16, 11, 13) jumps
    phi 0.0000 -> 0.8000 in ONE step (T 173.00 -> 183.00 C) with
    dphi/dT = 0.0000 1/C at the step start: the pointwise apparent-cp latent
    sink is entirely absent for the window-crossing step. Sweep for context:
    mult=1 resid=+0.0000 clamp=False; 50 +0.0051 False; 100 +0.0213 True;
    200 +0.1333 True; 400 +0.3017 True; 1000 +0.5598 True.

    HONESTY NOTE (see docs/superpowers/plans/s1-findings.md): the latent skip
    itself is worth only ~0.3 J here (one voxel: rho*L*dV); the ~389 J surplus
    is dominated by the spiked voxel saturating at temp_max=600 C while RF keeps
    depositing. So the residual assertion pins "the run went non-physical", and
    the phi-jump measurement above is what pins the latent-skip mechanism.
    This test is inverted into the regression test once the enthalpy update lands.
    """
    grid, part, p, q = spike_case()
    res = run(grid, part, p, qrf_override=q, max_time_s=120.0)
    assert res.clamp_bound is True
    assert res.energy_residual_frac > 0.10
