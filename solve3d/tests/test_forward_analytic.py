"""solve3d Phase A analytic benchmarks (S1 ports) + environment smoke test.

RUNS IN THE SPIKE ENV (dolfinx 0.11 complex build):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_forward_analytic.py
"""
from __future__ import annotations

import numpy as np


def test_env_imports():
    """Task 0: solve3d.forward imports dolfinx (through jit_fix) and exposes the
    Phase A public API."""
    from solve3d import forward

    assert forward.DOLFINX_VERSION.startswith("0.")
    assert forward.IS_COMPLEX is True, "Phase A requires the complex scalar build"
    assert hasattr(forward, "ForwardParams")
    assert hasattr(forward, "run_forward")
    p = forward.ForwardParams()
    # mirrors heatr3d.Params defaults that Phase A depends on
    assert p.phase_update == "enthalpy"
    assert p.eqs_update_interval_s == 0.0
    assert p.sigma_temp_coeff_per_K == 0.0
    assert p.freq_hz == 27.12e6
    assert p.dt_s == 0.05


# =========================================================================== #
# Task 3: enthalpy thermal-phase march, analytic benchmarks
# Ported from test_heatr3d_s1.py (the S1 standing benchmarks):
#   (a) test_adiabatic_uniform_heating_matches_analytic_plateau
#   (b) test_conduction_decay_matches_fourier_mode
#   (c) the standing energy gate (Result.energy_residual_frac)
# =========================================================================== #
import dataclasses  # noqa: E402

import pytest  # noqa: E402


def _uniform_box(n: int, L: float = 0.060):
    from solve3d import forward
    return forward.box_mesh(n, L=L)


def test_latent_plateau_matches_the_analytic_enthalpy_budget():
    """(a) Whole domain = part, uniform source, convection OFF: T(t) is a pure
    source integration of H, including the latent plateau, and it is EXACT for
    the enthalpy scheme by construction. This pins the wiring -- property maps,
    lumped volumes, dt bookkeeping and the H<->T inversion.

    Sizing copied from the heatr3d S1 test: q = 2.0e6 W/m^3 for 100 s deposits
    2.0e8 J/m^3 and lands MID-PLATEAU, so the latent barrier is actually
    exercised. Constant-property arm (cp_liquid = cp_solid, k_liquid = k_solid,
    rho_liquid = rho_s) -- exactly the assumptions the closed form makes.
    """
    from solve3d import forward

    p0 = forward.ForwardParams(conv_h=0.0)
    rho_s = p0.rho_powder + p0.rho_rel * (p0.rho_solid - p0.rho_powder)
    p = dataclasses.replace(p0, cp_liquid=p0.cp_solid, k_liquid=p0.k_solid,
                            rho_liquid=rho_s)
    rho_cp = rho_s * p.cp_solid
    rho_L = rho_s * p.latent_j_per_kg
    q_val, t_end = 2.0e6, 100.0
    H_end = forward.enthalpy_from_T(np.array([p.preheat_c]), rho_cp, rho_L, p) \
        + q_val * t_end
    T_exact = float(forward.T_from_enthalpy(H_end, rho_cp, rho_L, p)[0])
    assert p.t_pc_c - p.dt_pc_c / 2 < T_exact < p.t_pc_c + p.dt_pc_c / 2

    msh = _uniform_box(8)
    out = forward.march_enthalpy(msh, p, in_part=lambda mp: np.ones(mp.shape[1], bool),
                                 q_uniform=q_val, max_time_s=t_end,
                                 phi_target=2.0)
    assert out["clamp_bound"] is False
    assert abs(out["part_mean_T_c"] - T_exact) < 0.05
    assert out["part_std_T_c"] < 1e-6
    assert abs(out["energy_residual_frac"]) < 1e-9


def test_conduction_decay_matches_fourier_mode():
    """(b) No source, no convection, uniform powder: the lowest cosine mode
    compatible with zero-flux walls decays as exp(-alpha k^2 t).

    Gate (plan Task 3): the observed decay RATE matches the analytic eigenvalue
    within 2 %. The rate is read as -log(amp/amp0)/t_end so it is independent of
    the cell-centred/nodal sampling factor that the heatr3d S1 test had to
    correct for."""
    from solve3d import forward

    n, L, t_end = 24, 0.060, 400.0
    p = forward.ForwardParams(conv_h=0.0)
    msh = _uniform_box(n, L=L)
    kx = np.pi / L
    out = forward.march_enthalpy(
        msh, p, in_part=None, q_uniform=0.0, max_time_s=t_end, phi_target=2.0,
        T0_fn=lambda x: p.preheat_c + 5.0 * np.cos(kx * (x[0] + L / 2.0)))

    alpha = p.k_powder / (p.rho_powder * p.cp_powder)
    lam_exact = alpha * kx ** 2
    amp0 = 0.5 * (out["T0_max_c"] - out["T0_min_c"])
    amp = 0.5 * (out["T_max_c"] - out["T_min_c"])
    lam_num = -np.log(amp / amp0) / t_end
    assert abs(lam_num - lam_exact) / lam_exact < 0.02, (lam_num, lam_exact)
    # zero-flux walls, no source, no sink: the volume-weighted mean is conserved
    assert abs(out["domain_mean_T_c"] - out["T0_domain_mean_c"]) < 1e-9


def test_energy_audit_on_a_heat_and_melt_case():
    """(c) The standing S1 conservation gate, ported: a run that heats a part
    through the melt window with convection ON must book |residual_frac| < 1e-2.

    The audit is NOT vacuous: stored energy is banked heatr3d-style (sensible
    with THIS step's blended rho*cp, plus latent from the phi change), not read
    back off the enthalpy variable the march advances."""
    from solve3d import forward

    p = forward.ForwardParams()
    msh = _uniform_box(12)
    half = 0.010
    out = forward.march_enthalpy(
        msh, p, in_part=lambda mp: (np.abs(mp[0]) <= half) & (np.abs(mp[1]) <= half),
        q_uniform=2.0e6, max_time_s=600.0, phi_target=0.90)
    assert out["reached"] is True, "the melt window must actually be crossed"
    assert out["part_mean_phi"] >= 0.90
    assert abs(out["energy_residual_frac"]) < 1e-2, out["energy_residual_frac"]
