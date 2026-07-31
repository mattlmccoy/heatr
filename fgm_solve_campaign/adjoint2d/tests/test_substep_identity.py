"""The prototype substep must reproduce `rfam_eqs_coupled.thermal_step` exactly.

This is the inner half of the L0 gate: if the substep is not bit-identical the
whole-march identity cannot be, and bisecting a whole march is expensive.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import forward as fwd
from adjoint2d.pins import build_case, load_cfg
from adjoint2d.prod import rfam

CFG = ("/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/"
       "code/geo-prewarp/outputs_eqs/fgm_calibrated_control/configs/square_m0p8775.yaml")


def _therm_params(case):
    p = case.pins
    dens = case.cfg["densification"]
    ph = case.cfg["thermal"]["phase_change"]
    phase_user = rfam.PhaseConfig(
        model="comsol_heaviside", solidus_c=0.0, liquidus_c=0.0,
        t_pc_c=p.t_pc_c, dt_pc_c=p.dt_pc_c, smooth_shape="linear", tanh_beta=8.0,
    )
    return rfam.ThermalParams(
        dx=case.dx, dy=case.dy, dt=p.dt_sub, max_dt_step_c=p.max_dt_step_c,
        temp_min_c=p.temp_min_c, temp_max_c=p.temp_max_c, phase_cfg=phase_user,
        latent_heat=p.latent_heat, ambient_c=p.ambient_c, convection_model="constant",
        h_const=p.h_const, conv_plate_distance_m=0.01, conv_chimney_height_m=0.01,
        conv_pressure_pa=1.0133e5, conv_external_temp_k=p.ambient_c + 273.15,
        rho_powder=p.rho_powder, k_powder=p.k_powder, cp_powder=p.cp_powder,
        rho_solid=p.rho_solid, rho_liquid=p.rho_liquid, k_solid=p.k_solid,
        k_liquid=p.k_liquid, cp_solid=p.cp_solid, cp_liquid=p.cp_liquid,
        rho_rel_initial=p.rho_rel_init,
        dens_k0=float(dens["k0_per_s"]), dens_ea=float(dens["activation_energy_j_per_mol"]),
        dens_model="physics_dual",
        dens_phi_exponent=float(dens.get("phi_exponent", 1.0)),
        dens_rho_exponent=p.dens_rho_exponent,
        dens_max_delta_per_step=p.dens_max_delta_per_substep,
        dens_surface_tension_n_per_m=p.dens_surface_tension,
        dens_particle_radius_m=p.dens_particle_radius_m,
        dens_eta_ref_pa_s=p.dens_eta_ref_pa_s, dens_eta_ref_temp_k=p.dens_eta_ref_temp_k,
        dens_eta_activation_j_per_mol=p.dens_eta_activation_j_per_mol,
        dens_geom_factor=p.dens_geom_factor,
        dens_k0_ss=p.dens_k0_ss, dens_ea_ss=p.dens_ea_ss,
        dens_k0_liq=float(dens.get("k0_liq_per_s", 0.08)),
        dens_ea_liq=0.0,
        dens_phi_threshold=p.dens_phi_threshold,
        dens_phi_solid_exponent=p.dens_phi_solid_exponent,
        dens_phi_liq_exponent=p.dens_phi_liq_exponent,
        dens_liquid_rate_mode="viscous_capillary",
        dA=case.dA,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_substep_bit_identical(seed):
    case = build_case(load_cfg(CFG))
    p = case.pins
    rng = np.random.default_rng(seed)
    ny, nx = case.part_mask.shape
    # A state that spans the melt window so the phase ramp and both
    # densification mechanisms are exercised.
    T = 23.0 + 200.0 * rng.random((ny, nx))
    rho = np.zeros((ny, nx))
    rho[case.part_mask] = 0.55 + 0.4 * rng.random(int(case.part_mask.sum()))
    Q = np.zeros((ny, nx))
    Q[case.doped_mask] = 1e7 * rng.random(int(case.doped_mask.sum()))

    tp = _therm_params(case)
    T_ref, rho_ref, phi_ref, _xc, _diag = rfam.thermal_step(
        T.copy(), rho.copy(), Q, case.part_mask, case.expo, tp
    )
    T_new, rho_new, phi_new, _c = fwd.substep(T.copy(), rho.copy(), Q, case)
    assert np.max(np.abs(T_new - T_ref)) == 0.0
    assert np.max(np.abs(phi_new - phi_ref)) == 0.0
    assert np.max(np.abs(rho_new - rho_ref)) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:warnings"]))
