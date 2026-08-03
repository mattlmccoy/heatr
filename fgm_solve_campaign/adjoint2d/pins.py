"""Configuration pins for the prototype, read from a production YAML config.

Only the branch of the production engine that the pinned campaign configuration
actually exercises is supported. Every unsupported branch raises loudly at
construction time rather than being silently ignored, because a silently
ignored branch is exactly how a "bit-identical" claim goes wrong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .prod import rfam

# Take the constants from the production module so they can never drift.
EPS0 = rfam.EPS0
R_GAS = rfam.R_GAS


class UnsupportedConfig(RuntimeError):
    """The config uses a production branch the prototype does not replicate."""


@dataclass(frozen=True)
class Pins:
    # electric
    omega: float
    power_factor: float
    max_qrf: float
    update_interval: int
    v_hi: float
    v_lo: float
    sigma_v: float
    eps_v: float
    sigma_d0: float
    eps_d: float
    zero_qrf_outside_doped: bool
    # thermal
    n_steps: int
    dt: float
    n_substeps: int
    dt_sub: float
    max_dt_step_c: float
    temp_min_c: float
    temp_max_c: float
    ambient_c: float
    h_const: float
    latent_heat: float
    t_pc_c: float
    dt_pc_c: float
    # material
    rho_powder: float
    k_powder: float
    cp_powder: float
    rho_solid: float
    rho_liquid: float
    k_solid: float
    k_liquid: float
    cp_solid: float
    cp_liquid: float
    # densification (physics_dual)
    rho_rel_init: float
    dens_k0_ss: float
    dens_ea_ss: float
    dens_phi_solid_exponent: float
    dens_phi_threshold: float
    dens_phi_liq_exponent: float
    dens_rho_exponent: float
    dens_eta_ref_pa_s: float
    dens_eta_ref_temp_k: float
    dens_eta_activation_j_per_mol: float
    dens_geom_factor: float
    dens_surface_tension: float
    dens_particle_radius_m: float
    dens_max_delta_per_substep: float


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise UnsupportedConfig(msg)


def load_cfg(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def pins_from_cfg(cfg: dict[str, Any], dx: float, dy: float) -> Pins:
    elec = cfg["electric"]
    therm = cfg["thermal"]
    dens = cfg["densification"]
    mat = cfg["materials"]

    _require(not bool(elec.get("enforce_generator_power", False)),
             "enforce_generator_power=true adds the P_abs rank-one term; not in this layer")
    _require(not str(elec.get("qrf_file_npy", "")).strip(), "fixed Q_rf injection not supported")
    _require(float(elec.get("eqs_adaptive_rtol", 0.0) or 0.0) == 0.0, "adaptive EQS skip not supported")
    _require(str(elec.get("voltage_mode", "centered")).strip().lower() == "grounded",
             "only grounded voltage mode is pinned")
    _require("voltage_hi_v" not in elec and "voltage_lo_v" not in elec, "explicit hi/lo voltages not pinned")
    _require(not cfg.get("turntable", {}).get("enabled", False), "turntable not supported")
    _require(not cfg.get("optimizer", {}).get("enabled", False), "exposure optimizer not supported")
    _require(not cfg.get("heatsink", None), "heatsink field not supported")
    _require(not cfg.get("passive_conductor", None), "passive conductors not supported")
    _require(not therm.get("depth_correction", {}).get("enabled", False), "depth correction not supported")
    _require(str(therm.get("convection_model", "constant")).strip().lower() == "constant",
             "only constant convection is pinned")
    _require(list(therm.get("convective_boundaries", [])) == ["top"], "convective_boundaries must be ['top']")
    _require(therm.get("stop_after_phi_bar", None) is None, "production early stop not pinned")
    _require(str(cfg.get("physics_model", {}).get("family", "baseline")).strip().lower() == "baseline",
             "only the baseline physics family is supported")
    _require(str(mat["doped"].get("sigma_profile", "uniform")).lower() == "uniform",
             "surface sigma profile not supported")
    _require(float(mat["doped"].get("sigma_temp_coeff_per_K", 0.0)) == 0.0,
             "sigma temperature coefficient must be zero in this layer")
    _require(float(mat["doped"].get("sigma_density_coeff", 0.0)) == 0.0,
             "sigma density coefficient must be zero in this layer")
    ph = therm["phase_change"]
    _require(str(ph.get("model", "")).strip().lower() in {"comsol_heaviside", "heaviside"},
             "only the COMSOL Heaviside phase model is pinned")
    _require(str(ph.get("smooth_shape", "linear")).strip().lower() == "linear",
             "only the linear smooth shape is pinned")
    _require(str(dens.get("model", "")).strip().lower() in {"physics_dual", "dual_mechanism"},
             "only the physics_dual densification model is pinned")
    _require(str(dens.get("liquid_rate_mode", "")).strip().lower() in {"viscous_capillary", "capillary_viscous"},
             "only viscous_capillary liquid rate mode is pinned")
    _require(float(dens.get("activation_energy_liq_j_per_mol", 0.0)) == 0.0,
             "nonzero liquid activation energy not supported in this layer")

    powder, doped, virgin = mat["powder"], mat["doped"], mat["virgin"]
    dt = float(therm["dt_s"])
    k_max = max(float(powder["k_solid_w_per_mk"]), float(doped["k_solid_w_per_mk"]),
                float(doped["k_liquid_w_per_mk"]), 1e-12)
    rho_cp_min = max(min(
        float(powder["rho_solid_kg_per_m3"]) * float(powder["cp_solid_j_per_kgk"]),
        float(doped["rho_solid_kg_per_m3"]) * float(doped["cp_solid_j_per_kgk"]),
        float(doped["rho_liquid_kg_per_m3"]) * float(doped["cp_liquid_j_per_kgk"]),
    ), 1e-9)
    dt_stable = 0.24 * (min(dx, dy) ** 2) * rho_cp_min / k_max
    n_substeps = max(1, int(math.ceil(dt / max(dt_stable, 1e-12))))

    v0 = float(elec["voltage_v"])
    return Pins(
        omega=2.0 * math.pi * float(elec["frequency_hz"]),
        power_factor=float(elec.get("power_factor", 1.0)),
        max_qrf=float(elec.get("max_qrf_w_per_m3", 2.0e9)),
        update_interval=max(0, int(elec.get("update_interval", 1))),
        v_hi=v0,
        v_lo=0.0,
        sigma_v=max(float(virgin["sigma_s_per_m"]), 1e-8),
        eps_v=float(virgin["eps_r"]),
        sigma_d0=float(doped["sigma_s_per_m"]),
        eps_d=float(doped["eps_r"]),
        zero_qrf_outside_doped=bool(elec.get("zero_qrf_outside_doped", True)),
        n_steps=int(therm["n_steps"]),
        dt=dt,
        n_substeps=n_substeps,
        dt_sub=dt / float(n_substeps),
        max_dt_step_c=float(therm.get("max_deltaT_per_step_c", 3.0)),
        temp_min_c=float(therm.get("min_temp_c", -50.0)),
        temp_max_c=float(therm.get("max_temp_c", 450.0)),
        ambient_c=float(therm["ambient_c"]),
        h_const=float(therm.get("convection_h_w_per_m2k", 25.0)),
        latent_heat=float(ph["latent_heat_j_per_kg"]),
        t_pc_c=float(ph["t_pc_c"]),
        dt_pc_c=max(float(ph["dt_pc_c"]), 1e-9),
        rho_powder=float(powder["rho_solid_kg_per_m3"]),
        k_powder=float(powder["k_solid_w_per_mk"]),
        cp_powder=float(powder["cp_solid_j_per_kgk"]),
        rho_solid=float(doped["rho_solid_kg_per_m3"]),
        rho_liquid=float(doped["rho_liquid_kg_per_m3"]),
        k_solid=float(doped["k_solid_w_per_mk"]),
        k_liquid=float(doped["k_liquid_w_per_mk"]),
        cp_solid=float(doped["cp_solid_j_per_kgk"]),
        cp_liquid=float(doped["cp_liquid_j_per_kgk"]),
        rho_rel_init=float(dens.get("rho_rel_initial", 0.55)),
        dens_k0_ss=float(dens.get("k0_ss_per_s", max(0.1 * float(dens["k0_per_s"]), 1e-12))),
        dens_ea_ss=float(dens.get("activation_energy_ss_j_per_mol",
                                  max(0.6 * float(dens["activation_energy_j_per_mol"]), 0.0))),
        dens_phi_solid_exponent=float(dens.get("phi_solid_exponent", 1.0)),
        dens_phi_threshold=float(dens.get("phi_threshold", 0.02)),
        dens_phi_liq_exponent=float(dens.get("phi_liq_exponent", 1.0)),
        dens_rho_exponent=float(dens.get("rho_exponent", 1.0)),
        dens_eta_ref_pa_s=float(dens.get("eta_ref_pa_s", 8.0e3)),
        dens_eta_ref_temp_k=float(dens.get("eta_ref_temp_k", 458.15)),
        dens_eta_activation_j_per_mol=float(dens.get("eta_activation_j_per_mol", 6.0e4)),
        dens_geom_factor=float(dens.get("geom_factor", 0.05)),
        dens_surface_tension=float(dens.get("surface_tension_n_per_m", 0.03)),
        dens_particle_radius_m=float(dens.get("particle_radius_m", 35e-6)),
        dens_max_delta_per_substep=float(dens.get("max_delta_per_step", 0.05)) / float(n_substeps),
    )


@dataclass
class Case:
    cfg: dict[str, Any]
    pins: Pins
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    dA: float
    part_mask: np.ndarray
    doped_mask: np.ndarray
    fill_frac: np.ndarray
    elec_hi: np.ndarray
    elec_lo: np.ndarray
    expo: np.ndarray

    @property
    def n_part(self) -> int:
        return int(self.part_mask.sum())


def build_case(cfg: dict[str, Any]) -> Case:
    x, y, _poly, part_mask, doped_mask, hi, lo, fill_frac, _pid, _polys = rfam.make_domain(cfg)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    p = pins_from_cfg(cfg, dx, dy)
    conv_b = [str(v) for v in cfg["thermal"].get("convective_boundaries", [])]
    expo = rfam.boundary_exposure_coeff(part_mask.shape, dx, dy, conv_b)
    return Case(
        cfg=cfg, pins=p, x=x, y=y, dx=dx, dy=dy, dA=dx * dy,
        part_mask=part_mask, doped_mask=doped_mask, fill_frac=fill_frac,
        elec_hi=hi, elec_lo=lo, expo=expo,
    )
