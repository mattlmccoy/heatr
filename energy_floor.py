"""PA12 melt-energy floor and degradation-ceiling headroom (per unit volume).

Materials/energy calculation only -- no solve. Deepens
deck_figures_3d/fig5_energy_sls_vs_rfam.py with the DSC-measured latent heat.

Provenance (see docs/THEORY_REFERENCES.md tags):
  [MEASURED]  latent 101700 J/kg      -- configs/experimental_pa12_dsc_profile.yaml
  [MEASURED]  melt window 171/180.8/186 C -- same DSC profile
  [LIT]       cp_solid 1287, cp_liquid 2500 J/kgK -- RFAM_physics_from_literature.md:31
  [LIT]       rho_solid 490, rho_liquid 1010 kg/m3 -- RFAM_physics_from_literature.md:30
  [ASSUMED]   T_ceiling 250 C degradation limit -- thermal-ceiling workstream

Basis: energy per cm3 of CONSOLIDATED (dense) part, rho ~ rho_liquid, so the
numbers are comparable to fig5's SLS/RFAM per-cm3-of-part estimates.
"""
from __future__ import annotations

J_PER_M3_TO_J_PER_CM3 = 1.0e-6  # 1 m^3 = 1e6 cm^3


def floor_j_per_cm3(*, t_start_c, t_melt_c, cp_solid, latent_j_per_kg, rho):
    """Minimum energy to raise dense PA12 from t_start to full melt, per cm3.

    E = rho * (cp_solid * dT + L). Sensible heat (solid cp over dT) plus the
    DSC-measured latent. From a preheat start dT is small and L dominates.
    """
    dT = float(t_melt_c) - float(t_start_c)
    e_j_per_kg = cp_solid * dT + latent_j_per_kg
    return rho * e_j_per_kg * J_PER_M3_TO_J_PER_CM3


def headroom_j_per_cm3(*, t_melt_c, t_ceiling_c, cp_liquid, rho):
    """Sensible energy from the melt point to the degradation ceiling, per cm3.

    The operating band's upper wall: once fused, this much more local energy
    takes a voxel from melt to the 250 C burn limit. Energy beyond it is waste
    that drives degradation.
    """
    dT = float(t_ceiling_c) - float(t_melt_c)
    return rho * cp_liquid * dT * J_PER_M3_TO_J_PER_CM3


def ratio_above_floor(delivered_j_per_cm3, floor_j_per_cm3_value):
    """How many times the melt floor a delivered/absorbed energy sits at."""
    return float(delivered_j_per_cm3) / float(floor_j_per_cm3_value)
