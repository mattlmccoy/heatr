"""TDD for the PA12 melt-energy floor + degradation-ceiling headroom calc.

Deepens deck_figures_3d/fig5_energy_sls_vs_rfam.py with the DSC-measured latent
(configs/experimental_pa12_dsc_profile.yaml) and repo material props
(RFAM_physics_from_literature.md).
"""
import numpy as np

from energy_floor import floor_j_per_cm3, headroom_j_per_cm3, ratio_above_floor


def test_floor_preheat_start_is_latent_dominated():
    # From a 170 C preheat to melt end 186 C: sensible is tiny, latent dominates.
    # (1287*16 + 101700) J/kg * 1010 kg/m3 / 1e6 = 123.5 J/cm3
    val = floor_j_per_cm3(t_start_c=170.0, t_melt_c=186.0,
                          cp_solid=1287.0, latent_j_per_kg=101700.0, rho=1010.0)
    assert np.isclose(val, 123.5, atol=0.5)


def test_floor_room_temp_start_is_higher():
    # From room temp 25 C to melt end 186 C.
    # (1287*161 + 101700) * 1010 / 1e6 = 312.0 J/cm3
    val = floor_j_per_cm3(t_start_c=25.0, t_melt_c=186.0,
                          cp_solid=1287.0, latent_j_per_kg=101700.0, rho=1010.0)
    assert np.isclose(val, 312.0, atol=1.0)


def test_headroom_melt_to_ceiling():
    # Sensible energy from melt end 186 C to the 250 C degradation ceiling
    # (liquid cp 2500): 2500 * 64 * 1010 / 1e6 = 161.6 J/cm3 of headroom.
    val = headroom_j_per_cm3(t_melt_c=186.0, t_ceiling_c=250.0,
                             cp_liquid=2500.0, rho=1010.0)
    assert np.isclose(val, 161.6, atol=0.5)


def test_ratio_above_floor():
    # RFAM absorbed ~2400 J/cm3 vs a ~123.5 J/cm3 preheat floor ~= 19.4x.
    assert np.isclose(ratio_above_floor(2400.0, 123.5), 19.43, atol=0.05)
