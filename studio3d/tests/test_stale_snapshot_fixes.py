"""Stale melt-onset snapshot fixes (Tamper two-lobe incident, 2026-08-05).

heatr3d's Result fields T_phi90 / phi_final / T_max_c are MELT-ONSET reads
by that solver's documented convention (heatr3d.py:1248-1263): for densify
runs the march continues long past t90, so those fields describe a mid-run
state. The Studio consumed them as if they were end-of-run truth, which
produced two false readings on the real Tamper job (feb850ec):

  * the densified viewer showed a two-lobed part because ~1000 flange-rim
    voxels had phi_final < 0.5 AT t90, even though their END-state
    rho_final (mean 0.83) proves they melted and were consolidating;
  * the T ceiling gate read 239.1 C (t90 snapshot) while the march log of
    the identical drive shows ~317 C at the density stop and 360 C at the
    horizon - the ACCEPTED benefit verdict rested on that under-report.

These tests pin the Studio-side fixes: end-state classification from
rho_final, true-peak ceiling from T_final, and the benefit gate comparing
the true peak when it is available.
"""
from __future__ import annotations

import numpy as np
import pytest

import heatr3d as H
from studio3d.warped_mesh import build_densified_meshes
from studio3d.correction_gate import evaluate_correction


def _grid_part(n=16):
    grid = H.Grid(n)
    part = np.zeros((n, n, n), bool)
    part[5:11, 5:11, 5:9] = True
    return grid, part


def test_fused_by_end_state_rho_not_stale_phi():
    """A voxel whose END-state rho proves consolidation is SOLID, even when
    the stale melt-onset phi snapshot says it had not melted yet (the real
    Tamper rim: phi_final 0.4, rho_final 0.83)."""
    grid, part = _grid_part()
    p = H.Params()
    rho = np.where(part, 1.0, 0.55)
    phi = np.where(part, 1.0, 0.0).astype(float)
    # the "rim": stale phi says unmelted, end-state rho says consolidating
    rim = np.zeros_like(part)
    rim[5, 5:11, 5:9] = True
    phi[rim] = 0.40
    rho[rim] = 0.83
    solid, powder, info = build_densified_meshes(part, rho, phi, p, grid)
    assert info["n_powder"] == 0, (
        "rim voxels with end-state rho 0.83 are fused material, not loose "
        "powder; classification must come from rho_final, not the stale "
        "melt-onset phi snapshot")
    assert info["n_solid"] == int(part.sum())
    # and the porosity is not hidden: the under-consolidated count is loud
    assert info["n_under_consolidated"] == int(rim.sum())
    assert info["classification"] == "end-state rho_final"


def test_truly_unfused_voxels_stay_powder():
    """rho at the powder-bed initial (0.55) = never consolidated = powder,
    regardless of what the stale phi snapshot says."""
    grid, part = _grid_part()
    p = H.Params()
    rho = np.where(part, 1.0, 0.55)
    phi = np.where(part, 1.0, 0.0).astype(float)
    dead = np.zeros_like(part)
    dead[10, 5:11, 5:9] = True
    rho[dead] = 0.552
    phi[dead] = 0.9  # stale snapshot claims melted; end state says never fused
    solid, powder, info = build_densified_meshes(part, rho, phi, p, grid)
    assert info["n_powder"] == int(dead.sum())
    assert info["n_solid"] == int(part.sum() - dead.sum())


def _results(t_max_c=239.1, t_end_max_c=None, out_melt=0.03, sigma_t=28.5,
             sim=1062.0):
    g = {"energy_residual_ok": True, "clamp_bound": False,
         "T_max_C": t_max_c, "T_ceiling_C": 250.0,
         "T_ceiling_ok": t_max_c <= 250.0}
    if t_end_max_c is not None:
        g["T_end_max_C"] = t_end_max_c
    return {"grid_n": 64, "max_time_s": 1500.0, "stop_mean_rho": 0.98,
            "sim_time_s": sim, "t_phi90_s": 745.0, "sigma_T": sigma_t,
            "out_of_part_melt_frac": out_melt, "gates": g}


def test_gate_compares_true_peak_when_available():
    """The Tamper acceptance replayed with the true end-state peaks: the
    melt-onset snapshots tie (239 vs 240) but the corrected arm's TRUE peak
    regressed far past the before arm's. The gate must catch it."""
    before = _results(t_max_c=239.0, t_end_max_c=300.0)
    after = _results(t_max_c=240.0, t_end_max_c=340.0)
    v = evaluate_correction(before, after)
    assert v["verdict"] == "REJECTED"
    assert "T_ceiling" in v["failed_guards"]
    assert v["guards"]["T_ceiling"]["read"] == "end_state_peak"


def test_gate_falls_back_to_melt_onset_with_loud_note():
    """Legacy records without T_end_max_C still gate on the melt-onset read,
    and the verdict SAYS the read is melt-onset (never silently)."""
    before = _results(t_max_c=239.0)
    after = _results(t_max_c=241.0)
    v = evaluate_correction(before, after)
    assert v["guards"]["T_ceiling"]["read"] == "melt_onset_snapshot"
    assert "melt-onset" in v["guards"]["T_ceiling"]["note"]
