#!/usr/bin/env python3
"""S1 diagnosis: quantify per-step dT vs the melt window and the conduction
CFL number at the moment of blow-up, for the mechanism repro case.

Run: ./.venv312/bin/python scripts/analysis/s1_diagnose_instability.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from heatr3d import Grid, Params, make_geometry, phase_fraction  # noqa: E402,F401


def main() -> None:
    n = 32
    grid = Grid(n=n, L=0.060)
    part = make_geometry(grid, "sphere", diam=0.020)  # noqa: F841
    p = Params()
    # conduction CFL bound (secondary hypothesis check)
    alpha_max = p.k_liquid / (p.rho_liquid * p.cp_liquid)
    dt_cfl = grid.h ** 2 / (6.0 * alpha_max)
    print(f"h={grid.h*1e3:.3f} mm  dt_s={p.dt_s}  dt_CFL={dt_cfl:.3f} s  "
          f"CFL ratio={p.dt_s/dt_cfl:.3f} (<1 means conduction-stable)")
    for nn in (48, 96, 200):
        hh = 0.060 / nn
        print(f"  n={nn}: dt_CFL={hh**2/(6*alpha_max):.3f} s "
              f"ratio={p.dt_s/(hh**2/(6*alpha_max)):.3f}")
    # source-term dT for a corner-spike voxel vs the melt window
    for mult in (1, 10, 100, 400, 1000):
        q = p.power_density_w_per_m3 * mult
        dT = q * p.dt_s / (p.rho_solid * p.cp_solid)
        print(f"  spike x{mult:5d}: raw source dT/step = {dT:8.3f} C "
              f"(melt window = {p.dt_pc_c} C, clamp = {p.max_dt_step_c} C)")
    # latent barrier a window-crossing step must pay
    e_latent_per_c = p.latent_j_per_kg / p.dt_pc_c
    print(f"latent per C inside window: {e_latent_per_c:.0f} J/(kg C) vs "
          f"cp_solid {p.cp_solid} J/(kg C) -> apparent cp in-window ~x"
          f"{1 + e_latent_per_c/p.cp_solid:.1f}")


if __name__ == "__main__":
    main()
