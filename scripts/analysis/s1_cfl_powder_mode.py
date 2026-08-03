#!/usr/bin/env python3
"""S1 / THM-03: decisive test of the explicit-conduction CFL hypothesis at fine
grids, and the correction to the Task-3 conclusion that "CFL is innocent".

Task 3 ruled the conduction CFL innocent using
alpha_max = k_liquid/(rho_liquid*cp_liquid) = 7.85e-8 m^2/s. The fastest medium
in the domain is not the liquid but the POWDER BED (~98 % of the voxels):
alpha_powder = k_powder/(rho_powder*cp_powder) = 3.7504e-7 m^2/s, 4.78x larger.
The explicit 6-neighbour bound dt < h^2/(6 alpha) is therefore violated for
n > 178.9 at dt_s = 0.05 s, L = 0.060 m -- exactly the documented "blow-up at
grid >= 200" trigger.

Experiment: all-powder domain (no part), no source, no convection, initial
condition = a tiny 3-D checkerboard (the fastest-growing discrete mode), marched
through run() via the Task-6 T0_override hook. For pure conduction the
per-step amplification of that mode is 1 - 12*alpha*dt/h^2, so it DECAYS below
the threshold and GROWS above it. Measured vs predicted agree to 4 decimals at
n=176 (0.9362) and n=184 (1.1162), and the sign flips across n=178.9.

Run: ./.venv312/bin/python scripts/analysis/s1_cfl_powder_mode.py
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from heatr3d import Grid, Params, run  # noqa: E402


def main() -> None:
    p0 = dataclasses.replace(Params(), conv_h=0.0)
    alpha = p0.k_powder / (p0.rho_powder * p0.cp_powder)
    L = 0.060
    print(f"alpha_powder={alpha:.4e} m^2/s   dt_s={p0.dt_s} s   L={L} m")
    print(f"CFL threshold grid n = L/sqrt(6*alpha*dt) = "
          f"{L / np.sqrt(6 * alpha * p0.dt_s):.1f}")
    nsteps = 60
    amp0 = 1e-3
    for n in (128, 160, 176, 184, 200):
        grid = Grid(n=n, L=L)
        part = np.zeros((n, n, n), dtype=bool)
        i, j, k = np.indices((n, n, n))
        T0 = p0.preheat_c + amp0 * ((-1.0) ** (i + j + k))
        q = np.zeros((n, n, n))
        t0 = time.time()
        r = run(grid, part, p0, qrf_override=q, max_time_s=nsteps * p0.dt_s,
                phi_target=2.0, T0_override=T0)
        amp = float(np.abs(r.T_final - p0.preheat_c).max())
        ratio = p0.dt_s / (grid.h ** 2 / (6 * alpha))
        g_pred = abs(1.0 - 12 * alpha * p0.dt_s / grid.h ** 2)
        print(f"n={n:4d} h={grid.h*1e3:.4f}mm CFL_ratio={ratio:.3f} "
              f"predicted_growth/step={g_pred:.4f} "
              f"amp_{nsteps}={amp:.3e} "
              f"measured_growth/step={(amp/amp0)**(1.0/nsteps):.4f} "
              f"clamp_bound={r.clamp_bound} wall={time.time()-t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
