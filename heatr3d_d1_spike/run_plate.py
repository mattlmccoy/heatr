"""Task 1 gate: EQS parallel-plate characterization in dolfinx.

Uniform VIRGIN bed (no part), heatr3d electrode BCs (V = 860 V at y = -L/2,
V = 0 at y = +L/2, Neumann side walls). With uniform gamma the exact solution
is the linear ramp V(y) = 860 * (L/2 - y) / L and Vi == 0, so this isolates
formulation errors (wrong transpose, wrong BC placement, complex/real mixups)
from discretization error -- a P1 FEM reproduces a linear field exactly.

Gates (from the plan):
    max |V - linear(y)|  <  1e-6 * 860   over the CG1 dofs
    transverse std       <  1e-9 * 860   (std within each y-layer)
heatr3d passes the same characterization at 1.25e-11 V.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

import eqs_common as ec

N_CELLS = 24


def main() -> int:
    t0 = time.perf_counter()
    msh = ec.box_mesh(N_CELLS)
    mats = ec.materials(msh, in_part=None)          # uniform virgin bed
    t_mesh = time.perf_counter()
    Vr, Vi = ec.solve_eqs(msh, mats)
    wall_solve = time.perf_counter() - t_mesh
    wall_total = time.perf_counter() - t0

    W = Vr.function_space
    xyz = W.tabulate_dof_coordinates()
    y = xyz[:, 1]
    # solve_eqs returns Vr/Vi already split into two real-valued fields
    # (stored in the build's scalar type), so take the real part of both.
    vr = np.real(Vr.x.array)
    vi = np.real(Vi.x.array)

    exact = ec.V_LO + (ec.V_HI - ec.V_LO) * (y + ec.L_DOMAIN / 2) / ec.L_DOMAIN
    v_err_max = float(np.abs(vr - exact).max())
    vi_max = float(np.abs(vi).max())

    # transverse std: scatter within each y-layer (dof y values are exact
    # multiples of h on a structured box mesh; bin by rounded index)
    h = ec.L_DOMAIN / N_CELLS
    key = np.rint((y + ec.L_DOMAIN / 2) / h).astype(int)
    stds = [float(vr[key == k].std()) for k in np.unique(key)]
    transverse_std = float(np.max(stds))

    n_dofs = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)
    tol_v = 1e-6 * ec.V_LO
    tol_t = 1e-9 * ec.V_LO
    ok = (v_err_max < tol_v) and (transverse_std < tol_t)

    out = {"task1": {
        "scalar_path": ec.SCALAR_PATH,
        "n_cells_per_axis": N_CELLS,
        "n_dofs": n_dofs,
        "n_cells": int(msh.topology.index_map(msh.topology.dim).size_global),
        "v_err_max": v_err_max,
        "v_err_tol": tol_v,
        "transverse_std": transverse_std,
        "transverse_std_tol": tol_t,
        "vi_max_abs": vi_max,
        "wall_s": wall_total,
        "wall_solve_s": wall_solve,
        "gate_ok": bool(ok),
        "heatr3d_conventions": {
            "gamma": "sigma + 1j*omega*eps0*eps_r, omega=2*pi*27.12e6 "
                     "(heatr3d.build_gamma, binary blend path)",
            "bvp": "div(gamma grad V)=0; V=860 at y_min, V=0 at y_max, "
                   "Neumann x/z walls (heatr3d.solve_eqs_3d)",
            "qrf": "Q = 0.5*Re(gamma*|E|^2) = 0.5*sigma*|E|^2, E=-grad V, "
                   "clipped >=0, zeroed outside the doped region "
                   "(heatr3d.compute_qrf_3d, premix=False)",
            "power_renorm_basis": "Q scaled so integral(Q dV) = "
                                  "power_density_w_per_m3 * V_doped, with "
                                  "power_density_w_per_m3 = 10/(pi*0.01^2*0.02) "
                                  "= %.6g W/m^3 (heatr3d.py:59, :400-405)"
                                  % ec.POWER_DENSITY_W_PER_M3,
            "deviations": [
                "FEM Galerkin form with DG0 gamma replaces heatr3d's "
                "harmonic face-averaged finite volume (identical for the "
                "uniform-material plate case).",
                "heatr3d's cell-centered grid places electrodes at "
                "y=-+(L/2 - h/2), spanning L-h; the FEM mesh places them on "
                "the true faces y=-+L/2 (O(h/L) gauge difference).",
                "erf edge regularization and premix variants of build_gamma "
                "are not replicated (voxel-staircase remedies).",
            ],
        },
    }}
    p = Path(__file__).parent / "results.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(out)
    if MPI.COMM_WORLD.rank == 0:
        p.write_text(json.dumps(d, indent=1))
        print(json.dumps(out, indent=1))
    assert ok, (f"plate gate failed: v_err_max={v_err_max:.3e} (tol {tol_v:.1e}), "
                f"transverse_std={transverse_std:.3e} (tol {tol_t:.1e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
