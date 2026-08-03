"""Per-stage profile of the march_fast substep (single-threaded).

Times each kernel and each NumPy reduction in isolation so the optimisation
target is measured, not guessed.
"""
from __future__ import annotations

import time

import numpy as np

import heatr3d as h3

from .kernels import (compress_part, densify_kernel, faces_kernel,
                      props_kernel, step_kernel)


def profile(n: int = 48, reps: int = 60) -> None:
    grid = h3.Grid(n=n)
    part = np.ascontiguousarray(
        h3.make_geometry(grid, "square", diam=0.020, zspan=0.020), dtype=np.bool_)
    p = h3.Params(phase_update="enthalpy")
    shp = part.shape
    nx, ny, nz = shp
    Tp = np.zeros((nx + 2, ny + 2, nz + 2)); Tp[1:-1, 1:-1, 1:-1] = 120.0
    Tpn = np.zeros_like(Tp)
    rho_rel = np.full(shp, p.rho_rel)
    Qrf = np.zeros(shp); Qrf[part] = 1e5
    phi_old = np.empty(shp); phi_new = np.empty(shp)
    k = np.empty(shp); rho_cp = np.empty(shp); rho_L = np.empty(shp)
    rho_cpeff = np.ones(shp)
    kfx = np.zeros((nx + 1, ny, nz)); kfy = np.zeros((nx, ny + 1, nz))
    kfz = np.zeros((nx, ny, nz + 1))
    qconv = np.zeros(shp); esens = np.empty(shp)
    part_idx = np.flatnonzero(part.ravel()).astype(np.int64)
    pi, pj, pm = np.unravel_index(part_idx, shp)
    part_idx_p = (((pi + 1) * (ny + 2) + (pj + 1)) * (nz + 2) + (pm + 1)).astype(np.int64)
    npart = part_idx.size
    phi_part = np.empty(npart); lat_part = np.empty(npart)

    def props():
        props_kernel(Tp, part, rho_rel, phi_old, k, rho_cp, rho_L, rho_cpeff,
                     p.t_pc_c, p.dt_pc_c, p.latent_j_per_kg,
                     p.rho_powder, p.k_powder, p.cp_powder,
                     p.rho_solid, p.k_solid, p.cp_solid,
                     p.rho_liquid, p.k_liquid, p.cp_liquid, False)

    def faces():
        faces_kernel(k, kfx, kfy, kfz)

    def step():
        step_kernel(Tp, Tpn, kfx, kfy, kfz, Qrf, rho_cp, rho_L, rho_cpeff,
                    phi_new, qconv, esens, grid.h, p.dt_s, p.conv_h,
                    p.preheat_c, p.t_pc_c, p.dt_pc_c, p.max_dt_step_c,
                    p.temp_min_c, p.temp_max_c, True)

    def reduce_qconv():
        return float(qconv.sum())

    def reduce_esens():
        return float(esens.sum())

    def compress():
        compress_part(phi_new.ravel(), phi_old.ravel(), rho_L.ravel(),
                      part_idx, phi_part, lat_part)

    def part_reductions():
        return (float(lat_part.sum()), float(phi_part.mean()),
                float(rho_rel.ravel()[part_idx].mean()))

    def densify():
        densify_kernel(Tp.ravel(), part_idx_p, phi_new.ravel(),
                       rho_rel.ravel(), part_idx, p.dt_s,
                       p.dens_max_drho_rate * p.dt_s,
                       p.dens_k0_ss, p.dens_ea_ss, p.dens_phi_solid_exp,
                       p.dens_phi_threshold, p.dens_phi_liq_exp,
                       p.dens_geom_factor, p.dens_surface_tension,
                       p.dens_particle_radius_m, p.dens_eta_ref_pa_s,
                       p.dens_eta_ref_temp_k, p.dens_eta_activation,
                       p.dens_rho_exp, h3.R_GAS)

    stages = [("props_kernel", props), ("faces_kernel", faces),
              ("step_kernel", step),
              ("qconv.sum", reduce_qconv), ("esens.sum", reduce_esens),
              ("compress_part", compress), ("part reductions", part_reductions),
              ("densify_kernel", densify)]
    for name, fn in stages:      # warm / compile
        fn()
    total = 0.0
    print(f"n={n}, {int(np.prod(shp))} cells, {reps} reps, single-threaded")
    for name, fn in stages:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        dt = 1e3 * (time.perf_counter() - t0) / reps
        total += dt
        print(f"  {name:18s} {dt:8.4f} ms/step")
    print(f"  {'TOTAL':18s} {total:8.4f} ms/step")


if __name__ == "__main__":
    import sys
    profile(int(sys.argv[1]) if len(sys.argv) > 1 else 48)
