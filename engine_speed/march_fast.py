"""march_fast -- a drop-in fast replacement for heatr3d.run's thermal march.

PROTOTYPE. Not wired into the Studio and not to be adopted into the engine
until the graduation lane blesses it after S2. heatr3d.py is NOT modified and
NOT monkeypatched; this module imports it and calls its own EQS path
(build_gamma / solve_eqs_3d / compute_qrf_3d) unchanged, so the drive field is
heatr3d's by construction. Only the thermal+phase+densification march is
re-implemented, in single-threaded numba kernels written to be bit-identical
(see engine_speed/kernels.py for the association-order discipline).

SUPPORTED CONFIGURATION (anything else raises NotImplementedError rather than
silently running different physics):
  * power_schedule is None                (no dose-schedule multiplier)
  * heatsink_field is None                (no k-gain / cold-loss lattice)
  * powder_loss_mode is None              (no volumetric powder sink)
  * eps_perturb_field is None             (no EM permittivity perturbation)
  * premix_frac == 0.0                    (jet-only absorption)
  * edge_width_m == 0.0                   (binary material boundary)
  * Params.eqs_update_interval_s == 0.0   (Q_rf frozen; no in-march EQS re-solve)
  * Params.phase_update in ("enthalpy", "apparent_cp")
Everything that IS supported is reproduced exactly, including the THM-03
powder-bed CFL substepping, the THM-01 per-step dT cap, the THM-02
temp_min/temp_max clamp and their clamp_bound latch, the S1 energy audit, and
the melt-onset / stop_mean_rho break semantics (including which step appends to
phi_hist).
"""
from __future__ import annotations

import logging

import numpy as np

import heatr3d as h3

from .kernels import (compress_part, densify_kernel, faces_kernel,
                      props_kernel, step_kernel)

logger = logging.getLogger(__name__)

__all__ = ["march_fast", "UnsupportedConfig"]


class UnsupportedConfig(NotImplementedError):
    """Raised when a run() option outside march_fast's proven envelope is used."""


def _check_supported(p: h3.Params, **kw) -> None:
    bad = []
    if kw.get("power_schedule") is not None:
        bad.append("power_schedule")
    if kw.get("heatsink_field") is not None:
        bad.append("heatsink_field")
    if kw.get("powder_loss_mode") is not None:
        bad.append("powder_loss_mode")
    if kw.get("eps_perturb_field") is not None:
        bad.append("eps_perturb_field")
    if float(kw.get("premix_frac", 0.0)) != 0.0:
        bad.append("premix_frac")
    if float(kw.get("edge_width_m", 0.0)) != 0.0:
        bad.append("edge_width_m")
    if float(p.eqs_update_interval_s) != 0.0:
        bad.append("Params.eqs_update_interval_s (S4 in-march EQS re-solve)")
    if p.phase_update not in ("enthalpy", "apparent_cp"):
        bad.append(f"Params.phase_update={p.phase_update!r}")
    if bad:
        raise UnsupportedConfig(
            "march_fast does not implement: " + ", ".join(bad)
            + ". Use heatr3d.run for these; march_fast is a proven-envelope "
              "prototype, not a full replacement.")


def march_fast(grid: h3.Grid, part: np.ndarray, p: h3.Params,
               sat: np.ndarray | None = None,
               max_time_s: float = 1500.0, phi_target: float = 0.90,
               densify: bool = False, stop_mean_rho: float | None = None,
               power_schedule=None, verbose: bool = False,
               heatsink_field: np.ndarray | None = None,
               heatsink_h: float = 0.0, heatsink_kgain: float = 0.0,
               edge_width_m: float = 0.0,
               eps_perturb_field: np.ndarray | None = None,
               eps_perturb_value: float = 0.0,
               qrf_override: np.ndarray | None = None,
               powder_loss_mode: str | None = None,
               powder_loss_coeff: float = 0.0,
               powder_path_len_m: float = 0.0,
               powder_loss_region: str = "part",
               premix_frac: float = 0.0,
               premix_budget: str = "floor_added",
               T0_override: np.ndarray | None = None,
               qrf_gradient: str = "masked",
               t_start_s: float = 0.0) -> h3.Result:
    """Same signature and same Result as heatr3d.run, over the supported subset."""
    _check_supported(p, power_schedule=power_schedule,
                     heatsink_field=heatsink_field,
                     powder_loss_mode=powder_loss_mode,
                     eps_perturb_field=eps_perturb_field,
                     premix_frac=premix_frac, edge_width_m=edge_width_m)

    part = np.asarray(part)
    if T0_override is not None:
        T = np.array(T0_override, dtype=np.float64, copy=True)
        if T.shape != part.shape:
            raise ValueError("T0_override shape must match part.shape")
    else:
        T = np.full(part.shape, p.preheat_c, dtype=np.float64)
    rho_rel = np.full(part.shape, p.rho_rel, dtype=np.float64)

    # ---- drive field: heatr3d's OWN EQS path, unmodified -------------------- #
    if qrf_override is None:
        gamma = h3.build_gamma(part, p, sat, edge_width_m=edge_width_m, h=grid.h,
                               premix_frac=premix_frac,
                               premix_budget=premix_budget)
        V = h3.solve_eqs_3d(gamma, grid, p)
        n_eqs_solves = 1
        Qrf = h3.compute_qrf_3d(V, gamma, grid, p, part, premix=False,
                                qrf_gradient=qrf_gradient)
    else:
        Qrf = np.array(qrf_override, dtype=np.float64, copy=True)
        if Qrf.shape != part.shape:
            raise ValueError("qrf_override shape must match part.shape")
        Qrf[~part] = 0.0
        n_eqs_solves = 0

    nsteps = int(max_time_s / p.dt_s)
    h = grid.h
    dV = grid.dV

    # ---- THM-03 powder-bed CFL substepping (identical policy to run()) ------ #
    dt_stable = h3.dt_stable_thermal(grid, p)
    n_sub = h3.cfl_substeps(grid, p) if p.enforce_cfl else 1
    dt_sub = p.dt_s / n_sub
    cfl_violated = dt_sub > h3.CFL_SAFETY * dt_stable
    if cfl_violated:
        logger.warning(
            "THM-03 CFL VIOLATION: dt_s=%.4g s exceeds %.2f * h^2/(6 alpha_max) "
            "= %.4g s (h=%.4g m, alpha_max=%.4e m^2/s, grid n=%d). The explicit "
            "conduction update is UNSTABLE.", p.dt_s, h3.CFL_SAFETY,
            h3.CFL_SAFETY * dt_stable, h, h3.alpha_max_thermal(p), grid.n)
    elif n_sub > 1:
        logger.info("THM-03: substepping the thermal update %d x (dt_sub=%.4g s).",
                    n_sub, dt_sub)
    drho_cap = p.dens_max_drho_rate * dt_sub

    _has_part = bool(part.any())
    part_idx = np.flatnonzero(part.ravel()).astype(np.int64)
    npart = part_idx.size

    # ---- work buffers (allocated once, reused every substep) ---------------- #
    shp = part.shape
    nx, ny, nz = shp
    # Halo-padded temperature buffers. The halo is written ONCE (to zero, by
    # np.zeros) and never again, so every boundary face reads 0.0 and is
    # multiplied by a 0.0 face conductivity -- see kernels.py for why that is
    # bit-identical to heatr3d's boundary-slice zeroing.
    Tp = np.zeros((nx + 2, ny + 2, nz + 2), np.float64)
    Tp[1:-1, 1:-1, 1:-1] = T
    Tpn = np.zeros((nx + 2, ny + 2, nz + 2), np.float64)
    phi_old = np.empty(shp, np.float64)
    phi_new = np.empty(shp, np.float64)
    k_buf = np.empty(shp, np.float64)
    rho_cp = np.empty(shp, np.float64)
    rho_L = np.empty(shp, np.float64)
    # Face conductivities. Allocated with np.zeros so the domain-boundary faces
    # (index 0 and index n along the face axis) stay 0.0 for the whole march;
    # faces_kernel only ever writes the internal faces.
    kfx = np.zeros((nx + 1, ny, nz), np.float64)
    kfy = np.zeros((nx, ny + 1, nz), np.float64)
    kfz = np.zeros((nx, ny, nz + 1), np.float64)
    want_cpeff = (p.phase_update != "enthalpy")
    rho_cpeff = np.ones(shp, np.float64)
    # q_conv is nonzero only on the y_max plane; the rest stays zero forever, so
    # np.sum over the full buffer reproduces heatr3d's q_conv.sum() exactly.
    qconv = np.zeros(shp, np.float64)
    esens = np.empty(shp, np.float64)
    phi_part = np.empty(npart, np.float64)
    lat_part = np.empty(npart, np.float64)

    part_c = np.ascontiguousarray(part, dtype=np.bool_)
    Qrf_c = np.ascontiguousarray(Qrf, dtype=np.float64)
    use_enthalpy = (p.phase_update == "enthalpy")
    # flat part indices into the UNPADDED (nx,ny,nz) buffers and into the
    # PADDED temperature buffer, precomputed once
    pi, pj, pm = np.unravel_index(part_idx, shp)
    part_idx_p = (((pi + 1) * (ny + 2) + (pj + 1)) * (nz + 2)
                  + (pm + 1)).astype(np.int64)

    def _unpad(A):
        return np.ascontiguousarray(A[1:-1, 1:-1, 1:-1])

    # e_in is constant per substep (Qrf is frozen and pmult == 1.0), so heatr3d's
    # float((Qrf * 1.0).sum()) is evaluated once and re-added with the identical
    # float operations every substep.
    qrf_sum = float((Qrf * 1.0).sum())

    T_phi90 = None
    reached = False
    t90 = float("nan")
    phi_hist: list = []
    clamp_bound = False
    e_in = 0.0
    e_loss = 0.0
    e_stored_acc = 0.0
    latent = p.latent_j_per_kg

    for _it_sub in range(nsteps * n_sub):
        it, isub = divmod(_it_sub, n_sub)
        props_kernel(Tp, part_c, rho_rel, phi_old, k_buf, rho_cp, rho_L,
                     rho_cpeff, p.t_pc_c, p.dt_pc_c, latent,
                     p.rho_powder, p.k_powder, p.cp_powder,
                     p.rho_solid, p.k_solid, p.cp_solid,
                     p.rho_liquid, p.k_liquid, p.cp_liquid, want_cpeff)
        faces_kernel(k_buf, kfx, kfy, kfz)
        n_dT_clip, n_temp_clip, max_abs_dT = step_kernel(
            Tp, Tpn, kfx, kfy, kfz, Qrf_c, rho_cp, rho_L, rho_cpeff, phi_new,
            qconv, esens, h, dt_sub, p.conv_h, p.preheat_c,
            p.t_pc_c, p.dt_pc_c, p.max_dt_step_c, p.temp_min_c, p.temp_max_c,
            use_enthalpy)

        e_in += qrf_sum * dV * dt_sub
        e_loss += float(qconv.sum()) * dV * dt_sub

        if n_dT_clip > 0:
            clamp_bound = True
            logger.warning(
                "THM-01 per-step dT cap BOUND at it=%d: %.4f%% of cells clipped "
                "to +-%.3f C (max raw |dT|=%.3f C).",
                it, 100.0 * n_dT_clip / esens.size, p.max_dt_step_c, max_abs_dT)
        if n_temp_clip > 0:
            clamp_bound = True
            logger.warning(
                "THM-02 temperature clamp BOUND at it=%d: %.4f%% of cells hit "
                "[temp_min=%.1f, temp_max=%.1f] C.",
                it, 100.0 * n_temp_clip / esens.size, p.temp_min_c, p.temp_max_c)

        Tp, Tpn = Tpn, Tp                   # swap buffers; Tp is now the new field

        e_stored_acc += float(esens.sum()) * dV
        if _has_part:
            compress_part(phi_new.ravel(), phi_old.ravel(), rho_L.ravel(),
                          part_idx, phi_part, lat_part)
            e_stored_acc += float(lat_part.sum()) * dV
            mean_phi = float(phi_part.mean())
        else:
            mean_phi = 0.0

        if densify:
            densify_kernel(Tp.ravel(), part_idx_p, phi_new.ravel(),
                           rho_rel.ravel(), part_idx, dt_sub, drho_cap,
                           p.dens_k0_ss, p.dens_ea_ss, p.dens_phi_solid_exp,
                           p.dens_phi_threshold, p.dens_phi_liq_exp,
                           p.dens_geom_factor, p.dens_surface_tension,
                           p.dens_particle_radius_m, p.dens_eta_ref_pa_s,
                           p.dens_eta_ref_temp_k, p.dens_eta_activation,
                           p.dens_rho_exp, h3.R_GAS)

        if not reached and mean_phi >= phi_target:
            T_phi90 = _unpad(Tp)
            reached = True
            t90 = (it + (isub + 1) / n_sub) * p.dt_s
            if not densify:
                phi_hist.append(mean_phi)
                break
        if densify and stop_mean_rho is not None:
            # rho_rel was just updated in place; re-gather for the stop test.
            mean_rho = float(rho_rel.ravel()[part_idx].mean()) if _has_part else 0.0
            if mean_rho >= stop_mean_rho:
                phi_hist.append(mean_phi)
                break
        if isub < n_sub - 1:
            continue
        phi_hist.append(mean_phi)
        if verbose and it % 200 == 0:
            print(f"  t={it*p.dt_s:6.1f}s  "
                  f"Tmax={(Tp.ravel()[part_idx_p].max() if _has_part else float('nan')):6.1f}  "
                  f"phi={mean_phi:.3f}  "
                  f"rho={(rho_rel.ravel()[part_idx].mean() if _has_part else float('nan')):.3f}")

    T = _unpad(Tp)
    e_stored = e_stored_acc
    e_resid_frac = (e_in - e_stored - e_loss) / max(e_in, 1e-30)

    if T_phi90 is None:
        logger.warning(
            "MELT-ONSET FALLBACK: mean melt fraction never crossed "
            "phi_target=%.2f within max_time_s=%.1f s (final mean phi=%.4f).",
            phi_target, max_time_s, (phi_hist[-1] if phi_hist else float("nan")))
        T_phi90 = T.copy()
    sigma_T = float(T_phi90[part].std()) if _has_part else float("nan")
    print(f"  [s1-energy] in={e_in:.1f} J stored={e_stored:.1f} J "
          f"loss={e_loss:.1f} J residual_frac={e_resid_frac:+.4f}"
          f"{'  CLAMP-BOUND' if clamp_bound else ''}")
    return h3.Result(
        sigma_T=sigma_T, T_phi90=T_phi90, part=part, Qrf=Qrf,
        phi_final=h3.phase_fraction(T_phi90, p)[0], t_phi90_s=t90,
        reached=reached, phi_hist=phi_hist,
        T_max_c=(float(T_phi90[part].max()) if _has_part else float("nan")),
        rho_final=(rho_rel if densify else None),
        exposure_s=(nsteps * p.dt_s if densify else t90),
        clamp_bound=clamp_bound,
        energy_in_j=e_in, energy_stored_j=e_stored, energy_loss_j=e_loss,
        energy_residual_frac=e_resid_frac,
        T_final=T.copy(), n_substeps_used=n_sub, cfl_violated=cfl_violated,
        n_eqs_solves=n_eqs_solves, n_eqs_resolves_skipped=0)
