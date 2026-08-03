"""numba kernels for the engine_speed fast thermal march.

THREAD POLICY (machine-wide compute convention): every kernel here is
SINGLE-THREADED. There is no ``prange`` and no ``parallel=True`` anywhere in
this file, so the march occupies exactly one core regardless of
OMP_NUM_THREADS / NUMBA_NUM_THREADS. Nothing here needs a bigger scheduler slot.

BIT-IDENTITY DISCIPLINE: every expression below is written in the SAME
association order as the corresponding NumPy expression in heatr3d.run, because
IEEE-754 addition and multiplication are not associative. ``fastmath`` is OFF
(the default) for exactly this reason -- enabling it would let LLVM reassociate
and the parity gate would drop off the 1e-16 floor. Do not add fastmath=True.

TWO MEASURED OPTIMISATIONS, both bit-identity-preserving (see SPEED_REPORT.md):

1. HALO PADDING instead of boundary branches. The temperature field is carried
   in a (nx+2, ny+2, nz+2) buffer whose halo is permanently 0.0, and the six
   face-conductivity arrays are permanently 0.0 on the domain-boundary faces.
   A boundary face therefore contributes (0.0 * (0.0 - T)) / h^2 = -0.0, and
   x + (-0.0) == x for every float x, so this is arithmetically identical to
   heatr3d zeroing that boundary slice and adding it. The payoff is that the
   innermost loop has NO data-dependent branch and LLVM can vectorise it; a
   micro-benchmark on this machine put a vectorised f64 divide at ~0.07 ns,
   i.e. the original kernel was branch-bound, not division-bound.

2. FACE-CONDUCTIVITY CACHING. heatr3d evaluates the harmonic mean of every
   internal face TWICE, once from each side. _harmonic is EXACTLY symmetric:
   ``den = a + b`` is symmetric, ``0.5 * (a + b)`` is symmetric, and
   ``2.0 * a * b / den`` is too because ``2.0 * a`` is an exact power-of-two
   scaling, so (2a)*b and (2b)*a round the same real number identically. Each
   face is therefore computed once and read from both sides with no change of
   bits.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_MAXF = 1.7976931348623157e308


@njit(cache=True, inline="always")
def _nan_to_num(x: float) -> float:
    """np.nan_to_num defaults for a scalar float64."""
    if np.isnan(x):
        return 0.0
    if x == np.inf:
        return _MAXF
    if x == -np.inf:
        return -_MAXF
    return x


@njit(cache=True, inline="always")
def _clip(x: float, lo: float, hi: float) -> float:
    """np.clip semantics including NaN pass-through."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(cache=True)
def props_kernel(Tp, part, rho_rel, phi_out, k_out, rho_cp_out, rho_L_out,
                 rho_cpeff_out, t_pc, dt_pc, latent,
                 rho_powder, k_powder, cp_powder,
                 rho_solid, k_solid, cp_solid,
                 rho_liquid, k_liquid, cp_liquid, want_cpeff):
    """phase_fraction + the per-voxel property blend of heatr3d.run.

    Tp is the HALO-PADDED temperature field; domain cell (i,j,m) lives at
    Tp[i+1, j+1, m+1].

    rho_L_out[i] = rho_s_eff*latent inside the part, 0 outside -- which is both
    the enthalpy scheme's latent slope AND the exact factor the S1 stored-energy
    audit multiplies dphi by, so one buffer serves both.
    """
    nx, ny, nz = part.shape
    for i in range(nx):
        for j in range(ny):
            for m in range(nz):
                Tv = Tp[i + 1, j + 1, m + 1]
                arg = (Tv - t_pc) / dt_pc
                ph = _clip(arg + 0.5, 0.0, 1.0)
                phi_out[i, j, m] = ph
                if part[i, j, m]:
                    rr = rho_rel[i, j, m]
                    rho_s_eff = rho_powder + rr * (rho_solid - rho_powder)
                    k_s_eff = k_powder + rr * (k_solid - k_powder)
                    rho = (1.0 - ph) * rho_s_eff + ph * rho_liquid
                    kk = (1.0 - ph) * k_s_eff + ph * k_liquid
                    cp = (1.0 - ph) * cp_solid + ph * cp_liquid
                    rho_L_out[i, j, m] = rho_s_eff * latent
                else:
                    rho = rho_powder
                    kk = k_powder
                    cp = cp_powder
                    rho_L_out[i, j, m] = 0.0
                k_out[i, j, m] = kk
                rho_cp_out[i, j, m] = rho * cp
                if want_cpeff:
                    if abs(arg) <= 0.5:
                        dphi = 1.0 / dt_pc
                    else:
                        dphi = 0.0
                    v = rho * (cp + latent * dphi)
                    if v < 1e-9:            # np.maximum(rho*cp_eff, 1e-9)
                        v = 1e-9
                    rho_cpeff_out[i, j, m] = v


@njit(cache=True)
def faces_kernel(k, kfx, kfy, kfz):
    """Harmonic face conductivities, one evaluation per internal face.

    kfx[i,j,m] is the face between cells i-1 and i (so cell i's "minus" face is
    kfx[i] and its "plus" face is kfx[i+1]); kfx[0] and kfx[nx] are the domain
    boundary and are left at the 0.0 the caller allocated them with -- never
    written here, so they stay zero for the whole march.
    """
    nx, ny, nz = k.shape
    for i in range(1, nx):
        for j in range(ny):
            for m in range(nz):
                a = k[i - 1, j, m]
                b = k[i, j, m]
                den = a + b
                q = 2.0 * a * b / den
                kfx[i, j, m] = q if abs(den) > 1e-30 else 0.5 * (a + b)
    for i in range(nx):
        for j in range(1, ny):
            for m in range(nz):
                a = k[i, j - 1, m]
                b = k[i, j, m]
                den = a + b
                q = 2.0 * a * b / den
                kfy[i, j, m] = q if abs(den) > 1e-30 else 0.5 * (a + b)
    for i in range(nx):
        for j in range(ny):
            for m in range(1, nz):
                a = k[i, j, m - 1]
                b = k[i, j, m]
                den = a + b
                q = 2.0 * a * b / den
                kfz[i, j, m] = q if abs(den) > 1e-30 else 0.5 * (a + b)


@njit(cache=True)
def step_kernel(Tp, Tpn, kfx, kfy, kfz, Qrf, rho_cp, rho_L, rho_cpeff,
                phi_new_out, qconv_out, esens_out,
                h, dt_sub, conv_h, preheat, t_pc, dt_pc,
                max_dt_step, temp_min, temp_max, use_enthalpy):
    """One thermal substep, fused and branch-free in the innermost loop.

    Face-flux accumulation order is (ax0,-1),(ax0,+1),(ax1,-1),(ax1,+1),
    (ax2,-1),(ax2,+1) -- the exact order of heatr3d's ``for ax: for s in (-1,+1)``
    ``div += flux`` loop, because floating-point addition is not associative.

    Returns (n_dT_clip, n_temp_clip, max_abs_dT_raw) so the caller can reproduce
    the THM-01/THM-02 warnings and the clamp_bound latch exactly.
    """
    nx, ny, nz = rho_cp.shape
    hh = h * h
    lo = t_pc - dt_pc / 2.0
    hi_H_const = lo + dt_pc
    n_dT_clip = 0
    n_temp_clip = 0
    max_abs_dT = 0.0
    for i in range(nx):
        ip = i + 1
        for j in range(ny):
            jp = j + 1
            is_top = (j == ny - 1)
            for m in range(nz):
                mp = m + 1
                Tv = Tp[ip, jp, mp]
                div = (kfx[i, j, m] * (Tp[i, jp, mp] - Tv)) / hh
                div = div + (kfx[i + 1, j, m] * (Tp[ip + 1, jp, mp] - Tv)) / hh
                div = div + (kfy[i, j, m] * (Tp[ip, j, mp] - Tv)) / hh
                div = div + (kfy[i, j + 1, m] * (Tp[ip, jp + 1, mp] - Tv)) / hh
                div = div + (kfz[i, j, m] * (Tp[ip, jp, m] - Tv)) / hh
                div = div + (kfz[i, j, m + 1] * (Tp[ip, jp, mp + 1] - Tv)) / hh

                # open top face (y max) convection; identically zero elsewhere
                if is_top:
                    qc = conv_h * (Tv - preheat) / h
                    qconv_out[i, j, m] = qc
                else:
                    qc = 0.0

                num = (div + Qrf[i, j, m]) - qc

                rcp = rho_cp[i, j, m]
                if use_enthalpy:
                    # ``use_enthalpy`` is loop-invariant, so LLVM unswitches
                    # this out of the inner loop; the branch costs nothing and
                    # keeps the apparent-cp divide off the enthalpy path.
                    rL = rho_L[i, j, m]
                    if rL == 0.0:
                        # NO-LATENT FAST PATH -- exact, not an approximation.
                        # rho_L is identically 0 outside the part, which is
                        # ~96 % of the domain. With rL == 0:
                        #   H       = rcp*Tv + 0.0*frac = rcp*Tv + 0.0
                        #             (frac in [0,1] so 0.0*frac is +0.0; if Tv
                        #              is NaN both forms give NaN)
                        #   T_below = H/rcp
                        #   T_above = (H - 0.0)/rcp     == H/rcp
                        #   T_window= (H + 0.0)/(rcp+0.0) == H/rcp
                        # so all three np.where candidates collapse to H/rcp and
                        # the branch selection cannot matter. (The one signed-
                        # zero corner, H == -0.0, is unreachable here: H_lo =
                        # rcp*175 > 0, so H = -0.0 always takes the below branch
                        # in the general path too.) This drops one divide for
                        # frac and two of the three candidate divides.
                        H = rcp * Tv + 0.0
                        H = H + dt_sub * _nan_to_num(num)
                        T_new = H / rcp
                    else:
                        frac = _clip((Tv - lo) / dt_pc, 0.0, 1.0)
                        H = rcp * Tv + rL * frac
                        H = H + dt_sub * _nan_to_num(num)
                        H_lo = rcp * lo
                        H_hi = rcp * hi_H_const + rL
                        # All three candidates are evaluated eagerly, exactly as
                        # heatr3d's nested np.where does.
                        T_below = H / rcp
                        T_above = (H - rL) / rcp
                        T_window = (H + rL * lo / dt_pc) / (rcp + rL / dt_pc)
                        if H <= H_lo:
                            T_new = T_below
                        elif H >= H_hi:
                            T_new = T_above
                        else:
                            T_new = T_window
                    dT_raw = T_new - Tv
                else:
                    dT_raw = dt_sub * _nan_to_num(num / rho_cpeff[i, j, m])

                a = abs(dT_raw)
                if a > max_abs_dT:
                    max_abs_dT = a
                n_dT_clip += 1 if a > max_dt_step else 0
                dT = _clip(dT_raw, -max_dt_step, max_dt_step)
                T_cand = Tv + dT
                n_temp_clip += 1 if (T_cand > temp_max or T_cand < temp_min) else 0
                Tnew = _clip(T_cand, temp_min, temp_max)
                Tpn[ip, jp, mp] = Tnew
                esens_out[i, j, m] = rcp * (Tnew - Tv)
                phi_new_out[i, j, m] = _clip((Tnew - t_pc) / dt_pc + 0.5,
                                             0.0, 1.0)
    return n_dT_clip, n_temp_clip, max_abs_dT


@njit(cache=True)
def compress_part(phi_new_flat, phi_old_flat, rho_L_flat, part_idx,
                  phi_part_out, lat_part_out):
    """Gather the part-masked reductands in C order.

    NumPy's ``A[part]`` produces a compressed 1-D array in exactly this order,
    and ``.sum()``/``.mean()`` then run NumPy's pairwise summation over it. By
    materialising the SAME 1-D arrays and letting NumPy reduce them, the audit
    sums stay bit-identical instead of drifting by a reduction-order epsilon.
    """
    for q in range(part_idx.shape[0]):
        idx = part_idx[q]
        pn = phi_new_flat[idx]
        phi_part_out[q] = pn
        lat_part_out[q] = rho_L_flat[idx] * (pn - phi_old_flat[idx])


@njit(cache=True)
def densify_kernel(Tp_flat, part_idx_p, phi_flat, rho_rel_flat, part_idx,
                   dt_sub, drho_cap,
                   k0_ss, ea_ss, phi_solid_exp, phi_threshold, phi_liq_exp,
                   geom_factor, surface_tension, particle_radius,
                   eta_ref, eta_ref_temp_k, eta_activation, rho_exp, r_gas):
    """heatr3d.densify_rate + the clipped rho_rel update, on part voxels only.

    heatr3d evaluates densify_rate over the WHOLE domain and then indexes
    ``drho[part]``; the off-part values are discarded, so evaluating only the
    part voxels is arithmetically identical (every operation is elementwise).

    The exp/pow calls are deliberately left SCALAR. Vectorising them would swap
    libm's scalar exp for a vector-math implementation whose last bit can differ
    from NumPy's, which is exactly the kind of silent 1-ulp drift the parity
    gate exists to catch.
    """
    ea_over_r = eta_activation / r_gas
    gs = geom_factor * surface_tension
    thr_den = 1.0 - phi_threshold
    if thr_den < 1e-9:
        thr_den = 1e-9
    for q in range(part_idx.shape[0]):
        idx = part_idx[q]
        Tk = Tp_flat[part_idx_p[q]] + 273.15
        if Tk < 1.0:
            Tk = 1.0
        rr = rho_rel_flat[idx]
        base = _clip(1.0 - rr, 0.0, 1.0)
        rho_term = base if rho_exp == 1.0 else base ** rho_exp
        kss = k0_ss * np.exp((-ea_ss) / (r_gas * Tk))
        sd = _clip(1.0 - phi_flat[idx], 0.0, 1.0)
        ss_drive = sd if phi_solid_exp == 1.0 else sd ** phi_solid_exp
        eta = eta_ref * np.exp(ea_over_r * (1.0 / Tk - 1.0 / eta_ref_temp_k))
        if eta < 1e-12:
            eta = 1e-12
        kliq = gs / (eta * particle_radius)
        pa = _clip((phi_flat[idx] - phi_threshold) / thr_den, 0.0, 1.0)
        liq_drive = pa if phi_liq_exp == 1.0 else pa ** phi_liq_exp
        rate = (kss * ss_drive + kliq * liq_drive) * rho_term
        drho = _clip(dt_sub * rate, 0.0, drho_cap)
        rho_rel_flat[idx] = _clip(rr + drho, 0.0, 1.0)
