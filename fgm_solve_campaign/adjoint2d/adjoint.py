"""Reverse-mode gradient of the read-state objectives with respect to the
per-cell binder saturation.

Three pieces, each separately testable:

1. `substep_vjp`      the vector-Jacobian product of one thermal substep,
                      coupled in (T, rho), with every clip and cap handled as a
                      subgradient (gate the sensitivity by the clip-inactive
                      mask, keep the carried-identity term ungated).
2. `reverse_march`    walks the outer steps backwards, recomputing each outer
                      step's substeps from its checkpoint, and accumulates
                      dJ/dQ_rf separately for the two electrical states.
3. `eqs_vjp`          the EQS adjoint: solves A^T lambda = p with the SAME
                      assembled matrix the forward solved, then differentiates
                      the harmonic face conductances.

Complex arithmetic appears only in V, gamma and A. Everything downstream
(Q_rf, T, rho, phi, the objective) is real.
"""
from __future__ import annotations

import numpy as np

from . import eqs, forward as fwd, gradops, schedule as sch
from .pins import EPS0, R_GAS, Case


# ---------------------------------------------------------------------------
# 1. thermal substep VJP
# ---------------------------------------------------------------------------

def diffusion_vjp(g_div: np.ndarray, T: np.ndarray, k: np.ndarray,
                  dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Adjoint of `rfam_eqs_coupled.diffusion_divergence` in (T, k)."""
    g_T = np.zeros_like(T)
    g_k = np.zeros_like(k)

    # x faces: face c sits between columns c and c+1
    g_face_x = (g_div[:, 1:] - g_div[:, :-1]) / dx
    kxf = 0.5 * (k[:, 1:] + k[:, :-1])
    dT_x = (T[:, 1:] - T[:, :-1])
    g_T[:, 1:] += g_face_x * (-kxf / dx)
    g_T[:, :-1] += g_face_x * (kxf / dx)
    g_kx = g_face_x * (-dT_x / dx)
    g_k[:, 1:] += 0.5 * g_kx
    g_k[:, :-1] += 0.5 * g_kx

    g_face_y = (g_div[1:, :] - g_div[:-1, :]) / dy
    kyf = 0.5 * (k[1:, :] + k[:-1, :])
    dT_y = (T[1:, :] - T[:-1, :])
    g_T[1:, :] += g_face_y * (-kyf / dy)
    g_T[:-1, :] += g_face_y * (kyf / dy)
    g_ky = g_face_y * (-dT_y / dy)
    g_k[1:, :] += 0.5 * g_ky
    g_k[:-1, :] += 0.5 * g_ky
    return g_T, g_k


def substep_vjp(c: fwd.SubstepCache, case: Case,
                gT_out: np.ndarray, gR_out: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dJ/dT_in, dJ/drho_in, dJ/dQ) for one substep."""
    p = case.pins
    pm = case.part_mask
    gT = np.array(gT_out, dtype=float, copy=True)
    gR_in = np.zeros_like(gR_out)

    # --- rho_new -----------------------------------------------------------
    gR_in += np.where(pm, gR_out * c.rho_new_range, gR_out)
    g_drho = np.where(pm, gR_out * c.rho_new_range, 0.0)

    # drho = clip(dt*rate, 0, cap)     (subgradient: zero where the cap binds)
    g_rate = g_drho * c.drho_clip_mask * p.dt_sub

    # rate = (kss*ss + kliq*liq) * rho_term
    g_rho_term = g_rate * (c.kdens_ss * c.ss_drive + c.kdens_liq * c.liq_drive)
    g_kss = g_rate * c.ss_drive * c.rho_term
    g_ss = g_rate * c.kdens_ss * c.rho_term
    g_kliq = g_rate * c.liq_drive * c.rho_term
    g_liq = g_rate * c.kdens_liq * c.rho_term

    # rho_term = clip(1-rho, 0, 1)**e_r
    e_r = p.dens_rho_exponent
    omr = np.clip(1.0 - c.rho_in, 0.0, 1.0)
    m_r = (1.0 - c.rho_in > 0.0) & (1.0 - c.rho_in < 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_rho_term = np.where(omr > 0.0, e_r * np.power(omr, e_r - 1.0), 0.0)
    gR_in += g_rho_term * (-d_rho_term) * m_r

    # temperature-dependent kinetics (through Tk)
    inv_rt2 = 1.0 / (R_GAS * c.Tk * c.Tk)
    g_Tk = g_kss * (c.kdens_ss * p.dens_ea_ss * inv_rt2)
    eta_ea = max(p.dens_eta_activation_j_per_mol, 0.0)
    g_Tk = g_Tk + g_kliq * (c.kdens_liq * eta_ea * inv_rt2)

    # ss_drive = clip(1-phi_out, 0, 1)**e_s
    e_s = p.dens_phi_solid_exponent
    omp = np.clip(1.0 - c.phi_out, 0.0, 1.0)
    m_s = (1.0 - c.phi_out > 0.0) & (1.0 - c.phi_out < 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_ss = np.where(omp > 0.0, e_s * np.power(omp, e_s - 1.0), 0.0)
    g_phi_out = g_ss * (-d_ss) * m_s

    # liq_drive = phi_act**e_l ; phi_act = clip((phi_out-thr)/(1-thr), 0, 1)
    e_l = p.dens_phi_liq_exponent
    with np.errstate(divide="ignore", invalid="ignore"):
        d_liq = np.where(c.phi_act > 0.0, e_l * np.power(c.phi_act, e_l - 1.0), 0.0)
    g_phi_act = g_liq * d_liq
    g_phi_out = g_phi_out + g_phi_act * c.phi_act_range / max(1.0 - p.dens_phi_threshold, 1e-9)

    # phi_out = clip((T_new - t_pc)/dt_pc + 0.5, 0, 1)
    gT += g_phi_out * c.phi_out_range / p.dt_pc_c
    gT += g_Tk * c.Tk_active

    # --- T_new = clip(T + dT_step, tmin, tmax) -----------------------------
    g_Tc = gT * c.T_range_mask
    gT_in = np.array(g_Tc, copy=True)          # carried identity term
    g_dTs = g_Tc
    g_dTdt = g_dTs * c.dT_clip_mask * p.dt_sub

    # dTdt = num/den
    g_num = g_dTdt / c.den
    g_den = -g_dTdt * c.num / (c.den * c.den)
    g_prod = g_den * c.den_active
    g_rho_f = g_prod * c.cp_eff
    g_cp_eff = g_prod * c.rho_f
    g_cp_base = g_cp_eff                        # latent term has zero subgradient

    g_Q = np.array(g_num, copy=True)
    g_div = g_num
    g_qconv = -g_num
    gT_in += g_qconv * (p.h_const * case.expo)

    g_T_from_div, g_k_f = diffusion_vjp(g_div, c.T_in, c.k_f, case.dx, case.dy)
    gT_in += g_T_from_div

    # --- material fields back to phi_p and rho_in --------------------------
    rho_s_eff = p.rho_powder + c.rho_in * (p.rho_solid - p.rho_powder)
    k_s_eff = p.k_powder + c.rho_in * (p.k_solid - p.k_powder)
    g_rho_local = np.where(pm, g_rho_f, 0.0)
    g_k_local = np.where(pm, g_k_f, 0.0)
    g_cp_local = np.where(pm, g_cp_base, 0.0)

    g_phi_p = (g_rho_local * (p.rho_liquid - rho_s_eff)
               + g_k_local * (p.k_liquid - k_s_eff)
               + g_cp_local * (p.cp_liquid - p.cp_solid))
    gR_in += (g_rho_local * (1.0 - c.phi_p) * (p.rho_solid - p.rho_powder)
              + g_k_local * (1.0 - c.phi_p) * (p.k_solid - p.k_powder))

    # phi_p = clip((T - t_pc)/dt_pc + 0.5, 0, 1)
    argp = (c.T_in - p.t_pc_c) / p.dt_pc_c + 0.5
    m_p = (argp > 0.0) & (argp < 1.0)
    gT_in += g_phi_p * m_p / p.dt_pc_c

    return gT_in, gR_in, g_Q


# ---------------------------------------------------------------------------
# 2. reverse outer march
# ---------------------------------------------------------------------------

def reverse_march(case: Case, tr: fwd.Trajectory,
                  seeds: dict[int, np.ndarray]
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate the drive sensitivities in one backward sweep.

    Returns `(gRaw_a, gRaw_b, gP_step)`:

      gRaw_a, gRaw_b   dJ/d(Q_rf raw) for electrical state A and state B,
                       already carrying the power scale p_k and the cap
                       subgradient mask of every step that used that state.
                       This is what `eqs_vjp` needs, pre-masked.
      gP_step          dJ/dp evaluated PER OUTER STEP, the inner product of the
                       adjoint state with that step's injected-power pattern.
                       `schedule.accumulate_to_segments` folds it onto the
                       piecewise-constant segments; keeping it per step means
                       the segmentation is not baked into the physics.

    With no schedule (p_full is None) every p_k is exactly 1.0 and the mask is
    exactly the active set `eqs_vjp` used to compute internally, so the
    conductivity gradient is bit-for-bit unchanged.
    """
    p = case.pins
    ui = p.update_interval
    shape = case.part_mask.shape
    gQ_a = np.zeros(shape)
    gQ_b = np.zeros(shape)
    gT = np.zeros(shape)
    gR = np.zeros(shape)
    gP_step = np.zeros(max(tr.n_outer, 1))

    p_full = tr.p_full
    horizon = int(tr.p_horizon) if tr.p_horizon else tr.n_outer
    q_cache: dict[tuple[bool, int], tuple[np.ndarray, np.ndarray]] = {}

    last = max(seeds) if seeds else -1
    for it in range(last, -1, -1):
        if it in seeds:
            gT = gT + seeds[it]
        use_b = ui > 0 and it >= ui
        raw = tr.state_b.Qrf_raw if use_b else tr.state_a.Qrf_raw
        if p_full is None:
            p_k, key = 1.0, (use_b, 0)
        else:
            p_k = float(p_full[min(it, horizon - 1)])
            key = (use_b, sch.segment_index(it, horizon, int(tr.n_seg)))
        if key not in q_cache:
            q_cache[key] = fwd.scaled_Qrf(case, raw, p_k)
        Q, active = q_cache[key]
        T = tr.ckpt_T[it]
        rho = tr.ckpt_rho[it]
        caches: list[fwd.SubstepCache] = []
        for _ in range(p.n_substeps):
            T, rho, _phi, c = fwd.substep(T, rho, Q, case, keep_cache=True)
            caches.append(c)
        acc = np.zeros(shape)
        for c in reversed(caches):
            gT, gR, gQ = substep_vjp(c, case, gT, gR)
            acc += gQ
        acc_masked = acc * active
        # dQ/dp_k = raw where the cap does not bind; dQ/draw = p_k there.
        gP_step[it] = float(np.sum(acc_masked * raw))
        if use_b:
            gQ_b += p_k * acc_masked
        else:
            gQ_a += p_k * acc_masked
    return gQ_a, gQ_b, gP_step


# ---------------------------------------------------------------------------
# 3. EQS adjoint
# ---------------------------------------------------------------------------

def eqs_vjp(case: Case, st: fwd.ElectricState, gQ: np.ndarray,
            Gx, Gy, pre_masked: bool = False) -> np.ndarray:
    """dJ/dsigma for one electrical state, given dJ/dQ_rf.

    `pre_masked` says the caller has already applied the cap subgradient mask
    and any power scale, which is what `reverse_march` does when a temporal
    schedule is active (the mask then depends on the step's own p_k and cannot
    be reconstructed here).
    """
    p = case.pins
    # Q_rf = doped ? clip(max(raw,0), 0, max_qrf) : 0, raw = 0.5*pf*sigma*|E|^2
    if pre_masked:
        g_raw = np.asarray(gQ, dtype=float)
    else:
        active = case.doped_mask & (st.Qrf_raw > 0.0) & (st.Qrf_raw < p.max_qrf)
        g_raw = gQ * active

    # direct sigma dependence
    dJ_dsigma = g_raw * (0.5 * p.power_factor * st.e2)

    # dependence through the field
    coef = g_raw * (0.5 * p.power_factor * np.real(st.gamma))
    px = -(Gx.T @ (coef * np.conj(st.Ex)).ravel())
    py = -(Gy.T @ (coef * np.conj(st.Ey)).ravel())
    pvec = px + py
    if not np.any(pvec):
        return dJ_dsigma

    lam = st.op.solve_transpose(pvec).reshape(case.part_mask.shape)

    g = st.gamma
    free = st.op.free
    inv = {"up": st.op.inv_h2[1], "down": st.op.inv_h2[1],
           "left": st.op.inv_h2[0], "right": st.op.inv_h2[0]}
    V = st.V
    for d in eqs.DIRECTIONS:
        di, dj = eqs._OFFSET[d]
        ny, nx = case.part_mask.shape
        i0, i1 = max(0, -di), ny - max(0, di)
        j0, j1 = max(0, -dj), nx - max(0, dj)
        gk = g[i0:i1, j0:j1]
        gn = g[i0 + di:i1 + di, j0 + dj:j1 + dj]
        Vk = V[i0:i1, j0:j1]
        Vn = V[i0 + di:i1 + di, j0 + dj:j1 + dj]
        lk = lam[i0:i1, j0:j1]
        fk = free[i0:i1, j0:j1]
        den2 = (gk + gn) ** 2
        base = -lk * inv[d] * (Vk - Vn) * fk
        dgf_dk = 2.0 * gn * gn / den2
        dgf_dn = 2.0 * gk * gk / den2
        dJ_dsigma[i0:i1, j0:j1] += 2.0 * np.real(base * dgf_dk)
        np.add.at(dJ_dsigma,
                  (slice(i0 + di, i1 + di), slice(j0 + dj, j1 + dj)),
                  2.0 * np.real(base * dgf_dn))
    return dJ_dsigma


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------

def gradient(case: Case, s: np.ndarray, tr: fwd.Trajectory,
             seeds: dict[int, np.ndarray], grad_ops=None,
             with_schedule: bool = False):
    """dJ/ds over the whole domain (zero outside the part).

    With `with_schedule=True` also returns dJ/dp_k, the temporal power-schedule
    gradient, folded from the per-outer-step sensitivities onto the
    piecewise-constant segments. Both come out of the SAME backward sweep; the
    schedule gradient costs one extra inner product per outer step.
    """
    p = case.pins
    Gx, Gy = grad_ops if grad_ops is not None else gradops.gradient_matrices(case.x, case.y)
    gQ_a, gQ_b, gP_step = reverse_march(case, tr, seeds)

    ds = np.zeros(case.part_mask.shape)
    if np.any(gQ_a):
        dsig_a = eqs_vjp(case, tr.state_a, gQ_a, Gx, Gy, pre_masked=True)
        ds += dsig_a * case.fill_frac * (p.sigma_d0 - p.sigma_v)
    if np.any(gQ_b):
        dsig_b = eqs_vjp(case, tr.state_b, gQ_b, Gx, Gy, pre_masked=True)
        _sig_b, inrange = fwd.sigma_state_b(case, s)
        ds += dsig_b * inrange * p.sigma_d0
    if not with_schedule:
        return ds
    n_seg = int(tr.n_seg) if tr.n_seg else 1
    horizon = int(tr.p_horizon) if tr.p_horizon else tr.n_outer
    dp = sch.accumulate_to_segments(gP_step, horizon, n_seg)
    return ds, dp
