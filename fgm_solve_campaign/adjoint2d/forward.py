"""Differentiable re-implementation of the production 2-D forward march.

The design variable is the per-cell binder saturation `s` (dimensionless,
`sigma = sigma_d0 * s` inside the part). It is injected exactly the way the
production two-sided per-node hook `fgm_feedback.sat_map_npz_direct` injects it:
permittivity blends by GEOMETRY FILL ONLY, so `s` moves conductivity and not
permittivity (`rfam_eqs_coupled.py:342`, `eps_geometry_only = True`).

Two electrical states exist in the pinned configuration, and only two, because
the conductivity temperature and density coefficients are both zero:

  state A  the startup assembly, `sigma = sigma_v + (fill*s)*(sigma_d0-sigma_v)`
           used for outer steps 0 .. update_interval-1
  state B  the re-assembly performed on every `update_interval` tick,
           `sigma[part] = clip(sigma_d0*s, 1e-4*sigma_d0, 25*sigma_d0)`
           used from outer step `update_interval` to the end

The production engine recomputes state B on every tick, which reproduces the
same numbers; the prototype computes it once. That is a caching change, not a
physics change, and it is covered by the L0 bit-identity gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import eqs, schedule as sch
from .pins import EPS0, R_GAS, Case, Pins
from .prod import rfam


# ----------------------------------------------------------------------------
# the temporal power-scheduling actuator
# ----------------------------------------------------------------------------

def scaled_Qrf(case: Case, Qrf_raw: np.ndarray, p_k: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Injected heating for a drive at power scale `p_k`, and its active mask.

    CONVENTION (see `schedule.py`): p multiplies POWER. The EQS problem is
    linear in the potential at fixed material properties, so p is exactly the
    drive change V -> V*sqrt(p) and Q_rf scales by p with no re-solve. The
    `max_qrf` cap and the doped mask are applied AFTER the scaling, which is
    where they would act on a real re-solve.

    The mask is the subgradient gate: the cells where Q_rf still responds to
    the drive, that is where the scaled raw value is strictly inside the cap.
    """
    p = case.pins
    raw = float(p_k) * np.asarray(Qrf_raw, dtype=float)
    Q = np.clip(raw, 0.0, p.max_qrf)
    # The doped mask is part of the active set in every pinned configuration
    # (`zero_qrf_outside_doped` is true), matching `adjoint.eqs_vjp`.
    active = case.doped_mask & (raw > 0.0) & (raw < p.max_qrf)
    if p.zero_qrf_outside_doped:
        Q = np.where(case.doped_mask, Q, 0.0)
    return Q, active


# ----------------------------------------------------------------------------
# electrical state
# ----------------------------------------------------------------------------

@dataclass
class ElectricState:
    sigma: np.ndarray
    eps_r: np.ndarray
    gamma: np.ndarray
    op: eqs.EqsOperator
    V: np.ndarray
    Ex: np.ndarray
    Ey: np.ndarray
    e2: np.ndarray          # |Ex|^2 + |Ey|^2, real
    Qrf_raw: np.ndarray     # before the max_qrf clip and the doped mask
    Qrf: np.ndarray
    clip_hi_active: np.ndarray
    pos_active: np.ndarray


def sigma_state_a(case: Case, s: np.ndarray, float32_sat: bool) -> np.ndarray:
    p = case.pins
    eff = case.fill_frac * s
    if float32_sat:
        eff = eff.astype(np.float32)
    return p.sigma_v + eff * (p.sigma_d0 - p.sigma_v)


def sigma_state_b(case: Case, s: np.ndarray, float32_sat: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """State-B conductivity.

    The production `sigma` array inherits float32 from the state-A expression
    (`sigma_v + eff_fill*(sigma_d0-sigma_v)` with `eff_fill` cast to float32 by
    `_FgmFeedback.effective_fill`), so the state-B in-place assignment is also
    rounded to float32. That rounding is reproduced only in identity mode; the
    differentiable mode keeps float64 so the design variable enters smoothly.
    """
    p = case.pins
    dtype = np.float32 if float32_sat else np.float64
    sig = np.full(case.part_mask.shape, p.sigma_v, dtype=dtype)
    raw = p.sigma_d0 * s[case.part_mask]
    lo, hi = 1e-4 * p.sigma_d0, 25.0 * p.sigma_d0
    inrange = (raw > lo) & (raw < hi)
    sig[case.part_mask] = np.clip(raw, lo, hi)
    inrange_full = np.zeros(case.part_mask.shape, dtype=bool)
    inrange_full[case.part_mask] = inrange
    return sig, inrange_full


def eps_field(case: Case, s: np.ndarray | None = None, covary: bool = False,
              float32_sat: bool = False) -> np.ndarray:
    """Permittivity field.

    DEFAULT (covary = False) reproduces the two-sided per-node direct hook,
    where permittivity blends by GEOMETRY FILL ONLY
    (`rfam_eqs_coupled.py:342`, `eps_geometry_only = True`). This is the
    production behaviour of the hook the prototype uses and it is unchanged.

    OPT-IN (covary = True) reproduces the OTHER production hook,
    `fgm_feedback.saturation_map_npz`, in which the saturation scales
    permittivity as well as conductivity. That is the channel the step-2
    calibrated control actually ran in, so it is needed to put the strongest
    published control on the table. It is a separate, clearly-labelled arm and
    it is never the default.
    """
    p = case.pins
    if not covary:
        return p.eps_v + case.fill_frac * (p.eps_d - p.eps_v)
    if s is None:  # pragma: no cover
        raise ValueError("covary requires the saturation field")
    eff = case.fill_frac * s
    if float32_sat:
        eff = eff.astype(np.float32)
    return p.eps_v + eff * (p.eps_d - p.eps_v)


def solve_electric(case: Case, sigma: np.ndarray, eps_r: np.ndarray) -> ElectricState:
    p = case.pins
    gamma = np.asarray(sigma, dtype=float) + 1j * p.omega * EPS0 * np.asarray(eps_r, dtype=float)
    op = eqs.assemble(gamma, case.elec_hi, case.elec_lo, p.v_hi, p.v_lo, case.dx, case.dy)
    V = op.solve()
    Ex, Ey, _E_mag, Qrf_raw = rfam.compute_electric_fields(V, case.x, case.y, gamma, p.power_factor)
    e2 = np.real(Ex * np.conj(Ex) + Ey * np.conj(Ey))
    pos_active = (0.5 * p.power_factor * np.real(gamma * (Ex * np.conj(Ex) + Ey * np.conj(Ey)))) > 0.0
    Qrf = np.clip(np.asarray(Qrf_raw, dtype=float), 0.0, p.max_qrf)
    clip_hi_active = Qrf_raw >= p.max_qrf
    if p.zero_qrf_outside_doped:
        Qrf = np.where(case.doped_mask, Qrf, 0.0)
    return ElectricState(
        sigma=sigma, eps_r=eps_r, gamma=gamma, op=op, V=V, Ex=Ex, Ey=Ey, e2=e2,
        Qrf_raw=np.asarray(Qrf_raw, dtype=float), Qrf=Qrf,
        clip_hi_active=clip_hi_active, pos_active=pos_active,
    )


# ----------------------------------------------------------------------------
# phase fraction (COMSOL Heaviside, linear ramp) -- matches rfam.phase_fraction
# ----------------------------------------------------------------------------

def phase_fraction(T: np.ndarray, p: Pins) -> tuple[np.ndarray, np.ndarray]:
    T_eval = np.array(T, dtype=float, copy=True)
    arg = (T_eval - p.t_pc_c) / p.dt_pc_c
    phi = np.clip(arg + 0.5, 0.0, 1.0)
    dphi_dT = np.where(np.abs(arg) <= 0.5, 1.0 / p.dt_pc_c, 0.0)
    return phi, dphi_dT


# ----------------------------------------------------------------------------
# one thermal substep, with the cache the reverse sweep needs
# ----------------------------------------------------------------------------

@dataclass
class SubstepCache:
    T_in: np.ndarray
    rho_in: np.ndarray
    phi_p: np.ndarray
    dphi_dT: np.ndarray
    rho_f: np.ndarray
    k_f: np.ndarray
    cp_eff: np.ndarray
    den: np.ndarray
    den_active: np.ndarray
    num: np.ndarray
    dT_clip_mask: np.ndarray
    T_range_mask: np.ndarray
    T_new: np.ndarray
    phi_out: np.ndarray
    phi_out_range: np.ndarray
    Tk: np.ndarray
    Tk_active: np.ndarray
    rate: np.ndarray
    kdens_ss: np.ndarray
    kdens_liq: np.ndarray
    ss_drive: np.ndarray
    liq_drive: np.ndarray
    phi_act: np.ndarray
    phi_act_range: np.ndarray
    rho_term: np.ndarray
    drho_clip_mask: np.ndarray
    rho_new_range: np.ndarray
    frac_dT_clipped: float
    frac_temp_cap: float
    p_conv_loss_w_per_m: float


def substep(T: np.ndarray, rho: np.ndarray, Q: np.ndarray, case: Case,
            keep_cache: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, SubstepCache | None]:
    p = case.pins
    pm = case.part_mask
    phi_p, dphi_dT = phase_fraction(T, p)

    rho_s_eff = p.rho_powder + rho * (p.rho_solid - p.rho_powder)
    k_s_eff = p.k_powder + rho * (p.k_solid - p.k_powder)
    rho_local = (1.0 - phi_p) * rho_s_eff + phi_p * p.rho_liquid
    k_local = (1.0 - phi_p) * k_s_eff + phi_p * p.k_liquid
    cp_local = (1.0 - phi_p) * p.cp_solid + phi_p * p.cp_liquid

    rho_f = np.full_like(T, p.rho_powder)
    k_f = np.full_like(T, p.k_powder)
    cp_base = np.full_like(T, p.cp_powder)
    rho_f[pm] = rho_local[pm]
    k_f[pm] = k_local[pm]
    cp_base[pm] = cp_local[pm]
    cp_eff = cp_base + p.latent_heat * dphi_dT

    div_term = rfam.diffusion_divergence(T, k_f, case.dx, case.dy)
    q_conv = p.h_const * case.expo * (T - p.ambient_c)

    num = div_term + Q - q_conv
    prod = rho_f * cp_eff
    den = np.maximum(prod, 1e-9)
    den_active = prod > 1e-9
    dTdt = num / den
    dTdt = np.nan_to_num(dTdt, nan=0.0, posinf=0.0, neginf=0.0)

    dT_step_raw = np.asarray(p.dt_sub * dTdt, dtype=float)
    cap = abs(float(p.max_dt_step_c))
    dT_step = np.clip(dT_step_raw, -cap, cap)
    dT_clip_mask = np.abs(dT_step_raw) <= cap
    T_candidate = np.asarray(T, dtype=float) + dT_step
    frac_temp_cap = float(np.mean((T_candidate > p.temp_max_c) | (T_candidate < p.temp_min_c)))
    T_new = np.clip(T_candidate, p.temp_min_c, p.temp_max_c)
    T_range_mask = (T_candidate >= p.temp_min_c) & (T_candidate <= p.temp_max_c)

    phi_out_raw = (T_new - p.t_pc_c) / p.dt_pc_c + 0.5
    phi_out = np.clip(phi_out_raw, 0.0, 1.0)
    phi_out_range = (phi_out_raw > 0.0) & (phi_out_raw < 1.0)

    Tk_raw = np.array(T_new, dtype=float, copy=True) + 273.15
    Tk = np.maximum(Tk_raw, 1.0)
    Tk_active = Tk_raw > 1.0

    one_minus_rho = np.clip(1.0 - rho, 0.0, 1.0)
    rho_term = np.power(one_minus_rho, p.dens_rho_exponent)

    kdens_ss = p.dens_k0_ss * np.exp(-p.dens_ea_ss / (R_GAS * Tk))
    ss_drive = np.power(np.clip(1.0 - phi_out, 0.0, 1.0), p.dens_phi_solid_exponent)
    phi_thr = p.dens_phi_threshold
    phi_act_raw = (phi_out - phi_thr) / max(1.0 - phi_thr, 1e-9)
    phi_act = np.clip(phi_act_raw, 0.0, 1.0)
    phi_act_range = (phi_act_raw > 0.0) & (phi_act_raw < 1.0)
    liq_drive = np.power(phi_act, p.dens_phi_liq_exponent)

    eta_ref = max(p.dens_eta_ref_pa_s, 1e-9)
    eta_ea = max(p.dens_eta_activation_j_per_mol, 0.0)
    t_ref = max(p.dens_eta_ref_temp_k, 1.0)
    eta = eta_ref * np.exp(eta_ea / R_GAS * (1.0 / Tk - 1.0 / t_ref))
    kdens_liq = p.dens_geom_factor * p.dens_surface_tension / (
        np.maximum(eta, 1e-12) * max(p.dens_particle_radius_m, 1e-9)
    )
    rate = (kdens_ss * ss_drive + kdens_liq * liq_drive) * rho_term

    drho_raw = p.dt_sub * rate
    drho = np.clip(drho_raw, 0.0, p.dens_max_delta_per_substep)
    drho_clip_mask = (drho_raw > 0.0) & (drho_raw < p.dens_max_delta_per_substep)

    rho_new = np.array(rho, copy=True)
    cand = rho_new[pm] + drho[pm]
    rho_new[pm] = np.clip(cand, 0.0, 1.0)
    rho_new_range = np.zeros_like(pm)
    rho_new_range[pm] = (cand > 0.0) & (cand < 1.0)

    frac_dT = float(np.mean(np.abs(dT_step_raw) > cap))

    cache = None
    if keep_cache:
        cache = SubstepCache(
            T_in=T, rho_in=rho, phi_p=phi_p, dphi_dT=dphi_dT, rho_f=rho_f, k_f=k_f,
            cp_eff=cp_eff, den=den, den_active=den_active, num=num,
            dT_clip_mask=dT_clip_mask, T_range_mask=T_range_mask, T_new=T_new,
            phi_out=phi_out, phi_out_range=phi_out_range, Tk=Tk, Tk_active=Tk_active,
            rate=rate, kdens_ss=kdens_ss, kdens_liq=kdens_liq, ss_drive=ss_drive,
            liq_drive=liq_drive, phi_act=phi_act, phi_act_range=phi_act_range,
            rho_term=rho_term, drho_clip_mask=drho_clip_mask, rho_new_range=rho_new_range,
            frac_dT_clipped=frac_dT, frac_temp_cap=frac_temp_cap,
            # `rfam_eqs_coupled.py:2108`, the convective-loss power the energy
            # bookkeeping integrates. Only the positive part is counted, exactly
            # as production does.
            p_conv_loss_w_per_m=float(np.sum(np.maximum(q_conv, 0.0)) * case.dA),
        )
    return T_new, rho_new, phi_out, cache


# ----------------------------------------------------------------------------
# outer march
# ----------------------------------------------------------------------------

@dataclass
class Trajectory:
    time_s: np.ndarray
    mean_T_part_c: np.ndarray
    ui_rms_part: np.ndarray
    mean_phi_part: np.ndarray
    mean_rho_rel_part: np.ndarray
    sigma_T: np.ndarray
    n_outer: int
    stopped_early: bool
    T_final: np.ndarray
    rho_final: np.ndarray
    phi_final: np.ndarray
    P_abs_A: float
    P_abs_B: float
    frac_dT_clipped_max: float
    frac_temp_cap_max: float
    frac_qrf_cap: float
    # Standing energy-residual gate bookkeeping, one entry per outer step.
    # residual = energy_in - energy_out - energy_stored (rfam_eqs_coupled.py:3217).
    energy_in_J_per_m: np.ndarray = field(repr=False, default=None)
    energy_out_J_per_m: np.ndarray = field(repr=False, default=None)
    energy_stored_J_per_m: np.ndarray = field(repr=False, default=None)
    # Temporal power-scheduling actuator. `p_seg` is the design vector,
    # `p_full` its per-outer-step expansion over the NOMINAL horizon.
    p_seg: np.ndarray = field(repr=False, default=None)
    n_seg: int = 0
    p_horizon: int = 0
    p_full: np.ndarray = field(repr=False, default=None)
    state_a: ElectricState = field(repr=False, default=None)
    state_b: ElectricState = field(repr=False, default=None)
    ckpt_T: list = field(repr=False, default_factory=list)
    ckpt_rho: list = field(repr=False, default_factory=list)

    def T_at_end(self, n: int) -> np.ndarray:
        """Temperature field at the END of outer step n (0-based)."""
        if n == self.n_outer - 1:
            return self.T_final
        if not self.ckpt_T:
            raise RuntimeError("trajectory was run without checkpoints")
        return self.ckpt_T[n + 1]


def _stats(T: np.ndarray, phi: np.ndarray, rho: np.ndarray, case: Case) -> tuple[float, float, float, float]:
    pm = case.part_mask
    tp = T[pm]
    t_avg = float(np.mean(tp))
    denom = max(t_avg - case.pins.ambient_c, 1e-9)
    ui_rms = float(np.sqrt(np.mean((tp - t_avg) ** 2)) / denom)
    return t_avg, ui_rms, float(np.mean(phi[pm])), float(np.mean(rho[pm]))


def forward(
    case: Case,
    s: np.ndarray,
    *,
    float32_sat: bool = False,
    keep_checkpoints: bool = False,
    stop_after_phi: float | None = 0.90,
    stop_margin_steps: int = 2,
    n_steps: int | None = None,
    eps_covary: bool = False,
    shape_stop_patience: int | None = None,
    p_seg: np.ndarray | None = None,
    n_seg: int | None = None,
    p_horizon: int | None = None,
) -> Trajectory:
    p = case.pins
    n_steps = int(p.n_steps if n_steps is None else n_steps)

    # --- temporal power schedule -------------------------------------------
    # p_seg is None means "no schedule", and that path is left BIT FOR BIT
    # unchanged so every previously reported number is unaffected.
    if p_seg is None:
        p_full = None
        n_seg_i = 0
        horizon = int(n_steps)
    else:
        n_seg_i = int(len(p_seg) if n_seg is None else n_seg)
        horizon = int(n_steps if p_horizon is None else p_horizon)
        p_full = sch.expand(p_seg, horizon, n_seg_i)

    eps_r = eps_field(case, s, covary=eps_covary, float32_sat=float32_sat)
    st_a = solve_electric(case, sigma_state_a(case, s, float32_sat), eps_r)
    sig_b, _inrange = sigma_state_b(case, s, float32_sat)
    st_b = solve_electric(case, sig_b, eps_r)

    T = np.full(case.part_mask.shape, p.ambient_c, dtype=float)
    rho = np.zeros(case.part_mask.shape, dtype=float)
    rho[case.part_mask] = p.rho_rel_init
    phi = np.zeros_like(T)

    ui = p.update_interval
    pm_e = case.part_mask
    rec_T: list[float] = []
    rec_ui: list[float] = []
    rec_phi: list[float] = []
    rec_rho: list[float] = []
    ck_T: list[np.ndarray] = []
    ck_rho: list[np.ndarray] = []
    frac_dT_max = 0.0
    frac_tc_max = 0.0
    cross_step: int | None = None
    stopped_early = False

    # --- standing energy-residual gate bookkeeping -------------------------
    # Reproduces rfam_eqs_coupled.py:3149-3163 and :3210-3217. The stored energy
    # is accumulated incrementally with BEGINNING-of-outer-step material
    # properties, which is what makes it consistent with the Forward Euler march.
    rec_e_in: list[float] = []
    rec_e_out: list[float] = []
    rec_e_stored: list[float] = []
    e_in_acc = 0.0
    e_out_acc = 0.0
    e_stored_acc = 0.0
    eb_T_prev = T.copy()
    eb_phi_prev = phi.copy()
    eb_rho_prev = rho.copy()

    # Shape-fidelity early stop. J(t) = sum over the whole domain of
    # (phi - chi)^2 falls while the part melts and rises once melt spills into
    # the bed, so the march can be truncated `patience` steps after the running
    # minimum. The truncation is exact for the objective as long as the minimum
    # is interior, which `optimal_stop` re-checks and reports.
    chi = case.part_mask.astype(float)
    j_min = np.inf
    j_first: float | None = None
    j_argmin = 0
    shape_stopped = False

    Q = st_a.Qrf
    q_cache: dict[tuple[bool, int], np.ndarray] = {}
    for it in range(n_steps):
        if p_full is None:
            if ui > 0 and it > 0 and (it % ui == 0):
                Q = st_b.Qrf
        else:
            use_b = ui > 0 and it >= ui
            k = sch.segment_index(it, horizon, n_seg_i)
            key = (use_b, k)
            if key not in q_cache:
                raw = (st_b if use_b else st_a).Qrf_raw
                q_cache[key] = scaled_Qrf(case, raw, float(p_full[min(it, horizon - 1)]))[0]
            Q = q_cache[key]
        # T checkpoints are always kept: the objectives read T at outer-step
        # ends. rho checkpoints are only needed to restart the reverse sweep.
        ck_T.append(T.copy())
        if keep_checkpoints:
            ck_rho.append(rho.copy())
        p_conv_acc = 0.0
        for _ in range(p.n_substeps):
            T, rho, phi, c = substep(T, rho, Q, case, keep_cache=True)
            frac_dT_max = max(frac_dT_max, c.frac_dT_clipped)
            frac_tc_max = max(frac_tc_max, c.frac_temp_cap)
            p_conv_acc += c.p_conv_loss_w_per_m

        # Q is constant across the substeps of one outer step, so the production
        # substep average of p_qrf reduces exactly to sum(Q)*dA.
        e_in_acc += float(np.sum(np.maximum(Q, 0.0)) * case.dA) * p.dt
        e_out_acc += (p_conv_acc / float(p.n_substeps)) * p.dt
        d_T = T - eb_T_prev
        d_phi = phi - eb_phi_prev
        rs_bos = p.rho_powder + eb_rho_prev * (p.rho_solid - p.rho_powder)
        rloc_bos = np.where(pm_e, (1.0 - eb_phi_prev) * rs_bos + eb_phi_prev * p.rho_liquid,
                            p.rho_powder)
        cploc_bos = np.where(pm_e, (1.0 - eb_phi_prev) * p.cp_solid + eb_phi_prev * p.cp_liquid,
                             p.cp_powder)
        e_stored_acc += (np.sum(rloc_bos * cploc_bos * d_T)
                         + np.sum(np.where(pm_e, rloc_bos * p.latent_heat * d_phi, 0.0))) * case.dA
        eb_T_prev = T.copy()
        eb_phi_prev = phi.copy()
        eb_rho_prev = rho.copy()
        rec_e_in.append(e_in_acc)
        rec_e_out.append(e_out_acc)
        rec_e_stored.append(float(e_stored_acc))

        t_avg, ui_rms, phi_bar, rho_bar = _stats(T, phi, rho, case)
        rec_T.append(t_avg)
        rec_ui.append(ui_rms)
        rec_phi.append(phi_bar)
        rec_rho.append(rho_bar)
        if shape_stop_patience is not None:
            dphi = np.clip((T - p.t_pc_c) / p.dt_pc_c + 0.5, 0.0, 1.0) - chi
            j_now = float(np.sum(dphi * dphi))
            if j_first is None:
                j_first = j_now
            # `<=` so a flat prefix (nothing melted yet, J pinned at the part
            # cell count) does not freeze the running argmin and trip the
            # patience counter before the objective has started to move.
            if j_now <= j_min:
                j_min, j_argmin = j_now, it
            if (j_min < j_first) and (it - j_argmin >= int(shape_stop_patience)):
                shape_stopped = True
                stopped_early = True
                break
        if stop_after_phi is not None:
            if cross_step is None and phi_bar >= stop_after_phi:
                cross_step = it
            if cross_step is not None and it >= cross_step + stop_margin_steps:
                stopped_early = True
                break

    n_out = len(rec_T)
    arr_T = np.asarray(rec_T)
    arr_ui = np.asarray(rec_ui)
    q_all = np.concatenate([st_a.Qrf_raw[case.doped_mask], st_b.Qrf_raw[case.doped_mask]])
    return Trajectory(
        time_s=(np.arange(n_out) + 1) * p.dt,
        mean_T_part_c=arr_T,
        ui_rms_part=arr_ui,
        mean_phi_part=np.asarray(rec_phi),
        mean_rho_rel_part=np.asarray(rec_rho),
        sigma_T=arr_ui * (arr_T - p.ambient_c),
        n_outer=n_out,
        stopped_early=stopped_early or shape_stopped,
        T_final=T, rho_final=rho, phi_final=phi,
        P_abs_A=float(np.sum(st_a.Qrf[case.doped_mask]) * case.dA),
        P_abs_B=float(np.sum(st_b.Qrf[case.doped_mask]) * case.dA),
        frac_dT_clipped_max=frac_dT_max,
        frac_temp_cap_max=frac_tc_max,
        frac_qrf_cap=float(np.mean(q_all >= p.max_qrf)),
        energy_in_J_per_m=np.asarray(rec_e_in, dtype=float),
        energy_out_J_per_m=np.asarray(rec_e_out, dtype=float),
        energy_stored_J_per_m=np.asarray(rec_e_stored, dtype=float),
        p_seg=None if p_seg is None else np.asarray(p_seg, dtype=float).copy(),
        n_seg=n_seg_i, p_horizon=horizon, p_full=p_full,
        state_a=st_a, state_b=st_b,
        ckpt_T=ck_T, ckpt_rho=ck_rho,
    )
