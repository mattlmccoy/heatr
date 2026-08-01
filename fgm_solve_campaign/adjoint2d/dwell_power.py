"""SECONDARY arm only: a power schedule p(t) on top of a dwell schedule.

Power scheduling is DEPRIORITIZED as an actuator. Dwell time is the
dose-steering knob and it is the one the hardware story is built around. This
module exists to run ONE labelled counterexample check, on the cross, because
that is the single shape where an optimized p(t) previously produced a real
generator OFF period whose benefit survived a dose-matched control
(`TEMPORAL_SCHEDULING_REPORT.md` Sections 5 and 8). The narrow question is
whether p(t) adds anything ON TOP OF an optimized dwell schedule, or whether
the dwell schedule has already taken that gain.

INJECTION CONVENTION, inherited unchanged from `schedule.py`: p multiplies
POWER, and because the electro-quasi-static problem is linear in the potential
at fixed material properties, that IS the drive change V -> V * sqrt(p) with no
re-solve.

THE ONE CAVEAT, and it is enforced rather than documented away. The dwell
kernel applies the `max_qrf` cap PER POSITION in the lab frame, before the
angle average. A power scale applied to the already-averaged part-frame field
therefore cannot re-apply that cap where it would bind. `scheduled_forward`
REFUSES to run when any position sits at the cap, so the convention is never
silently violated. At every operating point in this campaign the measured cap
fraction is zero.
"""
from __future__ import annotations

import numpy as np

from . import adjoint as adj, gradops
from . import forward as fwd
from . import schedule as sch
from .dwell_kernel import DwellKernel


def _assert_cap_slack(kern: DwellKernel) -> None:
    cap = kern.case0.pins.max_qrf
    for a in kern.per_angle:
        for st in (a.st_a, a.st_b):
            if st is not None and float(np.max(st.Qrf_raw)) >= cap:
                raise ValueError(
                    "the max_qrf cap binds at one of the candidate positions, "
                    "so a power scale on the averaged field is not the drive "
                    "change it claims to be; refuse rather than mis-scale.")


def scheduled_forward(kern: DwellKernel, s: np.ndarray, p_full: np.ndarray,
                      n_steps: int, keep_checkpoints: bool = False,
                      shape_stop_patience: int | None = None) -> fwd.Trajectory:
    """The dwell-weighted quasi-static march with a per-step power scale."""
    case = kern.case0
    p = case.pins
    n_steps = int(n_steps)
    pf = np.asarray(p_full, dtype=float).ravel()
    if pf.size < n_steps:
        raise ValueError(f"p_full has {pf.size} entries for {n_steps} steps")
    Q_a, Q_b = kern.averaged_Q(s)
    _assert_cap_slack(kern)

    T = np.full(case.part_mask.shape, p.ambient_c, dtype=float)
    rho = np.zeros(case.part_mask.shape, dtype=float)
    rho[case.part_mask] = p.rho_rel_init
    phi = np.zeros_like(T)

    ui = p.update_interval
    pm_e = case.part_mask
    rec_T, rec_ui, rec_phi, rec_rho = [], [], [], []
    rec_e_in, rec_e_out, rec_e_stored = [], [], []
    e_in_acc = e_out_acc = e_stored_acc = 0.0
    eb_T_prev, eb_phi_prev, eb_rho_prev = T.copy(), phi.copy(), rho.copy()
    frac_dT_max = frac_tc_max = 0.0
    stopped_early = False
    chi = case.part_mask.astype(float)
    j_min, j_first, j_argmin = np.inf, None, 0
    ck_T, ck_rho = [], []

    for it in range(n_steps):
        use_b = ui > 0 and it >= ui
        Q = float(pf[it]) * (Q_b if use_b else Q_a)
        ck_T.append(T.copy())
        if keep_checkpoints:
            ck_rho.append(rho.copy())
        p_conv_acc = 0.0
        for _ in range(p.n_substeps):
            T, rho, phi, c = fwd.substep(T, rho, Q, case, keep_cache=True)
            frac_dT_max = max(frac_dT_max, c.frac_dT_clipped)
            frac_tc_max = max(frac_tc_max, c.frac_temp_cap)
            p_conv_acc += c.p_conv_loss_w_per_m
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
        eb_T_prev, eb_phi_prev, eb_rho_prev = T.copy(), phi.copy(), rho.copy()
        rec_e_in.append(e_in_acc)
        rec_e_out.append(e_out_acc)
        rec_e_stored.append(float(e_stored_acc))

        t_avg, ui_rms, phi_bar, rho_bar = fwd._stats(T, phi, rho, case)
        rec_T.append(t_avg)
        rec_ui.append(ui_rms)
        rec_phi.append(phi_bar)
        rec_rho.append(rho_bar)
        if shape_stop_patience is not None:
            dphi = np.clip((T - p.t_pc_c) / p.dt_pc_c + 0.5, 0.0, 1.0) - chi
            j_now = float(np.sum(dphi * dphi))
            if j_first is None:
                j_first = j_now
            if j_now <= j_min:
                j_min, j_argmin = j_now, it
            if (j_min < j_first) and (it - j_argmin >= int(shape_stop_patience)):
                stopped_early = True
                break

    n_out = len(rec_T)
    arr_T = np.asarray(rec_T)
    arr_ui = np.asarray(rec_ui)
    dmask = case.doped_mask
    tr = fwd.Trajectory(
        time_s=(np.arange(n_out) + 1) * p.dt,
        mean_T_part_c=arr_T, ui_rms_part=arr_ui,
        mean_phi_part=np.asarray(rec_phi), mean_rho_rel_part=np.asarray(rec_rho),
        sigma_T=arr_ui * (arr_T - p.ambient_c),
        n_outer=n_out, stopped_early=stopped_early,
        T_final=T, rho_final=rho, phi_final=phi,
        P_abs_A=float(np.sum(Q_a[dmask]) * case.dA),
        P_abs_B=float(np.sum(Q_b[dmask]) * case.dA),
        frac_dT_clipped_max=frac_dT_max, frac_temp_cap_max=frac_tc_max,
        frac_qrf_cap=0.0,
        energy_in_J_per_m=np.asarray(rec_e_in, dtype=float),
        energy_out_J_per_m=np.asarray(rec_e_out, dtype=float),
        energy_stored_J_per_m=np.asarray(rec_e_stored, dtype=float),
        state_a=None, state_b=None, ckpt_T=ck_T, ckpt_rho=ck_rho,
    )
    tr.Q_avg_a = Q_a
    tr.Q_avg_b = Q_b
    tr.p_full_used = pf[:n_out].copy()
    return tr


def scheduled_gradients(kern: DwellKernel, s: np.ndarray, tr: fwd.Trajectory,
                        seeds: dict[int, np.ndarray], p_full: np.ndarray,
                        n_seg: int, grad_ops=None, n_window: int | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dJ/ds, dJ/dw, dJ/dp_seg), all three from ONE reverse march.

    With the cap slack, `dQ_step/dp_it = Q_avg` and `dQ_step/dQ_avg = p_it`, so
    the power gradient is one extra inner product per outer step and the other
    two are the unscheduled ones with the step's own scale folded in. The
    per-step power sensitivities are folded onto the piecewise-constant
    segments by the already-tested `schedule.accumulate_to_segments`.
    """
    case = kern.case0
    p = case.pins
    ui = p.update_interval
    shape = case.part_mask.shape
    pf = np.asarray(p_full, dtype=float).ravel()
    gQ_a = np.zeros(shape)
    gQ_b = np.zeros(shape)
    gT = np.zeros(shape)
    gR = np.zeros(shape)
    gP_step = np.zeros(max(tr.n_outer, 1))
    last = max(seeds) if seeds else -1
    for it in range(last, -1, -1):
        if it in seeds:
            gT = gT + seeds[it]
        use_b = ui > 0 and it >= ui
        Q_avg = tr.Q_avg_b if use_b else tr.Q_avg_a
        p_it = float(pf[min(it, pf.size - 1)])
        Q = p_it * Q_avg
        T = tr.ckpt_T[it]
        rho = tr.ckpt_rho[it]
        caches = []
        for _ in range(p.n_substeps):
            T, rho, _phi, c = fwd.substep(T, rho, Q, case, keep_cache=True)
            caches.append(c)
        acc = np.zeros(shape)
        for c in reversed(caches):
            gT, gR, gQ = adj.substep_vjp(c, case, gT, gR)
            acc += gQ
        gP_step[it] = float(np.sum(acc * Q_avg))
        if use_b:
            gQ_b += p_it * acc
        else:
            gQ_a += p_it * acc

    # the map and dwell gradients, from the same gQ fields
    Gx, Gy = (grad_ops if grad_ops is not None
              else gradops.gradient_matrices(case.x, case.y))
    gw = np.array([float(np.sum(gQ_a * qa) + np.sum(gQ_b * qb))
                   for qa, qb in zip(kern._Qk_a, kern._Qk_b)])
    ds = np.zeros(shape)
    for wk, a in zip(kern._w(), kern.per_angle):
        if wk == 0.0:
            continue
        pm_j = a.case.part_mask
        for gQ_part, st, active, is_b in ((gQ_a, a.st_a, a.active_a, False),
                                          (gQ_b, a.st_b, a.active_b, True)):
            if not np.any(gQ_part):
                continue
            g_lab = a.R_to_part.apply_T(gQ_part) * wk
            dsig = adj.eqs_vjp(a.case, st, g_lab * active, Gx, Gy, pre_masked=True)
            if is_b:
                g_slab = dsig * a.inrange_b * p.sigma_d0
            else:
                g_slab = dsig * a.case.fill_frac * (p.sigma_d0 - p.sigma_v)
            ds += a.R_to_lab.apply_T(np.where(pm_j, g_slab, 0.0))
    win = int(n_window if n_window is not None else tr.n_outer)
    gp = sch.accumulate_to_segments(gP_step, win, int(n_seg))
    return ds, gw, gp
