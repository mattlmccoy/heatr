"""The SEQUENTIAL dwell march in the PART frame, and its two gradients.

WHAT IT ADDS TO `dwell_march.py`. That module executes a program by choosing
ONE stored per-position heating field per outer step, which forces every switch
onto the control grid and makes the objective a step function of the switch
times, with a gradient that is zero almost everywhere. Here a step that
STRADDLES a switch gets the exact time average over that step,

    Q_i = sum_k f_ik Q_(a_k)

with `f` the overlap matrix of `seq_dwell.step_mix`. On switch times that lie
on the control grid this reduces to the same one-hot selection, bit for bit
(`tests/test_seq_dwell_march.py`), so nothing is changed about the physics and
the only new content is that the switch times are now continuous variables.

THE TWO GRADIENTS COME OUT OF ONE REVERSE MARCH, exactly as in `dwell_kernel`.
The reverse sweep produces `acc_i = dJ/dQ_i` per outer step. From there:

    dJ/df_ik = <acc_i, Q_(a_k)>            (then `seq_dwell.mix_vjp` to durations)
    dJ/dQ_a  = sum over steps and segments at angle a of f_ik acc_i
               (then the per-position electro-quasi-static adjoint chain to s)

The heating state (A before the first `update_interval` tick, B after) is a
property of the STEP, so the same segment contributes to different stored
fields in different steps. That is carried explicitly rather than approximated.

WHAT IS NOT MODELLED. The move between positions is instantaneous and free.
The part frame is exact for the design, and its agreement with the production
engine is established separately (`DWELL_SCHEDULE_REPORT.md` Section 4, J to
0.09 percent on the schedules both can express).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import adjoint as adj, gradops
from . import forward as fwd
from . import seq_dwell as sq
from .dwell_kernel import DwellKernel


def _mixed_Q(kern: DwellKernel, seg_angle_idx: Sequence[int], f_row: np.ndarray,
             use_b: bool) -> np.ndarray:
    """The heating injected at one outer step: the exact within-step average."""
    store = kern._Qk_b if use_b else kern._Qk_a
    out = None
    for k, a in enumerate(seg_angle_idx):
        w = float(f_row[k])
        if w == 0.0:
            continue
        out = (w * store[a]) if out is None else out + w * store[a]
    return np.zeros(kern.case0.part_mask.shape) if out is None else out


def sequential_forward(kern: DwellKernel, seg_angle_idx: Sequence[int],
                       durations: Sequence[float], n_steps: int, *,
                       keep_checkpoints: bool = False,
                       shape_stop_patience: int | None = None) -> fwd.Trajectory:
    """March the thermal problem through an ORDERED list of holds.

    `kern` must already hold the per-position part-frame heating fields, that
    is `DwellKernel.averaged_Q(s)` (or `.forward(s)`) must have been run for the
    map being scored. Nothing is interpolated and no field is rotated here.
    """
    if not kern._Qk_a:
        raise RuntimeError(
            "sequential_forward needs the per-position heating fields; call "
            "DwellKernel.averaged_Q(s) (or .forward(s)) for this map first.")
    seg = [int(a) for a in seg_angle_idx]
    if any(a < 0 or a >= kern.n_angles for a in seg):
        raise ValueError(f"segment angle indices {seg} outside the candidate set")
    case = kern.case0
    p = case.pins
    n_steps = int(n_steps)
    f = sq.step_mix(durations, p.dt, n_steps)
    if f.shape[1] != len(seg):
        raise ValueError(f"{f.shape[1]} durations against {len(seg)} segments")

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
    ck_T: list[np.ndarray] = []
    ck_rho: list[np.ndarray] = []

    for it in range(n_steps):
        use_b = ui > 0 and it >= ui
        Q = _mixed_Q(kern, seg, f[it], use_b)
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
    dm_mask = case.doped_mask
    seg_frac = f[:n_out].sum(axis=0) / max(n_out, 1)
    w_ang = np.zeros(kern.n_angles)
    for k, a in enumerate(seg):
        w_ang[a] += seg_frac[k]
    Q_eff_a = sum(w * q for w, q in zip(w_ang, kern._Qk_a))
    Q_eff_b = sum(w * q for w, q in zip(w_ang, kern._Qk_b))
    tr = fwd.Trajectory(
        time_s=(np.arange(n_out) + 1) * p.dt,
        mean_T_part_c=arr_T, ui_rms_part=arr_ui,
        mean_phi_part=np.asarray(rec_phi), mean_rho_rel_part=np.asarray(rec_rho),
        sigma_T=arr_ui * (arr_T - p.ambient_c),
        n_outer=n_out, stopped_early=stopped_early,
        T_final=T, rho_final=rho, phi_final=phi,
        P_abs_A=float(np.sum(Q_eff_a[dm_mask]) * case.dA),
        P_abs_B=float(np.sum(Q_eff_b[dm_mask]) * case.dA),
        frac_dT_clipped_max=frac_dT_max, frac_temp_cap_max=frac_tc_max,
        frac_qrf_cap=0.0,
        energy_in_J_per_m=np.asarray(rec_e_in, dtype=float),
        energy_out_J_per_m=np.asarray(rec_e_out, dtype=float),
        energy_stored_J_per_m=np.asarray(rec_e_stored, dtype=float),
        state_a=None, state_b=None, ckpt_T=ck_T, ckpt_rho=ck_rho,
    )
    tr.Q_avg_a = Q_eff_a
    tr.Q_avg_b = Q_eff_b
    tr.seq_mix = f
    tr.seq_angle_idx = tuple(seg)
    tr.seq_durations = np.asarray(durations, dtype=float).copy()
    tr.seq_segment_fractions = seg_frac
    tr.realized_weights_executed = w_ang
    return tr


# ---------------------------------------------------------------------------
# the reverse march
# ---------------------------------------------------------------------------

def _reverse_march(kern: DwellKernel, tr: fwd.Trajectory,
                   seeds: dict[int, np.ndarray],
                   seeds_rho: dict[int, np.ndarray] | None = None
                   ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Per-step `dJ/dQ_i`, plus the per-angle accumulations for each state.

    Structurally `AveragedKernel._reverse_march`, with the single averaged
    heating array replaced by the per-step mix and with the per-step
    accumulation kept, because the duration gradient needs it step by step.
    """
    case = kern.case0
    p = case.pins
    ui = p.update_interval
    shape = case.part_mask.shape
    seg = list(tr.seq_angle_idx)
    f = tr.seq_mix
    gQ_ang_a = [np.zeros(shape) for _ in range(kern.n_angles)]
    gQ_ang_b = [np.zeros(shape) for _ in range(kern.n_angles)]
    g_mix = np.zeros((f.shape[0], len(seg)))
    gT = np.zeros(shape)
    gR = np.zeros(shape)
    sr = seeds_rho or {}
    last = max(list(seeds) + list(sr)) if (seeds or sr) else -1
    for it in range(last, -1, -1):
        if it in seeds:
            gT = gT + seeds[it]
        if it in sr:
            gR = gR + sr[it]
        use_b = ui > 0 and it >= ui
        Q = _mixed_Q(kern, seg, f[it], use_b)
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
        store = kern._Qk_b if use_b else kern._Qk_a
        tgt = gQ_ang_b if use_b else gQ_ang_a
        for k, a in enumerate(seg):
            g_mix[it, k] = float(np.sum(acc * store[a]))
            w = float(f[it, k])
            if w != 0.0:
                tgt[a] += w * acc
    return g_mix, gQ_ang_a, gQ_ang_b


def sequential_gradients(kern: DwellKernel, s: np.ndarray, tr: fwd.Trajectory,
                         seeds: dict[int, np.ndarray], grad_ops=None,
                         seeds_rho: dict[int, np.ndarray] | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """(dJ/ds, dJ/d(durations)) from ONE reverse march."""
    if kern._cache_s is None or not np.array_equal(
            kern._cache_s, np.asarray(s, dtype=float)):
        raise RuntimeError(
            "sequential_gradients must be called against the map of the most "
            "recent forward; the cached per-position electrical states belong "
            "to a different map.")
    if not hasattr(tr, "seq_mix"):
        raise RuntimeError("that trajectory did not come from sequential_forward")
    case = kern.case0
    p = case.pins
    Gx, Gy = (grad_ops if grad_ops is not None
              else gradops.gradient_matrices(case.x, case.y))
    g_mix, gQ_a, gQ_b = _reverse_march(kern, tr, seeds, seeds_rho)
    g_d = sq.mix_vjp(g_mix, tr.seq_durations, p.dt, g_mix.shape[0])

    ds = np.zeros(case.part_mask.shape)
    for a_i, a in enumerate(kern.per_angle):
        pm_j = a.case.part_mask
        for gQ_part, st, active, is_b in ((gQ_a[a_i], a.st_a, a.active_a, False),
                                          (gQ_b[a_i], a.st_b, a.active_b, True)):
            if not np.any(gQ_part):
                continue
            g_lab = a.R_to_part.apply_T(gQ_part)
            dsig = adj.eqs_vjp(a.case, st, g_lab * active, Gx, Gy, pre_masked=True)
            if is_b:
                g_slab = dsig * a.inrange_b * p.sigma_d0
            else:
                g_slab = dsig * a.case.fill_frac * (p.sigma_d0 - p.sigma_v)
            ds += a.R_to_lab.apply_T(np.where(pm_j, g_slab, 0.0))
    return ds, g_d
