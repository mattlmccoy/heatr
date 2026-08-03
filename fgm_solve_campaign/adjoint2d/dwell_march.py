"""TIME-RESOLVED execution of a dwell program, in the PART frame.

WHY THIS EXISTS. The dwell solve is done against the quasi-static weighted
angle average, which is the limit of an infinitely fast cycle. That limit has
to be checked against the schedule the machine would actually run. The
continuous-rotation pass checked its own quasi-static step against the
production engine's turntable, and that check cost it a bilinear interpolation
error of 0.007 to 0.010 percent of dose PER ROTATION EVENT
(`CONTINUOUS_ROTATION_REPORT.md` Section 6.1), large enough to fail the
standing energy gate at high event counts and to swamp the physics being
measured.

This march avoids that entirely by staying in the PART frame. The part does not
move here; the heating pattern switches. Each candidate position's part-frame
heating field was already computed once by `DwellKernel.averaged_Q`, so
executing a program is only a matter of choosing which stored field to inject
at each outer step. NO field is ever interpolated, so the energy bookkeeping is
exactly the static forward's and the ONLY difference from the quasi-static
forward is the finite cycle time, which is the thing being measured.

WHAT IT IS NOT. It is not the production engine. It shares `adjoint2d`'s
forward with the rest of this prototype, and the prototype's agreement with the
production engine is established separately (`CONTINUOUS_ROTATION_REPORT.md`
Section 5 reproduces the engine's static numbers to the printed digits). It
also treats the move between positions as instantaneous.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import forward as fwd
from .dwell import TurntableProgram
from .dwell_kernel import DwellKernel


def program_step_positions(prog: TurntableProgram, angles_deg: Sequence[float],
                           dt_s: float, n_steps: int) -> np.ndarray:
    """Per-outer-step index into the FULL candidate angle set.

    Steps past the end of the program hold the last commanded position, which
    is what a machine that has finished its program and not been told to stop
    would do, and it keeps a horizon longer than the exposure well defined.
    """
    ang = np.asarray(angles_deg, dtype=float).ravel()
    dt = float(dt_s)
    out = np.zeros(int(n_steps), dtype=int)
    last = 0
    for m in prog.moves:
        j = int(np.argmin(np.abs(ang - float(m["position_deg"]))))
        i0 = int(round(float(m["move_at_s"]) / dt))
        i1 = int(round((float(m["move_at_s"]) + float(m["dwell_s"])) / dt))
        if i0 >= int(n_steps):
            break
        out[i0:min(i1, int(n_steps))] = j
        last = j
    end = int(round(sum(float(m["dwell_s"]) for m in prog.moves) / dt))
    if end < int(n_steps):
        out[end:] = last
    return out


def program_forward(kern: DwellKernel, pos_index: np.ndarray, n_steps: int,
                    shape_stop_patience: int | None = None) -> fwd.Trajectory:
    """March the thermal problem switching heating by the dwell program.

    `kern` must already have run `averaged_Q(s)` for the map being scored, so
    the per-position part-frame heating fields are in hand. `pos_index[i]` is
    the candidate-position index in force at outer step `i`.

    Structurally `rot_kernel.AveragedKernel.forward`, with the single averaged
    heating array replaced by a per-step selection from the per-position ones.
    The two electrical states switch at the same `update_interval` tick.
    """
    if not kern._Qk_a:
        raise RuntimeError(
            "program_forward needs the per-position heating fields; call "
            "DwellKernel.averaged_Q(s) (or .forward(s)) for this map first.")
    case = kern.case0
    p = case.pins
    n_steps = int(n_steps)
    idx = np.asarray(pos_index, dtype=int).ravel()
    if idx.size < n_steps:
        raise ValueError(f"pos_index has {idx.size} entries for {n_steps} steps")

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
    dwell_steps = np.zeros(kern.n_angles, dtype=int)

    for it in range(n_steps):
        k = int(idx[it])
        dwell_steps[k] += 1
        use_b = ui > 0 and it >= ui
        Q = kern._Qk_b[k] if use_b else kern._Qk_a[k]
        ck_T.append(T.copy())
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
    w_real = dwell_steps / max(int(dwell_steps.sum()), 1)
    Q_eff_a = sum(w * q for w, q in zip(w_real, kern._Qk_a))
    Q_eff_b = sum(w * q for w, q in zip(w_real, kern._Qk_b))
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
        state_a=None, state_b=None, ckpt_T=ck_T, ckpt_rho=[],
    )
    tr.Q_avg_a = Q_eff_a
    tr.Q_avg_b = Q_eff_b
    tr.dwell_steps_executed = dwell_steps
    tr.realized_weights_executed = w_real
    return tr
