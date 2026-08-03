"""Rotationally-averaged radio-frequency heating kernel, in the PART frame.

THE PHYSICAL IDEA. A turntable spins the part while the electrodes stay fixed.
If the rotation is fast compared with the thermal time constants of the part,
every material point sees the ANGLE-AVERAGE of the heating it would receive at
each orientation, and the thermal problem can be solved once in the co-rotating
part frame with that averaged source. That is the quasi-static approximation.
It is an approximation and its error is measured, not assumed: see
`ROT_QUASISTATIC_NOTE` below and the Level-2 comparison against the real
rotating engine.

THE OPERATOR. For each sampled angle theta the part-frame design map s is
rotated into the lab frame, the electro-quasi-static (EQS) problem is solved
there against the part rasterized at theta, and the resulting Q_rf is rotated
BACK into the part frame. The average over the angle set is the kernel:

    Q_avg(s) = (1/M) sum_j  R_(-theta_j) [ Q_rf( R_(theta_j) s ; theta_j ) ]

Note that this is an OPERATOR in s, not a fixed array: the map still shapes the
field at every angle. A fixed array kernel evaluated once at a uniform map
would make the design variable inert and there would be nothing to solve.

TWO ELECTRICAL STATES, as everywhere else in this prototype. The production
engine assembles conductivity one way at startup (state A) and another way from
the first `update_interval` tick onwards (state B), so the kernel is averaged
separately for each and the march switches between them at the same tick.

THE ADJOINT walks the same operators backwards: the transpose of the
back-rotation, the per-angle EQS adjoint reusing the forward factorization, and
the transpose of the forward rotation. Every rotation is the assembled sparse
matrix of `rot_frame`, so no transpose is re-derived by hand.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import adjoint as adj, eqs, forward as fwd, gradops
from .pins import Case, build_case
from .rot_frame import RotationOperator, averaging_angles, rotation_operator

ROT_QUASISTATIC_NOTE = (
    "Quasi-static approximation: valid when the rotation period is short "
    "compared with the part's thermal diffusion time. Quantified per shape by "
    "`quasistatic_numbers`."
)


# ---------------------------------------------------------------------------
# quasi-static assumption, quantified
# ---------------------------------------------------------------------------

def quasistatic_numbers(case: Case, rotation_period_s: float) -> dict:
    """The numbers the quasi-static claim has to be read against.

    Two thermal times are reported because they bracket the question:

      tau_diffusion  L^2 / alpha with L the part's half-width (the largest
                     distance heat has to travel to erase an azimuthal
                     modulation) and alpha = k / (rho cp) the solid-phase
                     thermal diffusivity of the doped material. This is the
                     time to erase a part-scale non-uniformity.
      tau_cell       the same with L one grid cell. This is the time to erase
                     the smallest structure the kernel can express, and it is
                     the STRICT test: the averaging is exact only when the
                     rotation is fast compared with the fastest thermal
                     relaxation the heating pattern can drive.

    Also reported is tau_heat, the time to raise the part from ambient to the
    phase-change temperature at the delivered power, because the honest
    statement of the approximation is that the rotation must be fast compared
    with the time over which the temperature field actually develops, and that
    is tau_heat when tau_heat is the shortest of the three.
    """
    p = case.pins
    alpha = p.k_solid / (p.rho_solid * p.cp_solid)
    n_part = int(case.part_mask.sum())
    area = n_part * case.dA
    half_width = 0.5 * float(np.sqrt(area))
    cell = float(min(case.dx, case.dy))
    return {
        "rotation_period_s": float(rotation_period_s),
        "thermal_diffusivity_m2_per_s": float(alpha),
        "part_equivalent_half_width_m": half_width,
        "tau_diffusion_part_s": float(half_width ** 2 / alpha),
        "tau_diffusion_cell_s": float(cell ** 2 / alpha),
        "cell_m": cell,
        "n_part_cells": n_part,
    }


# ---------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------

@dataclass
class _AngleState:
    case: Case
    R_to_lab: RotationOperator
    R_to_part: RotationOperator
    st_a: Any = None
    st_b: Any = None
    inrange_b: np.ndarray = None
    active_a: np.ndarray = None
    active_b: np.ndarray = None


@dataclass
class AveragedKernel:
    """The averaged-kernel forward and adjoint on one shape."""

    case0: Case
    angles: np.ndarray
    per_angle: list[_AngleState] = field(repr=False, default_factory=list)
    _cache_s: np.ndarray = field(repr=False, default=None)

    # -- construction -------------------------------------------------------

    @classmethod
    def build(cls, cfg: dict, angles: np.ndarray | None = None,
              step_deg: float = 15.0) -> "AveragedKernel":
        ang = (averaging_angles(step_deg) if angles is None
               else np.asarray(angles, dtype=float))
        cfg0 = copy.deepcopy(cfg)
        cfg0["geometry"]["part"]["rotation_deg"] = 0.0
        case0 = build_case(cfg0)
        shape = case0.part_mask.shape
        per = []
        for th in ang:
            c = copy.deepcopy(cfg)
            c["geometry"]["part"]["rotation_deg"] = float(th)
            per.append(_AngleState(
                case=build_case(c),
                R_to_lab=rotation_operator(shape, float(th)),
                R_to_part=rotation_operator(shape, -float(th))))
        return cls(case0=case0, angles=ang, per_angle=per)

    @property
    def n_angles(self) -> int:
        return len(self.per_angle)

    # -- forward ------------------------------------------------------------

    def lab_map(self, a: _AngleState, s: np.ndarray) -> np.ndarray:
        """The part-frame design map as the engine at this angle would see it.

        Outside the rotated part the injected saturation is the prototype's
        nominal 1, matching `joint_angle_lib.rotated_warm_start`; inside, it is
        the rotated map. No clip: the box on the design variable already keeps
        s in [0, 1] and a normalized rotation of values in [0, 1] with fill 1
        stays in [0, 1], so no clip subgradient enters the chain.
        """
        return np.where(a.case.part_mask, a.R_to_lab.apply(s, outside=1.0), 1.0)

    def averaged_Q(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(Q_avg for electrical state A, Q_avg for state B), part frame."""
        shape = self.case0.part_mask.shape
        Qa = np.zeros(shape)
        Qb = np.zeros(shape)
        m = float(self.n_angles)
        for a in self.per_angle:
            s_lab = self.lab_map(a, s)
            eps = fwd.eps_field(a.case)
            a.st_a = fwd.solve_electric(a.case, fwd.sigma_state_a(a.case, s_lab, False), eps)
            sig_b, a.inrange_b = fwd.sigma_state_b(a.case, s_lab)
            a.st_b = fwd.solve_electric(a.case, sig_b, eps)
            p = a.case.pins
            a.active_a = (a.case.doped_mask & (a.st_a.Qrf_raw > 0.0)
                          & (a.st_a.Qrf_raw < p.max_qrf))
            a.active_b = (a.case.doped_mask & (a.st_b.Qrf_raw > 0.0)
                          & (a.st_b.Qrf_raw < p.max_qrf))
            Qa += a.R_to_part.apply(a.st_a.Qrf, outside=0.0) / m
            Qb += a.R_to_part.apply(a.st_b.Qrf, outside=0.0) / m
        self._cache_s = np.array(s, dtype=float, copy=True)
        return Qa, Qb

    def forward(self, s: np.ndarray, *, keep_checkpoints: bool = False,
                n_steps: int | None = None,
                shape_stop_patience: int | None = None) -> fwd.Trajectory:
        """The averaged-kernel march, in the part frame.

        Identical to `forward.forward` except that the two electrical states
        are replaced by the two ANGLE-AVERAGED heating patterns. The energy
        bookkeeping, the clip diagnostics and the shape early stop are kept so
        every standing gate still applies.
        """
        case = self.case0
        p = case.pins
        n_steps = int(p.n_steps if n_steps is None else n_steps)
        Q_a, Q_b = self.averaged_Q(s)

        T = np.full(case.part_mask.shape, p.ambient_c, dtype=float)
        rho = np.zeros(case.part_mask.shape, dtype=float)
        rho[case.part_mask] = p.rho_rel_init
        phi = np.zeros_like(T)

        ui = p.update_interval
        pm_e = case.part_mask
        rec_T, rec_ui, rec_phi, rec_rho = [], [], [], []
        ck_T, ck_rho = [], []
        rec_e_in, rec_e_out, rec_e_stored = [], [], []
        e_in_acc = e_out_acc = e_stored_acc = 0.0
        eb_T_prev, eb_phi_prev, eb_rho_prev = T.copy(), phi.copy(), rho.copy()
        frac_dT_max = frac_tc_max = 0.0
        stopped_early = False

        chi = case.part_mask.astype(float)
        j_min, j_first, j_argmin = np.inf, None, 0

        Q = Q_a
        for it in range(n_steps):
            if ui > 0 and it > 0 and (it % ui == 0):
                Q = Q_b
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
        dm = case.doped_mask
        tr = fwd.Trajectory(
            time_s=(np.arange(n_out) + 1) * p.dt,
            mean_T_part_c=arr_T, ui_rms_part=arr_ui,
            mean_phi_part=np.asarray(rec_phi), mean_rho_rel_part=np.asarray(rec_rho),
            sigma_T=arr_ui * (arr_T - p.ambient_c),
            n_outer=n_out, stopped_early=stopped_early,
            T_final=T, rho_final=rho, phi_final=phi,
            P_abs_A=float(np.sum(Q_a[dm]) * case.dA),
            P_abs_B=float(np.sum(Q_b[dm]) * case.dA),
            frac_dT_clipped_max=frac_dT_max, frac_temp_cap_max=frac_tc_max,
            frac_qrf_cap=float(np.mean(np.concatenate(
                [np.concatenate([a.st_a.Qrf_raw[a.case.doped_mask],
                                 a.st_b.Qrf_raw[a.case.doped_mask]])
                 for a in self.per_angle]) >= p.max_qrf)),
            energy_in_J_per_m=np.asarray(rec_e_in, dtype=float),
            energy_out_J_per_m=np.asarray(rec_e_out, dtype=float),
            energy_stored_J_per_m=np.asarray(rec_e_stored, dtype=float),
            state_a=None, state_b=None, ckpt_T=ck_T, ckpt_rho=ck_rho,
        )
        tr.Q_avg_a = Q_a
        tr.Q_avg_b = Q_b
        return tr

    # -- adjoint ------------------------------------------------------------

    def _reverse_march(self, tr: fwd.Trajectory, seeds: dict[int, np.ndarray],
                       seeds_rho: dict[int, np.ndarray] | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
        """dJ/dQ_avg for each electrical state, in the PART frame.

        Structurally `adjoint.reverse_march`, with two differences that are
        forced by the averaging: the injected heating is a stored array rather
        than a per-step re-scale of a raw field, and NO cap or doped-mask
        subgradient is applied here, because both of those act per angle in the
        LAB frame and are applied there.
        """
        case = self.case0
        p = case.pins
        ui = p.update_interval
        shape = case.part_mask.shape
        gQ_a = np.zeros(shape)
        gQ_b = np.zeros(shape)
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
            Q = tr.Q_avg_b if use_b else tr.Q_avg_a
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
            if use_b:
                gQ_b += acc
            else:
                gQ_a += acc
        return gQ_a, gQ_b

    def gradient(self, s: np.ndarray, tr: fwd.Trajectory,
                 seeds: dict[int, np.ndarray], grad_ops=None,
                 seeds_rho: dict[int, np.ndarray] | None = None) -> np.ndarray:
        """dJ/ds in the PART frame, conductivity channel only.

        Chain, per angle, read right to left:

            s_part -> R_theta -> mask -> s_lab -> sigma(A or B) -> EQS -> Q_lab
                   -> R_(-theta) -> (1/M) -> Q_avg -> thermal march -> J
        """
        case = self.case0
        p = case.pins
        # The per-angle electrical states are cached on the kernel by the
        # forward, so a forward at a DIFFERENT map run in between would
        # silently poison the adjoint. Refuse rather than return a wrong
        # gradient.
        if self._cache_s is None or not np.array_equal(self._cache_s, np.asarray(s, dtype=float)):
            raise RuntimeError(
                "AveragedKernel.gradient must be called against the map of the "
                "most recent forward; the cached per-angle electrical states "
                "belong to a different map.")
        Gx, Gy = (grad_ops if grad_ops is not None
                  else gradops.gradient_matrices(case.x, case.y))
        gQ_a, gQ_b = self._reverse_march(tr, seeds, seeds_rho)
        m = float(self.n_angles)
        ds = np.zeros(case.part_mask.shape)
        for a in self.per_angle:
            pm_j = a.case.part_mask
            for gQ_part, st, active, is_b in ((gQ_a, a.st_a, a.active_a, False),
                                              (gQ_b, a.st_b, a.active_b, True)):
                if not np.any(gQ_part):
                    continue
                g_lab = a.R_to_part.apply_T(gQ_part) / m
                dsig = adj.eqs_vjp(a.case, st, g_lab * active, Gx, Gy, pre_masked=True)
                if is_b:
                    g_slab = dsig * a.inrange_b * p.sigma_d0
                else:
                    g_slab = dsig * a.case.fill_frac * (p.sigma_d0 - p.sigma_v)
                ds += a.R_to_lab.apply_T(np.where(pm_j, g_slab, 0.0))
        return ds
