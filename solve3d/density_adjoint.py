"""solve3d Stage B B1: the rho+T co-state adjoint of the densify march.

RUNS IN THE SPIKE ENV ONLY (dolfinx 0.11 complex build).

WHAT THIS IS (spec 2026-08-07-stage-b-ceiling-coupled-dopant-design.md sec
"The core new machinery"): the densify end-state peak temperature is an
END-STATE quantity, so dThat_peak/ds must back-propagate THROUGH the densify
march. Two coupled co-states integrated backward over the time stepping:

  lambda_T   (temperature) -- seeded by the KS-peak softmax at the end-state.
  lambda_rho (density)     -- NEW. densify_rate depends on (T, rho) and rho feeds
                              the density-dependent properties (rho_s_eff, k_s_eff,
                              rho_L), so the peak's sensitivity to s travels partly
                              through the rho evolution. lambda_rho carries it.

COUPLING OFF (spec sec "Honest expectations and limits", last bullet): the
Stage A hold-out ran the densify gate with EQS-thermal coupling OFF, so the
drive Q_rf is solved ONCE at preheat and held fixed for the whole march. The
adjoint is derived for that same OFF forward -- the design enters only through
the initial Q, so the reverse sweep accumulates dJ/dF over all steps and hits
ONE gF_to_gq + ONE EQS vjp_q. Deriving a coupled adjoint but gating it against
an uncoupled forward would be a silent mismatch; this file states OFF and keeps
them consistent.

CONSISTENCY (assemble once, reuse): the forward densify march here reuses the
SAME assembled FEM forms as the melt-onset TransientCase (diff_form, conv_form,
F_form) and the SAME EQS adjoint (SteadyEqs.vjp_q, gF_to_gq). ks_peak_forward
and dks_peak_ds share ONE `_march` (store-everything on the coarse case), so the
forward the FD gate perturbs is byte-for-byte the forward the adjoint
differentiates. The per-step arithmetic mirrors forward.march_enthalpy's densify
branch (forward.py:760-822); the T-step VJP mirrors adjoint.step_vjp
(adjoint.py:822-858) with the density-property partials ADDED (adjoint.step_vjp
folds rho_s_eff/k_s_eff/rho_L away as constants because Phase A holds rho fixed);
the density terms mirror the 2-D coupled template fgm_solve_campaign/adjoint2d.

THE GATE (Task 3): dks_peak_ds vs central FD on the coarse case's high-
sensitivity design cells, at the frozen relative tolerance, NO widening. Mutation:
dropping lambda_rho must FAIL the gate (proves the density co-state is
load-bearing). No solve runs until this is green.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from solve3d import adjoint, ceiling, design_chain as dc, forward as fwd
from solve3d import stage_a_phase2 as p2

RESULTS = Path(__file__).resolve().parent / "results"

# Coarse gate case. Small square at the FIXED 0.40x drive; a fixed number of
# substeps (NO early rho-target stop, so the end-state is a SMOOTH function of
# the design -- a moving stop would inject a kink and forbid the 1e-6 gate). dt
# is chosen below the CFL limit so n_sub = 1 and the march arithmetic is the
# exact dt_sub = dt path.
COARSE_TARGET_NODES = 120
COARSE_LC0_M = 0.060 / 10.0
COARSE_DT_S = 1.0
COARSE_N_STEPS = 900
FILTER_RADIUS_M = 1.0e-3          # the frozen phase-2 design-chain filter radius


@dataclass
class _StepCache:
    """Everything one densify substep VJP needs (store-everything, coarse)."""
    T_in: np.ndarray
    rho_in: np.ndarray
    phi_in: np.ndarray
    m_phi_in: np.ndarray
    rho_s_eff: np.ndarray
    k_s_eff_cell: np.ndarray
    rho: np.ndarray
    cp: np.ndarray
    rho_cp: np.ndarray
    rho_L: np.ndarray
    frac: np.ndarray
    m_frac: np.ndarray
    phi_c: np.ndarray
    k_cells: np.ndarray
    num: np.ndarray
    H2: np.ndarray
    m_cap: np.ndarray
    m_tmp: np.ndarray
    T_out: np.ndarray
    phi_out: np.ndarray
    m_phi_out: np.ndarray
    drho_cap_mask: np.ndarray
    rho_clip_mask: np.ndarray
    pm: np.ndarray


@dataclass
class Case:
    tc: object                       # adjoint.TransientCase
    chain: dc.DesignChain
    dt: float
    n_steps: int
    F: np.ndarray | None = None      # fixed drive node-forcing of the last forward
    st: object | None = None         # last EQS SteadyState (for vjp_q)
    _v0: np.ndarray | None = field(default=None)

    # ---- design point / probes ------------------------------------------ #
    def design_point(self) -> np.ndarray:
        """A smooth, interior, non-uniform saturation on the part cells, kept in
        [0.05, 0.95] so no box clip enters the chain (the frozen 2-D choice)."""
        import dolfinx
        tc = self.tc
        mp = dolfinx.mesh.compute_midpoints(
            tc.msh, tc.msh.topology.dim,
            np.arange(tc.ncells, dtype=np.int32)).T[:, tc.eqs.part]
        x, y, z = mp[0], mp[1], mp[2]
        v = (0.70 + 0.18 * np.sin(np.pi * x / 0.010) * np.cos(np.pi * y / 0.010)
             + 0.06 * np.sin(np.pi * z / 0.030))
        return np.clip(v, 0.05, 0.95)

    def probe_indices(self, k: int = 4) -> list[int]:
        """The design cells the end-state KS peak is most sensitive to (largest
        |dks/ds|), the probes most likely to expose a wrong term."""
        v = self._v0 if self._v0 is not None else self.design_point()
        g = dks_peak_ds(self, v)
        return [int(i) for i in np.argsort(-np.abs(g))[:k]]


def probe_indices(case: "Case", k: int = 4) -> list[int]:
    """Module-level alias for case.probe_indices (the plan's da.probe_indices)."""
    return case.probe_indices(k)


def build_coarse_case(target_nodes: int = COARSE_TARGET_NODES,
                      lc0: float = COARSE_LC0_M, dt: float = COARSE_DT_S,
                      n_steps: int = COARSE_N_STEPS,
                      power_density: float | None = None) -> Case:
    """The coarse square at the FIXED 0.40x drive for the B1 FD gate.

    `power_density` overrides the chosen 0.40x drive for the Stage B4 drive-
    backoff (None keeps 0.40x, so every B1/B2/B3 call is unchanged)."""
    tc = adjoint.TransientCase.build(
        shape="square", target_nodes_in_part=int(target_nodes), lc0=float(lc0),
        p=p2.drive_params(dt_s=float(dt), power_density=power_density),
        max_time_s=float(dt) * float(n_steps))
    import dolfinx
    part_cent = np.asarray(dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim,
        np.arange(tc.ncells, dtype=np.int32)))[tc.eqs.part]
    chain = dc.DesignChain(part_cent, tc.eqs.vol[tc.eqs.part], FILTER_RADIUS_M, [0.0])
    case = Case(tc=tc, chain=chain, dt=float(dt), n_steps=int(n_steps))
    case._v0 = case.design_point()
    return case


# --------------------------------------------------------------------------- #
# Forward densify march (mirrors forward.march_enthalpy densify branch)
# --------------------------------------------------------------------------- #
def _peak_mask(tc) -> np.ndarray:
    m = tc.m_nodal > 0.5
    return m if m.any() else (tc.m_nodal > 0.0)


def _drive_F(case: Case, v: np.ndarray):
    """design v -> filtered map -> sigma -> ONE EQS solve -> Q -> node forcing F.

    Leaves case.st (the SteadyState + factorization) live so the reverse can
    reuse it for the EQS adjoint."""
    tc = case.tc
    s = case.chain.design_to_map(np.asarray(v, float), 0.0)     # filter (beta 0)
    sigma_part = tc.design_to_sigma(s)
    st = tc.eqs.forward(sigma_part)                             # sets sigma, LU, V
    tc.q_fn.x.array[:] = st.q.astype(fwd.dolfinx.default_scalar_type)
    F = fwd._assemble_real(tc.F_form)
    case.st = st
    case.F = F
    return F


def _substep_forward(case: Case, T_in, rho_in, F, keep_cache: bool):
    """One densify substep. Byte-mirror of forward.march_enthalpy's densify
    per-step block (forward.py:760-822) at n_sub = 1, coupling off."""
    tc, p = case.tc, case.tc.p
    dt = case.dt
    m_nodal = tc.m_nodal
    vol = tc.vol_nodal
    vol_safe = np.where(vol > 0, vol, 1.0)
    lo = p.t_pc_c - p.dt_pc_c / 2.0

    # phi from T_in (properties)
    arg_in = (T_in - p.t_pc_c) / p.dt_pc_c + 0.5
    phi_in = np.clip(arg_in, 0.0, 1.0)
    m_phi_in = (arg_in > 0.0) & (arg_in < 1.0)

    # density-dependent solid properties (forward.py:768-770)
    rho_s_eff = p.rho_powder + rho_in * (p.rho_solid - p.rho_powder)          # nodal
    cavg_rho = tc.cell_avg(rho_in)
    k_s_eff_cell = p.k_powder + cavg_rho * (p.k_solid - p.k_powder)           # cell

    rho_part = (1.0 - phi_in) * rho_s_eff + phi_in * p.rho_liquid
    cp_part = (1.0 - phi_in) * p.cp_solid + phi_in * p.cp_liquid
    rho = (1.0 - m_nodal) * p.rho_powder + m_nodal * rho_part
    cp = (1.0 - m_nodal) * p.cp_powder + m_nodal * cp_part
    rho_cp = rho * cp
    rho_L = m_nodal * rho_s_eff * p.latent_j_per_kg

    phi_c = tc.cell_avg(phi_in)
    k_part = (1.0 - phi_c) * k_s_eff_cell + phi_c * p.k_liquid
    k_cells = np.where(tc.doped_cells > 0.5, k_part, p.k_powder)

    tc.k_fn.x.array[:] = k_cells.astype(fwd.dolfinx.default_scalar_type)
    tc.T_fn.x.array[:] = T_in.astype(fwd.dolfinx.default_scalar_type)
    KT = fwd._assemble_real(tc.diff_form)
    C = (fwd._assemble_real(tc.conv_form) if p.conv_h != 0.0 else np.zeros_like(KT))
    num = -KT + F - C

    frac = np.clip((T_in - lo) / p.dt_pc_c, 0.0, 1.0)
    m_frac = ((T_in - lo) / p.dt_pc_c > 0.0) & ((T_in - lo) / p.dt_pc_c < 1.0)
    H = fwd.enthalpy_from_T(T_in, rho_cp, rho_L, p)
    H2 = H + dt * np.nan_to_num(num) / vol_safe
    T_new = fwd.T_from_enthalpy(H2, rho_cp, rho_L, p)
    dT_raw = T_new - T_in
    dT = np.clip(dT_raw, -p.max_dt_step_c, p.max_dt_step_c)
    m_cap = np.abs(dT_raw) <= p.max_dt_step_c
    T_cand = T_in + dT
    T_out = np.clip(T_cand, p.temp_min_c, p.temp_max_c)
    m_tmp = (T_cand >= p.temp_min_c) & (T_cand <= p.temp_max_c)

    # densify (forward.py:808-816); phi_now = phase_fraction(T_out)
    arg_out = (T_out - p.t_pc_c) / p.dt_pc_c + 0.5
    phi_out = np.clip(arg_out, 0.0, 1.0)
    m_phi_out = (arg_out > 0.0) & (arg_out < 1.0)
    rate = fwd.densify_rate(T_out, phi_out, rho_in, p)
    drho_raw = dt * rate
    drho_cap = p.dens_max_drho_rate * dt
    drho = np.clip(drho_raw, 0.0, drho_cap)
    drho_cap_mask = (drho_raw > 0.0) & (drho_raw < drho_cap)
    pm = m_nodal > 0.0
    rho_sum = rho_in + drho
    rho_new = np.clip(rho_sum, 0.0, 1.0)
    rho_clip_mask = (rho_sum > 0.0) & (rho_sum < 1.0)
    rho_out = np.where(pm, rho_new, rho_in)

    c = None
    if keep_cache:
        c = _StepCache(
            T_in=T_in, rho_in=rho_in, phi_in=phi_in, m_phi_in=m_phi_in,
            rho_s_eff=rho_s_eff, k_s_eff_cell=k_s_eff_cell, rho=rho, cp=cp,
            rho_cp=rho_cp, rho_L=rho_L, frac=frac, m_frac=m_frac, phi_c=phi_c,
            k_cells=k_cells, num=num, H2=H2, m_cap=m_cap, m_tmp=m_tmp,
            T_out=T_out, phi_out=phi_out, m_phi_out=m_phi_out,
            drho_cap_mask=drho_cap_mask, rho_clip_mask=rho_clip_mask, pm=pm)
    return T_out, rho_out, c


def _march(case: Case, v: np.ndarray, keep_cache: bool):
    """Fixed-horizon densify march. Returns (T_end, caches, diag)."""
    tc = case.tc
    F = _drive_F(case, v)
    T = np.full(tc.vol_nodal.size, tc.p.preheat_c, dtype=np.float64)
    rho = np.full(tc.vol_nodal.size, tc.p.rho_rel)
    caches: list[_StepCache] = []
    any_cap = False
    for _ in range(case.n_steps):
        T, rho, c = _substep_forward(case, T, rho, F, keep_cache)
        if keep_cache:
            caches.append(c)
        else:
            # cheap forward: watch the clips that would break the smooth gate
            pass
    return T, rho, caches, F


def ks_peak_forward(case: Case, v: np.ndarray) -> float:
    """Scalar end-state KS peak: march densify (fixed horizon, 0.40x, coupling
    off) and read ceiling.peak_temp on the end-state in-part temperature."""
    tc = case.tc
    T_end, _rho, _c, _F = _march(case, np.asarray(v, float), keep_cache=False)
    pk = ceiling.peak_temp(T_end, weights=tc.vol_nodal, mask=_peak_mask(tc))
    return float(pk["ks_aggregate_c"])


# --------------------------------------------------------------------------- #
# One densify substep VJP: (gT_out, gR_out) -> (gT_in, gR_in, g_F)
# --------------------------------------------------------------------------- #
def _dT_from_enthalpy(H, rho_cp, rho_L, p):
    """(dT/dH, dT/d(rho_cp), dT/d(rho_L)) for the exact piecewise-linear inverse
    T_from_enthalpy. rho_L enters T_from_enthalpy DIRECTLY (density adjoint) as
    well as through H; adjoint._dT_from_enthalpy drops the rho_L branch because
    Phase A holds rho fixed."""
    lo = p.t_pc_c - p.dt_pc_c / 2.0
    H_lo = rho_cp * lo
    H_hi = rho_cp * (lo + p.dt_pc_c) + rho_L
    below, above = H <= H_lo, H >= H_hi
    D = rho_cp + rho_L / p.dt_pc_c
    dT_dH = np.where(below | above, 1.0 / rho_cp, 1.0 / D)
    dT_drc = np.where(
        below, -H / rho_cp ** 2,
        np.where(above, -(H - rho_L) / rho_cp ** 2,
                 -(H + rho_L * lo / p.dt_pc_c) / D ** 2))
    dT_drL = np.where(below, 0.0,
                      np.where(above, -1.0 / rho_cp,
                               (lo * rho_cp - H) / (p.dt_pc_c * D * D)))
    return dT_dH, dT_drc, dT_drL


def _substep_vjp(case: Case, c: _StepCache, gT_out, gR_out):
    tc, p = case.tc, case.tc.p
    dt = case.dt
    m_nodal = tc.m_nodal
    vol_safe = np.where(tc.vol_nodal > 0, tc.vol_nodal, 1.0)

    gTo = np.array(gT_out, dtype=float, copy=True)
    gR_in = np.zeros_like(gR_out)

    # --- rho_out = where(pm, clip(rho_in+drho, 0, 1), rho_in) --------------
    gR_in += np.where(c.pm, 0.0, gR_out)                     # else identity branch
    g_rho_new = np.where(c.pm, gR_out, 0.0)
    g_rho_sum = g_rho_new * c.rho_clip_mask
    gR_in += g_rho_sum                                       # rho_in in rho_in+drho
    g_drho = g_rho_sum
    g_rate = g_drho * c.drho_cap_mask * dt                   # drho cap subgradient

    part = fwd.densify_rate_partials(c.T_out, c.phi_out, c.rho_in, p)
    gTo += g_rate * part["dT"]                               # rate via Tk -> T_out
    g_phi_out = g_rate * part["dphi"]
    gR_in += g_rate * part["drho"]                           # rate via rho_term
    gTo += g_phi_out * c.m_phi_out / p.dt_pc_c               # phi_out = phase_fraction(T_out)

    # --- T_out = clip(T_cand, tmin, tmax); T_cand = T_in + dT --------------
    g = gTo * c.m_tmp
    gT_in = np.array(g, copy=True)                           # carried identity
    g_dT = g * c.m_cap                                       # dT cap
    g_Tnew = g_dT
    gT_in -= g_dT                                            # dT_raw = T_new - T_in

    dT_dH, dT_drc, dT_drL = _dT_from_enthalpy(c.H2, c.rho_cp, c.rho_L, p)
    g_H2 = g_Tnew * dT_dH
    g_rho_cp = g_Tnew * dT_drc
    g_rho_L = g_Tnew * dT_drL                                # T_new direct rho_L path

    g_num = g_H2 * dt / vol_safe
    g_H = g_H2
    # H = enthalpy_from_T(T_in, rho_cp, rho_L): dH/dT_in, dH/drho_cp, dH/drho_L
    gT_in += g_H * (c.rho_cp + c.rho_L * c.m_frac / p.dt_pc_c)
    g_rho_cp = g_rho_cp + g_H * c.T_in
    g_rho_L = g_rho_L + g_H * c.frac

    # --- num = -KT + F - C -------------------------------------------------
    g_F = g_num.copy()
    tc.G_fn.x.array[:] = g_num.astype(fwd.dolfinx.default_scalar_type)
    tc.k_fn.x.array[:] = c.k_cells.astype(fwd.dolfinx.default_scalar_type)
    tc.T_fn.x.array[:] = c.T_in.astype(fwd.dolfinx.default_scalar_type)
    gT_in -= fwd._assemble_real(tc.diffG_form)               # -KT, K symmetric
    if p.conv_h != 0.0:
        gT_in -= fwd._assemble_real(tc.convG_form)           # -C, M_top symmetric
    g_k_cells = -np.real(fwd.fem.assemble_vector(tc.gk_form).array)

    # --- k_cells = where(doped, (1-phi_c) k_s_eff + phi_c k_liq, k_powder) --
    doped = tc.doped_cells > 0.5
    g_phi_c = g_k_cells * doped * (p.k_liquid - c.k_s_eff_cell)
    g_k_s_eff_cell = g_k_cells * doped * (1.0 - c.phi_c)
    # k_s_eff_cell = k_powder + cell_avg(rho_in) (k_solid - k_powder)
    gR_in += tc.cell_avg_T(g_k_s_eff_cell * (p.k_solid - p.k_powder))
    g_phi_in = tc.cell_avg_T(g_phi_c)

    # --- rho_cp = rho * cp ; rho, cp from phi_in and rho_s_eff -------------
    g_rho = g_rho_cp * c.cp
    g_cp = g_rho_cp * c.rho
    # rho = (1-m) rho_powder + m [(1-phi) rho_s_eff + phi rho_liquid]
    g_rho_local = g_rho * m_nodal
    g_phi_in += g_rho_local * (p.rho_liquid - c.rho_s_eff)
    g_rho_s_eff = g_rho_local * (1.0 - c.phi_in)
    # cp = (1-m) cp_powder + m [(1-phi) cp_solid + phi cp_liquid]
    g_phi_in += g_cp * m_nodal * (p.cp_liquid - p.cp_solid)
    # rho_L = m rho_s_eff latent
    g_rho_s_eff = g_rho_s_eff + g_rho_L * m_nodal * p.latent_j_per_kg
    # rho_s_eff = rho_powder + rho_in (rho_solid - rho_powder)
    gR_in += g_rho_s_eff * (p.rho_solid - p.rho_powder)

    # --- phi_in = clip((T_in - t_pc)/dt_pc + 0.5, 0, 1) -------------------
    gT_in += g_phi_in * c.m_phi_in / p.dt_pc_c
    return gT_in, gR_in, g_F


# --------------------------------------------------------------------------- #
# dks_peak_ds: the rho+T co-state, backward over the fixed-horizon march
# --------------------------------------------------------------------------- #
def dks_peak_ds(case: Case, v: np.ndarray,
                _drop_density_costate: bool = False) -> np.ndarray:
    """The adjoint gradient of the end-state KS peak w.r.t. the design v.

    `_drop_density_costate` ABLATES lambda_rho (zeros the density co-state after
    every substep, so it never propagates), which must break the FD gate -- the
    mutation proof that the density co-state is load-bearing."""
    tc = case.tc
    v = np.asarray(v, float)
    T_end, _rho, caches, F = _march(case, v, keep_cache=True)

    mask = _peak_mask(tc)
    gT = ceiling.peak_temp_vjp(T_end, weights=tc.vol_nodal, mask=mask)   # seed
    gR = np.zeros_like(gT)
    g_F = np.zeros_like(gT)

    for c in reversed(caches):
        gT, gR, gF = _substep_vjp(case, c, gT, gR)
        g_F += gF
        if _drop_density_costate:
            gR = np.zeros_like(gR)

    gQ = tc.gF_to_gq(g_F)                          # dJ/dF -> dJ/dQ_cell
    g_sigma_part = tc.eqs.vjp_q(case.st, gQ)       # EQS adjoint, reuses the LU
    g_s = tc.design_vjp(g_sigma_part)              # dsigma/ds = (doped - virgin)
    g_v = case.chain.design_vjp(v, g_s, 0.0)       # filter transpose (beta 0)
    return g_v


def march_fidelity_check(case: Case | None = None) -> dict:
    """Fidelity of the gate forward `_march` vs PRODUCTION forward.march_enthalpy
    (the physics the B2 arbiter ceiling_end_state_gate reads).

    The B2 solve minimizes a peak read from `_march`, but is judged on a peak read
    from march_densify (-> march_enthalpy). If the two forwards diverge, the solve
    would drive down a peak the arbiter does not see -- a silent forward mismatch.
    This runs BOTH with byte-matched inputs: SAME mesh (tc.msh), SAME drive Q (the
    single EQS solve), SAME dt with CFL substepping DISABLED so n_sub=1 (the exact
    dt_sub path `_march` uses), and the SAME fixed horizon (phi_target=2.0,
    stop_mean_rho above what the march reaches), then compares the end-state
    in-part T field and mean rho. Bit-identity is the target."""
    import dataclasses
    import json
    case = case or build_coarse_case()
    tc = case.tc
    v = case.design_point()
    # gate forward (also sets tc.q_fn = st.q via _drive_F inside _march)
    T_mine, rho_mine, _c, _F = _march(case, v, keep_cache=False)

    # production forward, matched: same q (tc.q_fn), same dt, n_sub forced to 1
    p_prod = dataclasses.replace(tc.p, enforce_cfl=False)
    out = fwd.march_enthalpy(
        tc.msh, p_prod, mats=tc.mats, q_dg0=tc.q_fn,
        max_time_s=case.dt * case.n_steps, phi_target=2.0, L=tc.L,
        sample_dt_s=None, densify=True, stop_mean_rho=0.999)
    T_prod = out["T"]
    rho_prod = out["rho_final"]

    mask = tc.m_nodal > 0.0
    Tm, Tp = T_mine[mask], T_prod[mask]
    denomT = float(np.max(np.abs(Tp))) or 1.0
    max_abs_dT = float(np.max(np.abs(Tm - Tp)))
    rel_T = max_abs_dT / denomT
    peak_mask = _peak_mask(tc)
    peak_mine = float(T_mine[peak_mask].max())
    peak_prod = float(out["true_peak_T_c"])
    mean_rho_mine = float(np.average(rho_mine[mask]))
    mean_rho_prod = float(np.average(rho_prod[mask]))
    rel_mean_rho = abs(mean_rho_mine - mean_rho_prod) / max(abs(mean_rho_prod), 1e-12)
    n_sub = int(out.get("n_substeps_used", 1))
    agree = bool(rel_T < 1e-6 and rel_mean_rho < 1e-6 and n_sub == 1)
    doc = {
        "what": "Stage B fidelity cross-check: density_adjoint._march (the FD-gate "
                "forward the co-state differentiates) vs production "
                "forward.march_enthalpy (densify=True) -- the forward the B2 "
                "arbiter ceiling_end_state_gate reads. Matched mesh, drive Q, dt, "
                "n_sub=1, fixed horizon. Bit-identity is the target.",
        "config": {"target_nodes": COARSE_TARGET_NODES, "lc0_m": COARSE_LC0_M,
                   "dt_s": case.dt, "n_steps": case.n_steps,
                   "n_part_nodes": int(mask.sum()),
                   "enforce_cfl_disabled_for_match": True},
        "n_substeps_used_prod": n_sub,
        "n_steps_taken_prod": int(out.get("n_steps_taken", -1)),
        "max_abs_dT_in_part_c": max_abs_dT,
        "rel_T_in_part": rel_T,
        "peak_true_c_mine": peak_mine,
        "peak_true_c_prod": peak_prod,
        "peak_abs_diff_c": abs(peak_mine - peak_prod),
        "mean_rho_mine": mean_rho_mine,
        "mean_rho_prod": mean_rho_prod,
        "rel_mean_rho": rel_mean_rho,
        "tolerance_rel": 1e-6,
        "agree": agree,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "stage_b_march_fidelity.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=float))
    tmp.replace(path)
    return doc


def fd_gate(case: Case | None = None, n_probes: int = 4, h: float = 1e-4,
            seed: int = 7) -> dict:
    """B1 gate artifact: dks_peak_ds vs central FD on the high-sensitivity design
    cells (frozen 1e-6 relative, NO widening) + the drop-lambda_rho mutation.

    COUPLING OFF: the gated forward solves the EQS ONCE at preheat and holds the
    drive fixed (the Stage A hold-out convention); the adjoint is derived for that
    same OFF forward. launch_ok is the precondition Task 7 (the heavy solve) reads."""
    import json
    case = case or build_coarse_case()
    v = case.design_point()
    g = dks_peak_ds(case, v)
    g_bad = dks_peak_ds(case, v, _drop_density_costate=True)
    idx = [int(i) for i in np.argsort(-np.abs(g))[:n_probes]]
    probes = {}
    worst = 0.0
    worst_mut = 0.0
    for i in idx:
        vp = v.copy(); vp[i] += h; vm = v.copy(); vm[i] -= h
        fd = (ks_peak_forward(case, vp) - ks_peak_forward(case, vm)) / (2 * h)
        re = abs(fd - g[i]) / max(1.0, abs(fd))
        re_mut = abs(fd - g_bad[i]) / max(1.0, abs(fd))
        worst = max(worst, re)
        worst_mut = max(worst_mut, re_mut)
        probes[str(i)] = {"fd": float(fd), "adjoint": float(g[i]),
                          "rel_err": float(re),
                          "rel_err_drop_lambda_rho": float(re_mut)}
    diag = diagnose_case(case)
    passed = bool(worst <= 1e-6)
    mutation_bites = bool(worst_mut > 1e-3)
    launch_ok = bool(passed and mutation_bites and diag["melted_any"]
                     and not diag["dt_step_cap_active"]
                     and not diag["temp_bound_cap_active"]
                     and not diag["drho_cap_active_in_part"])
    doc = {
        "what": "Stage B B1 FD gate: rho+T density co-state adjoint dks_peak/ds "
                "vs central finite differences on the coarse square's high-"
                "sensitivity design cells; frozen 1e-6 relative, no widening. "
                "Mutation: dropping lambda_rho must FAIL (rel_err > 1e-3).",
        "coupling": "off",
        "coupling_note": "EQS solved once at preheat, drive fixed for the march "
                         "(the Stage A hold-out convention); the adjoint is derived "
                         "for the same OFF forward -- consistent, not a mismatch.",
        "drive_a": p2.chosen_drive_a(),
        "power_density_w_per_m3": p2.chosen_drive_power_density(),
        "case": {"target_nodes": COARSE_TARGET_NODES, "lc0_m": COARSE_LC0_M,
                 "dt_s": case.dt, "n_steps": case.n_steps,
                 "march_time_s": case.dt * case.n_steps,
                 "n_part_nodes": diag["n_part_nodes"],
                 "n_design_cells": int(g.size),
                 "fixed_horizon_no_rho_stop": True,
                 "fixed_horizon_note": "the gate marches a FIXED number of steps "
                 "(no rho-target early stop) so the end-state is a SMOOTH function "
                 "of the design; a moving stop would inject a kink."},
        "end_state": {"peak_true_c": diag["peak_true_c"],
                      "peak_ks_c": diag["peak_ks_c"],
                      "mean_rho": diag["mean_rho_end"],
                      "max_rho": diag["max_rho_end"]},
        "clips_inactive": {
            "dt_step_cap": not diag["dt_step_cap_active"],
            "temp_bound": not diag["temp_bound_cap_active"],
            "drho_cap_in_part": not diag["drho_cap_active_in_part"]},
        "n_probes": n_probes, "probe_indices": idx, "h": h,
        "worst_rel_err": float(worst),
        "pass_rel_err": 1e-6,
        "fd_gate_passed": passed,
        "mutation_worst_rel_err_drop_lambda_rho": float(worst_mut),
        "mutation_bites": mutation_bites,
        "rho_target_reached": bool(diag["mean_rho_end"] >= _stop_target()),
        "launch_ok": launch_ok,
        "probes": probes,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "stage_b_density_adjoint_fd_gate.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=float))
    tmp.replace(path)
    return doc


def _stop_target() -> float:
    from solve3d import stage_a
    return float(stage_a.thermal_config()["rho_target"]["practical_ideal"])


def diagnose_case(case: Case) -> dict:
    """Forward-only health read of the coarse case: is it melted + densifying,
    and are the clips whose subgradients would break the 1e-6 gate INACTIVE at
    the design point?"""
    tc = case.tc
    v = case._v0 if case._v0 is not None else case.design_point()
    T, rho, caches, F = _march(case, v, keep_cache=True)
    mask = _peak_mask(tc)
    any_cap = any(bool((~c.m_cap).any() or (~c.m_tmp).any()
                       or (c.drho_cap_mask == False).all() and False)
                  for c in caches)
    dt_cap = any(bool((~c.m_cap).any()) for c in caches)
    t_cap = any(bool((~c.m_tmp).any()) for c in caches)
    drho_cap_hit = any(bool((~c.drho_cap_mask & c.pm).any()) for c in caches)
    pk = ceiling.peak_temp(T, weights=tc.vol_nodal, mask=mask)
    return {
        "n_steps": case.n_steps, "dt_s": case.dt,
        "march_time_s": case.dt * case.n_steps,
        "n_part_nodes": int(mask.sum()),
        "peak_true_c": pk["true_max_c"], "peak_ks_c": pk["ks_aggregate_c"],
        "mean_rho_end": float(np.average(rho[mask])),
        "max_rho_end": float(rho[mask].max()),
        "melted_any": bool((T[mask] >= tc.p.t_pc_c + tc.p.dt_pc_c).any()),
        "dt_step_cap_active": bool(dt_cap),
        "temp_bound_cap_active": bool(t_cap),
        "drho_cap_active_in_part": bool(drho_cap_hit),
    }
