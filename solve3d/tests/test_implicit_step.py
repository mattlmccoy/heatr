"""Phase-E implicit diffusion step -- FORWARD gates (plan 2026-08-11, Phase 1).

The Phase-E densify forward replaces the CFL-limited EXPLICIT enthalpy Euler with
an unconditionally-stable IMPLICIT backward-Euler solve (density_adjoint.
_implicit_T_step). Phase 1 changes only the FORWARD; its two correctness gates are:

  1. COARSE-EQUIVALENCE: on the coarse square (both schemes CFL-stable), the
     implicit end-state peak matches the explicit end-state peak to a stated tol.
     Proves the implicit scheme is the SAME physics (a consistent discretization),
     not a different model -- the coarse result must not move.

  2. TAMPER STABILITY (the point): on the fine Tamper mesh the explicit step is
     ~150x over CFL and its dt-cap clamp fires on ~54% of cells (the instability
     Phase 0 measured; the forward only survives because the clamp masks it). The
     implicit step clamps ~0% AND tracks a TRUSTED substepped-explicit reference.
     The DISCRIMINATOR is the clamp fraction: the peak alone is not, because the
     explicit clamp holds a wrong field near-right (Phase 0's warning).

The adjoint FD-gate is Phase 2-4; this file gates the forward only.

Run (spike env):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_implicit_step.py -x -q -s
"""
import numpy as np
import pytest

from solve3d import ceiling
from solve3d import density_adjoint as da
from solve3d import forward as fwd


def _march_peak_and_clamps(case, implicit):
    """March the densify forward step-by-step (n_sub=1) with the chosen diffusion
    scheme; return (true_peak_c, dt_cap_cell_count) where the dt-cap count is the
    union over the march of cells whose per-step |dT| rode the max_dt_step_c cap
    (the explicit-CFL-instability signature). Reuses the production _substep_forward
    (keep_cache=False) so the test measures the real step, not a re-derivation."""
    tc, p = case.tc, case.tc.p
    F = da._drive_F(case, case._v0 if case._v0 is not None else case.design_point())
    n = tc.vol_nodal.size
    T = np.full(n, p.preheat_c, dtype=np.float64)
    rho = np.full(n, p.rho_rel)
    ever_capped = np.zeros(n, dtype=bool)
    cap = p.max_dt_step_c
    for _ in range(case.n_steps):
        T_prev = T
        T, rho, _c = da._substep_forward(case, T, rho, F, keep_cache=False,
                                         implicit=implicit)
        ever_capped |= np.abs(T - T_prev) >= cap - 1e-9
    pk = ceiling.peak_temp(T, weights=tc.vol_nodal, mask=da._peak_mask(tc))
    return float(pk["true_max_c"]), int(ever_capped.sum())


# --------------------------------------------------------------------------- #
# Gate 1: coarse-equivalence (the coarse physics must not move)
# --------------------------------------------------------------------------- #
def test_implicit_matches_explicit_on_coarse_within_tol():
    case = da.build_coarse_case()          # square, 900 steps, dt=1.0, CFL-stable
    v = case.design_point()
    ks_exp = da.ks_peak_forward(case, v, implicit=False)
    ks_imp = da.ks_peak_forward(case, v, implicit=True)
    diff = abs(ks_imp - ks_exp)
    print(f"\n[coarse-equivalence] explicit ks={ks_exp:.4f} C  "
          f"implicit ks={ks_imp:.4f} C  |diff|={diff:.4f} C")
    # STATED TOL: backward-Euler vs forward-Euler differ at O(dt); at the coarse
    # dt=1.0 the observed gap is ~0.05 C (see the convergence probe, which shrinks
    # it ~linearly as dt halves). 1.0 C is the frozen coarse-equivalence band.
    assert diff <= 1.0, (ks_exp, ks_imp, diff)


# --------------------------------------------------------------------------- #
# Gate 2: Tamper stability (the payoff)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_implicit_forward_matches_substepped_reference_on_tamper():
    """REDUCED Tamper (production lc_part=2.5e-3 -- REQUIRED, coarsening hides the
    sliver-tet instability -- but a SHORT horizon so the CFL-substepped reference
    is affordable). Asserts (a) the implicit peak matches the trusted substepped-
    explicit reference to a stated tol AND (b) the implicit dt-cap clamp fires on
    FAR fewer cells than the explicit n_sub=1 step. The explicit n_sub=1 clamp
    fraction (~54%, Phase 0) is measured and reported here -- it is the RED
    evidence that the un-substepped explicit step is CFL-broken on this mesh."""
    from solve3d import design_chain as dc
    from solve3d.phase_e import run_tamper as rt

    tc, _info = rt.build_case(lc_part=2.5e-3, max_time_s=1800.0)
    part_cent = rt._part_centroids(tc)
    chain = dc.DesignChain(part_cent, tc.eqs.vol[tc.eqs.part],
                           da.FILTER_RADIUS_M, [0.0])
    p, vol = tc.p, tc.vol_nodal
    n_steps = 30
    v0 = np.ones(chain.n_design)

    # trusted reference: the EXPLICIT step, CFL-SUBSTEPPED at n_sub from
    # forward._stability_dt (dt_sub = dt/n_sub is CFL-stable -> accurate but slow).
    # Realized as a smaller-dt / more-steps EXPLICIT march (reuses the real code).
    k_max = np.where(tc.doped_cells > 0.5, max(tc.k_s_eff, p.k_liquid), p.k_powder)
    rho_cp0 = np.full(vol.size, p.rho_powder) * p.cp_powder
    dt_stable = fwd._stability_dt(tc.msh, tc.eqs.W, k_max, rho_cp0, vol, tc.eqs.Q0)
    n_sub = int(np.ceil(0.5 / (fwd.CFL_SAFETY * dt_stable)))
    assert n_sub > 100, ("this mesh must be deeply CFL-unstable for the test to "
                         f"be meaningful (n_sub={n_sub})")

    def _case(dt, nsteps):
        c = da.Case(tc=tc, chain=chain, dt=float(dt), n_steps=int(nsteps))
        c._v0 = v0
        return c

    ref_peak, _ = _march_peak_and_clamps(
        _case(0.5 / n_sub, n_steps * n_sub), implicit=False)   # substepped truth
    imp_peak, imp_caps = _march_peak_and_clamps(
        _case(0.5, n_steps), implicit=True)                    # the implicit fix
    exp_peak, exp_caps = _march_peak_and_clamps(
        _case(0.5, n_steps), implicit=False)                   # explicit n_sub=1

    ncells = vol.size
    print(f"\n[Tamper] n_cells={tc.ncells} n_nodes={ncells} n_sub(ref)={n_sub} "
          f"n_steps={n_steps}")
    print(f"[Tamper] peaks (C): reference(substepped)={ref_peak:.3f}  "
          f"implicit={imp_peak:.3f}  explicit_nsub1={exp_peak:.3f}")
    print(f"[Tamper] implicit-vs-ref={abs(imp_peak-ref_peak):.3f} C  "
          f"explicit_nsub1-vs-ref={abs(exp_peak-ref_peak):.3f} C")
    print(f"[Tamper] dt-cap clamp cells: implicit={imp_caps} "
          f"({100*imp_caps/ncells:.1f}%)  explicit_nsub1={exp_caps} "
          f"({100*exp_caps/ncells:.1f}%)")

    # (a) implicit tracks the substepped truth
    assert abs(imp_peak - ref_peak) <= 1.0, (imp_peak, ref_peak)
    # (b) the DISCRIMINATOR: implicit is CFL-stable (no clamping) where explicit
    # n_sub=1 is not. explicit must clamp a large fraction (the RED evidence);
    # implicit must clamp essentially none.
    assert exp_caps > 0.30 * ncells, ("explicit n_sub=1 must be CFL-broken on the "
                                      f"Tamper: only {exp_caps}/{ncells} clamped")
    assert imp_caps <= 0.01 * ncells, ("implicit must be CFL-stable: "
                                       f"{imp_caps}/{ncells} clamped")
