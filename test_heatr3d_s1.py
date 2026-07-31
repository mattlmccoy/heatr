"""S1 numerical-integrity tests for heatr3d (spec: docs/superpowers/specs/
2026-07-30-heatr3d-graduation-design.md, Gate S1)."""
import dataclasses

import numpy as np
import pytest

from heatr3d import Grid, Params, make_geometry, run


def small_sphere_case(n=32):
    grid = Grid(n=n, L=0.060)
    part = make_geometry(grid, "sphere", diam=0.020)
    p = Params()
    return grid, part, p


def test_energy_audit_fields_present_and_small_on_benign_run():
    grid, part, p = small_sphere_case()
    res = run(grid, part, p, max_time_s=30.0, verbose=False)
    # new provenance fields
    assert hasattr(res, "energy_in_j")
    assert hasattr(res, "energy_stored_j")
    assert hasattr(res, "energy_loss_j")
    assert hasattr(res, "energy_residual_frac")
    assert res.energy_in_j > 0
    # benign pre-melt run must conserve energy to a few percent
    assert abs(res.energy_residual_frac) < 0.05


def spike_case(n=32, spike_mult=400.0):
    """Uniform mild heating plus one interior hot column whose raw per-step dT
    exceeds the full melt window (dt_pc_c), forcing a window-crossing step."""
    grid, part, p = small_sphere_case(n)
    q = np.zeros((n, n, n))
    q[part] = p.power_density_w_per_m3
    ii = np.argwhere(part)
    c = ii[len(ii) // 2]
    q[c[0], c[1], c[2]] *= spike_mult
    return grid, part, p, q


def test_legacy_phase_update_skips_latent_on_window_crossing():
    """Legacy failure signature, calibrated at spike_mult=400.0 (n=32, 120 s).

    Measured (2026-07-30, ./.venv312): clamp_bound=True,
    energy_in=1287.9 J, stored=899.3 J, loss=0.023 J,
    energy_residual_frac=+0.3017. The spiked voxel (16, 11, 13) jumps
    phi 0.0000 -> 0.8000 in ONE step (T 173.00 -> 183.00 C) with
    dphi/dT = 0.0000 1/C at the step start: the pointwise apparent-cp latent
    sink is entirely absent for the window-crossing step. Sweep for context:
    mult=1 resid=+0.0000 clamp=False; 50 +0.0051 False; 100 +0.0213 True;
    200 +0.1333 True; 400 +0.3017 True; 1000 +0.5598 True.

    HONESTY NOTE (see docs/superpowers/plans/s1-findings.md): the latent skip
    itself is worth only ~0.3 J here (one voxel: rho*L*dV); the ~389 J surplus
    is dominated by the spiked voxel saturating at temp_max=600 C while RF keeps
    depositing. So the residual assertion pins "the run went non-physical", and
    the phi-jump measurement above is what pins the latent-skip mechanism.
    This test is inverted into the regression test once the enthalpy update lands.
    """
    grid, part, p, q = spike_case()
    res = run(grid, part, p, qrf_override=q, max_time_s=120.0)
    assert res.clamp_bound is True
    assert res.energy_residual_frac > 0.10


def test_enthalpy_roundtrip_and_window_crossing():
    from heatr3d import enthalpy_from_T, T_from_enthalpy
    p = Params()
    rho_cp = p.rho_solid * p.cp_solid          # J/(m^3 K), sensible slope
    rho_L = p.rho_solid * p.latent_j_per_kg    # J/m^3, latent plateau
    Ts = np.array([25.0, 175.0, 180.0, 185.0, 190.0, 250.0])
    H = enthalpy_from_T(Ts, rho_cp, rho_L, p)
    Tb = T_from_enthalpy(H, rho_cp, rho_L, p)
    assert np.allclose(Tb, Ts, atol=1e-9)
    # depositing exactly the latent plateau plus 20 C sensible from the window
    # start lands 20 C above the window end, never skipping the latent barrier
    H0 = enthalpy_from_T(np.array([p.t_pc_c - p.dt_pc_c / 2]), rho_cp, rho_L, p)
    H1 = H0 + rho_L + rho_cp * (p.dt_pc_c + 20.0)
    T1 = T_from_enthalpy(H1, rho_cp, rho_L, p)
    assert np.allclose(T1, p.t_pc_c + p.dt_pc_c / 2 + 20.0, atol=1e-9)


def bulk_crossing_case(n=32, mult=130.0):
    """Uniform heating of the WHOLE part, sized so every part voxel takes a
    melt-window-crossing step, with no numerical limiter binding.

    raw source dT/step = mult * 0.06920 C = 9.00 C at mult=130: below
    max_dt_step_c = 10.0 (so THM-01 never binds) yet large enough that a voxel
    sitting below the window start (175 C) lands well inside the window in one
    step with dphi/dT = 0 at the step start -- the exact mechanism traced in
    docs/superpowers/plans/s1-findings.md section 2 (173.00 -> 183.00 C).

    Applying it to all 624 part voxels (instead of the single spiked voxel of
    spike_case) makes the skipped latent heat GLOBALLY measurable: the part's
    full latent budget is rho_s_eff * L * V_part = 188.3 J against ~851 J of
    RF input over 1.0 s. In spike_case the skipped latent is ~0.3 J and is
    invisible next to clamp artifacts (Task-3 honesty note).
    """
    grid, part, p = small_sphere_case(n)
    q = np.zeros((n, n, n))
    q[part] = p.power_density_w_per_m3 * mult
    return grid, part, p, q


def test_enthalpy_update_conserves_energy_on_window_crossing():
    """The S1 fix heals the latent-skip energy error on a window-crossing run.

    DEVIATION from the plan's spike_case(200)/60 s version, with measurements
    (2026-07-30, ./.venv312, n=32): that case cannot isolate the phase
    mechanism at ANY (spike_mult, max_time_s). A single spiked voxel needs
    mult >= ~145 to cross the window, and its steady state sits
    mult*q1/(6*k_s_eff/h^2) = 974 C (mult=150) above its neighbours, so it
    always drives into the temp_max_c = 600 C clamp; and below saturation the
    skipped latent (~0.3 J) is far smaller than the clamp/audit error.
    Measured spike sweep (resid apparent_cp -> enthalpy): mult=100 t=3 s
    +0.0268 -> +0.0240; mult=150 t=10 s +0.0617 -> +0.0620; mult=200 t=60 s
    +0.1258 -> +0.1239 (T_max=600 C, clamp_bound both). The residual there is
    dominated by the clamps, not by latent, so it cannot show the fix.

    bulk_crossing_case makes the same mechanism global and limiter-free.
    Measured at mult=130, t_end=1.0 s: apparent_cp residual = -0.0711 (the
    latent-skip signature: the audit books 126 J of latent from the phi jump
    that the solver never paid, so stored > in), enthalpy = +0.0045, both with
    clamp_bound False and T_max ~180 C. That is the fix, isolated.
    """
    grid, part, p0, q = bulk_crossing_case()
    p = dataclasses.replace(p0, phase_update="enthalpy")   # Params is frozen
    res = run(grid, part, p, qrf_override=q, max_time_s=1.0, phi_target=2.0)
    assert res.clamp_bound is False             # no limiter is hiding anything
    assert res.T_max_c < p.temp_max_c - 1.0     # no temp-clamp saturation
    assert abs(res.energy_residual_frac) < 0.05
    # the part really did cross into the melt window; it pays the latent toll
    # now, so it sits mid-window instead of being snapped past it
    assert res.phi_final.max() > 0.4
    assert 175.0 < res.T_max_c < 185.0

    # differential control: the legacy scheme on the IDENTICAL case books the
    # latent-skip surplus the fix removes (this is what makes the assertion
    # above discriminating rather than vacuous).
    res_legacy = run(grid, part, p0, qrf_override=q, max_time_s=1.0,
                     phi_target=2.0)
    assert res_legacy.energy_residual_frac < -0.05
    assert res_legacy.phi_final.max() > res.phi_final.max()


def test_audit_stays_tight_through_melt_both_schemes():
    """Healthy uniform molten run: the audit must stay tight ABOVE the window.

    Task-4 finding 2: the v1 audit booked the whole run with initial-state
    solid properties (cp_solid, rho_s_eff at t=0) while the solver blends to
    cp_liquid=3279 / rho_liquid=1010, so it drifted to +0.3795 (both schemes)
    at t_end=3.0 s on this healthy, limiter-free case -- the standing gate was
    unusable exactly where the instability lives.

    Measured after the v2 per-step banking (2026-07-30, ./.venv312, n=32,
    bulk_crossing_case mult=130, phi_target=2.0, clamp_bound False both arms):
        t=3.0 s  apparent_cp  in=2553.131 stored=2553.246 resid=-4.5e-05
        t=3.0 s  enthalpy     in=2553.131 stored=2553.131 resid=-1.5e-16
    The enthalpy arm is exact by construction: its update deposits num*dt into
    the SAME H(T) the audit books (rho*cp sensible slope + rho_s_eff*L across
    the window, and phase_fraction's phi IS the enthalpy ramp fraction).

    This upgrade does NOT mask the legacy scheme defect: on the same case at
    t=1.0 s (mid-window, phi_mean 0.669) the legacy arm still books
    resid=-0.0802 (stored 919.3 > in 851.0) against the enthalpy arm's
    +2e-16 -- see test_enthalpy_update_conserves_energy_on_window_crossing,
    whose differential control asserts exactly that. The legacy residual
    happens to cancel to ~1e-4 once the part is FULLY molten (its skipped
    latent is offset by paying the resolved part of the latent at the blended
    liquid density rho ~ 740-1010 instead of rho_s_eff = 473.5), which is why
    this test's window-crossing sibling is the discriminating one.
    """
    grid, part, p, q = bulk_crossing_case()
    for scheme in ("apparent_cp", "enthalpy"):
        pp = dataclasses.replace(p, phase_update=scheme)
        res = run(grid, part, pp, qrf_override=q, max_time_s=3.0,
                  phi_target=2.0)
        assert abs(res.energy_residual_frac) < 0.02, scheme


def test_adiabatic_uniform_heating_matches_analytic_plateau():
    """Whole domain = part, uniform q, conv off: T(t) is analytic including
    the latent plateau. Exact for the enthalpy scheme by construction; this
    pins the wiring (property maps, dt, source bookkeeping).

    Sizing (deviation from the plan's q=2.0e5 / t_end=200 s, which deposits
    only 4.0e7 J/m^3 and lands at 56.8 C -- entirely BELOW the melt window, so
    it would never touch the latent plateau it is named for): q=2.0e6 W/m^3 for
    100.0 s deposits 2.0e8 J/m^3 and lands MID-PLATEAU at T_exact = 178.48 C
    (phi ~ 0.35), which is what the plan's step-3 note asks for.

    Two arms, both measured 2026-07-30 (./.venv312, n=16, dt=0.05, 2000 steps,
    clamp_bound False, energy residual ~1e-14 in both):
      * default property blending: T_num = 178.312174 vs T_exact = 178.482866,
        err = 1.71e-01 C  (< the plan's 0.5 C bound). The gap is NOT a wiring
        error: once phi > 0 the solver blends rho -> rho_liquid and
        cp -> cp_liquid, while the closed-form H(T) above assumes the fixed
        solid slope rho_s*cp_solid. That is a property-model difference and it
        is what this arm measures.
      * constant properties (cp_liquid=cp_solid, k_liquid=k_solid,
        rho_liquid=rho_s -- the plan's step-3 escape hatch, and exactly the
        assumption the analytic solution makes): T_num = T_exact to
        err = 3.13e-13 C, i.e. machine precision. This is the real wiring gate.
    """
    from heatr3d import T_from_enthalpy, enthalpy_from_T
    n = 16
    grid = Grid(n=n, L=0.060)
    part = np.ones((n, n, n), dtype=bool)
    p = dataclasses.replace(Params(), phase_update="enthalpy", conv_h=0.0)
    q_val = 2.0e6                                # W/m^3, uniform
    q = np.full((n, n, n), q_val)
    t_end = 100.0                                # lands mid-plateau (see above)
    # analytic: uniform state, no gradients -> pure source integration
    rho_s = p.rho_powder + p.rho_rel * (p.rho_solid - p.rho_powder)
    rho_cp = rho_s * p.cp_solid
    rho_L = rho_s * p.latent_j_per_kg
    H_end = enthalpy_from_T(np.array([p.preheat_c]), rho_cp, rho_L, p) \
        + q_val * t_end
    T_exact = float(T_from_enthalpy(H_end, rho_cp, rho_L, p)[0])
    assert p.t_pc_c - p.dt_pc_c / 2 < T_exact < p.t_pc_c + p.dt_pc_c / 2

    res = run(grid, part, p, qrf_override=q, max_time_s=t_end,
              phi_target=2.0)                    # never stop early
    assert abs(float(res.T_final.mean()) - T_exact) < 0.5
    assert float(res.T_final.std()) < 1e-6

    # constant-property arm: the analytic solution's own assumptions
    p_const = dataclasses.replace(p, cp_liquid=p.cp_solid, k_liquid=p.k_solid,
                                  rho_liquid=rho_s)
    res_c = run(grid, part, p_const, qrf_override=q, max_time_s=t_end,
                phi_target=2.0)
    assert abs(float(res_c.T_final.mean()) - T_exact) < 0.05
    assert float(res_c.T_final.std()) < 1e-6
    assert res_c.clamp_bound is False
    assert abs(res_c.energy_residual_frac) < 1e-9


def test_conduction_decay_matches_fourier_mode():
    """No source, no convection, uniform powder medium, initial condition =
    lowest cosine Fourier mode compatible with Neumann walls. The mode decays
    as exp(-alpha k^2 t) exactly; second-order spatial accuracy expected.

    Three nested references, all measured 2026-07-30 (./.venv312, n=24,
    L=0.060, t_end=400 s, 8000 forward-Euler steps of dt=0.05,
    alpha = k_powder/(rho_powder*cp_powder) = 3.7504e-07 m^2/s):

      continuous  amp_exact = 5 exp(-alpha k^2 t)      rel err -1.566e-03
      semi-disc.  amp0 exp(-lambda_d t)                rel err -1.054e-05
      fully disc. amp0 (1 - lambda_d dt)^nsteps        rel err +2.741e-13

    with the discrete decay rate lambda_d = alpha (2/h^2)(1 - cos(k h)) (the
    plan's step-4 documented form) and amp0 = 5 cos(pi/2n): the cell-centred
    grid never samples the mode's true peak, so the observed (max-min)/2 starts
    at 5 cos(pi/48) = 4.98929, not 5. That sampling factor (-2.14e-03) is why
    the raw discrete-rate comparison looks WORSE than the continuous one --
    the two errors partially cancel in the continuous form. Once both discrete
    effects are accounted for, the solver reproduces the analytic mode to
    2.7e-13: the discrete Laplacian eigenvalue, the zero-flux (Neumann) wall
    treatment, the harmonic face averaging on a uniform k, the uniform property
    maps and the explicit time integrator are all exactly as intended.

    Spatial convergence (documented, not asserted -- n=48 doubles runtime):
    the continuous-reference error is -1.566e-03 at n=24 and -3.992e-04 at
    n=48, ratio 3.92 ~ 4, i.e. second order in h as expected.
    """
    n = 24
    grid = Grid(n=n, L=0.060)
    part = np.zeros((n, n, n), dtype=bool)     # all powder, no part
    p = dataclasses.replace(Params(), conv_h=0.0)
    q = np.zeros((n, n, n))
    kx = np.pi / grid.L
    x = grid.x.reshape(-1, 1, 1)
    T0 = p.preheat_c + 5.0 * np.cos(kx * (x + grid.L / 2.0)) * np.ones((n, n, n))
    t_end = 400.0
    res = run(grid, part, p, qrf_override=q, max_time_s=t_end, phi_target=2.0,
              T0_override=T0)
    alpha = p.k_powder / (p.rho_powder * p.cp_powder)
    decay = np.exp(-alpha * kx ** 2 * t_end)
    amp_num = float((res.T_final.max() - res.T_final.min()) / 2.0)
    amp_exact = 5.0 * decay
    assert abs(amp_num - amp_exact) / amp_exact < 0.05
    # discrete references (see docstring): grid sampling factor + discrete rate
    amp0 = 5.0 * np.cos(np.pi / (2 * n))
    lam_d = alpha * (2.0 / grid.h ** 2) * (1.0 - np.cos(kx * grid.h))
    amp_semi = amp0 * np.exp(-lam_d * t_end)
    assert abs(amp_num - amp_semi) / amp_semi < 1e-4
    amp_full = amp0 * (1.0 - lam_d * p.dt_s) ** int(t_end / p.dt_s)
    assert abs(amp_num - amp_full) / amp_full < 1e-9
    # the mean is conserved exactly: zero-flux walls, no source, no sinks
    assert abs(float(res.T_final.mean()) - p.preheat_c) < 1e-9


def test_eqs_uniform_medium_is_parallel_plate():
    """Characterization: on a uniform medium the EQS solve must reproduce the
    parallel-plate field exactly (V linear in y, constant in x and z).

    Result: PASS on BOTH solver paths (measured 2026-07-30, ./.venv312):
      n=24 (N=13824 <= 50000 -> direct spsolve)
          max|V(y) - linspace(v_lo, v_hi, n)| = 1.25e-11 V (1.4e-14 relative)
          max transverse std = 9.46e-13 V,  max|Im V| = 8.3e-17 V
      n=40 (N=64000  > 50000 -> ILU-preconditioned BiCGSTAB)
          max|V(y) - linear| = 3.14e-06 V (3.7e-09 relative)
          max transverse std = 4.84e-06 V,  max|Im V| = 2.0e-16 V
    The n=40 arm matters because every production grid n >= 37 takes the
    iterative branch; its error floor is set by the solver's own rtol=1e-8, so
    it is asserted at 1e-6 relative, not at the direct path's 1e-9.

    FINDING (documented, not a failure -- see docs/superpowers/plans/
    s1-findings.md section 6): the Dirichlet rows sit at the CELL CENTRES of
    the first/last y layers, so the effective plate gap is (n-1)h, not L. The
    measured uniform field is |E_y| = v_lo/((n-1)h) = 14956.521739 V/m at n=24
    (ptp 2.6e-09), matching that to 13 digits and exceeding the nominal
    v_lo/L = 14333.33 V/m by L/(L-h) = 4.3%. The bias is grid dependent
    (0.5% at n=200), but it is a UNIFORM scale factor on E, and
    compute_qrf_3d renormalizes Q to a fixed total absorbed power, so it
    cancels out of Qrf on a uniform medium. It is not corrected here.
    """
    from heatr3d import build_gamma, solve_eqs_3d
    for n, tol_lin, tol_std in ((24, 1e-6, 1e-9), (40, 1e-6, 1e-7)):
        grid = Grid(n=n, L=0.060)
        part = np.zeros((n, n, n), dtype=bool)      # uniform virgin bed
        p = Params()
        gamma = build_gamma(part, p, None, h=grid.h)
        V = solve_eqs_3d(gamma, grid, p)
        Vr = np.real(V)
        # linear in y between the plates, uniform in x and z
        y_prof = Vr.mean(axis=(0, 2))
        y_lin = np.linspace(p.v_lo, p.v_hi, n)
        assert np.max(np.abs(y_prof - y_lin)) < tol_lin * abs(p.v_lo), n
        assert float(Vr.std(axis=(0, 2)).max()) < tol_std * abs(p.v_lo), n
        # uniform medium -> no phase lag anywhere
        assert float(np.max(np.abs(np.imag(V)))) < 1e-12 * abs(p.v_lo), n
        # uniform E_y at the cell-centre plate gap (n-1)h (see FINDING above)
        Ey = -np.gradient(Vr, grid.h, edge_order=1)[1]
        assert abs(float(Ey.mean()) - p.v_lo / ((n - 1) * grid.h)) < 1e-6, n
        assert float(np.ptp(Ey)) < 1e-2, n


def test_legacy_default_is_unchanged():
    """Default Params must still take the legacy apparent_cp path.

    DEVIATION from the plan's `np.array_equal` form, with evidence: heatr3d's
    default path is NOT bit-reproducible run-to-run. Repeating this identical
    n=32 / 20 s solve in one process, run 3 of 7 differed from runs 1-2 by
    max|dT| = 7.1e-15 C (2026-07-30, ./.venv312: numpy 1.26.4, scipy 1.13.1,
    OpenBLAS MAX_THREADS=3). The EQS solve is exactly reproducible (6/6
    np.array_equal on V), so the drift is inside the thermal loop; root-causing
    it is a separate S1 item, not this task. Asserting array_equal here would
    commit a known-flaky gate, so the tolerance is 1e-12 C -- still ~1e12 x
    tighter than any real scheme change (the enthalpy branch moves T by O(1) C).

    The bit-for-bit constraint itself was verified out-of-band across the
    Task-4 commit boundary: HEAD (5ae12e5..c4bd6b8 state) vs this working tree,
    default Params, n=32 sphere, 30 s -> np.array_equal True on T_phi90, Qrf,
    phi_final and an identical sigma_T = 21.19205274948017.
    """
    grid, part, p = small_sphere_case()
    r1 = run(grid, part, p, max_time_s=20.0)
    r2 = run(grid, part, p, max_time_s=20.0)
    a = r1.T_phi90 if r1.T_phi90 is not None else np.zeros(1)
    b = r2.T_phi90 if r2.T_phi90 is not None else np.zeros(1)
    assert np.allclose(a, b, rtol=0.0, atol=1e-12)
    assert Params().phase_update == "apparent_cp"
