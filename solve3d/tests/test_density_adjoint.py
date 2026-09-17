"""Stage B B1: FD gates for the rho+T density co-state adjoint.

The KS scalar key in solve3d.ceiling.peak_temp is ``ks_aggregate_c`` (the plan
draft said ``ks_c``; the real return dict uses ``ks_aggregate_c`` -- verified in
solve3d/ceiling.py:69). Every gate here reads the real key.
"""
from __future__ import annotations

import numpy as np

from solve3d import ceiling


def test_densify_rate_jac_matches_fd():
    from solve3d import forward as fwd
    p = fwd.ForwardParams(power_density_w_per_m3=636619.7723675814)
    T = np.array([250.0, 300.0, 400.0]); phi = np.array([0.2, 0.6, 1.0])
    rho = np.array([0.6, 0.8, 0.95])
    dT, drho = fwd.densify_rate_jac(T, phi, rho, p)     # partials of densify_rate
    h = 1e-3
    r0 = fwd.densify_rate(T, phi, rho, p)
    fd_T = (fwd.densify_rate(T + h, phi, rho, p) - r0) / h
    fd_r = (fwd.densify_rate(T, phi, rho + h, p) - r0) / h
    assert np.allclose(dT, fd_T, rtol=1e-4, atol=1e-12), (dT, fd_T)
    assert np.allclose(drho, fd_r, rtol=1e-4, atol=1e-12), (drho, fd_r)


def test_densify_rate_dphi_matches_fd():
    """The phi partial (Task 3 uses it; the forward's phi_now = phase_fraction(T)
    couples rho evolution back to T). Probed away from phi=1 where (1-phi)^0.8
    has an infinite slope kink."""
    from solve3d import forward as fwd
    p = fwd.ForwardParams(power_density_w_per_m3=636619.7723675814)
    T = np.array([250.0, 300.0, 400.0]); phi = np.array([0.2, 0.6, 0.85])
    rho = np.array([0.6, 0.8, 0.95])
    part = fwd.densify_rate_partials(T, phi, rho, p)
    h = 1e-6
    r0 = fwd.densify_rate(T, phi, rho, p)
    fd_phi = (fwd.densify_rate(T, phi + h, rho, p) - r0) / h
    assert np.allclose(part["dphi"], fd_phi, rtol=1e-4, atol=1e-10), (part["dphi"], fd_phi)


def test_dks_peak_ds_matches_fd_coarse():
    from solve3d import density_adjoint as da
    case = da.build_coarse_case()          # small mesh, few design cells, 0.40x
    v = case.design_point()                # a non-degenerate interior design vector
    g = da.dks_peak_ds(case, v)            # the adjoint gradient of the end-state KS peak
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)

    def ks_of(vv):
        return da.ks_peak_forward(case, vv)   # scalar end-state KS peak
    base = ks_of(v); h = 1e-4
    idx = da.probe_indices(case)           # a handful of high-sensitivity design cells
    for i in idx:
        vp = v.copy(); vp[i] += h; vm = v.copy(); vm[i] -= h
        fd = (ks_of(vp) - ks_of(vm)) / (2 * h)     # central difference
        assert abs(fd - g[i]) <= 1e-6 * max(1.0, abs(fd)) + 1e-9, (i, fd, g[i])


def test_march_checkpoints_lays_anchors_and_matches_forward():
    """The anchor march lays a (T,rho) checkpoint every k steps and reproduces the
    store-all march's end state bit-for-bit (deterministic explicit path)."""
    import math
    from solve3d import density_adjoint as da
    case = da.build_coarse_case(); v = case.design_point()
    k = 100
    T_end, rho_end, anchors, F = da._march_checkpoints(case, v, k)
    assert len(anchors) == math.ceil(case.n_steps / k)
    assert 0 in anchors                                   # first anchor is the initial state
    T0, r0 = anchors[0]
    assert T0.shape == T_end.shape and r0.shape == rho_end.shape
    T_ref, rho_ref, _caches, _F = da._march(case, v, keep_cache=True)
    assert np.array_equal(T_end, T_ref)                   # bit-identical end state
    assert np.array_equal(rho_end, rho_ref)


def test_dks_peak_ds_checkpointed_is_bit_identical():
    """THE acceptance gate: the checkpointed gradient equals the store-all gradient
    bit-for-bit on the explicit path, for several k including a non-divisor of
    n_steps (ragged final segment)."""
    from solve3d import density_adjoint as da
    case = da.build_coarse_case(); v = case.design_point()
    g_full = da.dks_peak_ds(case, v)
    for k in (50, 100, 250, 300):        # 250 does not divide 900 -> ragged tail
        g_ck = da.dks_peak_ds(case, v, checkpoint_interval=k)
        assert np.array_equal(g_ck, g_full), (k, float(np.abs(g_ck - g_full).max()))


def test_dks_peak_ds_checkpointed_matches_fd():
    """Non-vacuity: the checkpointed gradient still passes the frozen FD gate
    (<=1e-6), with NO tolerance widening -- mirrors test_dks_peak_ds_matches_fd_coarse."""
    from solve3d import density_adjoint as da
    case = da.build_coarse_case(); v = case.design_point()
    g = da.dks_peak_ds(case, v, checkpoint_interval=100)

    def ks_of(vv):
        return da.ks_peak_forward(case, vv)
    h = 1e-4
    for i in da.probe_indices(case):
        vp = v.copy(); vp[i] += h; vm = v.copy(); vm[i] -= h
        fd = (ks_of(vp) - ks_of(vm)) / (2 * h)
        assert abs(fd - g[i]) <= 1e-6 * max(1.0, abs(fd)) + 1e-9, (i, fd, g[i])


def test_drop_lambda_rho_fails_fd_checkpointed():
    """Safety property survives checkpointing: ablating the density co-state must
    still break the FD gate even on the checkpointed path (the checkpointing does
    not mask a wrong gradient)."""
    from solve3d import density_adjoint as da
    case = da.build_coarse_case(); v = case.design_point()
    g_bad = da.dks_peak_ds(case, v, _drop_density_costate=True,
                           checkpoint_interval=100)
    h = 1e-4
    worst = 0.0
    for i in da.probe_indices(case):
        vp = v.copy(); vp[i] += h; vm = v.copy(); vm[i] -= h
        fd = (da.ks_peak_forward(case, vp) - da.ks_peak_forward(case, vm)) / (2 * h)
        worst = max(worst, abs(fd - g_bad[i]) / max(1.0, abs(fd)))
    assert worst > 1e-3, "checkpointed path must still detect the dropped co-state"


def test_march_matches_production_densify():
    """The gate forward _march must reproduce production march_enthalpy (densify)
    -- the forward the B2 arbiter reads -- or the solve would chase a peak the
    arbiter does not judge."""
    from solve3d import density_adjoint as da
    d = da.march_fidelity_check()
    assert d["n_substeps_used_prod"] == 1
    assert d["rel_T_in_part"] < 1e-6, d
    assert d["rel_mean_rho"] < 1e-6, d
    assert d["agree"]


def test_drop_lambda_rho_fails_fd():
    from solve3d import density_adjoint as da
    case = da.build_coarse_case(); v = case.design_point()
    g_bad = da.dks_peak_ds(case, v, _drop_density_costate=True)   # ablation flag
    base = da.ks_peak_forward(case, v); h = 1e-4
    worst = 0.0
    for i in da.probe_indices(case):
        vp = v.copy(); vp[i] += h; vm = v.copy(); vm[i] -= h
        fd = (da.ks_peak_forward(case, vp) - da.ks_peak_forward(case, vm)) / (2 * h)
        worst = max(worst, abs(fd - g_bad[i]) / max(1.0, abs(fd)))
    assert worst > 1e-3, "dropping the density co-state must break the gradient"


def test_peak_temp_vjp_matches_fd():
    rng = np.random.default_rng(0)
    n = 40
    T = 200.0 + 40.0 * rng.random(n)          # C, spread across a realistic band
    w = rng.random(n)
    mask = np.ones(n, dtype=bool)
    g = ceiling.peak_temp_vjp(T, weights=w, mask=mask)   # analytic dThat/dT_i
    assert g.shape == (n,)
    assert abs(g.sum() - 1.0) < 1e-10                     # softmax weights sum to 1
    base = ceiling.peak_temp(T, weights=w, mask=mask)["ks_aggregate_c"]
    h = 1e-4
    for i in (0, 7, 23, n - 1):
        Tp = T.copy(); Tp[i] += h
        fd = (ceiling.peak_temp(Tp, weights=w, mask=mask)["ks_aggregate_c"] - base) / h
        assert abs(fd - g[i]) < 1e-5, (i, fd, g[i])
