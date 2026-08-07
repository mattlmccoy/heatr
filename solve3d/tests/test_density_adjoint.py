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
