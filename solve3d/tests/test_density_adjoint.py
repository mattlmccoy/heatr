"""Stage B B1: FD gates for the rho+T density co-state adjoint.

The KS scalar key in solve3d.ceiling.peak_temp is ``ks_aggregate_c`` (the plan
draft said ``ks_c``; the real return dict uses ``ks_aggregate_c`` -- verified in
solve3d/ceiling.py:69). Every gate here reads the real key.
"""
from __future__ import annotations

import numpy as np

from solve3d import ceiling


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
