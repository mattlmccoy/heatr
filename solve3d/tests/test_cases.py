"""solve3d Task 1 support tests -- heatr3d-side anchor cases.

RUNS IN THE geo-prewarp VENV (heatr3d needs scipy; the dolfinx spike env has
neither heatr3d nor scipy):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ./.venv312/bin/python -m pytest solve3d/tests/test_cases.py
"""
from __future__ import annotations

import dataclasses

import numpy as np

import heatr3d
from solve3d import cases


def test_segmented_march_reproduces_a_monolithic_run():
    """cases.march_sampled chains heatr3d.run through T0_override so the
    part-mean heating CURVE can be sampled without touching heatr3d.py.

    That chaining is only legitimate if it is arithmetically the same march.
    With densify=False the entire solver state is T (rho_rel is constant and
    phase is a pure function of T) and the drive is frozen by qrf_override, so
    segmentation must be exact -- pinned here at a relative 1e-12, never
    array_equal (BLAS-noise finding from S1)."""
    n = 16
    grid = heatr3d.Grid(n=n)
    part = heatr3d.make_geometry(grid, "cylinder", diam=0.020)
    p = dataclasses.replace(heatr3d.Params(), phase_update="enthalpy")
    rng = np.random.default_rng(7)
    q = np.zeros(part.shape)
    q[part] = 2.0e6 * (1.0 + 0.2 * rng.random(int(part.sum())))

    mono = heatr3d.run(grid, part, p, qrf_override=q, max_time_s=300.0,
                       phi_target=0.90)
    assert mono.reached, "monolithic anchor must reach melt onset"

    seg = cases.march_sampled(grid, part, p, q, max_time_s=300.0,
                              phi_target=0.90, sample_dt_s=10.0)

    assert seg["reached"] is True
    assert abs(seg["t90_s"] - mono.t_phi90_s) <= 1e-12 * mono.t_phi90_s
    assert abs(seg["sigma_T_c"] - mono.sigma_T) <= 1e-12 * abs(mono.sigma_T)
    assert abs(seg["T_max_c"] - mono.T_max_c) <= 1e-12 * abs(mono.T_max_c)
    assert np.allclose(seg["T_phi90"], mono.T_phi90, rtol=1e-12, atol=0.0)
    # the sampled curve must start at preheat and be monotone in time
    t = np.asarray(seg["curve_t_s"])
    c = np.asarray(seg["curve_part_mean_T_c"])
    assert t[0] == 0.0 and abs(c[0] - p.preheat_c) < 1e-12
    assert np.all(np.diff(t) > 0)
    assert np.all(np.diff(c) > 0)
