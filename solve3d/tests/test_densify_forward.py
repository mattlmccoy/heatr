"""Stage A Task 1: the densify FORWARD port.

RUNS IN THE SPIKE ENV (dolfinx 0.11 complex build):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_densify_forward.py

The end-to-end cross-family EQUIVALENCE gate against heatr3d's own densify march
is the CLI driver `python -m solve3d.densify_forward --gate` (a ~minute small
case, not a unit test); its verdict lands in
solve3d/results/densify_equivalence.json and the mutation in
solve3d/results/densify_equivalence_mutation.json.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# The ported densification rate is bit-for-bit heatr3d.densify_rate
# --------------------------------------------------------------------------- #
def test_densify_rate_matches_heatr3d_bit_for_bit():
    import heatr3d
    from solve3d import forward

    rng = np.random.default_rng(0)
    T = rng.uniform(23.0, 300.0, size=500)
    phi = rng.uniform(0.0, 1.0, size=500)
    rho_rel = rng.uniform(0.4, 1.0, size=500)

    p_s = forward.ForwardParams()
    p_h = heatr3d.Params(phase_update="enthalpy")
    got = forward.densify_rate(T, phi, rho_rel, p_s)
    want = heatr3d.densify_rate(T, phi, rho_rel, p_h)
    # identical arithmetic + identical R_GAS -> exact equality
    assert np.array_equal(got, want)


def test_densify_rate_is_nonnegative_and_zero_at_full_density():
    from solve3d import forward
    p = forward.ForwardParams()
    T = np.array([200.0, 250.0, 300.0])
    phi = np.array([1.0, 1.0, 1.0])
    assert np.all(forward.densify_rate(T, phi, np.array([0.6, 0.7, 0.8]), p) > 0)
    # porosity gate: rho_rel = 1.0 -> (1 - rho)^exp = 0 -> zero rate
    assert np.allclose(forward.densify_rate(T, phi, np.ones(3), p), 0.0)


# --------------------------------------------------------------------------- #
# Off path is untouched; a zero-rate densify reproduces the constant-rho march
# --------------------------------------------------------------------------- #
def _small_case():
    from solve3d import forward
    msh = forward.box_mesh(8)
    pred = forward.in_part_predicate("circle")
    return forward, msh, pred


def test_densify_false_carries_no_densify_state():
    """The OFF path (densify=False, the default) returns the Phase A dict with
    NONE of the densify keys -- the branch is inert."""
    forward, msh, pred = _small_case()
    p = forward.ForwardParams(conv_h=0.0)
    out = forward.march_enthalpy(msh, p, in_part=pred, q_uniform=3.0e6,
                                 max_time_s=3.0, phi_target=2.0)
    for k in ("densify", "rho_final", "true_peak_T_c", "part_mean_rho",
              "rho_traj_t_s"):
        assert k not in out, f"densify key {k!r} leaked into the OFF path"


def test_zero_rate_densify_reproduces_constant_rho_march():
    """With the densification rate forced to zero (dens_k0_ss = 0,
    dens_geom_factor = 0) rho_rel never changes, so the per-step density-property
    recompute equals the OFF-path constants and the temperature trajectory is
    reproduced. This pins that the ONLY thing the densify branch changes is via
    drho -- the property/conduction/enthalpy arithmetic is untouched."""
    forward, msh, pred = _small_case()
    common = dict(conv_h=0.0)
    p_off = forward.ForwardParams(**common)
    p_on = forward.ForwardParams(dens_k0_ss=0.0, dens_geom_factor=0.0, **common)

    off = forward.march_enthalpy(msh, p_off, in_part=pred, q_uniform=3.0e6,
                                 max_time_s=4.0, phi_target=2.0)
    on = forward.march_enthalpy(msh, p_on, in_part=pred, q_uniform=3.0e6,
                                max_time_s=4.0, phi_target=2.0,
                                densify=True, stop_mean_rho=1.1)
    assert on["densify"] is True
    # rho held at the initial value everywhere (zero rate)
    assert np.allclose(on["rho_final"], p_on.rho_rel, atol=0.0)
    # temperature trajectory reproduced to tight tolerance (cell-average of a
    # constant rho re-derives k to within float round-off)
    np.testing.assert_allclose(on["T"], off["T"], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(on["T_max_c"], off["T_max_c"], rtol=1e-12)


def test_true_peak_is_the_trajectory_maximum_and_present():
    """The ceiling quantity true_peak_T_c is the running max of the in-part
    temperature and is >= the end-state part peak. In a monotone heating densify
    the two coincide; the guarantee is that the peak is never UNDER the
    end-state (the false-green a snapshot peak would create)."""
    forward, msh, pred = _small_case()
    p = forward.ForwardParams(conv_h=0.0)
    out = forward.march_enthalpy(msh, p, in_part=pred, q_uniform=3.0e6,
                                 max_time_s=6.0, phi_target=2.0,
                                 densify=True, stop_mean_rho=0.60)
    assert np.isfinite(out["true_peak_T_c"])
    assert out["true_peak_T_c"] >= out["T_end_max_c"] - 1e-9
    assert 0.0 <= out["true_peak_step_index"] <= out["n_steps_taken"]
    # rho evolved upward from the initial 0.55 under a real (nonzero) rate
    assert out["part_mean_rho"] >= p.rho_rel
