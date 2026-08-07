"""Stage B3 tests: AL pure-logic (multiplier/shift/mu/honest-null) + the
combined AL-gradient FD gate (reuses the FD-gated B1 dks_peak_ds).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_stage_b3.py -x -q
"""
import numpy as np

from solve3d import stage_b3 as b3


# --------------------------------------------------------------------------- #
# Task 1: pure-logic (no physics)
# --------------------------------------------------------------------------- #
def test_multiplier_update_inequality_kkt():
    # lambda_new = max(0, lambda + mu*g); stays >= 0, grows when violated (g>0),
    # decays when satisfied (g<0)
    assert b3.multiplier_update(lam=0.0, mu=1e3, g=0.5) == 0.5e3
    assert b3.multiplier_update(lam=100.0, mu=1e3, g=-1.0) == 0.0     # clipped at 0
    assert b3.multiplier_update(lam=2000.0, mu=1e3, g=-1.0) == 1000.0


def test_restoration_shift_targets_true_peak():
    # T_target = ceiling - Delta, Delta EMA-damped from (true_arbiter - ks_solve)
    d = b3.restoration_shift(ceiling=250.0, true_peak=250.69, ks_peak=248.45,
                             prev_delta=0.0, ema=0.5)
    assert abs(d["delta"] - 0.5 * (250.69 - 248.45)) < 1e-9   # first EMA step from 0
    assert abs(d["t_target"] - (250.0 - d["delta"])) < 1e-9


def test_mu_escalation_when_violation_stalls():
    # <50% drop -> grow
    assert b3.mu_escalation(mu=1e3, viol_prev=2.0, viol_now=1.9,
                            factor=5.0, shrink=0.5) == 5e3
    # good drop -> hold
    assert b3.mu_escalation(mu=1e3, viol_prev=2.0, viol_now=0.5,
                            factor=5.0, shrink=0.5) == 1e3


def test_al_gradient_factor():
    # active branch factor = max(0, lambda + mu*g); zero when lambda+mu*g <= 0
    assert b3.al_gradient_factor(lam=0.0, mu=1e3, g=0.01) == 10.0
    assert b3.al_gradient_factor(lam=0.0, mu=1e3, g=-0.01) == 0.0


def test_honest_null_not_spurious_when_uniform_feasible():
    # a feasible uniform map (under ceiling) => NEVER honest-null, regardless of
    # the shaped endpoint
    v = b3.honest_null_verdict(shaped_true_peak=250.69, uniform_true_peak=239.99,
                               ceiling=250.0)
    assert v["verdict"] != "no_feasible_dopant_at_this_drive"
    v2 = b3.honest_null_verdict(shaped_true_peak=252.0, uniform_true_peak=251.0,
                                ceiling=250.0)
    # only when even uniform is over
    assert v2["verdict"] == "no_feasible_dopant_at_this_drive"


# --------------------------------------------------------------------------- #
# Task 2: the combined AL-gradient FD gate (hinge active)
# --------------------------------------------------------------------------- #
def test_al_combined_grad_matches_fd():
    case = b3.build_al_coarse_case(lam=500.0, mu=1.0e4, t_target=247.8)
    v = case.design_point()
    J, g = b3.al_objective_and_grad(case, v)
    assert np.all(np.isfinite(g))
    h = 1e-4
    for i in case.probe_indices():
        vp = v.copy(); vp[i] += h
        vm = v.copy(); vm[i] -= h
        Jp, _ = b3.al_objective_and_grad(case, vp)
        Jm, _ = b3.al_objective_and_grad(case, vm)
        fd = (Jp - Jm) / (2 * h)
        assert abs(fd - g[i]) <= 1e-6 * max(1.0, abs(fd)) + 1e-9, (i, fd, g[i])
