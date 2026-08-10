"""Two-sided per-node dopant actuator: the cap/bounds pure logic, the
default-one-sided byte-identity guard, and the combined AL-gradient FD gate at a
node BOOSTED above 1.0 (the branch a correct one-sided gate does NOT exercise).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_two_sided.py -x -q
"""
import inspect

import numpy as np
import pytest

from solve3d import forward as fwd
from solve3d import stage_b3 as b3
from solve3d import stage_b4 as b4
from solve3d import two_sided as ts


# --------------------------------------------------------------------------- #
# Pure logic: the physical cap and the design box
# --------------------------------------------------------------------------- #
def test_default_bounds_are_one_sided():
    assert ts.design_bounds() == (0.0, 1.0)
    assert ts.design_bounds(ts.MAX_SAT_ONE_SIDED) == (0.0, 1.0)
    assert ts.is_two_sided(1.0) is False


def test_two_sided_bounds_open_above_one():
    assert ts.design_bounds(2.0) == (0.0, 2.0)
    assert ts.is_two_sided(2.0) is True
    assert ts.is_two_sided(ts.MAX_SAT_DEFAULT_TWO_SIDED) is True


def test_cap_rejects_below_one():
    with pytest.raises(ValueError):
        ts.design_bounds(0.8)


def test_cap_stays_under_sigma_coupling_clip():
    # the whole point of picking a defensible cap: a boosted node must never ride
    # the numerical sigma clip (which would zero its subgradient and silently
    # kill the two-sided gradient).
    p = fwd.ForwardParams()
    sig_at_cap = (p.sigma_virgin
                  + ts.MAX_SAT_DEFAULT_TWO_SIDED * (p.sigma_doped - p.sigma_virgin))
    assert sig_at_cap < fwd.SIGMA_COUPLING_CLIP_HI * p.sigma_doped


# --------------------------------------------------------------------------- #
# Default byte-identity: the heavy solve keeps max_sat=1.0 unless opted in
# --------------------------------------------------------------------------- #
def test_run_solve_al_b4_defaults_to_one_sided():
    sig = inspect.signature(b4.run_solve_al_b4)
    assert "max_sat" in sig.parameters
    assert sig.parameters["max_sat"].default == 1.0


# --------------------------------------------------------------------------- #
# The combined AL-gradient FD gate WITH a node boosted above 1.0
# --------------------------------------------------------------------------- #
def test_al_grad_two_sided_matches_fd_at_boosted_node():
    # Boost the nodes the AL CEILING term is most sensitive to (probe_indices)
    # ABOVE 1.0, PLUS the coldest node (the physical core-feed case), then FD-gate
    # at EXACTLY those boosted nodes. This is the ungameable test: the ceiling
    # gradient must be correct when it flows through a node whose sat > 1.0 -- the
    # branch a correct one-sided gate never exercises. (Probing NON-boosted high-
    # |g| nodes while other nodes are boosted instead measures the frozen gate's
    # own clip-kink fragility at a shifted operating point, not the two-sided
    # branch; that is already covered by test_stage_b3 at the un-boosted point.)
    case = b3.build_al_coarse_case(lam=500.0, mu=1.0e4, t_target=204.0)
    v = case.design_point()
    boost = list(dict.fromkeys([int(i) for i in case.probe_indices(6)]
                               + [int(np.argmin(v))]))
    for j in boost:
        v[j] = 1.5

    # the actuator does its job: sat=1.5 boosts conductivity ABOVE sigma_doped
    p = fwd.ForwardParams()
    sig = case.da_case.tc.design_to_sigma(np.array([1.5]))[0]
    assert sig > p.sigma_doped

    J, g = b3.al_objective_and_grad(case, v)
    assert np.all(np.isfinite(g))

    h = 1e-4                       # the frozen FD step (unchanged)
    worst = 0.0
    for i in boost:
        vp = v.copy(); vp[i] += h
        vm = v.copy(); vm[i] -= h
        fd = (b3.al_objective_and_grad(case, vp)[0]
              - b3.al_objective_and_grad(case, vm)[0]) / (2 * h)
        worst = max(worst, abs(fd - g[i]) / max(1.0, abs(fd)))
        assert abs(fd - g[i]) <= 1e-6 * max(1.0, abs(fd)) + 1e-9, (i, v[i], fd, g[i])
    print(f"\n[two-sided AL-grad FD gate] worst_rel_err={worst:.3e} "
          f"(frozen 1e-6, {len(boost)} nodes boosted to sat=1.5)")

    # mutation: dropping the AL/ceiling term must break the boosted node's
    # gradient -- proves the ceiling path is live at sat>1.0, not a no-op.
    _J, g_drop = b3.al_objective_and_grad(case, v, _drop_al_term=True)
    j = boost[0]
    assert abs(g[j] - g_drop[j]) > 1e-3 * max(1.0, abs(g[j])), (g[j], g_drop[j])
