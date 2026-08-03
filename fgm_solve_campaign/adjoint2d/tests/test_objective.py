"""Read-state logic and the seed derivatives, checked against finite differences.

These are the pieces that silently invert a result if they are wrong: the
melt-onset bracket, the implicit-function-theorem coefficients, and the two
seed gradients.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from adjoint2d import objective as obj


class _Case:
    def __init__(self, ny=6, nx=6, ambient=23.0, t_pc=180.0, dt_pc=10.0):
        m = np.zeros((ny, nx), dtype=bool)
        m[1:5, 1:5] = True
        self.part_mask = m
        self.pins = SimpleNamespace(ambient_c=ambient, t_pc_c=t_pc, dt_pc_c=dt_pc)


def _traj(phi, sigma):
    return SimpleNamespace(mean_phi_part=np.asarray(phi), sigma_T=np.asarray(sigma),
                           mean_T_part_c=23.0 + 200.0 * np.asarray(phi),
                           n_outer=len(phi), time_s=np.arange(len(phi)) + 1.0)


def test_bracket_and_theta_locate_the_crossing():
    tr = _traj([0.10, 0.60, 0.88, 0.92, 0.99], [1.0, 2.0, 3.0, 2.5, 2.0])
    rs = obj.read_states(tr)
    assert rs.melt_onset_index == 3
    assert rs.bracket_index == 2
    assert rs.theta == pytest.approx((0.90 - 0.88) / (0.92 - 0.88))
    assert rs.heating_peak_index == 2   # argmax of sigma_T among pre-melt steps


def test_melt_not_reached_is_loud():
    tr = _traj([0.1, 0.4, 0.7, 0.89], [1.0, 2.0, 3.0, 4.0])
    rs = obj.read_states(tr)
    assert rs.melt_onset_index is None
    m = obj.scored_metrics(tr, _Case())
    assert m["status"] == "NOT_REACHED"
    assert m["melt_onset_sigma_T_c"] is None


def test_sigma_T_seed_matches_finite_difference():
    case = _Case()
    rng = np.random.default_rng(0)
    T = 150.0 + 30.0 * rng.random(case.part_mask.shape)
    _J, g = obj.sigma_T_and_seed(T, case)
    d = np.zeros_like(T)
    d[case.part_mask] = rng.standard_normal(int(case.part_mask.sum()))
    eps = 1e-6
    fd = (obj.sigma_T_and_seed(T + eps * d, case)[0]
          - obj.sigma_T_and_seed(T - eps * d, case)[0]) / (2 * eps)
    ana = float(np.sum(g * d))
    assert abs(fd - ana) < 1e-7 * max(1.0, abs(ana))


def test_phi_bar_seed_matches_finite_difference_inside_the_ramp():
    case = _Case()
    rng = np.random.default_rng(1)
    T = 178.0 + 4.0 * rng.random(case.part_mask.shape)   # entirely inside the ramp
    _v, g = obj.phi_bar_and_seed(T, case)
    d = np.zeros_like(T)
    d[case.part_mask] = rng.standard_normal(int(case.part_mask.sum()))
    eps = 1e-6
    fd = (obj.phi_bar_and_seed(T + eps * d, case)[0]
          - obj.phi_bar_and_seed(T - eps * d, case)[0]) / (2 * eps)
    assert abs(fd - float(np.sum(g * d))) < 1e-9


def test_saturated_cells_contribute_zero_to_the_phi_bar_seed():
    """The phase ramp clip is a subgradient: cells pinned at phi = 1 must not
    move the melt-fraction sensitivity. This is the mechanism the L2 gate
    measured as the source of non-smoothness at the production ramp width."""
    case = _Case()
    T = np.full(case.part_mask.shape, 250.0)   # far above the ramp -> all phi = 1
    v, g = obj.phi_bar_and_seed(T, case)
    assert v == pytest.approx(1.0)
    assert np.all(g == 0.0)


def test_ift_theta_coefficients_match_a_direct_derivative():
    """dtheta/dp_n and dtheta/dp_{n+1} for theta = (0.9 - p_n)/(p_{n+1} - p_n)."""
    p_n, p_n1 = 0.885, 0.913

    def theta(a, b):
        return (obj.PHI_TARGET - a) / (b - a)

    th = theta(p_n, p_n1)
    slope = p_n1 - p_n
    ana_n = (th - 1.0) / slope
    ana_n1 = -th / slope
    h = 1e-7
    fd_n = (theta(p_n + h, p_n1) - theta(p_n - h, p_n1)) / (2 * h)
    fd_n1 = (theta(p_n, p_n1 + h) - theta(p_n, p_n1 - h)) / (2 * h)
    assert abs(fd_n - ana_n) < 1e-5 * abs(ana_n)
    assert abs(fd_n1 - ana_n1) < 1e-5 * abs(ana_n1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:warnings"]))
