"""RED-first tests for the shape-fidelity objective.

J(s, t_stop) = sum over the WHOLE domain of w(x) * (phi(x, t_stop) - chi_part(x))^2

It reads melt fraction, not temperature, so the latent plateau cannot flatter
it; it is summed over the bed as well as the part, so melt escaping into the
bed (part growth) costs; and under-melting the part costs symmetrically.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from adjoint2d import shape_objective as so


class _Case:
    def __init__(self, ny=8, nx=8, t_pc=180.0, dt_pc=10.0, ambient=23.0):
        m = np.zeros((ny, nx), dtype=bool)
        m[2:6, 2:6] = True          # 16 part cells of 64
        self.part_mask = m
        self.doped_mask = m
        self.dx = self.dy = 1.0e-3
        self.dA = self.dx * self.dy
        self.pins = SimpleNamespace(t_pc_c=t_pc, dt_pc_c=dt_pc, ambient_c=ambient)


def _T_for_phi(case, phi):
    """Temperature field that produces the requested melt fraction exactly."""
    p = case.pins
    return p.t_pc_c + p.dt_pc_c * (np.asarray(phi, dtype=float) - 0.5)


def test_perfect_match_gives_zero_objective():
    case = _Case()
    T = _T_for_phi(case, case.part_mask.astype(float))
    J, _g = so.shape_J_and_seed(T, case)
    assert J == pytest.approx(0.0)


def test_unmelted_part_costs_one_per_part_cell():
    case = _Case()
    T = np.full(case.part_mask.shape, case.pins.ambient_c)
    J, _g = so.shape_J_and_seed(T, case)
    assert J == pytest.approx(float(case.part_mask.sum()))


def test_bed_melt_costs_the_same_as_part_under_melt():
    """Symmetry is the whole point: melting one bed cell must cost exactly what
    leaving one part cell unmelted costs."""
    case = _Case()
    chi = case.part_mask.astype(float)
    grow = chi.copy()
    grow[1, 2] = 1.0                       # one extra melted bed cell
    shrink = chi.copy()
    shrink[2, 2] = 0.0                     # one part cell left unmelted
    j_grow = so.shape_J_and_seed(_T_for_phi(case, grow), case)[0]
    j_shrink = so.shape_J_and_seed(_T_for_phi(case, shrink), case)[0]
    assert j_grow == pytest.approx(1.0)
    assert j_shrink == pytest.approx(1.0)


def test_seed_matches_a_finite_difference():
    case = _Case()
    rng = np.random.default_rng(0)
    T = 176.0 + 8.0 * rng.random(case.part_mask.shape)   # inside the ramp
    _J, g = so.shape_J_and_seed(T, case)
    d = rng.standard_normal(T.shape)
    eps = 1e-6
    fd = (so.shape_J_and_seed(T + eps * d, case)[0]
          - so.shape_J_and_seed(T - eps * d, case)[0]) / (2 * eps)
    assert abs(fd - float(np.sum(g * d))) < 1e-8 * max(1.0, abs(fd))


def test_cells_outside_the_phase_ramp_have_zero_seed():
    """The phase ramp clip is a subgradient: fully melted and fully solid cells
    contribute no sensitivity."""
    case = _Case()
    T = np.where(case.part_mask, 400.0, -20.0)
    _J, g = so.shape_J_and_seed(T, case)
    assert np.all(g == 0.0)


def test_optimal_stop_is_the_argmin_along_the_trajectory():
    case = _Case()
    chi = case.part_mask.astype(float)
    frames = [np.full(chi.shape, case.pins.ambient_c),      # nothing melted
              _T_for_phi(case, chi),                        # perfect
              _T_for_phi(case, np.ones_like(chi))]          # whole bed melted
    tr = SimpleNamespace(n_outer=3, T_at_end=lambda n: frames[n],
                         mean_phi_part=np.array([0.0, 1.0, 1.0]),
                         time_s=np.array([1.0, 2.0, 3.0]))
    st = so.optimal_stop(tr, case)
    assert st.index == 1
    assert st.J == pytest.approx(0.0)
    assert st.at_horizon is False


def test_optimal_stop_flags_a_minimum_pinned_at_the_horizon():
    case = _Case()
    chi = case.part_mask.astype(float)
    frames = [np.full(chi.shape, case.pins.ambient_c),
              _T_for_phi(case, 0.3 * chi),
              _T_for_phi(case, 0.9 * chi)]
    tr = SimpleNamespace(n_outer=3, T_at_end=lambda n: frames[n],
                         mean_phi_part=np.array([0.0, 0.3, 0.9]),
                         time_s=np.array([1.0, 2.0, 3.0]))
    st = so.optimal_stop(tr, case)
    assert st.index == 2
    assert st.at_horizon is True


def test_region_metrics_count_growth_and_under_melt():
    case = _Case()
    chi = case.part_mask.astype(float)
    phi = chi.copy()
    phi[1, 2] = 1.0        # one bed cell melted   -> growth
    phi[2, 2] = 0.0        # one part cell unmelted -> under-melt
    T = _T_for_phi(case, phi)
    m = so.region_metrics(T, case)
    n_part = int(case.part_mask.sum())
    assert m["bed_melt_pct_of_part"] == pytest.approx(100.0 / n_part)
    assert m["part_under_melt_pct"] == pytest.approx(100.0 / n_part)
    # intersection 15, union 17
    assert m["IoU"] == pytest.approx(15.0 / 17.0)


def test_window_metrics_use_the_configured_melt_window():
    case = _Case()
    T = np.full(case.part_mask.shape, 100.0)
    tp = np.array([170.0, 180.0, 190.0, 300.0])
    idx = np.argwhere(case.part_mask)[:4]
    T[tuple(idx[:, 0]), tuple(idx[:, 1])] = tp
    w = so.window_metrics(T, case, lo=175.0, hi=185.0)
    n = int(case.part_mask.sum())
    # 12 remaining part cells sit at 100 C, so 13 of 16 are under 175 C
    assert w["under_pct"] == pytest.approx(100.0 * 13 / n)
    assert w["over_pct"] == pytest.approx(100.0 * 2 / n)
    assert w["p95_overshoot_c"] >= 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:warnings"]))
