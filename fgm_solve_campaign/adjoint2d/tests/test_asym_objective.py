"""RED-first tests for the DENSE-IF-AND-ONLY-IF-IN-BOUNDS objective.

    J_asym(s, t) = ( w_out * sum over the BED of phi^2
                   + w_in  * sum over the PART of hinge(rho_rel)^2 ) / n_part

    hinge(rho_rel) = max(0, floor - rho_rel) / (floor - rho_floor)

The two sides are deliberately NOT symmetric:

  * out of bounds is the HARD side. Any melt fraction in a bed cell is charged
    at full amplitude, and a fully melted bed cell costs w_out per part cell of
    normalization whatever the part is doing.
  * in bounds is the SOFT side with a FLOOR. Relative density at or above the
    floor costs exactly zero, so the objective never pays shape for density it
    does not need. Below the floor the cost grows quadratically to w_out's
    counterpart w_in at the powder floor.

The tests below pin that asymmetry, the hinge's exact zero above the floor, the
two seeds against cell-by-cell central differences, and the stop rule with its
guard flags.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from adjoint2d import asym_objective as ao


class _Case:
    """8 x 8 domain, 16 part cells, the campaign's powder floor and phase ramp."""

    def __init__(self, ny=8, nx=8, floor=0.55):
        m = np.zeros((ny, nx), dtype=bool)
        m[2:6, 2:6] = True
        self.part_mask = m
        self.doped_mask = m
        self.dx = self.dy = 1.0e-3
        self.dA = self.dx * self.dy
        self.n_part = int(m.sum())
        self.pins = SimpleNamespace(rho_rel_init=floor, dt=0.5,
                                    t_pc_c=180.0, dt_pc_c=10.0,
                                    ambient_c=20.0)


def _cold(case):
    return np.full(case.part_mask.shape, 100.0)


def _rho_at(case, value):
    return np.where(case.part_mask, value, 0.0)


# --- the hinge -------------------------------------------------------------

def test_the_hinge_is_exactly_zero_at_and_above_the_floor():
    case = _Case()
    h = ao.deficit(_rho_at(case, 0.85), case, floor=0.85)
    assert np.all(h == 0.0)
    h2 = ao.deficit(_rho_at(case, 0.97), case, floor=0.85)
    assert np.all(h2 == 0.0)


def test_the_hinge_is_one_at_the_powder_floor_and_linear_between():
    case = _Case(floor=0.55)
    h = ao.deficit(_rho_at(case, 0.55), case, floor=0.85)
    assert h[case.part_mask] == pytest.approx(1.0)
    mid = ao.deficit(_rho_at(case, 0.70), case, floor=0.85)
    assert mid[case.part_mask] == pytest.approx(0.5)


def test_the_hinge_is_zero_in_the_bed_whatever_the_stored_bed_value():
    case = _Case()
    a = ao.deficit(np.zeros(case.part_mask.shape), case, floor=0.85)
    b = ao.deficit(np.full(case.part_mask.shape, 0.99), case, floor=0.85)
    assert np.all(a[~case.part_mask] == 0.0)
    assert np.all(b[~case.part_mask] == 0.0)


def test_a_floor_at_or_below_the_powder_floor_is_rejected():
    case = _Case(floor=0.55)
    with pytest.raises(ValueError):
        ao.deficit(_rho_at(case, 0.6), case, floor=0.55)
    with pytest.raises(ValueError):
        ao.deficit(_rho_at(case, 0.6), case, floor=0.40)


# --- the objective ---------------------------------------------------------

def test_in_bounds_dense_and_no_bed_melt_gives_exactly_zero():
    case = _Case()
    T = _cold(case)
    rho = _rho_at(case, 0.90)
    J, _sT, _sr, parts = ao.J_and_seeds(T, rho, case, floor=0.85)
    assert J == pytest.approx(0.0)
    assert parts["J_out"] == pytest.approx(0.0)
    assert parts["J_in"] == pytest.approx(0.0)


def test_a_fully_unsintered_part_costs_exactly_w_in():
    case = _Case()
    J, _sT, _sr, parts = ao.J_and_seeds(_cold(case), _rho_at(case, 0.55), case,
                                        floor=0.85, w_out=1.0, w_in=1.0)
    assert parts["J_in"] == pytest.approx(1.0)
    assert J == pytest.approx(1.0)


def test_a_fully_melted_bed_cell_costs_w_out_over_n_part():
    case = _Case()
    T = _cold(case)
    T[0, 0] = 400.0                       # far above the phase ramp: phi = 1
    J, _sT, _sr, parts = ao.J_and_seeds(T, _rho_at(case, 0.90), case,
                                        floor=0.85, w_out=7.0, w_in=1.0)
    assert parts["J_out"] == pytest.approx(7.0 / case.n_part)
    assert J == pytest.approx(7.0 / case.n_part)


def test_melting_the_part_itself_is_never_charged_by_the_out_term():
    case = _Case()
    T = np.where(case.part_mask, 400.0, 100.0)
    _J, _sT, _sr, parts = ao.J_and_seeds(T, _rho_at(case, 0.90), case, floor=0.85)
    assert parts["J_out"] == pytest.approx(0.0)


def test_out_of_bounds_is_the_hard_side_at_matched_normalized_amplitude():
    """One fully melted bed cell against one fully unsintered part cell."""
    case = _Case()
    T_grow = _cold(case)
    T_grow[0, 0] = 400.0
    J_grow, *_ = ao.J_and_seeds(T_grow, _rho_at(case, 0.90), case,
                                floor=0.85, w_out=3.0, w_in=1.0)
    rho_thin = _rho_at(case, 0.90)
    rho_thin[2, 2] = 0.55
    J_thin, *_ = ao.J_and_seeds(_cold(case), rho_thin, case,
                                floor=0.85, w_out=3.0, w_in=1.0)
    assert J_grow == pytest.approx(3.0 * J_thin)


def test_density_above_the_floor_buys_nothing():
    """The asymmetry: pushing past the floor cannot pay for growth."""
    case = _Case()
    T = _cold(case)
    T[0, 0] = 400.0
    a, *_ = ao.J_and_seeds(T, _rho_at(case, 0.86), case, floor=0.85)
    b, *_ = ao.J_and_seeds(T, _rho_at(case, 1.00), case, floor=0.85)
    assert a == pytest.approx(b)


# --- the seeds -------------------------------------------------------------

def _J_only(T, rho, case, floor=0.85, w_out=2.0, w_in=1.0):
    return ao.J_and_seeds(T, rho, case, floor=floor, w_out=w_out, w_in=w_in)[0]


def test_the_temperature_seed_matches_a_cell_by_cell_central_difference():
    rng = np.random.default_rng(11)
    case = _Case()
    T = 175.0 + 8.0 * rng.standard_normal(case.part_mask.shape)
    rho = np.where(case.part_mask, 0.55 + 0.3 * rng.random(case.part_mask.shape), 0.0)
    _J, sT, _sr, _p = ao.J_and_seeds(T, rho, case, floor=0.85, w_out=2.0, w_in=1.0)
    eps = 1e-6
    for iy, ix in ((0, 0), (1, 4), (3, 3), (6, 2)):
        Tp = T.copy(); Tp[iy, ix] += eps
        Tm = T.copy(); Tm[iy, ix] -= eps
        fd = (_J_only(Tp, rho, case) - _J_only(Tm, rho, case)) / (2 * eps)
        assert sT[iy, ix] == pytest.approx(fd, rel=1e-5, abs=1e-10)


def test_the_density_seed_matches_a_cell_by_cell_central_difference():
    rng = np.random.default_rng(12)
    case = _Case()
    T = 175.0 + 8.0 * rng.standard_normal(case.part_mask.shape)
    rho = np.where(case.part_mask, 0.55 + 0.3 * rng.random(case.part_mask.shape), 0.0)
    _J, _sT, sr, _p = ao.J_and_seeds(T, rho, case, floor=0.85, w_out=2.0, w_in=1.0)
    eps = 1e-7
    for iy, ix in ((2, 2), (3, 4), (5, 5)):
        rp = rho.copy(); rp[iy, ix] += eps
        rm = rho.copy(); rm[iy, ix] -= eps
        fd = (_J_only(T, rp, case) - _J_only(T, rm, case)) / (2 * eps)
        assert sr[iy, ix] == pytest.approx(fd, rel=1e-5, abs=1e-10)


def test_the_density_seed_is_exactly_zero_where_the_hinge_is_inactive():
    case = _Case()
    rho = _rho_at(case, 0.90)
    rho[2, 2] = 0.60
    _J, _sT, sr, _p = ao.J_and_seeds(_cold(case), rho, case, floor=0.85)
    assert sr[2, 2] < 0.0
    assert np.count_nonzero(sr) == 1


def test_the_temperature_seed_is_exactly_zero_inside_the_part():
    case = _Case()
    T = np.where(case.part_mask, 180.0, 180.0)
    _J, sT, _sr, _p = ao.J_and_seeds(T, _rho_at(case, 0.9), case, floor=0.85)
    assert np.all(sT[case.part_mask] == 0.0)
    assert np.any(sT[~case.part_mask] != 0.0)


def test_the_seeds_scale_linearly_with_their_own_weight():
    case = _Case()
    T = _cold(case); T[0, 0] = 181.0
    rho = _rho_at(case, 0.70)
    _J, sT1, sr1, _p = ao.J_and_seeds(T, rho, case, floor=0.85, w_out=1.0, w_in=1.0)
    _J, sT5, sr5, _p = ao.J_and_seeds(T, rho, case, floor=0.85, w_out=5.0, w_in=3.0)
    assert sT5 == pytest.approx(5.0 * sT1)
    assert sr5 == pytest.approx(3.0 * sr1)


# --- the stop rule and its guards -----------------------------------------

class _Traj:
    def __init__(self, Ts, rhos, dt=0.5):
        self._T = Ts
        self._r = rhos
        self.n_outer = len(Ts)
        self.time_s = (np.arange(len(Ts)) + 1) * dt

    def T_at_end(self, n):
        return self._T[n]

    def rho_at_end(self, n):
        return self._r[n]


def _ramp_traj(case, n=12):
    """Density rises then the bed starts to melt: a genuinely interior argmin."""
    Ts, rhos = [], []
    for k in range(n):
        T = _cold(case).astype(float)
        if k >= 6:
            T[0, 0] = 170.0 + 4.0 * (k - 5)      # bed melt turns on late
        Ts.append(T)
        rhos.append(_rho_at(case, min(0.55 + 0.03 * k, 1.0)))
    return _Traj(Ts, rhos)


def test_the_stop_is_the_argmin_of_the_combined_objective():
    case = _Case()
    tr = _ramp_traj(case)
    st = ao.asym_stop(tr, case, floor=0.85)
    curve = ao.J_curve(tr, case, floor=0.85)
    assert st.index == int(np.argmin(curve))
    assert st.J == pytest.approx(float(np.min(curve)))
    assert 0 < st.index < tr.n_outer - 1
    assert st.at_horizon is False


def test_the_stop_flags_the_horizon_when_the_objective_never_turns():
    case = _Case()
    Ts = [_cold(case) for _ in range(6)]
    rhos = [_rho_at(case, 0.55 + 0.05 * k) for k in range(6)]
    st = ao.asym_stop(_Traj(Ts, rhos), case, floor=0.85)
    assert st.index == 5
    assert st.at_horizon is True


def test_the_stop_flags_a_dead_in_bounds_term():
    """If every in-bounds cell clears the floor the soft side is exactly zero,
    the objective degenerates to pure growth minimization and the read state is
    no longer meaningful. That must be visible in the data, not only here."""
    case = _Case()
    Ts = [_cold(case) for _ in range(4)]
    rhos = [_rho_at(case, 0.95) for _ in range(4)]
    st = ao.asym_stop(_Traj(Ts, rhos), case, floor=0.85)
    assert st.in_term_dead is True


def test_the_stop_reports_the_flat_onset_guard_gap():
    """The argmin of a plateaued curve is not a trustworthy read state. The
    first index within tol of the minimum is reported next to it."""
    case = _Case()
    tr = _ramp_traj(case)
    st = ao.asym_stop(tr, case, floor=0.85)
    assert st.flat_onset_index <= st.index
    assert st.flat_onset_gap_steps == st.index - st.flat_onset_index


def test_hinge_active_fraction_is_reported_at_the_stop():
    case = _Case()
    rho = _rho_at(case, 0.90)
    rho[2, 2] = 0.60
    rho[2, 3] = 0.60
    m = ao.region_metrics(_cold(case), rho, case, floor=0.85)
    assert m["hinge_active_frac"] == pytest.approx(2.0 / case.n_part)
    assert m["frac_part_at_or_above_floor"] == pytest.approx(1.0 - 2.0 / case.n_part)
    assert m["min_rho_rel_part"] == pytest.approx(0.60)
