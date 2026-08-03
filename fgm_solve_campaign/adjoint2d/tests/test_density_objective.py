"""RED-first tests for the density-region objective.

    J_rho(s, t_stop) = sum over the WHOLE domain of
                       (rho_norm(x, t_stop) - chi_part(x))^2

with rho_norm the relative density mapped from the powder floor rho_rel_init
to full density onto [0, 1].

The bed carries no densification state in the forward march
(`forward.substep` integrates rho only inside the part mask), so the bed is
extended at the powder floor, rho_norm = 0 there, and its contribution to the
whole-domain sum is identically zero. That asymmetry against the melt-region
objective is real and is asserted here rather than hidden.

Because relative density only ever increases, J_rho(t) is monotone
non-increasing and its argmin is ALWAYS the last stored step. That is the
saturation pathology, and the flat-onset stop rule is the guard.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from adjoint2d import density_objective as dobj


class _Case:
    def __init__(self, ny=8, nx=8, floor=0.55):
        m = np.zeros((ny, nx), dtype=bool)
        m[2:6, 2:6] = True          # 16 part cells of 64
        self.part_mask = m
        self.doped_mask = m
        self.dx = self.dy = 1.0e-3
        self.dA = self.dx * self.dy
        self.pins = SimpleNamespace(rho_rel_init=floor, dt=0.5)


# --- the normalization -----------------------------------------------------

def test_powder_floor_maps_to_zero_and_full_density_to_one():
    case = _Case()
    assert dobj.rho_norm(np.array([0.55]), case)[0] == pytest.approx(0.0)
    assert dobj.rho_norm(np.array([1.0]), case)[0] == pytest.approx(1.0)


def test_normalization_is_affine_and_uses_the_configured_floor():
    case = _Case(floor=0.40)
    got = dobj.rho_norm(np.array([0.40, 0.70, 1.0]), case)
    assert got == pytest.approx([0.0, 0.5, 1.0])


# --- the objective ---------------------------------------------------------

def test_fully_dense_part_gives_zero_objective():
    case = _Case()
    rho = np.where(case.part_mask, 1.0, 0.0)
    J, _g = dobj.rho_J_and_seed(rho, case)
    assert J == pytest.approx(0.0)


def test_part_at_the_powder_floor_costs_one_per_part_cell():
    case = _Case()
    rho = np.where(case.part_mask, 0.55, 0.0)
    J, _g = dobj.rho_J_and_seed(rho, case)
    assert J == pytest.approx(float(case.part_mask.sum()))


def test_the_bed_contributes_exactly_zero_whatever_its_stored_value():
    """The forward stores rho = 0 in the bed as bookkeeping, not as physics.

    The bed is undensified powder, so it is extended at the powder floor and
    its whole-domain contribution is identically zero. Changing the stored bed
    values must not move J at all.
    """
    case = _Case()
    base = np.where(case.part_mask, 0.8, 0.0)
    perturbed = base.copy()
    perturbed[~case.part_mask] = 0.9
    assert (dobj.rho_J_and_seed(base, case)[0]
            == pytest.approx(dobj.rho_J_and_seed(perturbed, case)[0]))


def test_the_seed_is_zero_in_the_bed():
    case = _Case()
    rho = np.where(case.part_mask, 0.8, 0.0)
    _J, g = dobj.rho_J_and_seed(rho, case)
    assert np.all(g[~case.part_mask] == 0.0)


def test_the_seed_matches_a_central_difference_cell_by_cell():
    case = _Case()
    rng = np.random.default_rng(11)
    rho = np.zeros(case.part_mask.shape)
    rho[case.part_mask] = 0.60 + 0.30 * rng.random(int(case.part_mask.sum()))
    _J, g = dobj.rho_J_and_seed(rho, case)
    eps = 1e-6
    for idx in ((2, 2), (3, 4), (5, 5)):
        rp = rho.copy(); rp[idx] += eps
        rm = rho.copy(); rm[idx] -= eps
        fd = (dobj.rho_J_and_seed(rp, case)[0] - dobj.rho_J_and_seed(rm, case)[0]) / (2 * eps)
        assert g[idx] == pytest.approx(fd, rel=1e-6, abs=1e-9)


# --- the saturation guard --------------------------------------------------

def test_flat_onset_stops_before_the_argmin_when_the_curve_plateaus():
    """A curve that falls then flattens must be read at the flat onset."""
    curve = np.concatenate([np.linspace(100.0, 10.0, 50),
                            10.0 - np.linspace(0.0, 0.05, 50)])
    st = dobj.flat_onset_stop(curve, tol=0.01, dt=0.5)
    assert st.argmin_index == len(curve) - 1
    assert st.index < 60
    assert st.index >= 45


def test_flat_onset_flags_the_horizon_when_the_curve_is_still_falling():
    curve = np.linspace(100.0, 10.0, 60)
    st = dobj.flat_onset_stop(curve, tol=0.01, dt=0.5)
    assert st.at_horizon is True
    assert st.index == 59


def test_flat_onset_on_a_curve_that_never_moves_reports_no_progress():
    curve = np.full(40, 7.0)
    st = dobj.flat_onset_stop(curve, tol=0.01, dt=0.5)
    assert st.no_progress is True
    assert st.index == 0


def test_flat_onset_index_is_the_first_index_within_tolerance():
    curve = np.array([10.0, 6.0, 3.0, 2.0, 1.02, 1.005, 1.0])
    st = dobj.flat_onset_stop(curve, tol=0.01, dt=1.0)
    # total decrease 9.0, so the band is 1.0 + 0.01*9.0 = 1.09 and the first
    # index inside it is 4 (J = 1.02), not 5.
    assert st.index == 4
    assert st.J == pytest.approx(1.02)


def test_tighter_tolerance_stops_later():
    curve = np.array([10.0, 6.0, 3.0, 2.0, 1.02, 1.005, 1.0])
    loose = dobj.flat_onset_stop(curve, tol=0.05, dt=1.0)
    tight = dobj.flat_onset_stop(curve, tol=0.0005, dt=1.0)
    assert loose.index <= tight.index
    assert tight.index == 6


# --- region metrics --------------------------------------------------------

def test_densified_region_uses_the_normalized_half_level():
    case = _Case(floor=0.55)
    rho = np.where(case.part_mask, 0.55, 0.0)
    rho[2, 2] = 0.78       # rho_norm = 0.5111, densified
    rho[2, 3] = 0.77       # rho_norm = 0.4889, not densified
    m = dobj.density_region_metrics(rho, case)
    assert m["densified_cells"] == 1
    assert m["IoU_rho"] == pytest.approx(1.0 / 16.0)


def test_stricter_density_levels_are_reported_because_the_half_level_saturates():
    """MEASURED: at the flat-onset read state every arm has the whole part above
    the half level, so `IoU_rho` is 1.0000 for all of them and discriminates
    nothing. Stricter levels are reported alongside it."""
    case = _Case(floor=0.55)
    rho = np.where(case.part_mask, 0.55, 0.0)
    flat = np.argwhere(case.part_mask)
    for k, (i, j) in enumerate(flat):
        rho[i, j] = 0.55 + 0.45 * (k + 1) / len(flat)   # rho_norm 1/16 .. 1
    m = dobj.density_region_metrics(rho, case)
    # normalized densities are k/16 for k = 1 .. 16, so k = 8 .. 16 clear 0.5
    assert m["dense_frac_0p50"] == pytest.approx(9 / 16)
    assert m["dense_frac_0p90"] == pytest.approx(2 / 16)
    assert m["dense_frac_0p50"] == pytest.approx(m["IoU_rho"])


def test_densified_bed_is_impossible_and_is_reported_as_such():
    case = _Case()
    rho = np.where(case.part_mask, 1.0, 0.0)
    m = dobj.density_region_metrics(rho, case)
    assert m["densified_bed_cells"] == 0
    assert m["IoU_rho"] == pytest.approx(1.0)
