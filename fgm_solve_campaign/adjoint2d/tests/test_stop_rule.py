"""RED-first tests for the v2.1.0 DEFAULT READ STATE of solve-mode runs.

THE DECISION BEING WIRED (DENSE_IFF_INBOUNDS_REPORT.md Sections 1 and 6): the
stop is the argmin of the dense-if-and-only-if-in-bounds objective J_asym at the
production out-of-bounds price w_out = 2.0 and density floor 0.85 relative
density, NOT the argmin of the melt-region objective J_phi. The melt objective
remains the MAP driver; the asymmetric objective owns the STOP only.

Both stops and both objective values are computed on every run so a v2.0.x
comparison stays possible, and `stop_rule = "j_phi"` restores the v2.0.x read
state exactly.

Acronyms: J_phi = the melt-region shape-fidelity objective. J_asym = the
asymmetric dense-if-and-only-if-in-bounds objective. bpp = bits per pixel.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from adjoint2d import stop_rule as sr


# ---------------------------------------------------------------------------
# 1. the frozen production constants
# ---------------------------------------------------------------------------

def test_the_default_rule_is_j_asym():
    assert sr.DEFAULT_STOP_RULE == "j_asym"
    assert set(sr.STOP_RULES) == {"j_asym", "j_phi"}


def test_the_production_price_and_floor_are_the_reports_recommended_values():
    # DENSE_IFF_INBOUNDS_REPORT.md Section 6: "the production out-of-bounds
    # price is w_out = 2 to 3, not 1"; Section 7: "Recommendation: 0.85".
    assert sr.W_OUT_PRODUCTION == 2.0
    assert sr.FLOOR_RHO_REL_PRODUCTION == 0.85
    assert sr.W_IN_PRODUCTION == 1.0


def test_validate_accepts_both_rules_and_rejects_anything_else():
    assert sr.validate_stop_rule("j_asym") == "j_asym"
    assert sr.validate_stop_rule("j_phi") == "j_phi"
    with pytest.raises(ValueError, match="stop_rule"):
        sr.validate_stop_rule("melt")
    with pytest.raises(ValueError, match="stop_rule"):
        sr.validate_stop_rule("")


# ---------------------------------------------------------------------------
# 2. selection, pure logic
# ---------------------------------------------------------------------------

def _phi_rec(index=100, J_phi=12.0, J_asym=0.30):
    return {"index": index, "time_s": 0.5 * (index + 1), "at_horizon": False,
            "J_phi": J_phi, "J_asym": J_asym}


def _asym_rec(index=160, J_phi=15.0, J_asym=0.21):
    return {"index": index, "time_s": 0.5 * (index + 1), "at_horizon": False,
            "J_phi": J_phi, "J_asym": J_asym,
            "J_asym_out": 0.09, "J_asym_in": 0.12,
            "stop_is_first_step": False, "in_term_dead": False,
            "flat_onset_gap_steps": 21}


def test_j_asym_rule_selects_the_asym_argmin():
    rec = sr.select_stop("j_asym", _phi_rec(), _asym_rec())
    assert rec["stop_rule"] == "j_asym"
    assert rec["index"] == 160
    assert rec["time_s"] == pytest.approx(80.5)
    assert rec["at_horizon"] is False


def test_j_phi_rule_selects_the_melt_argmin_the_v2_0_x_read_state():
    rec = sr.select_stop("j_phi", _phi_rec(), _asym_rec())
    assert rec["stop_rule"] == "j_phi"
    assert rec["index"] == 100
    assert rec["time_s"] == pytest.approx(50.5)


def test_both_stops_and_both_objective_values_are_always_recorded():
    """Legacy comparability: neither rule may discard the other's numbers."""
    for rule in ("j_asym", "j_phi"):
        rec = sr.select_stop(rule, _phi_rec(), _asym_rec())
        assert rec["j_phi_stop"]["index"] == 100
        assert rec["j_asym_stop"]["index"] == 160
        assert rec["j_phi_stop"]["J_phi"] == pytest.approx(12.0)
        assert rec["j_phi_stop"]["J_asym"] == pytest.approx(0.30)
        assert rec["j_asym_stop"]["J_phi"] == pytest.approx(15.0)
        assert rec["j_asym_stop"]["J_asym"] == pytest.approx(0.21)


def test_the_stop_gap_is_reported_signed_asym_minus_phi():
    rec = sr.select_stop("j_asym", _phi_rec(index=100), _asym_rec(index=160))
    assert rec["stop_gap_steps"] == 60
    rec2 = sr.select_stop("j_asym", _phi_rec(index=200), _asym_rec(index=160))
    assert rec2["stop_gap_steps"] == -40


def test_guard_flags_travel_with_the_asym_record():
    a = _asym_rec()
    a["in_term_dead"] = True
    a["stop_is_first_step"] = True
    rec = sr.select_stop("j_asym", _phi_rec(), a)
    assert rec["j_asym_stop"]["in_term_dead"] is True
    assert rec["j_asym_stop"]["stop_is_first_step"] is True
    assert rec["j_asym_stop"]["flat_onset_gap_steps"] == 21


def test_an_unknown_rule_is_rejected_by_select():
    with pytest.raises(ValueError, match="stop_rule"):
        sr.select_stop("argmax", _phi_rec(), _asym_rec())


# ---------------------------------------------------------------------------
# 3. dual_stop against a real objective pair on a synthetic trajectory
# ---------------------------------------------------------------------------

class _Case:
    """8 x 8 domain with 16 part cells, the campaign's powder floor and ramp."""

    def __init__(self, ny=8, nx=8, floor=0.55):
        m = np.zeros((ny, nx), dtype=bool)
        m[2:6, 2:6] = True
        self.part_mask = m
        self.doped_mask = m
        self.dx = self.dy = 1.0e-3
        self.dA = self.dx * self.dy
        self.n_part = int(m.sum())
        self.pins = SimpleNamespace(rho_rel_init=floor, dt=0.5, t_pc_c=180.0,
                                    dt_pc_c=10.0, ambient_c=20.0)


class _Traj:
    """A trajectory whose part melts first and whose bed melts later.

    That is the ordering the report measures on every real arm, so the J_phi
    argmin lands EARLIER than the J_asym argmin.
    """

    def __init__(self, case, n=12):
        self.case = case
        self.n_outer = n
        self.time_s = np.arange(1, n + 1) * 0.5
        pm = case.part_mask
        self._T = []
        self._rho = []
        for k in range(n):
            T = np.full(pm.shape, 100.0)
            # the part crosses the phase ramp early
            T[pm] = 100.0 + 12.0 * k
            # the bed follows, later
            T[~pm] = 100.0 + 7.0 * max(k - 4, 0)
            self._T.append(T)
            rho = np.where(pm, min(0.55 + 0.03 * k, 0.99), 0.0)
            self._rho.append(rho)

    def T_at_end(self, k):
        return self._T[int(k)]

    def rho_at_end(self, k):
        return self._rho[int(k)]


def test_dual_stop_computes_both_argmins_and_both_cross_read_values():
    from adjoint2d import asym_objective as ao
    from adjoint2d import topopt_objective as tobj

    case = _Case()
    tr = _Traj(case)
    chi = case.part_mask.astype(float)

    rec = sr.dual_stop(tr, case, chi)

    sp = tobj.optimal_stop(tr, case, chi)
    st = ao.asym_stop(tr, case, floor=sr.FLOOR_RHO_REL_PRODUCTION,
                      w_out=sr.W_OUT_PRODUCTION, w_in=sr.W_IN_PRODUCTION)
    assert rec["j_phi_stop"]["index"] == int(sp.index)
    assert rec["j_asym_stop"]["index"] == int(st.index)
    # the CROSS reads: each objective evaluated at the OTHER's stop
    assert rec["j_phi_stop"]["J_asym"] == pytest.approx(
        ao.J_and_seeds(tr.T_at_end(sp.index), tr.rho_at_end(sp.index), case,
                       floor=sr.FLOOR_RHO_REL_PRODUCTION,
                       w_out=sr.W_OUT_PRODUCTION, w_in=sr.W_IN_PRODUCTION)[0])
    assert rec["j_asym_stop"]["J_phi"] == pytest.approx(
        tobj.J_and_seed(tr.T_at_end(st.index), case, chi)[0])
    # and the default rule is the asymmetric one
    assert rec["stop_rule"] == "j_asym"
    assert rec["index"] == int(st.index)


def test_dual_stop_under_j_phi_reproduces_the_v2_0_x_stop_exactly():
    """The legacy path: byte-for-byte the melt argmin, no asym influence."""
    from adjoint2d import topopt_objective as tobj

    case = _Case()
    tr = _Traj(case)
    chi = case.part_mask.astype(float)
    sp = tobj.optimal_stop(tr, case, chi)

    rec = sr.dual_stop(tr, case, chi, rule="j_phi")
    assert rec["index"] == int(sp.index)
    assert rec["time_s"] == pytest.approx(float(sp.time_s))
    assert rec["J_phi_at_stop"] == pytest.approx(float(sp.J))


def test_dual_stop_records_the_price_and_floor_it_used():
    case = _Case()
    tr = _Traj(case)
    chi = case.part_mask.astype(float)
    rec = sr.dual_stop(tr, case, chi, w_out=3.0, floor=0.90)
    assert rec["w_out"] == 3.0
    assert rec["floor_rho_rel"] == 0.90
    assert rec["w_in"] == 1.0


def test_a_harder_out_of_bounds_price_never_stops_later():
    """The monotonicity the trade-curve identity guarantees (report Section 6)."""
    case = _Case()
    tr = _Traj(case)
    chi = case.part_mask.astype(float)
    prev = None
    for w in (0.5, 1.0, 2.0, 5.0, 20.0):
        i = sr.dual_stop(tr, case, chi, w_out=w)["j_asym_stop"]["index"]
        if prev is not None:
            assert i <= prev
        prev = i
