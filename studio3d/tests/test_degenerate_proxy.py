"""RED-first gate for the degenerate-proxy refusal on the inversion rung.

TAMPER_DIAGNOSIS.md fix 2(a)/2(b). The rung is KEPT, not retired, but it must
refuse to run on a proxy that carries no rankable structure -- which is exactly
what the BEFORE arm's rho_final is when the arm stopped ON a density target.

All thresholds and the fixtures behind them are real measurements captured by
capture_tamper_fixtures.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from studio3d.correction import (PROXY_FLOORS, DegenerateProxyError,
                                 choose_inversion_proxy, proxy_degeneracy)

FIX = Path(__file__).resolve().parent / "fixtures" / "tamper"
MANIFEST = json.loads((FIX / "manifest.json").read_text())


def _fields(job: str, arm: str = "uncorrected"):
    return MANIFEST["jobs"][job]["arms"][arm]["fields"]


# --------------------------------------------------------------------------- #
# 2(a) never use rho_final from a density-stopped arm
# --------------------------------------------------------------------------- #
def test_density_stopped_arm_switches_proxy_to_T_phi90():
    """The Tamper before arm stopped on stop_mean_rho=0.98 and ended at
    rho_final_mean=0.98. Its rho_final is flat BY CONSTRUCTION."""
    before = json.loads((FIX / "feb850ec_uncorrected_results.json").read_text())
    assert before["stop_mean_rho"] == 0.98
    assert before["rho_final_mean"] == 0.98
    name, why = choose_inversion_proxy(before)
    assert name == "T_phi90"
    assert "density" in why.lower()


def test_arm_that_did_NOT_reach_the_density_stop_may_use_rho_final():
    """A horizon-stopped arm's rho_final still carries structure, so the rule
    is targeted, not a blanket ban."""
    before = json.loads((FIX / "feb850ec_uncorrected_results.json").read_text())
    before = dict(before, rho_final_mean=0.61, stop_mean_rho=0.98)
    name, _ = choose_inversion_proxy(before)
    assert name == "rho_final"


def test_proxy_switch_survives_a_missing_stop_target():
    before = json.loads((FIX / "feb850ec_uncorrected_results.json").read_text())
    before = dict(before)
    before["stop_mean_rho"] = None
    name, _ = choose_inversion_proxy(before)
    assert name == "rho_final"


# --------------------------------------------------------------------------- #
# 2(b) degeneracy refusal on the CHOSEN proxy
# --------------------------------------------------------------------------- #
def test_saturated_rho_final_is_degenerate_on_every_real_harm_job():
    """The statistic that actually separates the archive: fraction of in-part
    voxels sitting exactly at the proxy maximum."""
    for job in ("feb850ec", "c474d787", "d4d50045"):
        d = proxy_degeneracy(_fields(job)["rho_final"])
        assert d["degenerate"], f"{job} rho_final should be refused: {d}"
        assert d["reason"] == "saturated"


def test_healthy_tube_rho_final_is_not_degenerate():
    d = proxy_degeneracy(_fields("2ffdfc3c")["rho_final"])
    assert not d["degenerate"], d


def test_T_phi90_is_healthy_on_every_captured_job():
    """After the 2(a) switch the chosen proxy is T_phi90, and it is
    non-degenerate on all four captured parts -- which is why the switch, not
    the floor, is the primary fix."""
    for job in ("feb850ec", "c474d787", "d4d50045", "2ffdfc3c"):
        d = proxy_degeneracy(_fields(job)["T_phi90"])
        assert not d["degenerate"], f"{job} T_phi90 unexpectedly refused: {d}"


def test_spread_floor_alone_would_MISS_the_worst_job():
    """Documented negative result, pinned so it cannot be quietly forgotten.

    A p98-p2 spread floor placed between the Tamper (0.2551) and the healthy
    tube (0.3794) -- the calibration originally proposed -- does NOT catch
    d4d50045, whose rho_final spread is 0.3511 yet which emitted the WORST map
    in the archive (87.7 % of the part zeroed). The saturation statistic
    catches all three. This is why both are checked.
    """
    tamper = _fields("feb850ec")["rho_final"]["spread"]
    tube = _fields("2ffdfc3c")["rho_final"]["spread"]
    worst = _fields("d4d50045")["rho_final"]["spread"]
    hypothetical_floor = 0.5 * (tamper + tube)
    assert tamper < hypothetical_floor, "sanity: the Tamper would be caught"
    assert worst > hypothetical_floor, (
        "d4d50045 would be MISSED by a spread-only floor -- the point of "
        "this test")
    assert proxy_degeneracy(_fields("d4d50045")["rho_final"])["degenerate"]


def test_saturation_statistic_is_grid_independent():
    """A smooth linear ramp must NOT be refused at any grid. A bare
    frac-at-max floor false-positives here (0.1667 at n=16); the atom ratio is
    1.00 at every grid."""
    for n in (16, 32, 64):
        part = np.zeros((n, n, n), bool)
        lo, hi = n // 4, 3 * n // 4
        part[lo:hi, lo:hi, lo:hi] = True
        ramp = np.zeros((n, n, n))
        ramp[part] = (150.0 + 100.0 * np.indices((n, n, n))[0] / n)[part]
        d = proxy_degeneracy(ramp, part=part, proxy_name="T_phi90")
        assert not d["degenerate"], f"smooth ramp refused at n={n}: {d}"
        assert d["atom_ratio"] < 2.0


def test_a_genuinely_flat_proxy_is_refused_by_the_spread_floor():
    """The spread floor is kept for the case it does cover: a proxy whose whole
    in-part range is inside one melt window carries no rankable structure."""
    flat = {"p2": 179.0, "p98": 181.0, "spread": 2.0, "min": 178.0,
            "max": 182.0, "mean": 180.0, "std": 0.5, "frac_at_max": 0.0001,
            "atom_ratio": 1.0}
    d = proxy_degeneracy(flat, proxy_name="T_phi90")
    assert d["degenerate"]
    assert d["reason"] == "flat"


def test_floors_are_pre_registered_in_the_docstring():
    import studio3d.correction as C
    doc = C.__doc__ or ""
    assert "pre-registered 2026-08-04, never retuned against outcomes" in doc
    # both measured numbers behind the floor must be quoted
    assert "0.2551" in doc and "0.3794" in doc
    assert "42.2" in doc and "187.3" in doc
    assert PROXY_FLOORS["pre_registered"].startswith("2026-08-04")


def test_docstring_records_the_rungs_known_ceiling():
    """Even non-degenerate, whole-part inversion reads the interior (the
    cylinder-null mechanism), so it is a quick-look; the benefit gate is what
    protects the user. That limit must be written down where the rung lives."""
    import studio3d.correction as C
    doc = (C.__doc__ or "").lower()
    assert "quick-look" in doc
    assert "cylinder-null" in doc
    assert "benefit gate" in doc


# --------------------------------------------------------------------------- #
# refusal behaviour
# --------------------------------------------------------------------------- #
def test_refusal_raises_a_typed_error_naming_the_statistic():
    d = proxy_degeneracy(_fields("feb850ec")["rho_final"])
    with pytest.raises(DegenerateProxyError) as e:
        raise DegenerateProxyError(
            f"proxy refused: {d['reason']} ({d['frac_at_max']:.4f})")
    assert "saturated" in str(e.value)


def test_proxy_degeneracy_reports_the_numbers_it_judged_on():
    d = proxy_degeneracy(_fields("feb850ec")["rho_final"])
    for k in ("spread", "frac_at_max", "floor_spread", "floor_frac_at_max",
              "degenerate", "reason"):
        assert k in d, k


def test_proxy_degeneracy_can_be_computed_from_a_raw_array():
    rng = np.random.default_rng(0)
    part = np.zeros((8, 8, 8), bool); part[2:6, 2:6, 2:6] = True
    a = np.zeros((8, 8, 8)); a[part] = 1.0            # fully saturated
    d = proxy_degeneracy(a, part=part)
    assert d["degenerate"] and d["reason"] == "saturated"
    b = np.zeros((8, 8, 8)); b[part] = rng.uniform(0.2, 1.0, int(part.sum()))
    assert not proxy_degeneracy(b, part=part)["degenerate"]
