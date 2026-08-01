"""Red-green tests for the pure bookkeeping of the FGM four-arm rerun.

No physics here: only percent-change arithmetic, sign-survival logic and the
standing-gate predicate. Run:

    ./.venv312/bin/python -m pytest heatr3d_eqs02_rerank/test_fgm_arms.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fgm_arms import benefit_pct, gate_pass, sign_verdict  # noqa: E402


# --------------------------------------------------------------------------
# benefit_pct: signed percent change of sigma_T relative to a reference arm
# --------------------------------------------------------------------------
def test_benefit_pct_improvement_is_negative():
    assert benefit_pct(15.148, 32.921) == pytest.approx(-54.0, abs=0.05)


def test_benefit_pct_degradation_is_positive():
    assert benefit_pct(40.0, 32.0) == pytest.approx(25.0)


def test_benefit_pct_zero_when_equal():
    assert benefit_pct(20.0, 20.0) == 0.0


def test_benefit_pct_rejects_nonpositive_reference():
    with pytest.raises(ValueError):
        benefit_pct(1.0, 0.0)


# --------------------------------------------------------------------------
# sign_verdict: does a published benefit's SIGN survive re-measurement?
# --------------------------------------------------------------------------
def test_sign_survives_when_both_negative():
    assert sign_verdict(-31.9, -12.4) == "survives"


def test_sign_reverses_when_measured_is_a_penalty():
    assert sign_verdict(-31.9, +8.3) == "reverses"


def test_sign_null_when_measured_is_within_noise_band():
    # |measured| below the band is neither a benefit nor a penalty
    assert sign_verdict(-31.9, -0.4, band_pct=1.0) == "null"


def test_sign_verdict_band_is_inclusive_of_larger_magnitudes():
    assert sign_verdict(-31.9, -1.5, band_pct=1.0) == "survives"


# --------------------------------------------------------------------------
# gate_pass: the standing gates on every march
# --------------------------------------------------------------------------
GOOD = {"reached_phi90": True, "energy_residual_frac": -4.0e-13,
        "clamp_bound": False, "cfl_violated": False}


def test_gate_pass_on_clean_march():
    assert gate_pass(GOOD) is True


@pytest.mark.parametrize("key,bad", [("reached_phi90", False),
                                     ("clamp_bound", True),
                                     ("cfl_violated", True),
                                     ("energy_residual_frac", 1e-3)])
def test_gate_fails_on_each_violation(key, bad):
    g = dict(GOOD, **{key: bad})
    assert gate_pass(g) is False


def test_gate_fails_on_nan_energy_residual():
    assert gate_pass(dict(GOOD, energy_residual_frac=math.nan)) is False
