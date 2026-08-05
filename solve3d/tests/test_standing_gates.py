"""The forward's STANDING GATES, emitted with every scored arm.

WHY THIS EXISTS. The Studio's scheduler could not tell "the acceptance gates
did not pass" (a real but disappointing solve) from "the forward was not
physical" (a result that must not be shown at all), because
`studio_solve.score_arm` reported the objective and the shape metrics but not
the march's own gates. A run whose temperature clamp latched looked exactly
like a run that simply did not beat uniform.

The rule this module encodes: ABSENCE IS NOT HEALTH. A missing gate key reads
as violated, never as fine. A diagnostic that cannot distinguish "I checked
and it is good" from "I got nothing back" is worse than no diagnostic.

Pure numpy and pure dicts, so it runs in either environment and needs no mesh.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from solve3d import gates as sg

ROOT = Path(__file__).resolve().parents[2]


def _clean(**kw) -> dict:
    out = {"energy_residual_frac": 1.2e-13, "clamp_bound": False,
           "cfl_violated": False, "T_max_c": 200.0}
    out.update(kw)
    return out


# --------------------------------------------------------------------------- #
# The happy path, and that it is not trivially true
# --------------------------------------------------------------------------- #
def test_a_clean_march_is_reported_physical():
    g = sg.standing_gates(_clean())
    assert g["forward_physical"] is True
    assert g["clamp_bound"] is False
    assert g["cfl_violated"] is False
    assert g["energy_residual_frac"] == pytest.approx(1.2e-13)
    assert g["peak_T_c"] == pytest.approx(200.0)
    assert g["peak_over_ceiling"] is False
    assert g["reason"] == []


@pytest.mark.parametrize("bad,key", [
    ({"clamp_bound": True}, "clamp_bound"),
    ({"cfl_violated": True}, "cfl_violated"),
    ({"energy_residual_frac": 3.07e-3}, "energy_residual_frac"),
])
def test_each_violation_alone_makes_the_forward_unphysical(bad, key):
    """One tripped gate is enough. Reported individually so the caller can say
    WHICH, not just that something is wrong."""
    g = sg.standing_gates(_clean(**bad))
    assert g["forward_physical"] is False
    assert key in g["reason"]


# --------------------------------------------------------------------------- #
# Fail-closed: the no-false-green rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("missing", ["energy_residual_frac", "clamp_bound",
                                     "cfl_violated"])
def test_a_missing_gate_key_reads_as_violated_not_as_healthy(missing):
    d = _clean()
    d.pop(missing)
    g = sg.standing_gates(d)
    assert g["forward_physical"] is False, (
        f"a march that never reported {missing} must not read as healthy")
    assert missing in g["reason"]
    assert g["complete"] is False


def test_an_empty_march_output_is_not_healthy():
    g = sg.standing_gates({})
    assert g["forward_physical"] is False
    assert g["complete"] is False


def test_a_nan_energy_residual_is_violated():
    g = sg.standing_gates(_clean(energy_residual_frac=float("nan")))
    assert g["forward_physical"] is False
    assert "energy_residual_frac" in g["reason"]


# --------------------------------------------------------------------------- #
# The ceiling, and the Studio's stale-snapshot lesson
# --------------------------------------------------------------------------- #
def test_peak_over_the_ceiling_is_flagged_but_is_not_a_physics_violation():
    """Over the ceiling is a PROCESS verdict (the part cooked), not a broken
    forward. The Studio needs those separated: a drive-limited part can be
    over the ceiling while the numerics are perfect."""
    g = sg.standing_gates(_clean(T_max_c=360.0))
    assert g["peak_over_ceiling"] is True
    assert g["forward_physical"] is True
    assert "peak_over_ceiling" not in g["reason"]


def test_an_explicit_trajectory_peak_overrides_the_end_state_peak():
    """studio3d ab08872: taking a melt-onset snapshot as end-of-run truth
    understated the peak. When the caller knows the true trajectory maximum it
    must win over the march's end-state T_max_c."""
    g = sg.standing_gates(_clean(T_max_c=200.0), peak_T_c=360.0)
    assert g["peak_T_c"] == pytest.approx(360.0)
    assert g["peak_over_ceiling"] is True
    assert g["peak_source"] == "trajectory_maximum"


def test_the_end_state_peak_is_labelled_as_such():
    g = sg.standing_gates(_clean(T_max_c=200.0))
    assert g["peak_source"] == "march_end_state"


# --------------------------------------------------------------------------- #
# Cross-component contract: the ceiling is ONE number
# --------------------------------------------------------------------------- #
def test_the_ceiling_matches_the_studio_lane_value():
    """The Studio owns the process ceiling. Duplicating the literal here would
    let the two lanes drift silently, so the value is checked against theirs
    the same way the STL refusal names are checked against the library."""
    src = (ROOT / "studio3d" / "runner.py").read_text()
    assert f"T_ceiling_C\": {sg.T_CEILING_C}" in src or \
        f"{sg.T_CEILING_C}" in src, "solve3d and studio3d disagree on the ceiling"
    assert sg.T_CEILING_C == 250.0


def test_the_energy_residual_tolerance_is_stated_not_invented():
    assert sg.ENERGY_RESIDUAL_TOL == 1e-6
    # Phase E's own marches land at ~1e-13, four orders inside this
    d = json.loads((ROOT / "solve3d" / "phase_e" / "results"
                    / "phase_e_pyramid.json").read_text())
    got = d["arms"]["uniform_baseline"]["gates"]["energy_residual_frac"]
    assert abs(got) < sg.ENERGY_RESIDUAL_TOL
