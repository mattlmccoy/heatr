"""Badges: registry lookup + per-run flag derivation (graduation spec sec 3).

Registry facts are hand-maintained in badge_registry.json; these tests pin the
lookup contract and the run-flag rules. Fixture values for run flags are taken
from the REAL captured run outputs_eqs/_heatr3d/8a16b9459f84/results.json
(reached_phi90 true, no energy audit keys = legacy run) plus constructed
variants for each tripped gate.
"""
from __future__ import annotations

import pytest

from heatr3d_workbench import badges


def test_registry_loads_and_has_required_classes():
    reg = badges.load_registry()
    for cls in ("thermal", "eqs", "shape", "shrinkage", "solved_map"):
        assert cls in reg["classes"], cls
        entry = reg["classes"][cls]
        assert entry["label"]
        assert entry["evidence"]


def test_exploratory_wording_pinned():
    # Matt-approved Q6 wording, verbatim.
    reg = badges.load_registry()
    assert reg["classes"]["thermal"]["label"] == (
        "exploratory (S1-verified numerics, convergence bands pending)")


def test_solved_map_badge_is_sim_only():
    b = badges.badge_for("solved_map")
    assert "sim-only" in b["label"]


def test_unknown_class_never_renders_healthy():
    b = badges.badge_for("no_such_class")
    assert b["level"] == "unknown"
    assert "unknown" in b["label"].lower()


# ---- per-run flags ---------------------------------------------------------

LEGACY_RESULTS = {  # captured shape: 8a16b9459f84 has no energy audit keys
    "sigma_T": 22.7, "reached_phi90": True, "T_max_C": 335.4,
}


def test_legacy_run_energy_gate_is_not_recorded_not_healthy():
    flags = badges.run_flags(LEGACY_RESULTS)
    e = next(f for f in flags if f["id"] == "energy_gate")
    assert e["state"] == "not_recorded"          # absence is never green


def test_energy_gate_pass_and_fail():
    ok = badges.run_flags({**LEGACY_RESULTS, "energy_residual_frac": 3e-3})
    assert next(f for f in ok if f["id"] == "energy_gate")["state"] == "pass"
    bad = badges.run_flags({**LEGACY_RESULTS, "energy_residual_frac": 0.5})
    assert next(f for f in bad if f["id"] == "energy_gate")["state"] == "fail"


def test_melt_onset_fallback_flag():
    flags = badges.run_flags({**LEGACY_RESULTS, "reached_phi90": False})
    m = next(f for f in flags if f["id"] == "melt_onset_fallback")
    assert m["state"] == "warn"
    ok = badges.run_flags(LEGACY_RESULTS)
    assert next(f for f in ok if f["id"] == "melt_onset_fallback")["state"] == "pass"


def test_clamp_and_cfl_flags_fail_loudly():
    flags = badges.run_flags({**LEGACY_RESULTS, "clamp_bound": True,
                              "cfl_violated": True})
    assert next(f for f in flags if f["id"] == "clamp_bound")["state"] == "fail"
    assert next(f for f in flags if f["id"] == "cfl")["state"] == "fail"


def test_ceiling_flag_at_250c():
    hot = badges.run_flags({**LEGACY_RESULTS, "T_max_C": 260.0})
    assert next(f for f in hot if f["id"] == "ceiling_250c")["state"] == "warn"
    cool = badges.run_flags({**LEGACY_RESULTS, "T_max_C": 200.0})
    assert next(f for f in cool if f["id"] == "ceiling_250c")["state"] == "pass"


def test_any_failed_flag_marks_run_banner():
    flags = badges.run_flags({**LEGACY_RESULTS, "cfl_violated": True})
    assert badges.banner_state(flags) == "fail"
    flags2 = badges.run_flags({**LEGACY_RESULTS, "energy_residual_frac": 1e-3})
    assert badges.banner_state(flags2) in ("ok", "warn")
