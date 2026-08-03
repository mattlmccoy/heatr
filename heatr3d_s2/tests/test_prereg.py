"""S2 Task 0: the pre-registration artifact.

Committed BEFORE any campaign run. This test is what stops the grid ladder,
band rule, PASS thresholds and densify registration from being chosen after
seeing a convergence curve.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
PREREG = RESULTS / "s2_preregistration.json"


def _doc() -> dict:
    assert PREREG.exists(), (
        f"{PREREG.name} missing -- S2 Task 0 must pre-register BEFORE any "
        "campaign run")
    return json.loads(PREREG.read_text())


def test_shapes_and_grid_ladder_are_pinned_with_a_run_order():
    d = _doc()
    assert d["shapes"] == ["circle", "square", "lshape"]
    g = d["grids"]
    assert g["full_physics"] == [48, 64, 80, 96]
    assert g["eqs_only"] == [96, 112, 128]
    # partial completion must still yield a 3-grid band
    assert g["run_order"][:3] == [48, 64, 80]
    assert g["run_order"][-1] == 96
    assert max(g["full_physics"]) <= 96, "EQS-01 full-physics ceiling"
    assert max(g["eqs_only"]) <= 128, "EQS-01 EQS-only ceiling"
    assert all(n < 200 for n in g["full_physics"] + g["eqs_only"])


def test_quantity_hierarchy_matches_the_recorded_objective():
    d = _doc()
    q = d["quantities"]
    verdict = [k for k, v in q.items() if v["role"] == "verdict"]
    coprimary = [k for k, v in q.items() if v["role"] == "co-primary"]
    diagnostic = [k for k, v in q.items() if v["role"] == "diagnostic"]
    assert "sigma_T" in diagnostic and "sigma_T" not in verdict
    assert coprimary == ["t90"]
    for k in ("iou_phi0p8", "iou_phi0p9", "out_of_part_melt_fraction",
              "front_position_mm"):
        assert k in verdict, k


def test_band_rule_and_pass_thresholds_are_frozen_with_numbers():
    d = _doc()
    b = d["band_rule"]
    assert b["safety_factor"] == 1.5
    assert "finest" in b["statement"].lower()
    assert d["convergence"]["min_grids_for_a_claim"] >= 3
    for k, v in d["quantities"].items():
        if v["role"] == "diagnostic":
            assert v.get("pass_ceiling") is None, k
        else:
            assert isinstance(v["pass_ceiling"], float) and v["pass_ceiling"] > 0, k
            assert v["pass_ceiling_precedent"], k


def test_pass_criteria_reject_divergence_explicitly():
    d = _doc()
    c = d["pass_criteria"]
    assert c["requires_non_increasing_successive_changes"] is True
    assert c["diverging_is_FAIL"] is True
    assert c["richardson_is_reported_not_truth"] is True


def test_dual_read_states_are_defined_precisely():
    d = _doc()
    r = d["read_states"]
    assert set(r) == {"melt_onset", "heating_fixed_time"}
    assert r["melt_onset"]["definition"]
    # the fixed-time read must be BEFORE melt onset on every grid, else it is
    # not a heating-phase read
    for shape, t in r["heating_fixed_time"]["t_ref_s"].items():
        assert t > 0, shape
    assert r["heating_fixed_time"]["must_precede_melt_onset"] is True


def test_gauge_arms_and_the_decision_rule_are_pre_registered():
    d = _doc()
    g = d["gauge"]
    assert set(g["arms"]) == {"cell_centred_current", "face_gauge"}
    assert "grid-invarian" in g["decision_rule"].lower()
    assert g["observable"] == "raw_absorbed_power_pre_renormalization"
    assert g["why_not_total_absorbed_power"]
    assert g["default_flip_requires_matt"] is True


def test_densify_march_is_registered_with_a_zero_control_and_validity_bound():
    d = _doc()
    t = d["densify_march"]
    assert 0.0 in t["sigma_density_coeff_arms"]
    assert t["validity_bound_abs_a_per_K"] == 0.0044
    assert t["zero_control_must_reproduce"]
    assert t["eqs_update_interval_s"] > 0
    for gate in ("energy_audit", "clamp_bound", "cfl"):
        assert gate in t["standing_gates"]
    assert t["anti_circularity"]


def test_partial_completion_policy_exists():
    d = _doc()
    p = d["partial_completion_policy"]
    assert "NOT_RUN" in p["unrun"]
    assert p["min_grids_reported"] >= 3
