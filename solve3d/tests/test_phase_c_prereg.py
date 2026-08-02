"""Phase C Task 0: the PRE-REGISTRATION artifact.

Committed BEFORE any objective or solve code exists. This test is what stops
the campaign's budgets, arms, weights and acceptance bands from being chosen
after seeing a result.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
PREREG = RESULTS / "phase_c_preregistration.json"


def _doc() -> dict:
    assert PREREG.exists(), (
        f"{PREREG.name} missing -- Task 0 must pre-register BEFORE any "
        "objective/solve code exists")
    return json.loads(PREREG.read_text())


def test_case_and_budget_are_pinned_with_measured_cost():
    d = _doc()
    c = d["case"]
    assert c["shape"] == "circle"
    b = d["budget"]
    assert b["forward_equivalents_per_arm"] == 40
    # the adjoint cost must be MEASURED at solve scale, not inherited
    m = b["measured_at_solve_scale"]
    assert m["wall_forward_s"] > 0 and m["wall_gradient_s"] > 0
    assert m["gradient_forward_equivalents"] > 0
    assert b["gradient_evaluations_per_arm"] == int(
        40 // m["per_gradient_eval_forward_equivalents"])


def test_both_objective_weightings_are_pre_registered():
    d = _doc()
    o = d["objective"]
    assert o["symmetric_control"]["formula"]
    a = o["asymmetric"]
    assert a["phi_floor"] == 0.85
    assert a["w_out_over_w_in"] == 10.0
    assert a["sensitivity_arm_w_out_over_w_in"] == 3.0
    assert a["justification"] and a["not_tuned"] is True
    assert "d298c6d" in o["source"]


def test_arms_are_enumerated_with_a_priority_order():
    d = _doc()
    names = [a["name"] for a in d["arms"]]
    for required in ("uniform_baseline", "inversion_map", "solve_filter_only",
                     "solve_projection_beta_continuation"):
        assert required in names, required
    prios = [a["priority"] for a in d["arms"]]
    assert prios == sorted(prios), "arms must carry a run order"
    assert d["budget"]["unrun_arms_policy"]


def test_acceptance_bands_state_their_rule_and_their_measured_source():
    d = _doc()
    g = d["acceptance"]
    h = g["mesh_holdout"]
    assert h["solve_mesh"] == "phase_a_coarse" and h["score_mesh"] == "phase_a_mid"
    assert h["band_rule"]
    assert h["band_source"].endswith("phase_a_shape_gate.json")
    assert h["safety_factor"] == 1.5
    for k in ("jaccard_dist_phi0p9", "front_ssd_mm"):
        assert h["bands"][k] > 0.0, k
    # J has no Phase A measurement, so its band rule must be stated and its
    # number deferred to a measurement made BEFORE any solved map is scored
    assert h["bands_deferred"]["J_rel"]["rule"]
    assert h["bands_deferred"]["J_rel"]["measured_from"] == "uniform_arm_own_mid_move"
    s = g["smoothing_robustness"]
    assert s["tolerance_rel_J"] == 0.10
    assert "FROZEN_CONVENTIONS_2D" in s["citation"]
    assert 0.0 < s["perturbation_radius_m"] < d["design_chain"]["filter_radius_m"]


def test_solved_label_rule_is_explicit_and_conjunctive():
    d = _doc()
    r = d["acceptance"]["solved_label_rule"]
    assert r["all_of"] and len(r["all_of"]) >= 3
    assert r["null_result_is_a_valid_finding"] is True


def test_inversion_arm_provenance_and_its_drop_condition():
    d = _doc()
    arm = next(a for a in d["arms"] if a["name"] == "inversion_map")
    p = arm["provenance"]
    assert "FGM_BENEFIT_RERUN.md" in p["report"]
    assert p["artifact_status"] == "not_saved_regenerated_from_make_fgm"
    assert p["reproduction_check"]["target_mean_sat"] == 0.4074
    assert p["drop_condition"]
    assert p["transfer"]["method"] == "trilinear_then_clip"
    assert "robust.resample_map" in p["transfer"]["citation"]
