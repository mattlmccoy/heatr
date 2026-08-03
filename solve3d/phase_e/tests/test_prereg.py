"""Phase E opener: the pre-registration artifact, committed before any solve."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
PREREG = RESULTS / "phase_e_preregistration.json"


def _doc() -> dict:
    assert PREREG.exists(), "Task 0 must pre-register BEFORE any Phase E solve"
    return json.loads(PREREG.read_text())


def test_cases_are_the_library_stls_with_verified_dimensions():
    d = _doc()
    assert set(d["cases"]) == {"pyramid", "cube"}
    for name, c in d["cases"].items():
        assert c["source_stl"].startswith("shape_library_3d/stl/")
        assert c["target_volume_mm3"] == 4188.790204786391
        assert c["geometry_verification"]["tolerance_rel"] <= 1e-6


def test_instrument_choice_is_justified_by_s2_evidence():
    d = _doc()
    i = d["instrument"]
    assert i["engine"] == "solve3d"
    assert "commensurab" in i["why_not_heatr3d"].lower()
    assert i["conforming_mesh"] is True


def test_frozen_conventions_are_carried_verbatim():
    d = _doc()
    c = d["conventions"]
    assert c["filter_radius_m"] == 1.0e-3
    assert c["phi_floor"] == 0.85
    assert c["w_out_over_w_in"] == 10.0
    assert c["budget_forward_equivalents_per_arm"] == 40
    assert c["scale_first_step"] is True
    assert c["envelope_stop"] is True
    assert c["symmetric_control_scored"] is True


def test_the_s2_honesty_constraint_is_recorded():
    d = _doc()
    h = d["honesty_constraint_from_s2"]
    assert h["absolute_fidelity_is_ungated"] is True
    assert "solved-vs-uniform" in h["what_is_meaningful"]
    assert h["apex_holdout_may_fail_and_that_is_informative"] is True
    assert "S3" in h["escalation"]


def test_heuristic_arm_has_provenance_and_a_drop_condition():
    d = _doc()
    a = next(x for x in d["arms"] if x["name"] == "heuristic_grading_law")
    assert "6c2aab9" in a["provenance"]["commit"]
    assert a["provenance"]["transfer_drop_condition_rel"] == 0.02
    assert a["provenance"]["never_approximated"]


def test_acceptance_bands_are_per_shape_and_measured_not_assumed():
    d = _doc()
    acc = d["acceptance"]
    assert acc["bands_are_per_shape"] is True
    assert acc["band_rule"]
    assert acc["smoothing"]["tolerance_rel_J"] == 0.10
    assert acc["mesh_holdout"]["band_source"] == "measured_in_campaign"
