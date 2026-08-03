"""Phase B Task 0: the pre-registered FD/subgradient protocol.

Pure JSON/threshold checks -- runs in either environment.
The thresholds are the FROZEN 2-D ones (FROZEN_CONVENTIONS_2D.md section 5,
their commit b04e356). This test is what stops them drifting.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
PROTOCOL = RESULTS / "phase_b_protocol.json"


def _doc() -> dict:
    # HARD failure, not a skip: the protocol is a REQUIRED pre-registration
    # artifact. A gate whose protocol quietly does not exist is not a gate.
    assert PROTOCOL.exists(), (
        f"{PROTOCOL.name} missing -- Task 0 must pre-register the protocol "
        "BEFORE any adjoint code exists")
    return json.loads(PROTOCOL.read_text())


def test_epsilon_sweep_is_the_frozen_2d_sweep():
    d = _doc()
    assert d["fd"]["epsilons"] == [1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8]
    assert d["fd"]["scheme"] == "central"
    assert "gate_rho.py:47-52" in d["fd"]["citation"]


def test_pass_standards_are_the_frozen_2d_standards():
    d = _doc()
    assert d["thresholds"]["pass_rel_err"] == 1e-6
    assert d["thresholds"]["subgradient_pass_rel_err"] == 1e-5
    assert d["thresholds"]["transpose_rel_err"] == 1e-10
    for k in ("pass_rel_err", "subgradient_pass_rel_err", "transpose_rel_err"):
        assert "FROZEN_CONVENTIONS_2D.md" in d["thresholds"]["citations"][k]


def test_every_checklist_item_is_ported_with_a_citation():
    d = _doc()
    items = d["checklist"]
    assert len(items) == 10, "FROZEN_CONVENTIONS_2D.md section 5 has 10 items"
    for it in items:
        assert it["citation"].startswith("FROZEN_CONVENTIONS_2D.md"), it
        assert it["status"] in ("ported", "not_applicable_phase_b"), it
        if it["status"] == "not_applicable_phase_b":
            assert it["why"], it


def test_mutants_are_pre_registered_and_must_fail():
    d = _doc()
    names = {m["name"] for m in d["mutation_tests"]}
    assert names == {"renorm_frozen", "adjoint_dropped"}
    for m in d["mutation_tests"]:
        assert m["must"] == "FAIL the FD gate"


def test_fd_case_is_pinned_and_justified_with_measured_evidence():
    """The small case must be MEASURED to cross the melt window and to trigger
    at least two EQS re-solves -- not asserted."""
    d = _doc()
    c = d["fd_case"]
    assert c["measured"]["n_eqs_solves"] >= 3, "1 pre-loop + >= 2 re-solves"
    assert len(c["measured"]["resolve_times_s"]) >= 2
    assert 0.05 < c["measured"]["part_mean_phi"] < 0.95, "melt front inside the part"
    assert c["measured"]["n_nodes_in_melt_window"] > 0, "subgradient content"
    assert c["measured"]["wall_forward_s"] < 60.0, "central differences affordable"
    assert c["justification"]


def test_actuator_and_boundary_conventions_match_the_frozen_2d_ones():
    d = _doc()
    a = d["conventions"]
    assert a["actuator"] == "conductivity_only"
    assert a["eps_channel"] == "OFF"
    assert a["outside_part_saturation"] == 1.0
    assert "section 7" in a["citations"]["actuator"]
    assert "section 4" in a["citations"]["outside_part_saturation"]
