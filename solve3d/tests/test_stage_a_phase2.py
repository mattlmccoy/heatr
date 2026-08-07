"""Stage A phase 2 (dopant SHAPE-solve at the chosen fixed drive) pure-logic gates.

These are the FAST, dolfinx-free contracts of the phase-2 driver:
  * the fixed drive is READ from the Phase 1 result artifact (not restated), so
    it cannot drift from the drive Phase 1 certified feasible;
  * the degradation-ceiling verdict is read off the END-STATE true peak and is a
    forward gate only (a peak over the ceiling is infeasible, no dopant gradient
    involved);
  * the emitted output carries the solved map and the recommended drive in the
    Stage A output shape (recommended_power_settings, unchanged 2.0.0).

The FD gate of the adjoint at the fixed drive is NOT a unit test -- it is a
minutes-long dolfinx sweep run as the pre-launch verification gate
(`stage_a_phase2.py --fd-gate`); its numbers are recorded in
solve3d/results/stage_a_phase2_fd_gate.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solve3d import stage_a, stage_a_phase2 as p2

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_chosen_drive_reads_phase1_result_and_matches_recommended():
    """The fixed drive is the one Phase 1 wrote as the ONLY feasible drive."""
    pw = p2.chosen_drive_power_density()
    # exactly the recommended field in the Phase 1 artifact
    j1 = json.loads((RESULTS / "stage_a_task4_square.json").read_text())
    expect = float(j1["recommended_power_settings"]["power_density_w_per_m3"])
    assert pw == expect
    # and it is 0.40 x the Studio baseline, via the shared Stage A mapping
    assert pw == stage_a.recommended_power_settings(0.40)["power_density_w_per_m3"]


def test_chosen_drive_a_is_0p40():
    assert p2.chosen_drive_a() == 0.40


def test_ceiling_verdict_feasible_under_and_infeasible_over():
    """The ceiling is a FORWARD gate on the end-state true peak."""
    ok = p2.ceiling_verdict(240.1, 250.0)
    assert ok["feasible"] is True
    assert abs(ok["margin_c"] - 9.9) < 1e-9
    assert ok["true_peak_c"] == 240.1 and ok["ceiling_c"] == 250.0

    over = p2.ceiling_verdict(255.8, 250.0)
    assert over["feasible"] is False
    assert over["margin_c"] < 0.0


def test_ceiling_verdict_exactly_at_ceiling_is_feasible():
    v = p2.ceiling_verdict(250.0, 250.0)
    assert v["feasible"] is True and v["margin_c"] == 0.0


def test_stage_a_output_shape_carries_map_and_drive():
    """The emitted doc carries the solved map summary + the recommended drive."""
    solve_record = {"arm": "solve_filter_only", "J_asymmetric": 1.23e-4,
                    "part_mean_phi": 0.61, "map_stats": {"mean": 0.7}}
    ceiling_gate = p2.ceiling_verdict(238.0, 250.0)
    doc = p2.stage_a_output(
        part="square", solve_record=solve_record, chosen_drive_a=0.40,
        ceiling_gate=ceiling_gate, holdout_nodes=9000, fd_gate_passed=True)
    # the ONE Studio field, unchanged 2.0.0, equal to the chosen drive
    rps = doc["recommended_power_settings"]
    assert rps["power_density_w_per_m3"] == p2.chosen_drive_power_density()
    assert "rf_mode" not in rps        # no schema change
    # the solved map result is carried, not dropped
    assert doc["solve"]["J_asymmetric"] == 1.23e-4
    # acceptance records the end-state ceiling hold-out verdict
    assert doc["acceptance"]["end_state_ceiling_holdout"]["feasible"] is True
    assert doc["acceptance"]["fd_gate_passed"] is True
    assert doc["stage"] == "A_phase2_dopant_shape_solve"


def test_stage_a_output_honest_when_ceiling_violated():
    """If the solved map's end-state peak is over the ceiling, the doc says so
    and is_shippable is False -- never a false-green shippable map."""
    ceiling_gate = p2.ceiling_verdict(252.0, 250.0)
    doc = p2.stage_a_output(
        part="square", solve_record={"arm": "solve_filter_only"},
        chosen_drive_a=0.40, ceiling_gate=ceiling_gate, holdout_nodes=9000,
        fd_gate_passed=True)
    assert doc["acceptance"]["end_state_ceiling_holdout"]["feasible"] is False
    assert doc["acceptance"]["is_shippable"] is False


def test_stage_a_output_not_shippable_if_fd_gate_failed():
    """A map from an ungated gradient is never shippable, ceiling aside."""
    ceiling_gate = p2.ceiling_verdict(238.0, 250.0)
    doc = p2.stage_a_output(
        part="square", solve_record={"arm": "solve_filter_only"},
        chosen_drive_a=0.40, ceiling_gate=ceiling_gate, holdout_nodes=9000,
        fd_gate_passed=False)
    assert doc["acceptance"]["is_shippable"] is False
