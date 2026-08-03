"""Phase C post-hoc re-read: pins the conclusions drawn from stored artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _doc() -> dict:
    p = RESULTS / "phase_c_reread.json"
    assert p.exists(), "run solve3d.phase_c_reread"
    return json.loads(p.read_text())


def test_primary_scores_were_already_read_at_the_asymmetric_argmin():
    i1 = _doc()["item_1_stop_state_reread"]
    assert i1["already_scored_at_asymmetric_argmin"] is True
    assert i1["bound_tightens_via_read_state"] is False
    for k, s in i1["stops"].items():
        assert s["at_horizon_asymmetric"] is False, k


def test_the_objective_moves_the_stop_far_more_than_the_map_does():
    """The 2-D lane's finding, reproduced in 3-D."""
    i1 = _doc()["item_1_stop_state_reread"]
    assert i1["objective_moves_the_stop_steps"] > 10 * max(
        i1["map_moves_the_stop_steps"], 1)


def test_the_asymmetric_objective_stops_earlier_not_later():
    """It stops before the bed grows -- the sign is the whole point."""
    for s in _doc()["item_1_stop_state_reread"]["stops"].values():
        assert s["stop_shift_steps_asym_minus_sym"] < 0


def test_ranking_is_preserved_under_the_3x_weighting_at_both_meshes():
    i2 = _doc()["item_2_weight_sensitivity_by_rescoring"]
    assert i2["ranking_preserved_across_weightings"] is True
    for mesh, r in i2["rows"].items():
        for w in ("w10", "w3", "symmetric"):
            assert r[w]["solved_wins"], (mesh, w)
            assert r[w]["margin"] > 0.0


def test_the_rescoring_limit_is_recorded():
    i2 = _doc()["item_2_weight_sensitivity_by_rescoring"]
    assert i2["read_stage_only"] is True
    assert "does NOT establish what a 3x-weighted SOLVE would find" in i2["limit"]
