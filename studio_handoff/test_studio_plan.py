#!/usr/bin/env python3
"""RED/GREEN tests for the STUDIO ALPHA plan-card pure logic.

The plan card is what the Import & Plan flow renders after geometry intake:
detected symmetry, anisotropy per actuator mode, the classifier
recommendation with its advisory framing, the calibrated drive voltage, and
a predicted difficulty class. The difficulty bands reuse the classifier's
calibrated thresholds (geometry_actuator: A_MODE_SUFFICES = 0.50,
A_PHYSICAL_LIMIT = 0.80) applied to the BEST mode's residual anisotropy.

Fixture values are CAPTURED from the real keyhole intake record
(fgm_solve_campaign/out_intake/keyhole_novel.json), not invented.

Run: ./.venv312/bin/python -m pytest test_studio_plan.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import studio_plan as sp  # noqa: E402

KEYHOLE_JSON = BASE / "fgm_solve_campaign" / "out_intake" / "keyhole_novel.json"


def _keyhole():
    return json.loads(KEYHOLE_JSON.read_text())


def test_difficulty_bands_reuse_the_classifier_thresholds():
    assert sp.difficulty_class(0.30)["klass"] == "EXPECTED_TRACTABLE"
    assert sp.difficulty_class(0.65)["klass"] == "HARD_ACTUATOR_LOAD_BEARING"
    assert sp.difficulty_class(0.90)["klass"] == "AT_THE_PHYSICAL_LIMIT"
    # each class carries a plain-language basis naming the calibration
    for a in (0.30, 0.65, 0.90):
        basis = sp.difficulty_class(a)["basis"]
        assert "calibrat" in basis.lower()


def test_plan_card_from_the_real_keyhole_record():
    j = _keyhole()
    card = sp.plan_card(intake_info=j["intake"], symmetry=j["symmetry"],
                        anisotropy=j["anisotropy"],
                        recommendation=j["recommendation"], grid=j["grid"])
    assert card["part"]["name"] == "keyhole"
    assert abs(card["power"]["voltage_v"] - 2630.3288450001655) < 1e-6
    assert card["power"]["rf_mode"] == "constant"
    assert card["symmetry"]["point_group"] == "C1v"
    assert card["recommendation"]["actuator_class"] == "MAP_PLUS_MODE"
    assert card["recommendation"]["mode"] == "continuous"
    # advisory framing is mandatory on the card itself
    assert "1 of 2" in card["recommendation"]["advisory_disclaimer"]
    # anisotropy per mode passes through
    assert abs(card["anisotropy_per_mode"]["static"] - 0.6109954709090119) < 1e-9
    # difficulty from the best mode's residual (0.5156 -> load bearing band)
    assert card["difficulty"]["klass"] == "HARD_ACTUATOR_LOAD_BEARING"
    assert card["grid"] == 120
    assert "grid" in card["grid_qualifier"].lower()


def test_thresholds_are_pinned_to_the_classifier():
    sys.path.insert(0, str(BASE / "fgm_solve_campaign"))
    from adjoint2d import geometry_actuator as ga
    assert sp.A_MODE_SUFFICES == ga.A_MODE_SUFFICES
    assert sp.A_PHYSICAL_LIMIT == ga.A_PHYSICAL_LIMIT


def test_smoke_budget_labelling():
    assert sp.budget_label(40.0)["smoke"] is False
    lab = sp.budget_label(6.0)
    assert lab["smoke"] is True
    assert "not a quality solve" in lab["label"]


def test_refusal_message_is_user_facing():
    msg = sp.user_facing_refusal("the imported outline self-intersects: "
                                 "edge 3 ([0 0] to [1 1]) crosses edge 7")
    assert msg["refused"] is True
    assert "self-intersect" in msg["reason"]
    assert "fix" in msg["what_to_do"].lower()
    hole = sp.user_facing_refusal("clockwise interior loop ... hole ... "
                                  "UnsupportedGeometry")
    assert "hole" in hole["reason"]
