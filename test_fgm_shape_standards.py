#!/usr/bin/env python3
"""RED/GREEN tests for the per-shape FGM standards JSON generator.

The GUI needs one small static JSON with, per shape:
  - v_cal_v: calibrated drive voltage (dual-readstate campaign,
    outputs_eqs/geometry_dual_readstate/<shape>.json key "v_cal")
  - quick_look_gain: the window-reselected proportional gain m
    (FGM_WINDOW_RESELECTION.md winners table) + its verdict
  - solve_class: SOLVED / IMPROVED / MATCHED / NOT-RESCUED from the solve
    census (fgm_solve_campaign/out_lib/<shape>.json class block; MATCHED =
    census SOLVED that does not beat the stored historical mask)
  - solve_iou_4bpp: deliverable-arm IoU at grid 120
  - actuator_note: dwell/rotation upgrades (cross, star) and geometry limits

All fixtures are the REAL campaign outputs (data-contract rule: captured, not
invented). Expected numbers below are read from those files/reports.

Run: ./.venv312/bin/python -m pytest test_fgm_shape_standards.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from scripts.analysis.build_fgm_shape_standards import build_standards  # noqa: E402


def _std() -> dict:
    return build_standards(BASE)


def test_all_library_shapes_present():
    std = _std()
    shapes = std["shapes"]
    for s in ("square", "circle", "cross", "star", "T_shape", "L_shape",
              "rectangle", "diamond", "hexagon", "gt_logo"):
        assert s in shapes, f"missing {s}"


def test_v_cal_matches_the_dual_readstate_campaign():
    shapes = _std()["shapes"]
    # Captured from outputs_eqs/geometry_dual_readstate/square.json
    assert abs(shapes["square"]["v_cal_v"] - 2428.1732217858744) < 1e-6
    assert abs(shapes["circle"]["v_cal_v"] - 3399.6347837415706) < 1e-6


def test_quick_look_gain_is_the_window_reselected_winner():
    shapes = _std()["shapes"]
    # Captured from FGM_WINDOW_RESELECTION.md winners table.
    assert abs(shapes["circle"]["quick_look_gain"] - 0.6402) < 1e-4
    assert abs(shapes["cross"]["quick_look_gain"] - 1.1855) < 1e-4
    assert shapes["circle"]["quick_look_verdict"] == "BETTER"
    assert shapes["T_shape"]["quick_look_verdict"] == "HARMFUL"
    # gt_logo was not run (cv2 missing): nullable.
    assert shapes["gt_logo"]["quick_look_gain"] is None


def test_solve_class_mapping_including_matched():
    shapes = _std()["shapes"]
    assert shapes["ellipse"]["solve_class"] == "SOLVED"
    assert shapes["diamond"]["solve_class"] == "IMPROVED"
    assert shapes["cross"]["solve_class"] == "NOT-RESCUED"
    # square: census SOLVED but does not beat the stored historical mask
    assert shapes["square"]["solve_class"] == "MATCHED"
    assert shapes["gt_logo"]["solve_class"] is None


def test_dwell_and_rotation_upgrades_are_recorded():
    shapes = _std()["shapes"]
    assert "0.9829" in (shapes["cross"]["actuator_note"] or "")
    assert "0.953" in (shapes["star"]["actuator_note"] or "")


def test_written_file_round_trips(tmp_path):
    std = _std()
    out = tmp_path / "fgm_shape_standards.json"
    out.write_text(json.dumps(std, indent=1))
    back = json.loads(out.read_text())
    assert back["shapes"]["square"]["solve_class"] == "MATCHED"


if __name__ == "__main__":
    test_all_library_shapes_present()
    print("PASS")
