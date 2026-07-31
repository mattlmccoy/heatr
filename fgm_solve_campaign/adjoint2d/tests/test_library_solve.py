"""Red-first tests for the new pure logic of the shape-library solve.

Only pure logic is unit-tested here: the double-pass trigger, the best-arm
selection, and the shape classification. The solve itself is an integration run
and is verified by its own energy-residual and clip gates.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from adjoint2d import library_solve as ls

MAIN = Path("/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research"
            "/research/binderjet/code/geo-prewarp")


def test_needs_double_pass_triggers_strictly_above_fifteen_percent():
    assert ls.needs_double_pass(15.1) is True
    assert ls.needs_double_pass(15.0) is False
    assert ls.needs_double_pass(0.0) is False


def test_best_arm_by_J_picks_the_minimum():
    rows = {"a": {"J": 10.0}, "b": {"J": 3.0}, "c": {"J": 7.0}}
    assert ls.best_arm_by_J(rows) == "b"


def test_best_arm_by_J_returns_none_on_empty():
    assert ls.best_arm_by_J({}) is None


def test_classify_solved_requires_iou_at_least_point_nine_five():
    assert ls.classify(iou_solved=0.95, J_solved=1.0, iou_hist=0.99, J_hist=0.5) == "SOLVED"
    assert ls.classify(iou_solved=0.9499, J_solved=1.0, iou_hist=0.99, J_hist=0.5) != "SOLVED"


def test_classify_improved_when_solved_beats_history_on_both_metrics():
    # 5 percent better on J and 2 IoU points better
    assert ls.classify(iou_solved=0.80, J_solved=90.0,
                       iou_hist=0.78, J_hist=100.0) == "IMPROVED"


def test_classify_matched_inside_the_tie_band():
    assert ls.classify(iou_solved=0.700, J_solved=100.4,
                       iou_hist=0.699, J_hist=100.0) == "MATCHED"


def test_classify_not_rescued_when_history_wins_and_iou_is_low():
    assert ls.classify(iou_solved=0.50, J_solved=200.0,
                       iou_hist=0.60, J_hist=150.0) == "NOT RESCUED"


def test_shape_config_is_deterministic_and_exists():
    p = ls.shape_config("square")
    assert p.exists()
    assert p == ls.shape_config("square")


def test_stored_mask_catalog_is_deduplicated_and_nonempty():
    cat = ls.stored_mask_catalog("square")
    assert len(cat) >= 15
    labels = [c[0] for c in cat]
    assert len(labels) == len(set(labels))
    paths = [c[1] for c in cat]
    assert len(paths) == len(set(paths))
    assert any("oldgrid" in lb for lb in labels)
    assert any("cal" in lb for lb in labels)


def test_gt_logo_is_reported_as_skipped_not_silently_dropped():
    assert "gt_logo" not in ls.SHAPES
    assert ls.GT_LOGO_SKIP_REASON
