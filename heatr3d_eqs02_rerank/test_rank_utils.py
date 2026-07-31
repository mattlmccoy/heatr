"""Tests for the pure ranking / flip-detection logic (rank_utils)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rank_utils as R  # noqa: E402


def test_rank_ascending_returns_shapes_ordered_smallest_first():
    assert R.rank({"a": 3.0, "b": 1.0, "c": 2.0}) == ["b", "c", "a"]


def test_rank_descending_returns_shapes_ordered_largest_first():
    assert R.rank({"a": 3.0, "b": 1.0, "c": 2.0}, ascending=False) == ["a", "c", "b"]


def test_rank_is_deterministic_on_exact_ties_by_falling_back_to_name():
    assert R.rank({"b": 1.0, "a": 1.0}) == ["a", "b"]


def test_pair_inversions_is_empty_when_the_two_orders_agree():
    assert R.pair_inversions(["a", "b", "c"], ["a", "b", "c"]) == []


def test_pair_inversions_lists_every_pair_whose_order_swapped():
    # legacy a<b<c ; masked c<a<b  -> (a,c) and (b,c) both swap, (a,b) does not
    assert R.pair_inversions(["a", "b", "c"], ["c", "a", "b"]) == [("a", "c"), ("b", "c")]


def test_spearman_is_one_for_identical_orders():
    assert R.spearman(["a", "b", "c", "d"], ["a", "b", "c", "d"]) == pytest.approx(1.0)


def test_spearman_is_minus_one_for_a_fully_reversed_order():
    assert R.spearman(["a", "b", "c", "d"], ["d", "c", "b", "a"]) == pytest.approx(-1.0)


def test_compare_rankings_reports_no_flip_when_orders_match():
    legacy = {"a": 1.0, "b": 2.0}
    masked = {"a": 10.0, "b": 20.0}          # same order, different values
    out = R.compare_rankings(legacy, masked)
    assert out["flipped"] is False
    assert out["inversions"] == []
    assert out["rank_legacy"] == ["a", "b"] and out["rank_masked"] == ["a", "b"]


def test_compare_rankings_flags_a_flip_and_names_the_inverted_pair():
    out = R.compare_rankings({"a": 1.0, "b": 2.0}, {"a": 5.0, "b": 3.0})
    assert out["flipped"] is True
    assert out["inversions"] == [("a", "b")]
    assert out["spearman"] == pytest.approx(-1.0)


def test_compare_rankings_carries_the_per_shape_values_and_deltas():
    out = R.compare_rankings({"a": 2.0, "b": 4.0}, {"a": 3.0, "b": 2.0})
    assert out["values"]["a"] == {"legacy": 2.0, "masked": 3.0,
                                  "delta": 1.0, "rel": pytest.approx(0.5)}
    assert out["values"]["b"]["rel"] == pytest.approx(-0.5)


def test_compare_rankings_rejects_mismatched_shape_sets():
    with pytest.raises(ValueError):
        R.compare_rankings({"a": 1.0}, {"b": 1.0})


def test_compare_rankings_skips_shapes_whose_value_is_missing_in_either_arm():
    out = R.compare_rankings({"a": 1.0, "b": 2.0, "c": None},
                             {"a": 1.0, "b": 2.0, "c": 3.0})
    assert out["rank_legacy"] == ["a", "b"]
    assert out["skipped"] == ["c"]
