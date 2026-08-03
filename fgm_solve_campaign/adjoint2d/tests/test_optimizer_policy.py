"""RED-first tests for the v2.1.0 optimizer-by-objective-class policy.

THE DECISION BEING WIRED. Two measurements, not one:

  * On the CONSTRAINED (hinge / asymmetric) objective class, the method of
    moving asymptotes beat L-BFGS-B on 4 of 5 shapes at a matched 40
    forward-equivalents (triangle by 21.4 percent, cross 10.4, hexagon 2.1,
    L_shape 0.5; it lost on the square by 2.1 percent).
    DENSE_IFF_INBOUNDS_REPORT.md Section 8.
  * On the SMOOTH melt-region objective under the projection continuation the
    same comparison was NOT a clean win (4 of 6 with two catastrophic
    failures), so the smooth class keeps L-BFGS-B. MMA_RETEST_REPORT.md
    Section 1 item 3.

So the default optimizer is a function of the OBJECTIVE CLASS, and an explicit
configuration value always wins over the policy.

Acronyms: MMA = the method of moving asymptotes (Svanberg 1987). L-BFGS-B =
limited-memory Broyden-Fletcher-Goldfarb-Shanno with box constraints.
"""
from __future__ import annotations

import pytest

from adjoint2d import optimizer_policy as op


# ---------------------------------------------------------------------------
# 1. objective classification
# ---------------------------------------------------------------------------

def test_the_melt_region_objective_is_the_smooth_class():
    assert op.objective_class("j_phi") == "smooth"
    assert op.objective_class("melt") == "smooth"
    assert op.objective_class("shape") == "smooth"


def test_the_asymmetric_and_hinge_objectives_are_the_constrained_class():
    assert op.objective_class("j_asym") == "constrained"
    assert op.objective_class("asym") == "constrained"
    assert op.objective_class("hinge") == "constrained"


def test_an_unknown_objective_name_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError, match="objective"):
        op.objective_class("mystery")


# ---------------------------------------------------------------------------
# 2. the default, per class
# ---------------------------------------------------------------------------

def test_constrained_objectives_default_to_mma():
    r = op.resolve_optimizer("j_asym")
    assert r["optimizer"] == "mma"
    assert r["source"] == "policy"
    assert r["objective_class"] == "constrained"
    assert "4 of 5" in r["reason"]


def test_smooth_objectives_default_to_lbfgsb():
    r = op.resolve_optimizer("j_phi")
    assert r["optimizer"] == "lbfgsb"
    assert r["source"] == "policy"
    assert r["objective_class"] == "smooth"


def test_the_two_classes_get_different_defaults():
    assert (op.resolve_optimizer("j_asym")["optimizer"]
            != op.resolve_optimizer("j_phi")["optimizer"])


# ---------------------------------------------------------------------------
# 3. the configuration override always wins, either way
# ---------------------------------------------------------------------------

def test_an_explicit_override_wins_on_the_constrained_class():
    r = op.resolve_optimizer("j_asym", override="lbfgsb")
    assert r["optimizer"] == "lbfgsb"
    assert r["source"] == "config"


def test_an_explicit_override_wins_on_the_smooth_class():
    r = op.resolve_optimizer("j_phi", override="mma")
    assert r["optimizer"] == "mma"
    assert r["source"] == "config"


def test_auto_is_the_same_as_no_override():
    assert (op.resolve_optimizer("j_asym", override="auto")
            == op.resolve_optimizer("j_asym"))
    assert (op.resolve_optimizer("j_phi", override="auto")
            == op.resolve_optimizer("j_phi"))


def test_an_unknown_override_is_rejected_with_the_allowed_set_named():
    with pytest.raises(ValueError, match="optimizer"):
        op.resolve_optimizer("j_phi", override="newton")


def test_the_allowed_optimizers_are_the_two_that_were_measured():
    assert set(op.OPTIMIZERS) == {"lbfgsb", "mma"}
    assert "auto" in op.OPTIMIZER_CHOICES
    assert set(op.OPTIMIZER_CHOICES) == {"auto", "lbfgsb", "mma"}


def test_every_resolution_carries_a_non_empty_stated_reason():
    for obj in ("j_phi", "j_asym"):
        for ov in (None, "auto", "lbfgsb", "mma"):
            r = op.resolve_optimizer(obj, override=ov)
            assert r["reason"]
            assert r["optimizer"] in op.OPTIMIZERS
