"""Unfreezing heatr3d's chamber, on the coupling-knobs pattern (699ed79).

WHY. The Studio adopted chamber tagging (their cf1a203) and correctly noted
that heatr3d's own chamber is pinned at 60 mm by the job wiring, so an
adaptive-chamber solved map would be VERIFIED in a mismatched frame. Their
intake still refuses parts over 60 mm. The Tamper-class fix is incomplete
until the verification side can follow the solve side into a bigger box.

heatr3d.Grid ALREADY takes L; nothing in the solver is hardcoded. What was
frozen is the job wiring, which never passed it. So this is a wiring change,
which is the least invasive point available and keeps heatr3d.py itself
untouched.

THE DISCIPLINE, from the coupling-knobs precedent: the new knob is inert at
its default. `chamber` absent, or "frozen", must reproduce today's 60 mm grid
BIT-IDENTICALLY -- pinned here, not asserted in a comment.

ONE RULE, NO FORKS. The adaptive size must come from the SAME solve3d/chamber.py
both lanes read. A second copy of the rule in the heatr3d path would be a fork
waiting to drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import chamber as ch

import heatr3d as H
import heatr3d_job as J


# --------------------------------------------------------------------------- #
# The default is bit-identical: the knob is inert until asked for
# --------------------------------------------------------------------------- #
def test_the_grid_default_is_still_the_frozen_sixty_millimetre_chamber():
    assert H.Grid(n=48).L == pytest.approx(0.060)


@pytest.mark.parametrize("cfg", [{}, {"chamber": "frozen"},
                                 {"chamber_m": 0.060}])
def test_absent_or_frozen_chamber_reproduces_todays_grid_bit_identically(cfg):
    """`tobytes()` equality, not approx: a resolved chamber that merely rounds
    to 60 mm would silently re-baseline every published heatr3d figure."""
    base = H.Grid(n=48)
    got = J.resolve_grid({**cfg, "n": 48})
    assert got.L == base.L
    assert got.h == base.h
    assert got.x.tobytes() == base.x.tobytes()
    assert got.z.tobytes() == base.z.tobytes()
    assert got.dV == base.dV


def test_the_resolved_chamber_is_reported_so_a_run_records_its_frame():
    spec = J.resolve_chamber({"n": 48})
    assert spec["L_m"] == pytest.approx(0.060)
    assert spec["mode"] == "frozen"
    assert spec["tag"] == "ch060"


# --------------------------------------------------------------------------- #
# The adaptive path, and that it does not fork the rule
# --------------------------------------------------------------------------- #
def test_an_explicit_chamber_is_honoured():
    g = J.resolve_grid({"n": 48, "chamber_m": 0.085})
    assert g.L == pytest.approx(0.085)
    assert g.h == pytest.approx(0.085 / 48)


def test_adaptive_uses_the_shared_rule_not_a_local_copy():
    """Same spans through both entry points must give the same number."""
    spans = (0.0444, 0.0443349, 0.011)
    spec = J.resolve_chamber({"n": 48, "chamber": "adaptive",
                              "bbox_m": list(spans)})
    assert spec["L_m"] == pytest.approx(ch.chamber_for_bbox(*spans))
    assert spec["L_m"] == pytest.approx(0.085)
    assert spec["mode"] == "adaptive"
    assert spec["tag"] == "ch085"


def test_adaptive_without_a_bbox_is_refused_not_guessed():
    """Guessing a bbox would silently size the chamber off the wrong part."""
    with pytest.raises(ValueError):
        J.resolve_chamber({"n": 48, "chamber": "adaptive"})


def test_an_unknown_chamber_mode_is_refused():
    with pytest.raises(ValueError):
        J.resolve_chamber({"n": 48, "chamber": "whatever"})


# --------------------------------------------------------------------------- #
# Chamber-tagged identifiers on the heatr3d side too
# --------------------------------------------------------------------------- #
def test_the_run_id_carries_the_chamber_tag():
    assert "ch085" in J.run_identifier("tamper", {"n": 48, "chamber_m": 0.085})
    assert "ch060" in J.run_identifier("tamper", {"n": 48})
    assert (J.run_identifier("tamper", {"n": 48, "chamber_m": 0.085})
            != J.run_identifier("tamper", {"n": 48}))


# --------------------------------------------------------------------------- #
# Commensurability: a grown chamber at fixed n COARSENS the cells
# --------------------------------------------------------------------------- #
def test_a_grown_chamber_at_fixed_n_coarsens_the_cell_and_says_so():
    """The S2 lesson. Holding n while growing L is the trap: the part loses
    resolution. The helper must report the cell size so a caller cannot miss
    it, and `n_for_h` is the way to hold resolution instead."""
    frozen = J.resolve_grid({"n": 48})
    grown = J.resolve_grid({"n": 48, "chamber_m": 0.085})
    assert grown.h > frozen.h
    assert J.n_for_h(0.085, frozen.h) == 68           # ceil(0.085 / 0.00125)
    matched = J.resolve_grid({"n": J.n_for_h(0.085, frozen.h),
                              "chamber_m": 0.085})
    assert matched.h <= frozen.h
    assert matched.h == pytest.approx(frozen.h, rel=0.02)


def test_n_for_h_respects_the_standard_grid_ceiling():
    """HEATR_STANDARD_PARAMETERS.md caps the grid. Silently exceeding it to
    hold resolution would trade a stated limit for an unstated one."""
    with pytest.raises(ValueError):
        J.n_for_h(0.300, 0.00125)          # would need n = 240, over the cap
